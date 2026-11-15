import numpy as np
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class Quant(BaseEstimator, TransformerMixin):
    """Quant feature transformer (Dempster et al.). Deterministic: no randomness here."""

    def __init__(self, depth: int = 6, div: int = 4, batch_size: int | None = 1024):
        self.depth = depth
        self.div = div
        self.batch_size = batch_size

    # ---- hooks that improvements override ----
    def _repeats_for(self, depth: int) -> int:
        """How many copies of each feature block at this interval depth (sampling-frequency weighting)."""
        return 1

    def _div_for(self, depth: int) -> int:
        """Quantile divisor at this depth (quantiles per interval = 1 + (m - 1) // div)."""
        return self.div

    # ---- internals ----
    @staticmethod
    def _to_tensor(X):
        """Coerce 2D (N, L) or 3D (N, C, L) array-like input into a float32 3D PyTorch tensor."""
        if not isinstance(X, torch.Tensor):
            X = np.asarray(X)
        X = torch.as_tensor(X, dtype=torch.float32)
        return X.unsqueeze(1) if X.ndim == 2 else X

    @staticmethod
    def _generate_representations(X: torch.Tensor):
        """Derive the four signal views used by QUANT: raw, smoothed 1st diff, 2nd diff, and FFT magnitude."""
        # Pad replicate edges by 2 before a 5-point moving average to smooth high-frequency noise in 1st diff
        first_diff = F.avg_pool1d(F.pad(X.diff(), (2, 2), mode="replicate"), 5, 1)
        second_diff = torch.diff(X, n=2)
        series_fft = torch.fft.rfft(X).abs()
        return [X, first_diff, second_diff, series_fft]

    def _make_intervals(self, input_length: int):
        """Build hierarchical dyadic and half-step shifted [start, end) intervals up to `self.depth`."""
        # Cap tree depth so intervals don't split beyond single time steps
        exponent = min(self.depth, int(np.log2(input_length)) + 1)
        intervals, depths = [], []
        for level, dyad_step in enumerate((2 ** np.arange(exponent)).tolist(), start=1):
            # Partition [0, input_length] into `dyad_step` contiguous bins
            idx = torch.linspace(0, input_length, dyad_step + 1).long()
            dyadic = torch.stack((idx[:-1], idx[1:]), dim=1)
            intervals.append(dyadic)
            depths.append(torch.full((len(dyadic),), level))

            # Add half-interval shifted windows at depth >= 2 (when interval width > 1) to reduce boundary effects
            if dyad_step > 1 and dyadic.diff().median() > 1:
                shift = int(np.ceil(input_length / (dyad_step * 2)))
                shifted = dyadic[:-1] + shift
                intervals.append(shifted)
                depths.append(torch.full((len(shifted),), level))
        return torch.cat(intervals), torch.cat(depths)

    def _extract_quantiles(self, X_slice, div: int):
        """Extract evenly spaced quantiles on an interval, subtracting the interval mean from odd quantiles."""
        size = X_slice.shape[-1]
        if size == 1:
            return X_slice.flatten(start_dim=1)
        num_quantiles = 1 + (size - 1) // div
        if num_quantiles == 1:
            # Degenerate short interval: fall back to the median (0.5 quantile)
            q = torch.quantile(X_slice, torch.tensor([0.5]), dim=-1).permute(1, 2, 0)
            return q.flatten(start_dim=1)

        points = torch.linspace(0, 1, num_quantiles)
        # torch.quantile places quantile dimension first (Q, N, C); permute to (N, C, Q)
        q = torch.quantile(X_slice, points, dim=-1).permute(1, 2, 0)
        # Center alternating quantiles (indices 1, 3, 5, ...) around the interval mean to capture shape vs. offset
        q[..., 1::2] -= X_slice.mean(dim=-1, keepdim=True)
        return q.flatten(start_dim=1)

    def _transform_batch(self, X):
        """Extract and concatenate quantile features across all representations and intervals for one batch."""
        feats = []
        reps = self._generate_representations(X)
        for rep, intervals, depths in zip(reps, self.intervals_, self.interval_depths_):
            for (start, end), depth in zip(intervals.tolist(), depths.tolist()):
                copies = self._repeats_for(depth)
                if copies == 0:
                    continue
                f = self._extract_quantiles(rep[..., start:end], self._div_for(depth))
                feats.extend([f] * copies)
        return torch.cat(feats, dim=-1)

    # ---- sklearn API ----
    def fit(self, X, y=None):
        """Precompute interval boundaries for each of the four representations based on series length."""
        X = self._to_tensor(X)
        # Dry-run representations on a single sample to capture each view's exact output length
        reps = self._generate_representations(X[:1])
        made = [self._make_intervals(r.shape[-1]) for r in reps]
        self.intervals_ = [m[0] for m in made]
        self.interval_depths_ = [m[1] for m in made]
        return self

    def transform(self, X):
        """Transform time series in mini-batches to bound peak memory usage, returning a NumPy array."""
        check_is_fitted(self, "intervals_")
        X = self._to_tensor(X)
        bs = self.batch_size or len(X)
        return torch.cat(
            [self._transform_batch(X[i : i + bs]) for i in range(0, len(X), bs)]
        ).numpy()

    def fit_transform(self, X, y=None):
        """Fit interval geometry and transform X in one step."""
        return self.fit(X, y).transform(X)
