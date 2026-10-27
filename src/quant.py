import numpy as np
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class Quant(BaseEstimator, TransformerMixin):
    def __init__(self, depth: int = 6, div: int = 4, batch_size: int | None = 1024):
        self.depth = depth
        self.div = div
        self.batch_size = batch_size

    @staticmethod
    def _to_tensor(X):
        if not isinstance(X, torch.Tensor):
            X = np.asarray(X)
        X = torch.as_tensor(X, dtype=torch.float32)
        return X.unsqueeze(1) if X.ndim == 2 else X

    @staticmethod
    def _generate_representations(X: torch.Tensor):
        first_diff = X.diff()
        first_diff_padded = F.pad(first_diff, (2, 2), mode="replicate")
        first_diff_smoothed = F.avg_pool1d(first_diff_padded, kernel_size=5, stride=1)
        second_diff = torch.diff(X, n=2)
        series_fft = torch.fft.rfft(X).abs()

        return [X, first_diff_smoothed, second_diff, series_fft]

    def _make_intervals(self, input_length: int):
        exponent = min(self.depth, int(np.log2(input_length)) + 1)
        intervals = []

        for dyad_step in (2 ** np.arange(exponent)).tolist():
            indices = torch.linspace(0, input_length, dyad_step + 1).long()
            dyad_intervals = torch.stack((indices[:-1], indices[1:]), dim=1)
            intervals.append(dyad_intervals)

            if dyad_step > 1 and dyad_intervals.diff().median() > 1:
                shift = int(np.ceil(input_length / (dyad_step * 2)))
                shifted_intervals = dyad_intervals[:-1] + shift
                intervals.append(shifted_intervals)

        return torch.cat(intervals)

    def _extract_quantiles(self, X_slice):
        size = X_slice.shape[-1]
        if size == 1:
            return X_slice.flatten(start_dim=1)

        num_quantiles = 1 + (size - 1) // self.div
        if num_quantiles == 1:
            quantile = torch.quantile(X_slice, torch.tensor([0.5]), dim=-1).permute(
                1, 2, 0
            )
            return quantile.flatten(start_dim=1)

        quantile_points = torch.linspace(0, 1, num_quantiles)
        quantiles = torch.quantile(X_slice, quantile_points, dim=-1).permute(1, 2, 0)
        interval_mean = X_slice.mean(dim=-1, keepdim=True)
        quantiles[..., 1::2] -= interval_mean

        return quantiles.flatten(start_dim=1)

    def _transform_batch(self, X):
        feats = []
        for rep, intervals in zip(self._generate_representations(X), self.intervals_):
            for interval_start, interval_end in intervals.tolist():
                feats.append(
                    self._extract_quantiles(rep[..., interval_start:interval_end])
                )
        return torch.cat(feats, dim=-1)

    def fit(
        self, X: torch.Tensor | np.ndarray, y: torch.Tensor | np.ndarray | None = None
    ):
        X = self._to_tensor(X)
        reps = self._generate_representations(X[:1])
        self.intervals_ = [self._make_intervals(rep.shape[-1]) for rep in reps]
        return self

    def transform(self, X: torch.Tensor | np.ndarray):
        check_is_fitted(self, "intervals_")
        X = self._to_tensor(X)
        batch_size = self.batch_size or len(X)
        outputs = [
            self._transform_batch(X[i : i + batch_size])
            for i in range(0, len(X), batch_size)
        ]
        return torch.cat(outputs).numpy()

    def fit_transform(
        self, X: torch.Tensor | np.ndarray, y: torch.Tensor | np.ndarray | None = None
    ):
        return self.fit(X, y).transform(X)
