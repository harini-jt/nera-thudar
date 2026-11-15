import math
import numpy as np
import torch
import torch.nn.functional as F


def make_intervals(input_length, depth):
    """Build dyadic [start, end) interval pairs plus half-step shifted intervals for levels > 1."""
    # Cap tree depth so intervals do not subdivide below single time steps
    exponent = min(depth, int(np.log2(input_length)) + 1)
    intervals = []

    for n in 2 ** torch.arange(exponent):
        # Partition [0, input_length] into `n` contiguous bins
        indices = torch.linspace(0, input_length, n + 1).long()
        intervals_n = torch.stack((indices[:-1], indices[1:]), 1)
        intervals.append(intervals_n)

        # Add half-interval shifted windows at depth >= 2 (when median width > 1) to reduce boundary effects
        if n > 1 and intervals_n.diff().median() > 1:
            shift = int(math.ceil(input_length / n / 2))
            intervals.append((intervals_n[:-1] + shift))

    return torch.cat(intervals)


def f_quantile(X, div=4):
    """Extract evenly spaced quantiles across the last axis, subtracting the interval mean from odd quantiles."""
    n = X.shape[-1]
    if n == 1:
        # Degenerate width-1 interval: flatten (N, C, 1) to (N, 1, C)
        return X.view(X.shape[0], 1, X.shape[1] * X.shape[2])
    else:
        num_quantiles = 1 + (n - 1) // div
        if num_quantiles == 1:
            # Short interval with fewer than `div + 1` points: fall back to the median (0.5 quantile)
            quantiles = X.quantile(torch.tensor([0.5]), dim=-1).permute(1, 2, 0)
            return quantiles.view(
                quantiles.shape[0], 1, quantiles.shape[1] * quantiles.shape[2]
            )
        else:
            # Compute `num_quantiles` evenly spaced quantiles; permute (Q, N, C) -> (N, C, Q)
            quantiles = X.quantile(torch.linspace(0, 1, num_quantiles), dim=-1).permute(
                1, 2, 0
            )
            # Center odd-indexed quantiles (1, 3, 5, ...) around the interval mean to separate shape from offset
            quantiles[..., 1::2] = quantiles[..., 1::2] - X.mean(-1, keepdims=True)
            return quantiles.view(
                quantiles.shape[0], 1, quantiles.shape[1] * quantiles.shape[2]
            )


class IntervalModel:
    """Extracts quantile features across all dyadic and shifted intervals for a single representation."""

    def __init__(self, input_length, depth=6, div=4):
        assert div >= 1
        assert depth >= 1

        self.div = div

        # Precompute [start, end) interval boundaries for the given representation length
        self.intervals = make_intervals(
            input_length=input_length,
            depth=depth,
        )

    def fit(self, X, Y):
        """No-op for interface compatibility; interval geometry is determined at initialization."""
        pass

    def transform(self, X):
        """Slice X along each precomputed interval, extract quantiles, and concatenate along the feature axis."""
        features = []
        for a, b in self.intervals:
            features.append(f_quantile(X[..., a:b], div=self.div).squeeze(1))
        return torch.cat(features, -1)

    def fit_transform(self, X, Y):
        """Fit (no-op) and transform X in one call."""
        self.fit(X, Y)
        return self.transform(X)


class QuantBase:
    """Reference QUANT feature extractor (Dempster et al.) over four time-series representations."""

    def __init__(self, depth=6, div=4):
        assert depth >= 1
        assert div >= 1

        self.depth = depth
        self.div = div

        # Four signal views: raw series, smoothed 1st diff, 2nd diff, and FFT magnitude spectrum
        self.representation_functions = (
            lambda X: X,
            lambda X: F.avg_pool1d(F.pad(X.diff(), (2, 2), "replicate"), 5, 1),
            lambda X: X.diff(n=2),
            lambda X: torch.fft.rfft(X).abs(),
        )
        self.models = {}

        self.fitted = False

    def transform(self, X):
        """Apply each representation transform and concatenate interval quantiles across all four views."""
        assert self.fitted, "not fitted"
        features = []
        for index, function in enumerate(self.representation_functions):
            Z = function(X)
            features.append(self.models[index].transform(Z))
        return torch.cat(features, -1)

    def fit_transform(self, X, Y):
        """Initialize an IntervalModel per representation length and extract concatenated training features."""
        features = []
        for index, function in enumerate(self.representation_functions):
            Z = function(X)
            self.models[index] = IntervalModel(
                input_length=Z.shape[-1], depth=self.depth, div=self.div
            )

            features.append(self.models[index].fit_transform(Z, Y))

        self.fitted = True
        return torch.cat(features, -1)
