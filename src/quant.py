import numpy as np
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import ExtraTreesClassifier


class Quant(BaseEstimator, TransformerMixin):
    def __init__(self, depth: int = 6, div: int = 4):
        self.depth = depth
        self.div = div
        self.intervals_cache = {}  # Stores the intervals created for each dataset

    def _generate_representations(self, X: torch.Tensor):
        first_diff = torch.diff(X, dim=-1)
        first_diff_padded = F.pad(first_diff, (2, 2), mode="replicate")
        first_diff_smoothed = F.avg_pool1d(first_diff_padded, kernel_size=5, stride=1)
        second_diff = torch.diff(X, n=2, dim=-1)
        series_fft = torch.fft.rfft(X, dim=-1).abs()

        return [X, first_diff_smoothed, second_diff, series_fft]

    def _make_intervals(self, input_length):
        exponent = min(self.depth, int(np.log2(input_length)) + 1)
        intervals = []

        for dyad_step in 2 ** np.arange(exponent):
            indices = torch.linspace(0, input_length, dyad_step + 1).long()
            dyad_intervals = torch.stack((indices[:-1], indices[1:]), dim=1)
            intervals.append(dyad_intervals)

            if dyad_step > 1 and dyad_intervals.diff().median() > 1:
                shift = np.ceil(input_length / (dyad_step * 2))
                shifted_intervals = dyad_intervals[:-1] + shift
                intervals.append(shifted_intervals)

        return torch.cat(intervals)

    def _extract_quantiles(self, X_slice):
        size = X_slice.shape[-1]
        if size == 1:
            return X_slice.flatten(start_dim=1)

        num_quantiles = 1 + (size - 1) // self.div
        if num_quantiles == 1:
            return X_slice.median(dim=-1).values.flatten(start_dim=1)

        quantile_points = torch.linspace(0, 1, num_quantiles)
        quantiles = torch.quantile(X_slice, quantile_points, dim=-1).permute(1, 2, 0)
        interval_mean = X_slice.mean(dim=-1, keepdim=True)
        quantiles[..., 1::2] -= interval_mean

        return quantiles.flatten(start_dim=1)

    def fit_transform(
        self, X: torch.Tensor | np.ndarray, y: torch.Tensor | np.ndarray | None = None
    ):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        if X.ndim == 2:
            X = X.unsqueeze(1)

        # Generate representations
        representations = self._generate_representations(X)

        all_features = []
        for rep in representations:
            # Create intervals and store in cache
            length = rep.shape[-1]
            if length not in self.intervals_cache:
                self.intervals_cache[length] = self._make_intervals(length)
            intervals = self.intervals_cache[length]

            # Extract quantiles for each interval
            rep_features = []
            for interval_start, interval_end in intervals:
                X_slice = rep[..., int(interval_start), int(interval_end)]
                quantiles = self._extract_quantiles(X_slice)
                rep_features.append(quantiles)

            all_features.append(torch.cat(rep_features, dim=-1))

        # Concatenate all the features
        return torch.cat(all_features, dim=-1).numpy()
