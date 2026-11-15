import torch

from src.implementations.quant import Quant


class QuantWithIntervalStats(Quant):
    """Quant plus two extra features per interval: standard deviation and slope.

    Quantiles ignore the order of the values in an interval. The slope (least
    squares fit against time) adds the trend direction; the standard deviation
    adds the spread as a single number. Intervals of length 1 get no extras.
    """

    def _extract_quantiles(self, X_slice, div: int):
        """Append interval standard deviation and OLS linear slope to the base quantile vector."""
        # Extract base flattened quantiles of shape (batch, channels * num_quantiles)
        q = super()._extract_quantiles(X_slice, div)
        size = X_slice.shape[-1]
        if size == 1:
            return q

        # Population std (correction=0) across the interval; shape: (batch, channels)
        std = X_slice.std(dim=-1, correction=0)

        # Ordinary least squares slope via centered covariance: sum((y - y_mean) * (t - t_mean)) / sum((t - t_mean)^2)
        t = torch.arange(size, dtype=X_slice.dtype) - (size - 1) / 2
        centred = X_slice - X_slice.mean(dim=-1, keepdim=True)
        slope = (centred * t).sum(dim=-1) / (t * t).sum()

        # Concatenate along the feature dimension: [quantiles..., std (per channel), slope (per channel)]
        return torch.cat([q, std, slope], dim=-1)
