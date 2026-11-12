import torch

from src.implementations.quant import Quant


class QuantWithIntervalStats(Quant):
    """Quant plus two extra features per interval: standard deviation and slope.

    Quantiles ignore the order of the values in an interval. The slope (least
    squares fit against time) adds the trend direction; the standard deviation
    adds the spread as a single number. Intervals of length 1 get no extras.
    """

    def _extract_quantiles(self, X_slice, div: int):
        q = super()._extract_quantiles(X_slice, div)
        size = X_slice.shape[-1]
        if size == 1:
            return q
        std = X_slice.std(dim=-1, correction=0)
        t = torch.arange(size, dtype=X_slice.dtype) - (size - 1) / 2
        centred = X_slice - X_slice.mean(dim=-1, keepdim=True)
        slope = (centred * t).sum(dim=-1) / (t * t).sum()
        return torch.cat([q, std, slope], dim=-1)
