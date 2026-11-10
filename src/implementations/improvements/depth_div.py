from src.implementations.quant import Quant


class QuantWithDepthDiv(Quant):
    """Per-depth quantile divisor, e.g. {1: 2, 2: 2, 3: 2, 5: 8, 6: 8}.

    Smaller div means more quantiles (m/2 > m/4 > m/8). Unlisted depths use `div`.
    This changes the feature count, so compare against a matched-size control.
    """

    def __init__(self, depth=6, div=4, batch_size=1024, depth_div=None):
        super().__init__(depth=depth, div=div, batch_size=batch_size)
        self.depth_div = depth_div

    def _div_for(self, depth: int) -> int:
        return (self.depth_div or {}).get(depth, self.div)
