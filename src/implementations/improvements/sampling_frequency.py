from src.implementations.quant import Quant


class QuantWithSamplingFrequency(Quant):
    """Duplicates the features of chosen depths so the forest samples them more often.

    depth_repeats: {depth: copies}, e.g. {1: 2, 2: 2}. Unlisted depths keep one copy.
    """

    def __init__(self, depth=6, div=4, batch_size=1024, depth_repeats=None):
        super().__init__(depth=depth, div=div, batch_size=batch_size)
        # Store raw argument untouched to satisfy scikit-learn's BaseEstimator.__init__ cloning contract
        self.depth_repeats = depth_repeats

    def fit(self, X, y=None):
        """Validate repeat counts (0 drops a depth; >= 1 duplicates it) before fitting interval geometry."""
        # Note: in Python `isinstance(True, int)` is True; if booleans could ever slip in, check `not isinstance(r, bool)`
        for d, r in (self.depth_repeats or {}).items():
            if not (isinstance(r, int) and r >= 0):
                raise ValueError(f"depth_repeats[{d}] must be a non-negative int")
        return super().fit(X, y)

    def _repeats_for(self, depth: int) -> int:
        """Override base repeat lookup: return configured copy count for `depth`, defaulting to 1."""
        return (self.depth_repeats or {}).get(depth, 1)
