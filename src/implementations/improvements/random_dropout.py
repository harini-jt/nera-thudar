import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class RandomFeatureDropout(BaseEstimator, TransformerMixin):
    """Randomly drops a fraction of feature columns.

    The columns to keep are chosen once in `fit` (seeded), and the same
    columns are kept for train and test.
    """

    def __init__(self, rate: float = 0.2, seed: int = 0):
        self.rate = rate
        self.seed = seed

    def fit(self, X, y=None):
        if not 0 <= self.rate < 1:
            raise ValueError(f"rate must be in [0, 1), got {self.rate}")
        n_features = np.asarray(X).shape[1]
        n_keep = max(1, round(n_features * (1 - self.rate)))
        rng = np.random.default_rng(self.seed)
        self.keep_idx_ = np.sort(rng.choice(n_features, n_keep, replace=False))
        self.n_features_in_ = n_features
        return self

    def transform(self, X):
        check_is_fitted(self, "keep_idx_")
        X = np.asarray(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}"
            )
        return X[:, self.keep_idx_]
