import numpy as np
import pandas as pd

# Regime discretization boundaries (right-inclusive intervals) for dataset stratification
LENGTH_BINS = [0, 100, 300, 1000, np.inf]
LENGTH_LABELS = ["<=100", "101-300", "301-1000", ">1000"]
TRAIN_BINS = [0, 50, 200, 1000, np.inf]
TRAIN_LABELS = ["<=50", "51-200", "201-1000", ">1000"]

# Composite primary key uniquely identifying a single benchmark observation
KEY = ["dataset", "resample_id", "method", "seed"]


def bucket(values, bins, labels):
    """Discretize numeric series lengths or sample counts into labeled regime categories."""
    return pd.cut(values, bins=bins, labels=labels, right=True)


def load_results(paths) -> pd.DataFrame:
    """Concatenate one or more result CSVs and fail fast on missing schema columns or duplicate runs."""
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    missing = {"dataset", "method", "seed", "accuracy"} - set(df.columns)
    if missing:
        raise ValueError(f"Results are missing columns: {sorted(missing)}")

    # Guard against accidentally passing overlapping CSVs that would double-count datasets/seeds
    if set(KEY) <= set(df.columns):
        dup = df.duplicated(KEY, keep=False)
        if dup.any():
            raise ValueError(
                "Duplicate (dataset, resample_id, method, seed) rows:\n"
                + df.loc[dup, KEY].head(10).to_string(index=False)
            )
    return df


def load_paper_rows(path, resample: int) -> pd.DataFrame:
    """Official per-resample accuracies (wide: dataset, 0, 1, ...) as pseudo-method 'paper'."""
    wide = pd.read_csv(path)
    # Reshape wide resample columns into the long format matching `src.runner.COLUMNS`
    long = wide.melt(id_vars="dataset", var_name="resample_id", value_name="accuracy")
    long = long[long["resample_id"].astype(str) == str(resample)].copy()
    long["resample_id"] = resample
    long["method"], long["seed"] = "paper", 0
    return long[["dataset", "resample_id", "method", "seed", "accuracy"]]


def outcome_labels(delta, metric: str, tol: float):
    """Classify raw deltas (`improved - baseline`) into 'improved', 'regressed', or 'draw' using metric polarity."""
    from src.metrics import LOWER_IS_BETTER

    # Flip sign for lower-is-better metrics (e.g., log_loss, Brier, runtime) so positive `signed` always means better
    sign = -1.0 if metric in LOWER_IS_BETTER else 1.0
    signed = sign * np.asarray(delta, dtype=float)
    return np.where(
        signed > tol, "improved", np.where(signed < -tol, "regressed", "draw")
    )


def holm_adjust(p) -> np.ndarray:
    """Apply Holm-Bonferroni step-down family-wise error rate adjustment while preserving NaN entries."""
    p = np.asarray(p, dtype=float)
    adj = np.full(len(p), np.nan)
    # Sort only finite p-values in ascending order; NaNs remain untouched
    finite = np.where(np.isfinite(p))[0]
    order = finite[np.argsort(p[finite])]
    running, m = 0.0, len(order)
    for rank, idx in enumerate(order):
        # Enforce monotonicity across step-down multipliers (m - rank) and clamp at 1.0
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def bootstrap_ci(values, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0):
    """Compute a non-parametric percentile bootstrap confidence interval for the sample mean."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    # Draw (n_boot x len(v)) indices with replacement and compute bootstrap replicate means in one pass
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(
        np.quantile(means, 1 - alpha / 2)
    )
