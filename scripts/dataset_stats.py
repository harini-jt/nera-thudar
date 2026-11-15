"""Per-dataset statistics (size, length, classes, balance, normalisation, buckets).

python -m scripts.dataset_stats
python -m scripts.dataset_stats --datasets ItalyPowerDemand Strawberry --resample 0
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.common import LENGTH_BINS, LENGTH_LABELS, TRAIN_BINS, TRAIN_LABELS, bucket
from src.data_loader import list_datasets, load_dataset


def entropy_bits(counts) -> float:
    """Compute Shannon entropy in bits over class frequency counts."""
    n = counts.sum()
    return float(-sum((c / n) * math.log2(c / n) for c in counts if c))


def stats_for(name: str, resample: int, z_tol: float) -> dict:
    """Compute structural, class-balance, and normalization diagnostics for one dataset split."""
    X_tr, y_tr, X_te, y_te = load_dataset(name, resample)
    X_tr, X_te = np.asarray(X_tr, dtype=float), np.asarray(X_te, dtype=float)

    # Pool train and test sets to inspect global value range and per-series z-normalization
    allX = np.concatenate([X_tr, X_te])
    series_mean, series_std = allX.mean(axis=-1), allX.std(axis=-1)

    classes, counts = np.unique(y_tr, return_counts=True)
    n_classes_all = len(set(y_tr) | set(y_te))

    return {
        "dataset": name,
        "resample_id": resample,
        "n_train": len(y_tr),
        "n_test": len(y_te),
        "test_fraction": len(y_te) / (len(y_tr) + len(y_te)),
        "series_length": X_tr.shape[-1],
        "n_channels": X_tr.shape[1] if X_tr.ndim == 3 else 1,
        "train_test_length_match": X_tr.shape[-1] == X_te.shape[-1],
        "n_classes": n_classes_all,
        # Non-zero if a rare class appears in test without any training examples
        "classes_missing_from_train": n_classes_all - len(classes),
        "majority_fraction_train": counts.max() / counts.sum(),
        "imbalance_ratio_train": counts.max() / counts.min(),
        "min_train_per_class": int(counts.min()),
        # Normalized entropy in [0, 1] (1.0 = perfectly balanced classes; 0.0 = single-class degenerate)
        "class_entropy_norm": (
            entropy_bits(counts) / math.log2(len(classes)) if len(classes) > 1 else 0.0
        ),
        # Sample-to-dimension ratio; low values (< 1) indicate high-dimensional small-sample regimes
        "train_to_length_ratio": len(y_tr) / X_tr.shape[-1],
        "value_mean": float(allX.mean()),
        "value_std": float(allX.std()),
        "value_min": float(allX.min()),
        "value_max": float(allX.max()),
        "n_missing": int(np.isnan(allX).sum()),
        # Fraction of flat/zero-variance series (where z-normalization is undefined or produces NaNs)
        "constant_series_fraction": float((series_std < 1e-12).mean()),
        # Measure deviation from zero-mean, unit-variance per-series normalization
        "mean_abs_series_mean": float(np.abs(series_mean).mean()),
        "mean_abs_series_std_minus1": float(np.abs(series_std - 1).mean()),
        "is_znormalised": bool(
            np.abs(series_mean).mean() < z_tol and np.abs(series_std - 1).mean() < z_tol
        ),
        "error": "",
    }


def main():
    """Collect metadata across target datasets, assign size/length buckets, and write summary CSV."""
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="*")
    p.add_argument("--resample", type=int, default=0)
    p.add_argument("--znorm-tol", type=float, default=0.01)
    p.add_argument("--out", default="docs/dataset_stats.csv")
    a = p.parse_args()

    rows = []
    for name in a.datasets or list_datasets():
        try:
            rows.append(stats_for(name, a.resample, a.znorm_tol))
        except Exception as e:
            # Record load/shape errors inline so broken datasets still appear in the audit table
            rows.append(
                {"dataset": name, "error": f"{type(e).__name__}: {str(e)[:100]}"}
            )
            print(f"PROBLEM {name}: {e}")

    df = pd.DataFrame(rows)
    ok = df["error"] == ""

    # Assign categorical regime buckets only to rows that loaded cleanly
    df.loc[ok, "length_bucket"] = bucket(
        df.loc[ok, "series_length"], LENGTH_BINS, LENGTH_LABELS
    ).astype(str)
    df.loc[ok, "train_size_bucket"] = bucket(
        df.loc[ok, "n_train"], TRAIN_BINS, TRAIN_LABELS
    ).astype(str)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)

    # Print console overview for quick sanity checking
    good = df[ok]
    print(f"\n{len(good)}/{len(df)} datasets read; wrote {a.out}")
    print(
        good[["series_length", "n_train", "n_test", "n_classes"]]
        .describe()
        .round(1)
        .to_string()
    )
    print(
        "\nSeries length buckets:\n", good["length_bucket"].value_counts().to_string()
    )
    print(
        "\nTrain size buckets:\n", good["train_size_bucket"].value_counts().to_string()
    )
    print(f"\nNot z-normalised: {(~good['is_znormalised']).sum()} datasets")


if __name__ == "__main__":
    main()
