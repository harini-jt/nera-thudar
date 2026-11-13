"""Reference baselines on the same resample files as the QUANT runner.

Methods:
  majority    predicts the most frequent training class (probabilities = class priors)
  minirocket  MiniRocket -> StandardScaler -> RidgeClassifierCV (the aeon MiniRocketClassifier)

Output has the same columns as `src.runner`, so `scripts.compare_methods` and
`scripts.evaluation_report` can read it. Ridge gives no probabilities, so the
probability metrics (log loss, AUC, Brier, ECE) are left empty for minirocket.

Example:
  python -m scripts.run_baselines --methods majority minirocket --resample 0 \
      --out "results/{method}_resample{resample}.csv"
"""

import argparse
import json
import time
import traceback
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data_loader import list_datasets, load_dataset
from src.metrics import compute_metrics
from src.runner import COLUMNS, hardware_string, load_completed, resolve_out_path

PROBABILITY_METRICS = ["log_loss", "auc", "brier", "ece"]


def build(method: str, seed: int, n_jobs: int):
    if method == "majority":
        return DummyClassifier(strategy="prior")
    if method == "minirocket":
        from aeon.transformations.collection.convolution_based import MiniRocket

        return make_pipeline(
            MiniRocket(n_kernels=10_000, random_state=seed, n_jobs=n_jobs),
            StandardScaler(with_mean=False),
            RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)),
        )
    raise SystemExit(f"Unknown method '{method}'. Available: majority, minirocket")


def as_3d(X):
    X = np.asarray(X, dtype=np.float32)
    return X[:, None, :] if X.ndim == 2 else X


def run_single(data, method, seed, n_jobs, hardware, dataset, resample_id) -> dict:
    X_train, y_train, X_test, y_test = data
    X_train, X_test = as_3d(X_train), as_3d(X_test)
    model = build(method, seed, n_jobs)

    t0 = perf_counter()
    model.fit(X_train if method != "majority" else X_train[:, 0, :1], y_train)
    fit_time = perf_counter() - t0

    t0 = perf_counter()
    if method == "majority":
        proba = model.predict_proba(X_test[:, 0, :1])
    else:
        preds = model.predict(X_test)
        proba = (preds[:, None] == model.classes_[None, :]).astype(float)
    predict_time = perf_counter() - t0

    metrics = compute_metrics(y_test, proba, model.classes_)
    if method != "majority":
        metrics.update({m: float("nan") for m in PROBABILITY_METRICS})
        n_features = int(model[-1].n_features_in_)
    else:
        n_features = 0

    return {
        "dataset": dataset,
        "resample_id": resample_id,
        "method": method,
        "seed": seed,
        "model_config": json.dumps({"n_jobs": n_jobs}),
        **metrics,
        "fit_time_s": round(fit_time, 4),
        "predict_time_s": round(predict_time, 4),
        "predict_time_per_series_ms": round(predict_time / len(y_test) * 1000, 4),
        "n_features": n_features,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "series_length": X_train.shape[-1],
        "n_classes": len(model.classes_),
        "hardware": hardware,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--methods", nargs="+", default=["majority", "minirocket"])
    p.add_argument("--datasets", nargs="*", help="default: every dataset directory")
    p.add_argument("--seeds", nargs="+", type=int, default=[7, 11, 42])
    p.add_argument("--resample", type=int, default=0)
    p.add_argument("--out", default="results/{method}_resample{resample}.csv")
    p.add_argument("--n-jobs", type=int, default=-1)
    args = p.parse_args()

    t_start = time.perf_counter()
    datasets = args.datasets or list_datasets()
    hardware = hardware_string()

    for method in args.methods:
        out_file = resolve_out_path(args.out, method, args.resample)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        failure_log = out_file.with_suffix(".failures.log")
        completed = load_completed(out_file)
        # The majority baseline has no randomness, so one row per dataset is enough.
        seeds = [0] if method == "majority" else args.seeds
        total, done = len(datasets) * len(seeds), len(completed)
        print(f"\n--- [{method}] {total} runs planned, {done} already done in {out_file} ---", flush=True)

        if method == "minirocket":  # numba compilation, not counted as runtime
            rng = np.random.default_rng(0)
            build(method, 0, args.n_jobs).fit(
                rng.standard_normal((24, 1, 64)).astype("float32"), np.array(["a", "b"] * 12)
            )

        for dataset in datasets:
            todo = [s for s in seeds if (dataset, args.resample, method, s) not in completed]
            if not todo:
                continue
            try:
                data = load_dataset(dataset, args.resample)
                for seed in todo:
                    row = run_single(data, method, seed, args.n_jobs, hardware, dataset, args.resample)
                    pd.DataFrame([row], columns=COLUMNS).to_csv(
                        out_file,
                        mode="a",
                        index=False,
                        header=not out_file.exists() or out_file.stat().st_size == 0,
                    )
                    done += 1
                    print(
                        f"[{done}/{total}] {dataset} {method} s{seed}: "
                        f"acc={row['accuracy']:.4f} fit={row['fit_time_s']}s",
                        flush=True,
                    )
            except Exception:
                with open(failure_log, "a") as f:
                    f.write(f"{dataset} {method}\n{traceback.format_exc()}\n")
                print(f"FAILED {dataset} {method}", flush=True)

    print(f"\nTotal execution time: {time.perf_counter() - t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
