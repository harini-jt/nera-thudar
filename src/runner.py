import argparse
import json
import os
import platform
import time
import traceback
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from src.data_loader import list_datasets, load_dataset
from src.implementations.pipelines import PIPELINES, get_pipeline, method_config
from src.metrics import METRIC_NAMES, compute_metrics

# Fixed CSV schema ensuring consistent column ordering across incremental appends
COLUMNS = [
    "dataset",
    "resample_id",
    "method",
    "seed",
    "model_config",
    *METRIC_NAMES,
    "fit_time_s",
    "predict_time_s",
    "predict_time_per_series_ms",
    "n_features",
    "n_train",
    "n_test",
    "series_length",
    "n_classes",
    "hardware",
]


def hardware_string() -> str:
    """Capture a lightweight hardware/runtime fingerprint for reproducibility across machines."""
    return (
        f"{platform.processor()} | {os.cpu_count()} cores | torch {torch.__version__}"
    )


def warm_up(method: str, cfg: dict) -> None:
    """Throw-away fit so one-off start-up costs (PyTorch init, thread pools) don't pollute timing."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((24, 1, 64)).astype("float32")
    y = np.array(["a", "b"] * 12)
    model = get_pipeline(method, seed=0, **{**cfg, **method_config(method)})
    model.fit(X, y)
    model.predict_proba(X)


def run_single(
    data,
    method: str,
    seed: int,
    cfg: dict,
    hardware: str,
    dataset: str,
    resample_id: int,
) -> dict:
    """Fit and evaluate a single (dataset, method, seed) run and return a flat metrics dictionary."""
    X_train, y_train, X_test, y_test = data
    # A method's own fixed settings (e.g. criterion="gini") take priority over command-line defaults.
    model = get_pipeline(method, seed=seed, **{**cfg, **method_config(method)})

    # Time training and inference separately with high-resolution wall-clock timers
    t0 = perf_counter()
    model.fit(X_train, y_train)
    fit_time = perf_counter() - t0

    t0 = perf_counter()
    proba = model.predict_proba(X_test)
    predict_time = perf_counter() - t0

    classes = model.classes_
    return {
        "dataset": dataset,
        "resample_id": resample_id,
        "method": method,
        "seed": seed,
        "model_config": json.dumps({**cfg, **method_config(method)}, sort_keys=True),
        **compute_metrics(y_test, proba, classes),
        "fit_time_s": round(fit_time, 4),
        "predict_time_s": round(predict_time, 4),
        "predict_time_per_series_ms": round(predict_time / len(y_test) * 1000, 4),
        # Inspect the final estimator in the sklearn Pipeline for post-transform feature count
        "n_features": int(model[-1].n_features_in_),
        "n_train": len(y_train),
        "n_test": len(y_test),
        "series_length": X_train.shape[-1],
        "n_classes": len(classes),
        "hardware": hardware,
    }


def load_completed(path: Path) -> set:
    """Read an existing results CSV to enable crash-safe resumption without duplicating runs."""
    if not path.exists() or path.stat().st_size == 0:
        return set()
    df = pd.read_csv(path)
    return set(zip(df.dataset, df.resample_id, df.method, df.seed))


def resolve_out_path(out_arg: str, method: str, resample_id: int) -> Path:
    """Returns exact CSV Path. Injects method name if out_arg is a directory or template."""
    p = Path(out_arg)
    if "{method}" in out_arg:
        return Path(out_arg.format(method=method, resample=resample_id))
    if p.suffix != ".csv":
        return p / f"{method}_resample{resample_id}.csv"
    return p


def run_experiments_for_method(
    method: str, datasets, seeds, resample_id, out_file: Path, cfg: dict
):
    """Execute all pending (dataset, seed) runs for one method, checkpointing each row to disk."""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    failure_log = out_file.with_suffix(".failures.log")
    hardware = hardware_string()
    completed = load_completed(out_file)

    total = len(datasets) * len(seeds)
    done = len(completed)
    print(
        f"\n--- [{method}] {total} runs planned, {done} already completed in {out_file} ---",
        flush=True,
    )

    warm_up(method, cfg)

    for dataset in datasets:
        # Filter out seeds already recorded in out_file before loading the dataset from disk
        todo_seeds = [
            s for s in seeds if (dataset, resample_id, method, s) not in completed
        ]
        if not todo_seeds:
            continue
        try:
            data = load_dataset(dataset, resample_id)
        except Exception:
            # Isolate dataset I/O errors so a single corrupt archive doesn't abort the benchmark sweep
            with open(failure_log, "a") as f:
                f.write(f"{dataset} (load)\n{traceback.format_exc()}\n")
            print(f"FAILED to load {dataset}", flush=True)
            continue

        for seed in todo_seeds:
            try:
                row = run_single(
                    data, method, seed, cfg, hardware, dataset, resample_id
                )
            except Exception:
                # Isolate model/runtime failures per seed and log full tracebacks for debugging
                with open(failure_log, "a") as f:
                    f.write(
                        f"{dataset} {method} seed={seed}\n{traceback.format_exc()}\n"
                    )
                print(f"FAILED {dataset} {method} seed={seed}", flush=True)
                continue

            # Append immediately to CSV so progress survives interruptions; write header only on new/empty file
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


def main():
    """Parse CLI arguments, validate requested pipelines, and orchestrate benchmark sweeps."""
    start = datetime.now().astimezone()
    t0 = time.perf_counter()

    p = argparse.ArgumentParser(
        description="Run Quant variants on time series benchmarks."
    )
    p.add_argument("--datasets", nargs="*", help="default: every dataset directory")
    p.add_argument(
        "--methods",
        nargs="+",
        default=["quant_base"],
        help=f"any of {sorted(PIPELINES)}",
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[7, 11, 42])
    p.add_argument("--resample", type=int, default=0)
    p.add_argument(
        "--out",
        default="results/",
        help="Exact output CSV file path",
    )
    # Quant feature extraction and ExtraTrees classifier hyperparameters
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--div", type=int, default=4)
    p.add_argument("--trees", type=int, default=200)
    p.add_argument("--max-features", type=float, default=0.1)
    p.add_argument("--criterion", default="entropy")
    p.add_argument("--n-jobs", type=int, default=-1)
    args = p.parse_args()

    # Fail fast before loading data if any requested method name is invalid
    for m in args.methods:
        if m not in PIPELINES:
            raise SystemExit(f"Unknown method '{m}'. Available: {sorted(PIPELINES)}")

    cfg = {
        "depth": args.depth,
        "div": args.div,
        "n_estimators": args.trees,
        "max_features": args.max_features,
        "criterion": args.criterion,
        "n_jobs": args.n_jobs,
    }
    datasets = args.datasets or list_datasets()
    print(
        f"Started {start:%Y-%m-%d %H:%M:%S %Z} | {len(datasets)} datasets", flush=True
    )

    for method in args.methods:
        out_file = resolve_out_path(args.out, method, args.resample)
        run_experiments_for_method(
            method, datasets, args.seeds, args.resample, out_file, cfg
        )

    print(f"\nTotal execution time: {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
