import os

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import platform
from time import perf_counter
import torch
import traceback

import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline

from src.improvements.quant_with_sampling_frequency import QuantWithSamplingFrequency
from src.quant import Quant
from src.data_loader import ROOT_DATASET_DIR, load_dataset, list_datasets

from datetime import datetime
import time

COLUMNS = [
    "dataset",
    "resample_id",
    "method",
    "model_config",
    "seed",
    "accuracy",
    "balanced_accuracy",
    "log_loss",
    "auc",
    "fit_time_s",
    "predict_time_s",
    "n_train",
    "n_test",
    "series_length",
    "hardware",
]


def build_pipeline(
    seed: int = 42,
    depth: int = 6,
    div: int = 4,
    n_estimators: int = 200,
    max_features: float = 0.1,
    n_jobs: int = 1,
):
    """Constructs the Quant + ExtraTreesClassifier pipeline per the paper's defaults."""
    return make_pipeline(
        Quant(depth=depth, div=div),
        ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_features=max_features,
            criterion="entropy",
            random_state=seed,
            n_jobs=n_jobs,  # Uses all available CPU cores
        ),
    )


def build_improvement_pipeline(
    seed: int = 42,
    depth: int = 6,
    div: int = 4,
    n_estimators: int = 200,
    max_features: float = 0.1,
    n_jobs: int = 1,
    depth_repeats: dict[int, int] | None = {1: 2, 2: 2},
):
    """Constructs the Quant + ExtraTreesClassifier pipeline per the paper's defaults."""
    return make_pipeline(
        QuantWithSamplingFrequency(depth, div, depth_repeats=depth_repeats),
        ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_features=max_features,
            criterion="entropy",
            random_state=seed,
            n_jobs=n_jobs,  # Uses all available CPU cores
        ),
    )


def hardware_string() -> str:
    return (
        f"{platform.processor()} | {os.cpu_count()} cores | torch {torch.__version__}"
    )


def run_single_dataset(
    dataset: str,
    resample_id: int,
    method: str,
    model_config: dict,
    hardware: str,
    improvement: bool = False,
    depth_repeats: dict[int, int] | None = {1: 2, 2: 2},
    n_jobs: int = 1,
) -> dict:
    X_train, y_train, X_test, y_test = load_dataset(dataset, resample_id)

    seed = 42  # TODO: Check if seed must be fixed or not

    if improvement:
        model = build_improvement_pipeline(
            seed, n_jobs=n_jobs, depth_repeats=depth_repeats, **model_config
        )
    else:
        model = build_pipeline(seed, n_jobs=n_jobs, **model_config)

    # Fit the train data on the model
    t0 = perf_counter()
    model.fit(X_train, y_train)
    fit_time = perf_counter() - t0

    # Predict the test data with the model
    t0 = perf_counter()
    proba = model.predict_proba(X_test)
    predict_time = perf_counter() - t0

    # Calculate scores
    classes = model.classes_
    preds = classes[proba.argmax(axis=1)]
    accuracy = accuracy_score(y_test, preds)
    balanced_accuracy = balanced_accuracy_score(y_test, preds)
    ll = log_loss(y_test, proba, labels=classes)
    try:
        if len(classes) == 2:
            auc = roc_auc_score(y_test, proba[:, 1])
        else:
            auc = roc_auc_score(y_test, proba, multi_class="ovr", labels=classes)
    except ValueError:
        auc = float("nan")

    return {
        "dataset": dataset,
        "resample_id": resample_id,
        "method": method,
        "model_config": json.dumps(
            {
                **model_config,
                **({"depth_repeats": depth_repeats} if improvement else {}),
            },
            sort_keys=True,
        ),
        "seed": seed,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "log_loss": ll,
        "auc": auc,
        "fit_time_s": round(fit_time, 3),
        "predict_time_s": round(predict_time, 3),
        "n_train": len(y_train),
        "n_test": len(y_test),
        "series_length": X_train.shape[-1],
        "hardware": hardware,
    }


def _init_worker():
    torch.set_num_threads(1)
    print(f"worker {os.getpid()} ready", flush=True)


def _task(args):
    dataset, resample_id = args[0], args[1]
    try:
        return run_single_dataset(*args), None
    except Exception:
        return None, f"{dataset} r{resample_id}\n{traceback.format_exc()}\n"


def run_experiments(
    datasets,
    num_resamples: int = 30,
    results_file: str = "results/reproduction_results.csv",
    method: str = "quant_base",
    model_config: dict | None = None,
    improvement: bool = False,
    depth_repeats: dict[int, int] | None = {1: 2, 2: 2},
    workers: int = 16,
):
    model_config = model_config or {}
    results_path = Path(results_file)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    failure_log = results_path.with_suffix(".failures.log")
    hardware = hardware_string()

    # 1. Read the existing csv to see what is already done
    completed = set()
    if results_path.exists() and results_path.stat().st_size > 0:
        df = pd.read_csv(results_path)
        if list(df.columns) != COLUMNS:  # old header would corrupt the file
            raise SystemExit(
                f"{results_path} has a different header; move it aside or use a new --out"
            )
        completed = set(zip(df["dataset"], df["resample_id"], df["method"]))

    def size(ds):  # biggest datasets first, so stragglers don't finish last
        p = ROOT_DATASET_DIR / ds / f"{ds}0_TRAIN.ts"
        return p.stat().st_size if p.exists() else 0

    n_jobs = 1 if workers > 1 else -1
    tasks = [
        (ds, r, method, model_config, hardware, improvement, depth_repeats, n_jobs)
        for ds in sorted(datasets, key=size, reverse=True)
        for r in range(num_resamples)
        if (ds, r, method) not in completed
    ]
    print(
        f"{len(tasks)} runs to do, {len(completed)} already done, {workers} worker(s)",
        flush=True,
    )

    def handle(row, err, i):
        if err:
            with open(failure_log, "a") as f:
                f.write(err)
            print(f"FAILED ({err.splitlines()[0]})", flush=True)
            return
        pd.DataFrame([row], columns=COLUMNS).to_csv(
            results_path,
            mode="a",
            index=False,
            header=not results_path.exists() or results_path.stat().st_size == 0,
        )
        print(
            f"[{i}/{len(tasks)}] {row['dataset']} r{row['resample_id']}: acc={row['accuracy']:.4f}",
            flush=True,
        )

    if workers == 1:
        for i, t in enumerate(tasks, 1):
            handle(*_task(t), i)
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as ex:
            futures = {ex.submit(_task, t): t for t in tasks}
            for i, fut in enumerate(as_completed(futures), 1):
                try:
                    handle(*fut.result(), i)
                except Exception as e:
                    t = futures[fut]
                    handle(None, f"{t[0]} r{t[1]}\nworker crash: {e!r}\n", i)


def main():
    start_datetime = datetime.now().astimezone()
    start_time = time.perf_counter()

    print("=" * 60)
    print(f"Experiment started : {start_datetime:%Y-%m-%d %H:%M:%S %Z}")
    print("=" * 60)

    p = argparse.ArgumentParser()

    def parse_repeats(s):
        return (
            {int(k): int(v) for k, v in (kv.split(":") for kv in s.split(","))}
            if s
            else None
        )

    p.add_argument(
        "--datasets", nargs="*", help="dataset names (default: all in data/)"
    )
    p.add_argument("--resamples", type=int, default=30)
    p.add_argument("--improvement", action="store_true")
    p.add_argument(
        "--repeats",
        type=parse_repeats,
        default="1:2,2:2",
        help="depth:copies, e.g. 1:2,2:2,3:2",
    )
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument(
        "--out",
        default="results/reproduction_results.csv",
        help="The location where you wanna store the evaluation results. (default: results/reproduction_results.csv)",
    )
    p.add_argument(
        "--method",
        default="quant",
        help="The name of the model used. It can be used to differentiate between the results. (default: quant)",
    )
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--div", type=int, default=4)
    p.add_argument("--trees", type=int, default=200)
    p.add_argument("--max-features", type=float, default=0.1)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--min-mb", type=float, default=0.0)
    p.add_argument("--max-mb", type=float, default=float("inf"))
    args = p.parse_args()

    def train_mb(ds):
        return (ROOT_DATASET_DIR / ds / f"{ds}0_TRAIN.ts").stat().st_size / 1e6

    datasets = (args.datasets or list_datasets())[args.shard :: args.num_shards]
    datasets = [d for d in datasets if args.min_mb <= train_mb(d) < args.max_mb]
    out = args.out
    if args.num_shards > 1:
        out = out.replace(".csv", f"_shard{args.shard}.csv")

    model_config = {
        "depth": args.depth,
        "div": args.div,
        "n_estimators": args.trees,
        "max_features": args.max_features,
    }
    run_experiments(
        datasets,
        args.resamples,
        out,
        args.method,
        model_config,
        args.improvement,
        args.repeats,
        args.workers,
    )

    end_datetime = datetime.now().astimezone()
    end_time = time.perf_counter()

    elapsed = end_time - start_time

    print("\n" + "=" * 60)
    print(f"Experiment started : {start_datetime:%Y-%m-%d %H:%M:%S %Z}")
    print(f"Experiment ended   : {end_datetime:%Y-%m-%d %H:%M:%S %Z}")
    print(f"Total time taken   : {elapsed:.2f} seconds")
    print("=" * 60)


if __name__ == "__main__":
    main()

# python -m src.runner --datasets ItalyPowerDemand --resamples 3 --out results/smoke.csv
# python -m src.runner --improvement --datasets ItalyPowerDemand --resamples 3 --out results/smoke.csv --method quant-sampling-frequency


# python -m src.runner --improvement --repeats 1:2,2:2 --min-mb 20 --workers 1 --method dup_d1-2 --out results/comparison-v5c-sampling-frequency-parallel/improvement_results.csv

# python -m src.runner --improvement --repeats 1:2,2:2 --max-mb 20 --workers 16 --method dup_d1-2 --out results/comparison-v5c-sampling-frequency-parallel/improvement_results.csv
