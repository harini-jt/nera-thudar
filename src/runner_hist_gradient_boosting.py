import argparse
import json
import os
from pathlib import Path
import platform
from time import perf_counter
import torch
import traceback

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline

from src.quant import Quant
from src.data_loader import load_dataset, list_datasets

from datetime import datetime
import time
from threadpoolctl import threadpool_limits

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
    learning_rate: float = 0.1,
    max_leaf_nodes: int = 31,
):
    """Unchanged QUANT feature extraction followed by HistGradientBoosting."""
    return make_pipeline(
        Quant(depth=depth, div=div),
        HistGradientBoostingClassifier(
            max_iter=n_estimators,
            learning_rate=learning_rate,
            max_leaf_nodes=max_leaf_nodes,
            random_state=seed,
        ),
    )


def hardware_string() -> str:
    return (
        f"{platform.processor()} | {os.cpu_count()} cores | torch {torch.__version__}"
    )


def run_single_dataset(
    dataset: str, resample_id: int, method: str, model_config: dict, hardware: str
) -> dict:
    X_train, y_train, X_test, y_test = load_dataset(dataset, resample_id)

    seed = 42  # TODO: Check if seed must be fixed or not
    model = build_pipeline(seed, **model_config)

    # Fit the train data on the model
    # t0 = perf_counter()
    # model.fit(X_train, y_train)
    # fit_time = perf_counter() - t0
    t0 = perf_counter()

    with threadpool_limits(limits=16):
        model.fit(X_train, y_train)

    fit_time = perf_counter() - t0

    # Predict the test data with the model
    # t0 = perf_counter()
    # proba = model.predict_proba(X_test)
    # predict_time = perf_counter() - t0
    t0 = perf_counter()

    with threadpool_limits(limits=16):
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
        "model_config": json.dumps(model_config, sort_keys=True),
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


def run_experiments(
    datasets,
    num_resamples: int = 30,
    results_file: str = "results/reproduction_results.csv",
    method: str = "quant_base",
    model_config: dict | None = None,
):
    model_config = model_config or {}
    results_path = Path(results_file)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    failure_log = results_path.with_suffix(".failures.log")
    hardware = hardware_string()

    # 1. Read the existing csv to see what is already done
    completed_runs = set()
    if results_path.exists() and results_path.stat().st_size > 0:
        df_existing = pd.read_csv(results_path)
        completed_runs = set(
            zip(
                df_existing["dataset"],
                df_existing["resample_id"],
                df_existing["method"],
            )
        )

    # Warm-up run to avoid torch start up cost
    run_single_dataset(datasets[0], 0, method, model_config, hardware)

    # 2. Main execution loop
    total = len(datasets) * num_resamples
    done = len(completed_runs)

    for dataset in datasets:
        for resample_id in range(num_resamples):
            # 2.1. Check if run already exists
            if (dataset, resample_id, method) in completed_runs:
                continue

            # 2.2. Check if the run failed
            try:
                row = run_single_dataset(
                    dataset, resample_id, method, model_config, hardware
                )
            except Exception:
                with open(failure_log, "a") as f:
                    f.write(f"{dataset} r{resample_id}\n{traceback.format_exc()}\n")
                print(
                    f"FAILED {dataset} r{resample_id} (see {failure_log})", flush=True
                )
                continue

            # 2.3. Add the run to the results
            pd.DataFrame([row], columns=COLUMNS).to_csv(
                results_path,
                mode="a",
                header=not results_path.exists() or results_path.stat().st_size == 0,
                index=False,
            )
            done += 1
            print(
                f"[{done}/{total}] {dataset} r{resample_id}: accuracy={row['accuracy']:.4f} "
                f"fit={row['fit_time_s']}s pred={row['predict_time_s']}s",
                flush=True,
            )


def main():
    start_datetime = datetime.now().astimezone()
    start_time = time.perf_counter()

    print("=" * 60)
    print(f"Experiment started : {start_datetime:%Y-%m-%d %H:%M:%S %Z}")
    print("=" * 60)

    p = argparse.ArgumentParser()
    p.add_argument(
        "--datasets",
        nargs="*",
        help="The datasets you wanna consider evaluating for the model. (default: all in the ROOT_DATASET_DIR)",
    )
    p.add_argument(
        "--resamples",
        type=int,
        default=30,
        help="The maximum number of resamples you wanna consider. (default: 30)",
    )
    p.add_argument(
        "--out",
        default="results/reproduction_results.csv",
        help="The location where you wanna store the evaluation results. (default: results/reproduction_results.csv)",
    )
    p.add_argument(
        "--method",
        default="quant_hist_gradient_boosting",
        help="The name of the model used. It can be used to differentiate between the results. (default: quant)",
    )
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--div", type=int, default=4)
    p.add_argument("--trees", type=int, default=200)
    p.add_argument("--max-features", type=float, default=0.1)
    p.add_argument("--learning-rate", type=float, default=0.1)
    p.add_argument("--max-leaf-nodes", type=int, default=31)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    args = p.parse_args()

    datasets = (args.datasets or list_datasets())[args.shard :: args.num_shards]
    out = args.out
    if args.num_shards > 1:
        out = out.replace(".csv", f"_shard{args.shard}.csv")

    model_config = {
        "depth": args.depth,
        "div": args.div,
        "n_estimators": args.trees,
        "learning_rate": args.learning_rate,
        "max_leaf_nodes": args.max_leaf_nodes,
    }
    run_experiments(datasets, args.resamples, out, args.method, model_config)

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
