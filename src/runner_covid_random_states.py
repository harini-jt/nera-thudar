import argparse
import json
import os
import platform
import traceback
from datetime import datetime
from pathlib import Path
from time import perf_counter

import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline

from src.data_loader import load_dataset
from src.quant import Quant

COLUMNS = [
    "dataset", "resample_id", "method", "model_config", "seed", "accuracy",
    "balanced_accuracy", "log_loss", "auc", "fit_time_s", "predict_time_s",
    "n_train", "n_test", "series_length", "hardware",
]


def build_pipeline(seed=42, depth=6, div=4, n_estimators=200, max_features=0.1):
    """Build the QUANT + ExtraTrees pipeline; QUANT is unchanged."""
    return make_pipeline(
        Quant(depth=depth, div=div),
        ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_features=max_features,
            criterion="entropy",
            random_state=seed,
            n_jobs=-1,
        ),
    )


def hardware_string():
    return f"{platform.processor()} | {os.cpu_count()} cores | torch {torch.__version__}"


def run_single_experiment(dataset, method, model_config, hardware, seed):
    # This script deliberately loads resample ID 0 only.
    X_train, y_train, X_test, y_test = load_dataset(dataset, 0)
    model = build_pipeline(seed=seed, **model_config)

    t0 = perf_counter()
    model.fit(X_train, y_train)
    fit_time = perf_counter() - t0

    t0 = perf_counter()
    proba = model.predict_proba(X_test)
    predict_time = perf_counter() - t0

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
        "resample_id": 0,
        "method": method,
        "model_config": json.dumps({"resample_id": 0, **model_config}, sort_keys=True),
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


def run_random_state_experiment(dataset, output_file, method, model_config, seed_start, seed_end):
    if seed_end < seed_start:
        raise ValueError("--seed-end must be >= --seed-start")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    failure_log = output_path.with_suffix(".failures.log")
    hardware = hardware_string()

    # Resume support: skip seeds already saved for this dataset/resample/method.
    completed_seeds = set()
    if output_path.exists() and output_path.stat().st_size:
        existing = pd.read_csv(output_path)
        required = {"dataset", "resample_id", "method", "seed"}
        missing = required - set(existing.columns)
        if missing:
            raise ValueError(f"Existing CSV is missing columns: {sorted(missing)}")
        matching = existing[
            (existing["dataset"] == dataset)
            & (existing["resample_id"] == 0)
            & (existing["method"] == method)
        ]
        completed_seeds = set(matching["seed"].astype(int))

    seeds = range(seed_start, seed_end + 1)
    total = len(seeds)
    done = sum(seed in completed_seeds for seed in seeds)

    print(f"Dataset: {dataset}")
    print("Resample ID: 0 only")
    print(f"Seeds: {seed_start}–{seed_end} ({total} runs)")
    print(f"Already completed: {done}")
    print(f"Output: {output_path}")

    for seed in seeds:
        if seed in completed_seeds:
            continue
        try:
            row = run_single_experiment(dataset, method, model_config, hardware, seed)
        except Exception:
            with open(failure_log, "a", encoding="utf-8") as f:
                f.write(f"{dataset} resample=0 seed={seed}\n{traceback.format_exc()}\n")
            print(f"FAILED {dataset} resample=0 seed={seed}; see {failure_log}", flush=True)
            continue

        write_header = not output_path.exists() or output_path.stat().st_size == 0
        pd.DataFrame([row], columns=COLUMNS).to_csv(
            output_path, mode="a", header=write_header, index=False
        )
        done += 1
        print(
            f"[{done}/{total}] {dataset} r0 seed={seed}: "
            f"accuracy={row['accuracy']:.4f}, fit={row['fit_time_s']}s, "
            f"predict={row['predict_time_s']}s",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Run QUANT + ExtraTrees on Covid resample 0 across random states."
    )
    parser.add_argument("--dataset", default="Covid3MonthDiscrete")
    parser.add_argument("--out", default="results/covid_loop_random_state_results.csv")
    parser.add_argument("--method", default="quant_covid_random_state")
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--div", type=int, default=4)
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--max-features", type=float, default=0.1)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=1000)
    args = parser.parse_args()

    started = datetime.now().astimezone()
    timer_start = perf_counter()
    print("=" * 60)
    print(f"Experiment started: {started:%Y-%m-%d %H:%M:%S %Z}")
    print("=" * 60)

    model_config = {
        "depth": args.depth,
        "div": args.div,
        "n_estimators": args.trees,
        "max_features": args.max_features,
    }
    run_random_state_experiment(
        args.dataset, args.out, args.method, model_config,
        args.seed_start, args.seed_end
    )

    ended = datetime.now().astimezone()
    print(f"Experiment ended: {ended:%Y-%m-%d %H:%M:%S %Z}")
    print(f"Total time: {perf_counter() - timer_start:.2f} seconds")


if __name__ == "__main__":
    main()


#                         count     mean      std      min   median      max
# accuracy          1001.000000 0.727600 0.018668 0.655738 0.721311 0.770492
# balanced_accuracy 1001.000000 0.633973 0.025301 0.540237 0.635008 0.692810
# log_loss          1001.000000 0.745640 0.012250 0.707890 0.745805 0.784277
# auc               1001.000000 0.779988 0.009106 0.754452 0.779875 0.809533
# fit_time_s        1001.000000 0.251707 0.034963 0.202000 0.244000 0.566000
# predict_time_s    1001.000000 0.092162 0.015645 0.070000 0.091000 0.291000