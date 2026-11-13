"""Runtime benchmark with a fixed thread count for every method.

Times the four stages the QUANT paper reports (transform train, classifier fit,
transform test, predict) with the same number of threads for every method,
after a warm-up call, repeated `--repeats` times. Accuracy is recorded too, as a
check that the timed model is the same model as in the accuracy runs.

Examples:
  # Paper claim check: plain QUANT, all 142 datasets, one core
  python -m scripts.benchmark_runtime --methods quant_base --datasets all \
      --repeats 1 --out results/runtime_1core_full.csv

  # Method comparison on the fixed 30-dataset subset, 3 repeats each
  python -m scripts.benchmark_runtime --repeats 3 --out results/runtime_1core_subset.csv
"""

import argparse
import os
import sys
import time
from pathlib import Path
from time import perf_counter

# Thread limits must be set before numpy/torch/numba are imported.
_threads = "1"
if "--threads" in sys.argv:
    _threads = sys.argv[sys.argv.index("--threads") + 1]
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_var] = _threads

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from scripts.run_baselines import as_3d, build as build_baseline  # noqa: E402
from src.data_loader import list_datasets, load_dataset  # noqa: E402
from src.implementations.pipelines import PIPELINES, get_pipeline  # noqa: E402
from src.runner import hardware_string  # noqa: E402

DEFAULT_METHODS = [
    "quant_base", "dup_d1-2", "dup_d1_4_d2_2", "more_quantiles_d1-2",
    "depth_div_m2_m8", "quant_stats", "random_dropout_0.1", "random_dropout_0.2",
    "minirocket",
]
COLUMNS = [
    "dataset", "method", "repeat", "threads", "accuracy", "n_features",
    "transform_train_s", "fit_s", "transform_test_s", "predict_s", "total_s",
    "n_train", "n_test", "series_length", "hardware",
]


def subset_by_cost(n: int, stats_path="docs/dataset_stats.csv") -> list[str]:
    """Every k-th dataset when sorted by n_train * series_length, so the subset spans small to large."""
    st = pd.read_csv(stats_path)
    st = st[st.resample_id == 0].assign(cost=lambda d: d.n_train * d.series_length)
    st = st.sort_values(["cost", "dataset"]).reset_index(drop=True)
    picks = np.linspace(0, len(st) - 1, n).round().astype(int)
    return st.loc[picks, "dataset"].tolist()


def build(method: str, threads: int):
    if method in PIPELINES:
        return get_pipeline(method, seed=42, n_jobs=threads)
    return build_baseline(method, 42, threads)


def time_once(method: str, threads: int, data) -> dict:
    X_train, y_train, X_test, y_test = data
    X_train, X_test = as_3d(X_train), as_3d(X_test)
    pipe = build(method, threads)
    transform, learner = pipe[:-1], pipe[-1]

    t0 = perf_counter()
    Z_train = transform.fit_transform(X_train, y_train)
    t1 = perf_counter()
    learner.fit(Z_train, y_train)
    t2 = perf_counter()
    Z_test = transform.transform(X_test)
    t3 = perf_counter()
    preds = learner.predict(Z_test)
    t4 = perf_counter()

    return {
        "accuracy": float((preds == np.asarray(y_test)).mean()),
        "n_features": int(Z_train.shape[1]),
        "transform_train_s": t1 - t0,
        "fit_s": t2 - t1,
        "transform_test_s": t3 - t2,
        "predict_s": t4 - t3,
        "total_s": t4 - t0,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "series_length": X_train.shape[-1],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    p.add_argument("--datasets", nargs="*", help="names, or 'all'; default: the fixed subset")
    p.add_argument("--subset-size", type=int, default=30)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--out", default="results/runtime_1core_subset.csv")
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    try:
        import numba

        numba.set_num_threads(args.threads)
    except ImportError:
        pass

    if not args.datasets:
        datasets = subset_by_cost(args.subset_size)
    elif args.datasets == ["all"]:
        datasets = list_datasets()
    else:
        datasets = args.datasets

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists() and out.stat().st_size:
        prev = pd.read_csv(out)
        done = set(zip(prev.dataset, prev.method, prev.repeat))
    hardware = hardware_string()
    total = len(datasets) * len(args.methods) * args.repeats
    print(f"{len(datasets)} datasets x {len(args.methods)} methods x {args.repeats} repeats, "
          f"{args.threads} thread(s), {len(done)} already done", flush=True)

    # Warm-up: one-off costs (imports, numba compilation) are not timed.
    rng = np.random.default_rng(0)
    warm = (rng.standard_normal((24, 1, 64)).astype("float32"), np.array(["a", "b"] * 12)) * 2
    for method in args.methods:
        time_once(method, args.threads, warm)

    t_start = time.perf_counter()
    count = len(done)
    # Repeats are the outer loop, so each repeat of a method is separated in time.
    for repeat in range(args.repeats):
        for dataset in datasets:
            data = None
            for method in args.methods:
                if (dataset, method, repeat) in done:
                    continue
                if data is None:
                    data = load_dataset(dataset, 0)
                row = time_once(method, args.threads, data)
                row.update(dataset=dataset, method=method, repeat=repeat,
                           threads=args.threads, hardware=hardware)
                pd.DataFrame([row], columns=COLUMNS).to_csv(
                    out, mode="a", index=False,
                    header=not out.exists() or out.stat().st_size == 0,
                )
                count += 1
                print(f"[{count}/{total}] r{repeat} {dataset} {method}: "
                      f"total={row['total_s']:.2f}s acc={row['accuracy']:.4f}", flush=True)
    print(f"\nBenchmark wall time: {time.perf_counter() - t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
