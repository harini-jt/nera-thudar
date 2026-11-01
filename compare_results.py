
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


METRICS = [
    "accuracy",
    "balanced_accuracy",
    "log_loss",
    "auc",
    "fit_time_s",
    "predict_time_s",
]

HIGHER_IS_BETTER = [
    "accuracy",
    "balanced_accuracy",
    "auc",
]


def load_results(path):
    df = pd.read_csv(path)

    required = ["dataset", "resample_id"] + METRICS
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    for metric in METRICS:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")

    return df


def compare_results(cpu, gpu, cpu_name, gpu_name, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n========== ROW COUNTS ==========")
    print(f"{cpu_name}: {len(cpu)} rows")
    print(f"{gpu_name}: {len(gpu)} rows")

    keys = ["dataset", "resample_id"]

    # Compare only matching dataset/resample combinations.
    cpu_keys = cpu[keys].drop_duplicates()
    gpu_keys = gpu[keys].drop_duplicates()

    matched_keys = cpu_keys.merge(gpu_keys, on=keys, how="inner")
    cpu_only = cpu_keys.merge(gpu_keys, on=keys, how="left", indicator=True)
    cpu_only = cpu_only[cpu_only["_merge"] == "left_only"].drop(columns="_merge")

    gpu_only = gpu_keys.merge(cpu_keys, on=keys, how="left", indicator=True)
    gpu_only = gpu_only[gpu_only["_merge"] == "left_only"].drop(columns="_merge")

    print("\n========== MATCHING COVERAGE ==========")
    print(f"Matching dataset/resample pairs: {len(matched_keys)}")
    print(f"Only in {cpu_name}: {len(cpu_only)}")
    print(f"Only in {gpu_name}: {len(gpu_only)}")

    # Merge matching rows.
    merged = cpu.merge(
        gpu,
        on=keys,
        how="inner",
        suffixes=("_cpu", "_gpu"),
    )

    if merged.empty:
        print("\nNo matching dataset/resample pairs found.")
        return

    summary = []

    print("\n========== METRIC COMPARISON ==========")

    for metric in METRICS:
        cpu_col = f"{metric}_cpu"
        gpu_col = f"{metric}_gpu"

        cpu_mean = merged[cpu_col].mean()
        gpu_mean = merged[gpu_col].mean()

        difference = gpu_mean - cpu_mean

        if metric in HIGHER_IS_BETTER:
            verdict = "GPU improved" if difference > 0 else "GPU worse"
        else:
            verdict = "GPU improved" if difference < 0 else "GPU worse"

        if pd.isna(difference) or difference == 0:
            verdict = "No difference / unavailable"

        print(f"\n{metric}")
        print(f"  {cpu_name} mean: {cpu_mean:.6f}")
        print(f"  {gpu_name} mean: {gpu_mean:.6f}")
        print(f"  Difference (GPU - CPU): {difference:.6f}")
        print(f"  Result: {verdict}")

        summary.append({
            "metric": metric,
            "cpu_mean": cpu_mean,
            "gpu_mean": gpu_mean,
            "difference_gpu_minus_cpu": difference,
            "result": verdict,
            "matched_rows": merged[[cpu_col, gpu_col]].dropna().shape[0],
        })

        # Graph 1-6: Compare means for each metric.
        plt.figure(figsize=(7, 5))

        means = [cpu_mean, gpu_mean]
        labels = [cpu_name, gpu_name]

        plt.bar(labels, means)
        plt.title(f"Comparison of {metric}")
        plt.ylabel(metric)
        plt.tight_layout()

        filename = output_dir / f"{metric}_comparison.png"
        plt.savefig(filename, dpi=150)
        plt.close()

    # Save numerical comparison.
    pd.DataFrame(summary).to_csv(
        output_dir / "metric_summary.csv",
        index=False,
    )

    # Save matching rows for further analysis.
    merged.to_csv(
        output_dir / "matched_results.csv",
        index=False,
    )

    cpu_only.to_csv(output_dir / "only_in_first_file.csv", index=False)
    gpu_only.to_csv(output_dir / "only_in_second_file.csv", index=False)

    print("\n========== OUTPUT ==========")
    print(f"Comparison files and graphs saved in: {output_dir.resolve()}")
    print("Check metric_summary.csv for the summary.")
    print("Check matched_results.csv for row-by-row comparisons.")


def main():
    parser = argparse.ArgumentParser(
        description="Compare CPU and GPU reproduction results."
    )

    parser.add_argument("--cpu", required=True, help="First CSV file")
    parser.add_argument("--gpu", required=True, help="Second CSV file")

    args = parser.parse_args()

    cpu_path = Path(args.cpu)
    gpu_path = Path(args.gpu)

    cpu = load_results(cpu_path)
    gpu = load_results(gpu_path)

    compare_results(
        cpu,
        gpu,
        cpu_path.stem,
        gpu_path.stem,
        Path("comparison_plots"),
    )


if __name__ == "__main__":
    main()



# ========== ROW COUNTS ==========
# reproduction_results: 4260 rows
# reproduction_results2: 109 rows

# ========== MATCHING COVERAGE ==========
# Matching dataset/resample pairs: 109
# Only in reproduction_results: 4151
# Only in reproduction_results2: 0

# ========== METRIC COMPARISON ==========

# accuracy
#   reproduction_results mean: 0.888780
#   reproduction_results2 mean: 0.888894
#   Difference (GPU - CPU): 0.000114
#   Result: GPU improved

# balanced_accuracy
#   reproduction_results mean: 0.882816
#   reproduction_results2 mean: 0.882896
#   Difference (GPU - CPU): 0.000079
#   Result: GPU improved

# log_loss
#   reproduction_results mean: 0.546041
#   reproduction_results2 mean: 0.533708
#   Difference (GPU - CPU): -0.012333
#   Result: GPU improved

# auc
#   reproduction_results mean: 0.978752
#   reproduction_results2 mean: 0.978518
#   Difference (GPU - CPU): -0.000234
#   Result: GPU worse

# fit_time_s
#   reproduction_results mean: 5.939624
#   reproduction_results2 mean: 33.353670
#   Difference (GPU - CPU): 27.414046
#   Result: GPU worse

# predict_time_s
#   reproduction_results mean: 0.489945
#   reproduction_results2 mean: 0.912890
#   Difference (GPU - CPU): 0.422945
#   Result: GPU worse
