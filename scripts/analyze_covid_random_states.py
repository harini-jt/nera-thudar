import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Summarise QUANT + ExtraTrees random-state experiments."
    )
    parser.add_argument(
        "--input",
        default="results/covid_loop_random_state_results.csv",
        help="Input CSV containing one row per random state.",
    )
    parser.add_argument(
        "--output",
        default="results/covid_random_state_analysis",
        help="Directory for summary files and figures.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    required = {
        "dataset", "resample_id", "method", "seed", "accuracy",
        "balanced_accuracy", "log_loss", "auc", "fit_time_s",
        "predict_time_s", "n_train", "n_test", "series_length",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing columns: {sorted(missing)}")

    metrics = [
        "accuracy", "balanced_accuracy", "log_loss", "auc",
        "fit_time_s", "predict_time_s",
    ]
    summary = df[metrics].agg(["count", "mean", "std", "min", "median", "max"]).T
    summary.to_csv(output_dir / "metric_summary.csv")

    best_accuracy = df.loc[df["accuracy"].idxmax()]
    worst_accuracy = df.loc[df["accuracy"].idxmin()]
    best_log_loss = df.loc[df["log_loss"].idxmin()]

    metadata = {
        "rows": len(df),
        "unique_seeds": df["seed"].nunique(),
        "seed_min": int(df["seed"].min()),
        "seed_max": int(df["seed"].max()),
        "unique_datasets": int(df["dataset"].nunique()),
        "unique_resamples": int(df["resample_id"].nunique()),
        "duplicate_dataset_resample_method_seed_rows": int(
            df.duplicated(["dataset", "resample_id", "method", "seed"]).sum()
        ),
        "missing_values_in_metrics": int(df[metrics].isna().sum().sum()),
        "best_accuracy_seed": int(best_accuracy["seed"]),
        "best_accuracy": float(best_accuracy["accuracy"]),
        "worst_accuracy_seed": int(worst_accuracy["seed"]),
        "worst_accuracy": float(worst_accuracy["accuracy"]),
        "best_log_loss_seed": int(best_log_loss["seed"]),
        "best_log_loss": float(best_log_loss["log_loss"]),
    }
    pd.Series(metadata, name="value").to_csv(output_dir / "experiment_metadata.csv")

    # One row per accuracy value: useful for seeing the discrete score distribution.
    df["accuracy"].value_counts().sort_index().rename_axis(
        "accuracy"
    ).reset_index(name="seed_count").to_csv(
        output_dir / "accuracy_frequency.csv", index=False
    )

    # Accuracy distribution.
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(df["accuracy"], bins=20, edgecolor="black")
    ax.set_title("Covid3MonthDiscrete: Accuracy Across Random States")
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Number of random states")
    fig.tight_layout()
    fig.savefig(figures_dir / "accuracy_distribution.png", dpi=160)
    plt.close(fig)

    # Accuracy against seed, with mean and median reference lines.
    fig, ax = plt.subplots(figsize=(12, 5))
    ordered = df.sort_values("seed")
    ax.plot(ordered["seed"], ordered["accuracy"], linewidth=0.8)
    ax.axhline(df["accuracy"].mean(), linestyle="--", label="Mean")
    ax.axhline(df["accuracy"].median(), linestyle=":", label="Median")
    ax.set_title("Accuracy by ExtraTrees Random State")
    ax.set_xlabel("Random state")
    ax.set_ylabel("Accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "accuracy_by_seed.png", dpi=160)
    plt.close(fig)

    report = [
        "# Covid3MonthDiscrete Random-State Analysis",
        "",
        f"- Input: `{input_path}`",
        f"- Rows: {len(df)}",
        f"- Unique seeds: {df['seed'].nunique()}",
        f"- Seed range: {df['seed'].min()}–{df['seed'].max()}",
        f"- Dataset(s): {', '.join(map(str, df['dataset'].unique()))}",
        f"- Resample ID(s): {', '.join(map(str, sorted(df['resample_id'].unique())))}",
        "",
        "## Metric summary",
        "",
        summary.to_markdown(floatfmt=".6f"),
        "",
        "## Accuracy extremes",
        "",
        f"- Best: seed {int(best_accuracy['seed'])}, accuracy {best_accuracy['accuracy']:.6f}",
        f"- Worst: seed {int(worst_accuracy['seed'])}, accuracy {worst_accuracy['accuracy']:.6f}",
        f"- Best log loss: seed {int(best_log_loss['seed'])}, log loss {best_log_loss['log_loss']:.6f}",
        "",
        "## Validation",
        "",
        f"- Duplicate run keys: {metadata['duplicate_dataset_resample_method_seed_rows']}",
        f"- Missing metric values: {metadata['missing_values_in_metrics']}",
        "",
        "These experiments vary classifier random_state on the same recorded dataset/resample. "
        "They do not measure variation across different train/test resamples.",
    ]
    (output_dir / "analysis_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )

    print(f"Analysed {len(df)} rows.")
    print(summary.to_string(float_format=lambda x: f"{x:.6f}"))
    print(f"\nOutputs saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
