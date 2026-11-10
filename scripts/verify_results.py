"""
Verify reproduction results against the official QUANT UCR results.

Examples
--------
Dataset-level mean comparison:
    python -m scripts.verify_results --my_results="results/run.csv" \
        --outputfolder="results/comparison" --mean

Resample-level comparison:
    python -m scripts.verify_results --my_results="results/run.csv" \
        --outputfolder="results/comparison" --resample


NOT SO GOOD  -  MAY BE DONT USE THIS ONE
Detailed analysis for one dataset:
    python -m scripts.verify_results --my_results="results/run.csv" \
        --outputfolder="results/comparison" --resample \
        --dataset="Covid3MonthDiscrete"

Dependencies:
    pandas, matplotlib
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = PROJECT_ROOT / "reference" / "quant" / "results"

MEAN_REFERENCE_FILE = REFERENCE_DIR / "accuracy_mean_ucr142.csv"
RESAMPLE_REFERENCE_FILE = REFERENCE_DIR / "accuracy_resamples_ucr142.csv"

EXPECTED_RESAMPLES = 30
DEFAULT_TIE_TOLERANCE = 0.001

REQUIRED_RESULT_COLUMNS = {"dataset", "resample_id", "accuracy"}


# ---------------------------------------------------------------------------
# Input and validation
# ---------------------------------------------------------------------------


def parse_arguments():
    """Read command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare reproduction results with official QUANT results."
    )

    parser.add_argument(
        "--my_results",
        required=True,
        help="Path to the reproduction results CSV.",
    )
    parser.add_argument(
        "--outputfolder",
        required=True,
        help="Directory where comparison outputs will be saved.",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--mean",
        action="store_true",
        help="Compare dataset-level mean accuracy.",
    )
    mode.add_argument(
        "--resample",
        action="store_true",
        help="Compare individual resample accuracies.",
    )

    parser.add_argument(
        "--dataset",
        help="Optionally analyse only one dataset in detail.",
    )
    parser.add_argument(
        "--tie_tolerance",
        type=float,
        default=DEFAULT_TIE_TOLERANCE,
        help=f"Absolute difference treated as a draw (default: {DEFAULT_TIE_TOLERANCE}).",
    )

    return parser.parse_args()


def load_reproduction_results(file_path):
    """Load reproduction results and check required columns."""
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Reproduction results file not found: {file_path}")

    results = pd.read_csv(file_path)
    missing = REQUIRED_RESULT_COLUMNS - set(results.columns)

    if missing:
        raise ValueError(
            "Reproduction CSV is missing required columns: "
            + ", ".join(sorted(missing))
        )

    results = results.copy()
    results["dataset"] = results["dataset"].astype("string").str.strip()
    results["resample_id"] = results["resample_id"].astype("string").str.strip()
    results["accuracy"] = pd.to_numeric(results["accuracy"], errors="coerce")

    return results


def validate_datasets(results):
    """
    Keep datasets with exactly 30 unique resamples and valid accuracy values.

    A dataset is skipped if it has a duplicate resample ID, a missing resample
    ID, a missing/non-numeric accuracy, or anything other than 30 rows.
    """
    valid_frames = []
    skipped = []

    for dataset, group in results.groupby("dataset", dropna=False):
        dataset_name = str(dataset) if pd.notna(dataset) else "<missing dataset>"
        reasons = []

        if pd.isna(dataset):
            reasons.append("Missing dataset name")

        if group["resample_id"].isna().any() or (group["resample_id"] == "").any():
            reasons.append("Missing resample ID")

        duplicate_count = group["resample_id"].duplicated(keep=False).sum()
        if duplicate_count:
            reasons.append(f"{duplicate_count} rows have duplicate resample IDs")

        if group["accuracy"].isna().any():
            reasons.append("Missing or non-numeric accuracy")

        unique_resamples = group["resample_id"].nunique(dropna=True)
        if len(group) != EXPECTED_RESAMPLES:
            reasons.append(f"Expected {EXPECTED_RESAMPLES} rows, found {len(group)}")
        if unique_resamples != EXPECTED_RESAMPLES:
            reasons.append(
                f"Expected {EXPECTED_RESAMPLES} unique resamples, "
                f"found {unique_resamples}"
            )

        if reasons:
            skipped.append(
                {
                    "dataset": dataset_name,
                    "available_rows": len(group),
                    "unique_resamples": unique_resamples,
                    "expected_resamples": EXPECTED_RESAMPLES,
                    "reason": "; ".join(reasons),
                }
            )
        else:
            valid_frames.append(group)

    valid = (
        pd.concat(valid_frames, ignore_index=True)
        if valid_frames
        else results.iloc[0:0].copy()
    )
    skipped_df = pd.DataFrame(
        skipped,
        columns=[
            "dataset",
            "available_rows",
            "unique_resamples",
            "expected_resamples",
            "reason",
        ],
    )

    return valid, skipped_df


def load_mean_reference():
    """Load the official dataset-level mean accuracy file."""
    if not MEAN_REFERENCE_FILE.is_file():
        raise FileNotFoundError(
            f"Official mean reference file not found: {MEAN_REFERENCE_FILE}"
        )

    reference = pd.read_csv(MEAN_REFERENCE_FILE)
    required = {"dataset", "accuracy"}
    missing = required - set(reference.columns)

    if missing:
        raise ValueError(
            "Official mean reference is missing columns: " + ", ".join(sorted(missing))
        )

    reference = reference[["dataset", "accuracy"]].copy()
    reference["dataset"] = reference["dataset"].astype("string").str.strip()
    reference["paper_accuracy"] = pd.to_numeric(
        reference.pop("accuracy"), errors="coerce"
    )
    return reference


def load_resample_reference():
    """Convert the official wide-format resample file into long format."""
    if not RESAMPLE_REFERENCE_FILE.is_file():
        raise FileNotFoundError(
            f"Official resample reference file not found: {RESAMPLE_REFERENCE_FILE}"
        )

    reference = pd.read_csv(RESAMPLE_REFERENCE_FILE)
    if "dataset" not in reference.columns:
        raise ValueError("Official resample reference must contain a 'dataset' column.")

    resample_columns = [column for column in reference.columns if str(column).isdigit()]
    if not resample_columns:
        raise ValueError("No numeric resample columns found in official reference.")

    reference = reference.melt(
        id_vars="dataset",
        value_vars=resample_columns,
        var_name="resample_id",
        value_name="paper_accuracy",
    )

    reference["dataset"] = reference["dataset"].astype("string").str.strip()
    reference["resample_id"] = reference["resample_id"].astype("string").str.strip()
    reference["paper_accuracy"] = pd.to_numeric(
        reference["paper_accuracy"], errors="coerce"
    )
    return reference


# ---------------------------------------------------------------------------
# Comparison calculations
# ---------------------------------------------------------------------------


def calculate_win_draw_loss(comparison, tolerance):
    """Add per-row difference and win/draw/loss labels."""
    comparison = comparison.copy()
    comparison["difference"] = comparison["my_accuracy"] - comparison["paper_accuracy"]
    comparison["absolute_difference"] = comparison["difference"].abs()

    comparison["result"] = "Draw"
    comparison.loc[comparison["difference"] > tolerance, "result"] = "Win"
    comparison.loc[comparison["difference"] < -tolerance, "result"] = "Loss"

    return comparison


def summarize_comparison(comparison):
    """Calculate overall comparison metrics."""
    if comparison.empty:
        return pd.DataFrame(
            [
                {
                    "matched_count": 0,
                    "mean_difference": None,
                    "mae": None,
                    "rmse": None,
                    "wins": 0,
                    "draws": 0,
                    "losses": 0,
                }
            ]
        )

    difference = comparison["difference"]
    return pd.DataFrame(
        [
            {
                "matched_count": len(comparison),
                "my_mean_accuracy": comparison["my_accuracy"].mean(),
                "paper_mean_accuracy": comparison["paper_accuracy"].mean(),
                "mean_difference": difference.mean(),
                "mae": comparison["absolute_difference"].mean(),
                "rmse": (difference.pow(2).mean()) ** 0.5,
                "wins": int((comparison["result"] == "Win").sum()),
                "draws": int((comparison["result"] == "Draw").sum()),
                "losses": int((comparison["result"] == "Loss").sum()),
            }
        ]
    )


def compare_mean_results(results, tolerance):
    """Aggregate reproduction results and compare dataset means."""
    mine = results.groupby("dataset", as_index=False).agg(
        my_accuracy=("accuracy", "mean"),
        my_std=("accuracy", "std"),
        resample_count=("accuracy", "count"),
    )
    paper = load_mean_reference()

    comparison = mine.merge(paper, on="dataset", how="inner")
    comparison = calculate_win_draw_loss(comparison, tolerance)

    unmatched_mine = mine[~mine["dataset"].isin(paper["dataset"])].copy()
    unmatched_paper = paper[~paper["dataset"].isin(mine["dataset"])].copy()

    return comparison, unmatched_mine, unmatched_paper


def compare_resample_results(results, tolerance):
    """Compare reproduction and official accuracy for each dataset/resample."""
    mine = results[["dataset", "resample_id", "accuracy"]].rename(
        columns={"accuracy": "my_accuracy"}
    )
    paper = load_resample_reference()

    comparison = mine.merge(
        paper,
        on=["dataset", "resample_id"],
        how="inner",
        validate="one_to_one",
    )
    comparison = calculate_win_draw_loss(comparison, tolerance)

    mine_keys = mine[["dataset", "resample_id"]].drop_duplicates()
    paper_keys = paper[["dataset", "resample_id"]].drop_duplicates()

    unmatched_mine = mine_keys.merge(
        paper_keys, on=["dataset", "resample_id"], how="left", indicator=True
    )
    unmatched_mine = unmatched_mine[unmatched_mine["_merge"] == "left_only"].drop(
        columns="_merge"
    )

    unmatched_paper = paper_keys.merge(
        mine_keys, on=["dataset", "resample_id"], how="left", indicator=True
    )
    unmatched_paper = unmatched_paper[unmatched_paper["_merge"] == "left_only"].drop(
        columns="_merge"
    )

    return comparison, unmatched_mine, unmatched_paper


def create_dataset_statistics(comparison):
    """Summarise resample-level differences for each dataset."""
    if comparison.empty:
        return pd.DataFrame()

    return (
        comparison.groupby("dataset", as_index=False)
        .agg(
            matched_resamples=("difference", "count"),
            my_mean_accuracy=("my_accuracy", "mean"),
            paper_mean_accuracy=("paper_accuracy", "mean"),
            mean_difference=("difference", "mean"),
            mae=("absolute_difference", "mean"),
            max_absolute_difference=("absolute_difference", "max"),
            wins=("result", lambda values: (values == "Win").sum()),
            draws=("result", lambda values: (values == "Draw").sum()),
            losses=("result", lambda values: (values == "Loss").sum()),
        )
        .sort_values("mae", ascending=False)
    )


# ---------------------------------------------------------------------------
# Output and plots
# ---------------------------------------------------------------------------


def save_csv(dataframe, output_dir, filename):
    """Save a dataframe to the CSV output directory."""
    dataframe.to_csv(output_dir / filename, index=False)


def plot_mean_comparison(comparison, figures_dir):
    """Create scatter and dataset-difference plots for mean mode."""
    if comparison.empty:
        return

    minimum = min(comparison["my_accuracy"].min(), comparison["paper_accuracy"].min())
    maximum = max(comparison["my_accuracy"].max(), comparison["paper_accuracy"].max())

    plt.figure(figsize=(8, 7))
    plt.scatter(comparison["paper_accuracy"], comparison["my_accuracy"], alpha=0.7)
    plt.plot([minimum, maximum], [minimum, maximum], linestyle="--")
    plt.xlabel("Official mean accuracy")
    plt.ylabel("Reproduction mean accuracy")
    plt.title("Dataset-level mean accuracy: reproduction vs QUANT")
    plt.tight_layout()
    plt.savefig(figures_dir / "mean_accuracy_scatter.png", dpi=200)
    plt.close()

    ordered = comparison.sort_values("difference")
    plt.figure(figsize=(12, max(6, len(ordered) * 0.18)))
    plt.barh(ordered["dataset"], ordered["difference"])
    plt.axvline(0, linestyle="--")
    plt.xlabel("Reproduction - official accuracy")
    plt.ylabel("Dataset")
    plt.title("Dataset-level accuracy differences")
    plt.tight_layout()
    plt.savefig(figures_dir / "dataset_accuracy_difference.png", dpi=200)
    plt.close()


def plot_resample_comparison(comparison, figures_dir):
    """Create distribution and dataset-level plots for resample mode."""
    if comparison.empty:
        return

    plt.figure(figsize=(8, 6))
    plt.hist(comparison["difference"].dropna(), bins=30)
    plt.axvline(0, linestyle="--")
    plt.xlabel("Reproduction - official accuracy")
    plt.ylabel("Number of resamples")
    plt.title("Distribution of resample accuracy differences")
    plt.tight_layout()
    plt.savefig(figures_dir / "resample_difference_distribution.png", dpi=200)
    plt.close()

    statistics = create_dataset_statistics(comparison)
    if not statistics.empty:
        ordered = statistics.sort_values("mean_difference")
        plt.figure(figsize=(12, max(6, len(ordered) * 0.18)))
        plt.barh(ordered["dataset"], ordered["mean_difference"])
        plt.axvline(0, linestyle="--")
        plt.xlabel("Mean resample difference")
        plt.ylabel("Dataset")
        plt.title("Mean resample accuracy difference by dataset")
        plt.tight_layout()
        plt.savefig(figures_dir / "dataset_mean_resample_difference.png", dpi=200)
        plt.close()

        top = statistics.head(20).sort_values("mae")
        plt.figure(figsize=(10, 7))
        plt.barh(top["dataset"], top["mae"])
        plt.xlabel("Mean absolute error")
        plt.ylabel("Dataset")
        plt.title("20 datasets with largest resample discrepancies")
        plt.tight_layout()
        plt.savefig(figures_dir / "top_dataset_discrepancies.png", dpi=200)
        plt.close()


def create_output_directories(output_folder):
    """Create and return the output, figures, and CSV directories."""
    root = Path(output_folder)
    figures = root / "figures"
    csvs = root / "csvs"

    figures.mkdir(parents=True, exist_ok=True)
    csvs.mkdir(parents=True, exist_ok=True)

    return figures, csvs


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------


def print_validation_report(total_datasets, valid_results, skipped):
    """Print dataset validation results."""
    valid_count = valid_results["dataset"].nunique()
    print("\nDataset validation")
    print("------------------")
    print(f"Total datasets:    {total_datasets:>6}")
    print(f"Eligible datasets: {valid_count:>6}")
    print(f"Skipped datasets:  {len(skipped):>6}")

    if not skipped.empty:
        print("\nSkipped datasets:")
        for row in skipped.itertuples(index=False):
            print(f"- {row.dataset}: {row.reason}")


def main():
    args = parse_arguments()

    if args.tie_tolerance < 0:
        raise ValueError("--tie_tolerance must be zero or greater.")

    figures_dir, csvs_dir = create_output_directories(args.outputfolder)

    results = load_reproduction_results(args.my_results)
    total_datasets = results["dataset"].nunique(dropna=True)

    valid_results, skipped = validate_datasets(results)
    save_csv(skipped, csvs_dir, "skipped_datasets.csv")
    print_validation_report(total_datasets, valid_results, skipped)

    if args.dataset:
        valid_results = valid_results[valid_results["dataset"] == args.dataset].copy()

        if valid_results.empty:
            available = sorted(
                results["dataset"].dropna().astype(str).unique().tolist()
            )
            raise ValueError(
                f"Dataset '{args.dataset}' is not eligible or was not found. "
                f"Check csvs/skipped_datasets.csv. "
                f"Available datasets include: {available[:10]}"
            )

    if valid_results.empty:
        print("\nNo eligible datasets to compare. See skipped_datasets.csv.")
        return

    if args.mean:
        comparison, unmatched_mine, unmatched_paper = compare_mean_results(
            valid_results, args.tie_tolerance
        )
        save_csv(comparison, csvs_dir, "dataset_mean_comparison.csv")
        save_csv(unmatched_mine, csvs_dir, "datasets_missing_from_reference.csv")
        save_csv(unmatched_paper, csvs_dir, "datasets_missing_from_reproduction.csv")

        summary = summarize_comparison(comparison)
        save_csv(summary, csvs_dir, "overall_summary.csv")
        plot_mean_comparison(comparison, figures_dir)

    else:
        comparison, unmatched_mine, unmatched_paper = compare_resample_results(
            valid_results, args.tie_tolerance
        )
        save_csv(comparison, csvs_dir, "resample_comparison.csv")
        save_csv(unmatched_mine, csvs_dir, "resamples_missing_from_reference.csv")
        save_csv(unmatched_paper, csvs_dir, "resamples_missing_from_reproduction.csv")

        statistics = create_dataset_statistics(comparison)
        save_csv(statistics, csvs_dir, "dataset_statistics.csv")

        summary = summarize_comparison(comparison)
        save_csv(summary, csvs_dir, "overall_summary.csv")
        plot_resample_comparison(comparison, figures_dir)

        # For a selected dataset, save its individual resample rows separately.
        if args.dataset:
            save_csv(
                comparison,
                csvs_dir,
                f"{args.dataset}_resample_details.csv",
            )

    print("\nComparison complete.")
    print(f"Matched comparisons: {len(comparison)}")
    print(f"Output directory: {Path(args.outputfolder).resolve()}")
    print(f"CSV files: {csvs_dir.resolve()}")
    print(f"Figures: {figures_dir.resolve()}")


if __name__ == "__main__":
    main()
