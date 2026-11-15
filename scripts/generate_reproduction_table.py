"""Generate a per-dataset Markdown reproduction table comparing our QUANT runs to Dempster et al. (2024).

Examples:
  # Default: multi-resample comparison against reference/quant_src_code/results/accuracy_resamples_ucr142.csv
  python -m scripts.generate_reproduction_table

  # Single-resample comparison (e.g. Resample 0 with 3 seeds)
  python -m scripts.generate_reproduction_table \
      --ours results/reproduction/quant_base_resample0.csv \
      --resample 0 \
      --out docs/reproduction_table_resample0.md
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from scripts.common import bootstrap_ci


def pair_with_paper(paper_path: Path, per_split: pd.DataFrame) -> pd.DataFrame:
    """Pair per-(dataset, resample_id) accuracies with either a wide resample CSV or a 30-resample mean CSV."""
    paper = pd.read_csv(paper_path)

    # Case 1: Summary CSV (dataset, accuracy, ...) -> aggregate our runs across resamples first, then merge
    if "accuracy" in paper.columns:
        ours = per_split.groupby("dataset", as_index=False).agg(
            ours_acc=("accuracy", "mean"),
            ours_std=("accuracy", "std"),
            n_resamples=("resample_id", "nunique"),
        )
        return (
            paper[["dataset", "accuracy"]]
            .rename(columns={"accuracy": "paper_acc"})
            .merge(ours, on="dataset")
        )

    # Case 2: Wide per-resample CSV (dataset, 0, 1, ..., 29) -> melt and match on exact (dataset, resample_id) pairs
    paper_long = paper.melt(
        id_vars="dataset", var_name="resample_id", value_name="paper_acc"
    )
    paper_long["resample_id"] = paper_long["resample_id"].astype(int)
    paired = per_split.merge(paper_long, on=["dataset", "resample_id"])
    return paired.groupby("dataset", as_index=False).agg(
        paper_acc=("paper_acc", "mean"),
        ours_acc=("accuracy", "mean"),
        ours_std=("accuracy", "std"),
        n_resamples=("resample_id", "nunique"),
    )


def main():
    """Merge our reproduction results with the paper's reference accuracies and write a Markdown report."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--paper",
        default="reference/quant_src_code/results/accuracy_resamples_ucr142.csv",
        help="Path to paper reference CSV (accuracy_resamples_ucr142.csv or accuracy_mean_ucr142.csv)",
    )
    p.add_argument(
        "--ours",
        default="results/reproduction/quant_base_all-resamples_seed-42.csv",
        help="Path to our QUANT reproduction results CSV",
    )
    p.add_argument(
        "--resample",
        type=int,
        default=None,
        help="Optional resample ID to filter (omit to average across all resamples in --ours)",
    )
    p.add_argument(
        "--tie-tolerance",
        type=float,
        default=0.0,
        help="Absolute accuracy difference threshold counted as a draw (default: 0.0)",
    )
    p.add_argument(
        "--out",
        default="docs/reproduction_table.md",
        help="Output Markdown file path",
    )
    args = p.parse_args()

    ours_raw = pd.read_csv(args.ours)
    if args.resample is not None and "resample_id" in ours_raw.columns:
        ours_raw = ours_raw[ours_raw["resample_id"] == args.resample]

    # Collapse random seeds within each (dataset, resample_id) first so seeds don't inflate resample count
    per_split = ours_raw.groupby(["dataset", "resample_id"], as_index=False)[
        "accuracy"
    ].mean()

    df = (
        pair_with_paper(Path(args.paper), per_split)
        .sort_values("dataset")
        .reset_index(drop=True)
    )
    if df.empty:
        raise SystemExit(
            "No overlapping datasets/resamples found between --paper and --ours."
        )

    df["delta"] = df["ours_acc"] - df["paper_acc"]
    tol = args.tie_tolerance
    df["outcome"] = np.where(
        df["delta"] > tol, "win", np.where(df["delta"] < -tol, "loss", "draw")
    )

    # Compute dataset-level paired summary statistics
    lo, hi = bootstrap_ci(df["delta"].to_numpy())
    try:
        p_val = float(wilcoxon(df["delta"]).pvalue)
    except ValueError:
        p_val = float("nan")

    wins = int((df["outcome"] == "win").sum())
    draws = int((df["outcome"] == "draw").sum())
    losses = int((df["outcome"] == "loss").sum())

    # Format display columns for clean Markdown rendering
    table = df[
        [
            "dataset",
            "n_resamples",
            "paper_acc",
            "ours_acc",
            "ours_std",
            "delta",
            "outcome",
        ]
    ].rename(
        columns={
            "dataset": "Dataset",
            "n_resamples": "Resamples",
            "paper_acc": "Paper Mean Acc",
            "ours_acc": "Ours Mean Acc",
            "ours_std": "Ours Std",
            "delta": "Delta (Ours - Paper)",
            "outcome": "Outcome",
        }
    )
    for col in ["Paper Mean Acc", "Ours Mean Acc", "Ours Std"]:
        table[col] = table[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
    table["Delta (Ours - Paper)"] = table["Delta (Ours - Paper)"].map(
        lambda x: f"{x:+.4f}"
    )

    split_label = (
        f"Resample {args.resample}"
        if args.resample is not None
        else f"up to {int(df['n_resamples'].max())} resamples"
    )
    lines = [
        f"# Per-Dataset QUANT Reproduction vs. Dempster et al. (2024) ({split_label})",
        "",
        f"- **Datasets compared:** {len(df)}",
        f"- **Paper Mean Accuracy:** {df['paper_acc'].mean():.4f}",
        f"- **Ours Mean Accuracy:** {df['ours_acc'].mean():.4f}",
        f"- **Mean $\\Delta$ (Ours $-$ Paper):** {df['delta'].mean():+.4f} "
        f"(95% CI [{lo:+.4f}, {hi:+.4f}], Wilcoxon $p = {p_val:.4g}$)",
        f"- **Win / Draw / Loss:** {wins} / {draws} / {losses}",
        "",
        table.to_markdown(index=False),
    ]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path} ({len(df)} datasets, W/D/L = {wins}/{draws}/{losses})")


if __name__ == "__main__":
    main()
