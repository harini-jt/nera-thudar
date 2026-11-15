"""Compute and print filled Markdown tables (Table 1 & Table 2) for README.md."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from scripts.common import bootstrap_ci, holm_adjust, outcome_labels

PAPER_RESAMPLES = Path("reference/quant_src_code/results/accuracy_resamples_ucr142.csv")

# Table 1 reproduction files and labels
TABLE1_RUNS = [
    (
        "**Resample 0** (3 seeds: `7, 11, 42`)",
        Path("results/reproduction/quant_base_resample0.csv"),
    ),
    (
        "**All 30 Resamples (`0..29`)** (`seed=42`)",
        Path("results/reproduction/quant_base_all-resamples_seed-42.csv"),
    ),
    (
        "**All 30 Resamples (`0..29`)** (unseeded)",
        Path("results/reproduction/quant_base_all-resamples_no-seed.csv"),
    ),
]

# Table 2 method order, category labels, and descriptions
TABLE2_SPECS = [
    ("Baseline", "quant_base", "QUANT (`depth=6, div=4`) + 200 ExtraTrees (`entropy`)"),
    ("Reference", "majority", "Prior-class dummy baseline"),
    ("Reference", "minirocket", "MiniRocket (`10k` kernels) + `RidgeClassifierCV`"),
    ("Criterion", "quant_gini", "Split criterion changed from `entropy` to `gini`"),
    (
        "Weighting",
        "dup_d1-2",
        "Duplicate interval features at depths `1, 2` ($\\times 2$)",
    ),
    (
        "Weighting",
        "dup_d1_4_d2_2",
        "Duplicate depth `1` ($\\times 4$) and depth `2` ($\\times 2$)",
    ),
    (
        "Resolution",
        "more_quantiles_d1-2",
        "Finer quantile grid (`div=2`) at depths `1, 2`",
    ),
    (
        "Resolution",
        "depth_div_m2_m8",
        "`div=2` at depths `1..3`, `div=8` at depths `5..6`",
    ),
    ("Features", "quant_stats", "Append interval standard deviation and OLS slope"),
    (
        "Regularization",
        "random_dropout_0.1",
        "Random `10%` feature column dropout before forest",
    ),
    (
        "Regularization",
        "random_dropout_0.2",
        "Random `20%` feature column dropout before forest",
    ),
]

TABLE2_FILES = [
    Path("results/reproduction/quant_base_resample0.csv"),
    *sorted(Path("results/other_baselines").glob("*.csv")),
    *sorted(Path("results/improvements").glob("*.csv")),
]


def format_p(p: float) -> str:
    if not np.isfinite(p):
        return "N/A"
    return "< 0.0001" if p < 1e-4 else f"{p:.4f}"


def build_table1() -> str:
    paper_wide = pd.read_csv(PAPER_RESAMPLES)
    paper_long = paper_wide.melt(
        id_vars="dataset", var_name="resample_id", value_name="paper_acc"
    )
    paper_long["resample_id"] = paper_long["resample_id"].astype(int)

    rows = []
    for label, path in TABLE1_RUNS:
        if not path.exists():
            rows.append(f"| {label} | — | *(missing `{path}`)* | — | — | — |")
            continue

        raw = pd.read_csv(path)
        per_split = raw.groupby(["dataset", "resample_id"], as_index=False)[
            "accuracy"
        ].mean()
        paired = per_split.merge(paper_long, on=["dataset", "resample_id"])
        ds = paired.groupby("dataset", as_index=False).agg(
            ours=("accuracy", "mean"),
            paper=("paper_acc", "mean"),
        )
        delta = ds["ours"] - ds["paper"]
        w = int((delta > 0).sum())
        d = int((delta == 0).sum())
        l = int((delta < 0).sum())
        rows.append(
            f"| {label} | {len(ds)} | {ds['ours'].mean():.4f} | "
            f"{ds['paper'].mean():.4f} | {delta.mean():+.4f} | {w} / {d} / {l} |"
        )

    header = [
        "## Table 1: QUANT Reproduction vs. Dempster et al. (2024)",
        "",
        "| Setting / Split | Datasets | Mean Accuracy (Ours) | Mean Accuracy (Paper) | Mean $\\Delta$ (Ours $-$ Paper) | Win / Draw / Loss vs. Paper |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    return "\n".join(header + rows)


def build_table2() -> str:
    frames = [pd.read_csv(p) for p in TABLE2_FILES if p.exists()]
    df = pd.concat(frames, ignore_index=True)
    if "resample_id" in df.columns:
        df = df[df["resample_id"] == 0]

    # Collapse across seeds per (dataset, method)
    ds_means = (
        df.groupby(["dataset", "method"], as_index=False)["accuracy"]
        .mean()
        .pivot(index="dataset", columns="method", values="accuracy")
    )
    base = ds_means["quant_base"].dropna()

    # First pass: collect raw Wilcoxon p-values across all non-baseline methods for joint Holm adjustment
    stats_by_method = {}
    raw_pvals = []
    eval_methods = [
        m for _, m, _ in TABLE2_SPECS if m != "quant_base" and m in ds_means
    ]

    for method in eval_methods:
        common = base.index.intersection(ds_means[method].dropna().index)
        b_vals = base.loc[common].to_numpy()
        m_vals = ds_means.loc[common, method].to_numpy()
        delta = m_vals - b_vals
        lo, hi = bootstrap_ci(delta, n_boot=10000, seed=0)
        outcomes = outcome_labels(delta, "accuracy", tol=0.0)
        try:
            p_raw = float(wilcoxon(delta).pvalue)
        except ValueError:
            p_raw = float("nan")

        raw_pvals.append(p_raw)
        stats_by_method[method] = {
            "mean_acc": m_vals.mean(),
            "mean_delta": delta.mean(),
            "ci": f"[{lo:+.4f}, {hi:+.4f}]",
            "wdl": (
                f"{int((outcomes == 'improved').sum())} / "
                f"{int((outcomes == 'draw').sum())} / "
                f"{int((outcomes == 'regressed').sum())}"
            ),
        }

    adj_pvals = dict(zip(eval_methods, holm_adjust(raw_pvals)))

    rows = []
    for category, method, desc in TABLE2_SPECS:
        if method == "quant_base":
            rows.append(
                f"| **{category}** | `{method}` | {desc} | {base.mean():.4f} | — | — | — | — |"
            )
        elif method not in stats_by_method:
            rows.append(
                f"| **{category}** | `{method}` | {desc} | *(missing)* | — | — | — | — |"
            )
        else:
            st = stats_by_method[method]
            rows.append(
                f"| **{category}** | `{method}` | {desc} | {st['mean_acc']:.4f} | "
                f"{st['mean_delta']:+.4f} | {st['ci']} | {st['wdl']} | {format_p(adj_pvals[method])} |"
            )

    header = [
        "### Table 2: Baselines & QUANT Variants on Resample 0 (`142` Datasets, Seeds `7, 11, 42`)",
        "",
        "| Category | Method (`--methods` key) | Description | Mean Accuracy | Mean $\\Delta$ vs. `quant_base` | 95% Bootstrap CI | Win / Draw / Loss | Wilcoxon $p$ (Holm) |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    return "\n".join(header + rows)


if __name__ == "__main__":
    print(build_table1())
    print("\n")
    print(build_table2())
