"""Compare a baseline method against improved methods (CSVs, statistics, figures, report).

Example:
  python -m scripts.compare_methods \
      --results results/quant_base_resample0.csv results/dup_d1-2_resample0.csv \
      --baseline quant_base
"""

import argparse
from pathlib import Path

import matplotlib

# Force non-interactive backend so figure generation works on headless servers/CI
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from scripts.common import (
    LENGTH_BINS,
    LENGTH_LABELS,
    TRAIN_BINS,
    TRAIN_LABELS,
    bootstrap_ci,
    bucket,
    holm_adjust,
    load_paper_rows,
    load_results,
    outcome_labels,
)

DEFAULT_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "f1_macro",
    "mcc",
    "log_loss",
    "auc",
    "brier",
    "ece",
    "fit_time_s",
    "predict_time_s",
    "predict_time_per_series_ms",
]
COLORS = {"improved": "tab:green", "draw": "tab:gray", "regressed": "tab:red"}


# ----------------------------- Computation -----------------------------
def compare_pair(mean, std, baseline, method, metrics, tol):
    """Align seed-averaged per-dataset scores between `baseline` and `method` and compute deltas/outcomes."""
    b, m = mean.xs(baseline, level="method"), mean.xs(method, level="method")
    bs, ms = std.xs(baseline, level="method"), std.xs(method, level="method")
    # Restrict comparison to the exact intersection of datasets evaluated by both methods
    common = b.index.intersection(m.index)
    print(
        f"{method}: {len(common)} datasets in common with {baseline} "
        f"({len(b.index.difference(m.index))} baseline-only, {len(m.index.difference(b.index))} method-only)"
    )
    frames = []
    for metric in metrics:
        f = pd.DataFrame(
            {
                "dataset": common,
                "method": method,
                "metric": metric,
                "baseline": b.loc[common, metric].values,
                "improved": m.loc[common, metric].values,
                "baseline_seed_std": bs.loc[common, metric].values,
                "improved_seed_std": ms.loc[common, metric].values,
            }
        ).dropna(subset=["baseline", "improved"])
        f["delta"] = f["improved"] - f["baseline"]
        f["relative_delta_pct"] = (
            100 * f["delta"] / f["baseline"].abs().replace(0, np.nan)
        )
        # Classify each dataset as improved/draw/regressed respecting metric direction and tie tolerance
        f["outcome"] = outcome_labels(f["delta"], metric, tol)
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def summarise(comp, n_boot, seed):
    """Aggregate per-dataset deltas into effect sizes, bootstrap CIs, win/draw/loss counts, and Holm-adjusted p-values."""
    rows = []
    for (method, metric), g in comp.groupby(["method", "metric"], sort=False):
        d = g["delta"].to_numpy()
        lo, hi = bootstrap_ci(d, n_boot, seed=seed)
        try:
            p = float(wilcoxon(d).pvalue)
        except ValueError:
            # Wilcoxon raises ValueError if all paired differences are identically zero
            p = float("nan")
        counts = g["outcome"].value_counts()
        rows.append(
            {
                "method": method,
                "metric": metric,
                "n_datasets": len(g),
                "baseline_mean": g["baseline"].mean(),
                "improved_mean": g["improved"].mean(),
                "mean_delta": d.mean(),
                "median_delta": np.median(d),
                "ci95_low": lo,
                "ci95_high": hi,
                # Standardized paired effect size (Cohen's d_z) across datasets
                "cohens_dz": (
                    d.mean() / d.std(ddof=1)
                    if len(d) > 1 and d.std(ddof=1) > 0
                    else float("nan")
                ),
                "improved": int(counts.get("improved", 0)),
                "draw": int(counts.get("draw", 0)),
                "regressed": int(counts.get("regressed", 0)),
                "wilcoxon_p": p,
            }
        )
    s = pd.DataFrame(rows)
    # Apply Holm-Bonferroni multiple-testing correction across evaluated methods within each metric
    s["wilcoxon_p_holm"] = np.nan
    for metric, idx in s.groupby("metric").groups.items():
        s.loc[idx, "wilcoxon_p_holm"] = holm_adjust(s.loc[idx, "wilcoxon_p"].to_numpy())
    return s


def seed_robustness(df, baseline, methods, metrics, tol):
    """Check whether improvements hold consistently within each individual random seed (not just seed-averaged)."""
    rows = []
    b = df[df.method == baseline].set_index(["dataset", "seed"])
    for method in methods:
        i = df[df.method == method].set_index(["dataset", "seed"])
        common = b.index.intersection(i.index)
        for metric in metrics:
            if not len(common):
                continue
            delta = (i.loc[common, metric] - b.loc[common, metric]).dropna()
            for seed, d in delta.groupby(level="seed"):
                lab = outcome_labels(d.to_numpy(), metric, tol)
                rows.append(
                    {
                        "method": method,
                        "metric": metric,
                        "seed": seed,
                        "n": len(d),
                        "mean_delta": d.mean(),
                        "improved": int((lab == "improved").sum()),
                        "draw": int((lab == "draw").sum()),
                        "regressed": int((lab == "regressed").sum()),
                    }
                )
    return pd.DataFrame(rows)


def bucket_summary(comp, stats):
    """Stratify performance deltas by series length and training set size regimes."""
    stats = stats.copy()
    if "length_bucket" not in stats:
        stats["length_bucket"] = bucket(
            stats["series_length"], LENGTH_BINS, LENGTH_LABELS
        ).astype(str)
    if "train_size_bucket" not in stats:
        stats["train_size_bucket"] = bucket(
            stats["n_train"], TRAIN_BINS, TRAIN_LABELS
        ).astype(str)
    merged = comp.merge(
        stats[["dataset", "length_bucket", "train_size_bucket"]],
        on="dataset",
        how="left",
    )
    rows = []
    for col in ["length_bucket", "train_size_bucket"]:
        for (method, metric, bk), g in merged.groupby(["method", "metric", col]):
            d = g["delta"].to_numpy()
            rows.append(
                {
                    "bucket_type": col,
                    "bucket": bk,
                    "method": method,
                    "metric": metric,
                    "n_datasets": len(d),
                    "mean_delta": d.mean(),
                    "sem": d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan,
                    "improved": int((g["outcome"] == "improved").sum()),
                    "regressed": int((g["outcome"] == "regressed").sum()),
                }
            )
    return pd.DataFrame(rows)


# ------------------------------- Figures -------------------------------
def plot_scatter(g, method, baseline, metric, path):
    """1:1 parity scatter plot colored by win/draw/loss outcome."""
    fig, ax = plt.subplots(figsize=(6, 6))
    for outcome, h in g.groupby("outcome"):
        ax.scatter(
            h["baseline"],
            h["improved"],
            s=20,
            alpha=0.75,
            c=COLORS[outcome],
            label=f"{outcome} ({len(h)})",
        )
    lo, hi = min(g.baseline.min(), g.improved.min()), max(
        g.baseline.max(), g.improved.max()
    )
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set(
        xlabel=f"{baseline} {metric}",
        ylabel=f"{method} {metric}",
        title=f"{method} vs {baseline}: {metric} per dataset",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_deltas(g, method, metric, path):
    """Distribution histogram + sorted bar chart with non-overlapping annotations."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.5))
    a1.hist(g["delta"], bins=30, edgecolor="black", color="#3182bd")
    a1.axvline(0, ls="--", c="k")
    a1.set(
        xlabel=f"delta {metric}",
        ylabel="datasets",
        title="Distribution of per-dataset change",
    )
    s = g.sort_values("delta").reset_index(drop=True)
    a2.bar(s.index, s["delta"], color=[COLORS[o] for o in s["outcome"]], width=1.0)
    a2.axhline(0, c="k", lw=1)

    # Label the 3 largest regressions and 3 largest improvements without overlapping bars
    top_bottom_indices = list(s.index[:3]) + list(s.index[-3:])
    for idx in top_bottom_indices:
        val = s.loc[idx, "delta"]
        name = s.loc[idx, "dataset"]
        va = "bottom" if val >= 0 else "top"
        y_offset = max(0.005, abs(val) * 0.05) * (1 if val >= 0 else -1)
        a2.annotate(
            name,
            (idx, val + y_offset),
            fontsize=7,
            rotation=45,
            ha="left",
            va=va,
            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7),
        )

    a2.set(
        xlabel="datasets (sorted)",
        ylabel=f"delta {metric}",
        title=f"{method}: sorted changes",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_per_dataset_bar_chart(g, method, metric, path):
    """Full horizontal bar chart listing all datasets on Y-axis aligned cleanly."""
    s = g.sort_values("delta").reset_index(drop=True)
    # Scale figure height dynamically with dataset count so all ~142 labels remain legible
    fig_height = max(10, len(s) * 0.16)
    fig, ax = plt.subplots(figsize=(8, fig_height))

    ax.barh(s["dataset"], s["delta"], color="#4682b4", height=0.7)
    ax.axvline(0, ls="--", c="#2b5c8f", lw=1)

    ax.set_xlabel(f"Mean {metric} change ({method} - baseline)", fontsize=10)
    ax.set_ylabel("Dataset", fontsize=10)
    ax.set_title(f"Mean {metric} change by dataset", fontsize=12)
    ax.tick_params(axis="y", labelsize=7)

    # Clean x limits padding
    min_d, max_d = s["delta"].min(), s["delta"].max()
    pad = (max_d - min_d) * 0.05 if max_d != min_d else 0.01
    ax.set_xlim(min_d - pad, max_d + pad)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_buckets(bs, method, metric, path):
    """Grouped regime bar plots (mean delta +/- SEM) across series length and train size buckets."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, col, order in zip(
        axes, ["length_bucket", "train_size_bucket"], [LENGTH_LABELS, TRAIN_LABELS]
    ):
        g = bs[(bs.bucket_type == col) & (bs.method == method) & (bs.metric == metric)]
        g = g.set_index("bucket").reindex([o for o in order if o in set(g.bucket)])
        ax.bar(g.index, g["mean_delta"], yerr=g["sem"], capsize=4, color="tab:blue")
        ax.axhline(0, c="k", lw=1)
        ax.set_xticks(range(len(g)))
        ax.set_xticklabels(
            [f"{i}\n(n={int(n)})" for i, n in zip(g.index, g["n_datasets"])]
        )
        ax.set(title=col.replace("_", " "), ylabel=f"mean delta {metric}")
    fig.suptitle(f"{method}: change by dataset size")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_forest(summary, metric, path):
    """Forest plot comparing mean effect sizes and 95% bootstrap CIs across all evaluated methods."""
    s = summary[summary.metric == metric].reset_index(drop=True)
    if s.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 1.2 + 0.6 * len(s)))
    ax.errorbar(
        s["mean_delta"],
        range(len(s)),
        xerr=[s["mean_delta"] - s["ci95_low"], s["ci95_high"] - s["mean_delta"]],
        fmt="o",
        capsize=4,
    )
    ax.axvline(0, ls="--", c="k")
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels(s["method"])
    ax.set(xlabel=f"mean delta {metric} (95% bootstrap CI)", title="Effect vs baseline")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


# ------------------------------- Markdown Generator -------------------------------
def generate_markdown_report(
    summary: pd.DataFrame, df: pd.DataFrame, baseline: str, resample: int, methods: list
) -> str:
    """Format the statistical summary table into a GitHub-flavored Markdown report."""
    seed_counts = df.groupby("method")["seed"].nunique().to_dict()

    md = [
        f"# Benchmark Comparison: {', '.join(methods)} vs `{baseline}` (Resample {resample})\n",
        f"**Seeds per method:** `{seed_counts}`  ",
        "**Notes:** $\\Delta = \\text{method} - \\text{baseline}$. Metrics are direction-aware (lower is better for log loss, Brier, ECE, and fit/predict execution times).\n",
        "## Summary Results\n",
    ]

    disp = summary.copy()
    disp["ci95"] = disp.apply(
        lambda r: f"[{r['ci95_low']:+.5f}, {r['ci95_high']:+.5f}]", axis=1
    )
    disp["win_draw_loss"] = disp.apply(
        lambda r: f"{r['improved']} / {r['draw']} / {r['regressed']}", axis=1
    )
    disp["mean_delta"] = disp["mean_delta"].apply(lambda v: f"{v:+.5f}")
    disp["baseline_mean"] = disp["baseline_mean"].apply(lambda v: f"{v:.5f}")
    disp["improved_mean"] = disp["improved_mean"].apply(lambda v: f"{v:.5f}")
    disp["wilcoxon_p_holm"] = disp["wilcoxon_p_holm"].apply(
        lambda v: f"{v:.5f}" if pd.notna(v) else "N/A"
    )

    cols_map = {
        "method": "Method",
        "metric": "Metric",
        "n_datasets": "Datasets",
        "baseline_mean": "Baseline Mean",
        "improved_mean": "Method Mean",
        "mean_delta": "Mean Δ",
        "ci95": "95% Bootstrap CI",
        "win_draw_loss": "Win / Draw / Loss",
        "wilcoxon_p_holm": "Wilcoxon p (Holm)",
    }

    out_table = disp[list(cols_map.keys())].rename(columns=cols_map)
    md.append(out_table.to_markdown(index=False))
    return "\n".join(md)


# --------------------------------- Main ---------------------------------
def main():
    """Load result logs, compute paired statistics and regime breakdowns, and export CSVs/plots/report."""
    p = argparse.ArgumentParser(
        description="Compare multiple Quant experiment CSVs against a baseline."
    )
    p.add_argument(
        "--results", nargs="+", required=True, help="One or more result CSV paths"
    )
    p.add_argument(
        "--baseline",
        default="quant_base",
        help="Baseline method name (default: quant_base)",
    )
    p.add_argument(
        "--methods", nargs="*", help="default: every method present except baseline"
    )
    p.add_argument("--metrics", nargs="*", default=DEFAULT_METRICS)
    p.add_argument("--plot-metrics", nargs="*", default=["accuracy", "log_loss"])
    p.add_argument("--resample", type=int, default=0)
    p.add_argument("--stats", default="docs/dataset_stats.csv")
    p.add_argument(
        "--include-paper", default=None, help="Paper per-resample accuracy CSV"
    )
    p.add_argument("--tie-tolerance", type=float, default=0.0)
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--out", default="results/comparison")
    p.add_argument("--no-figures", action="store_true")
    a = p.parse_args()

    out = Path(a.out)
    (out / "csvs").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    df = load_results(a.results)
    if "resample_id" in df:
        df = df[df["resample_id"] == a.resample]
    if a.include_paper:
        df = pd.concat(
            [df, load_paper_rows(a.include_paper, a.resample)], ignore_index=True
        )

    if a.baseline not in set(df["method"]):
        raise SystemExit(
            f"Baseline '{a.baseline}' not found in provided CSVs. Methods present: {sorted(set(df['method']))}"
        )

    methods = a.methods or sorted(set(df["method"]) - {a.baseline})
    metrics = [m for m in a.metrics if m in df.columns and df[m].notna().any()]
    df = df[df["method"].isin([a.baseline, *methods])]

    # Collapse across random seeds first so datasets are the independent statistical units
    g = df.groupby(["dataset", "method"])
    mean, std = g[metrics].mean(), g[metrics].std(ddof=0)
    print(df.groupby("method")["seed"].nunique().rename("seeds").to_string())

    comp = pd.concat(
        [
            compare_pair(mean, std, a.baseline, m, metrics, a.tie_tolerance)
            for m in methods
        ],
        ignore_index=True,
    )
    if comp.empty:
        raise SystemExit(
            "No datasets in common between baseline and evaluated methods."
        )

    summary = summarise(comp, a.n_boot, seed=0)
    robust = seed_robustness(df, a.baseline, methods, metrics, a.tie_tolerance)

    comp.to_csv(out / "csvs" / "per_dataset_comparison.csv", index=False)
    summary.to_csv(out / "csvs" / "summary.csv", index=False)
    robust.to_csv(out / "csvs" / "seed_robustness.csv", index=False)

    buckets = pd.DataFrame()
    if Path(a.stats).exists():
        buckets = bucket_summary(comp, pd.read_csv(a.stats))
        buckets.to_csv(out / "csvs" / "bucket_summary.csv", index=False)
    else:
        print(f"{a.stats} not found: skipping size-bucket analysis")

    if not a.no_figures:
        for method in methods:
            for metric in a.plot_metrics:
                gm = comp[(comp.method == method) & (comp.metric == metric)]
                if gm.empty:
                    continue
                plot_scatter(
                    gm,
                    method,
                    a.baseline,
                    metric,
                    out / "figures" / f"{method}_{metric}_scatter.png",
                )
                plot_deltas(
                    gm,
                    method,
                    metric,
                    out / "figures" / f"{method}_{metric}_deltas.png",
                )
                plot_per_dataset_bar_chart(
                    gm,
                    method,
                    metric,
                    out / "figures" / f"{method}_{metric}_per_dataset.png",
                )
                if not buckets.empty:
                    plot_buckets(
                        buckets,
                        method,
                        metric,
                        out / "figures" / f"{method}_{metric}_buckets.png",
                    )
        for metric in a.plot_metrics:
            plot_forest(summary, metric, out / "figures" / f"forest_{metric}.png")

    report_content = generate_markdown_report(
        summary, df, a.baseline, a.resample, methods
    )
    (out / "report.md").write_text(report_content, encoding="utf-8")

    print("\n" + report_content)
    print(f"\nWrote CSVs, figures, and report.md to {out.resolve()}")


if __name__ == "__main__":
    main()
