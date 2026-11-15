"""Consistency check between two runs of the same method (e.g. two machines or builds).

  python -m scripts.cross_machine --a results/quant_base.csv --b results/quant_base_mac.csv \
      --label-a Windows --label-b Mac --out results/cross_machine
"""

import argparse
import itertools
from pathlib import Path

import matplotlib

# Use non-interactive backend so the script runs cleanly in headless/CLI environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from scripts.common import bootstrap_ci

# Composite join key identifying an identical experimental run across two result logs
KEY = ["dataset", "resample_id", "method", "seed"]


def seed_gap_per_dataset(df: pd.DataFrame) -> pd.Series:
    """Mean |accuracy difference| between every pair of seeds, per dataset (one machine)."""
    # Pivot to (dataset x seed) so all within-machine seed pairs can be differenced in vectorized form
    w = df.pivot_table(index="dataset", columns="seed", values="accuracy")
    pairs = list(itertools.combinations(w.columns, 2))
    if not pairs:
        return pd.Series(dtype=float)
    return pd.concat([(w[i] - w[j]).abs() for i, j in pairs], axis=1).mean(axis=1)


def main():
    """Compare two benchmark result CSVs, quantify machine vs. seed variance, and emit tables/plots."""
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--label-a", default="A")
    p.add_argument("--label-b", default="B")
    p.add_argument("--out", default="results/cross_machine")
    p.add_argument("--top", type=int, default=15)
    args = p.parse_args()
    la, lb = args.label_a, args.label_b
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Inner-join both runs on shared experimental keys so comparisons are strictly paired
    A, B = pd.read_csv(args.a), pd.read_csv(args.b)
    keys = [k for k in KEY if k in A.columns and k in B.columns]
    m = A.merge(B, on=keys, suffixes=("_a", "_b"))
    if m.empty:
        raise SystemExit(
            f"No matching rows on keys {keys}. Check method names and seeds."
        )
    # Restrict within-machine seed-gap calculations to the exact intersection of matched seeds
    seeds = sorted(m["seed"].unique())
    A = A[A["seed"].isin(seeds)]
    B = B[B["seed"].isin(seeds)]

    m["diff"] = m["accuracy_a"] - m["accuracy_b"]
    m["abs_diff"] = m["diff"].abs()
    # Convert fractional accuracy gap into minimum count of changed test predictions
    m["net_flips"] = (
        m["abs_diff"] * m["n_test_a"]
    ).round()  # lower bound on disagreements

    # ---------------- per-dataset table ----------------
    g = m.groupby("dataset")
    ds = pd.DataFrame(
        {
            "n_test": g["n_test_a"].first(),
            "mean_acc_a": g["accuracy_a"].mean(),
            "mean_acc_b": g["accuracy_b"].mean(),
            "mean_diff": g["diff"].mean(),
            "mean_abs_diff": g["abs_diff"].mean(),
            "max_abs_diff": g["abs_diff"].max(),
            "mean_net_flips": g["net_flips"].mean(),
            "pct_identical_runs": g["abs_diff"].apply(lambda x: (x == 0).mean() * 100),
        }
    )
    # Compare cross-machine discrepancy against intrinsic ExtraTrees random-seed variance
    ds["seed_gap_a"] = seed_gap_per_dataset(A)
    ds["seed_gap_b"] = seed_gap_per_dataset(B)
    ds["machine_over_seed_gap"] = ds["mean_abs_diff"] / ds[
        ["seed_gap_a", "seed_gap_b"]
    ].mean(axis=1).replace(0, np.nan)
    ds = ds.sort_values("mean_net_flips", ascending=False)
    ds.to_csv(out / "per_dataset.csv")
    m.to_csv(out / "per_run.csv", index=False)

    # ---------------- summary numbers ----------------
    d = m["abs_diff"]
    ds_mean_diff = ds["mean_diff"].to_numpy()
    # Non-parametric bootstrap CI and paired Wilcoxon test across datasets for systematic bias
    lo, hi = bootstrap_ci(ds_mean_diff)
    se = ds_mean_diff.std(ddof=1) / np.sqrt(len(ds_mean_diff))
    try:
        p_sign = float(wilcoxon(ds_mean_diff).pvalue)
    except ValueError:
        # Raised when all differences are identically zero
        p_sign = float("nan")
    seed_a, seed_b = ds["seed_gap_a"].mean(), ds["seed_gap_b"].mean()
    summary = {
        "runs_matched": len(m),
        "datasets": m["dataset"].nunique(),
        "seeds": seeds,
        "identical_runs_pct": 100 * (d == 0).mean(),
        "mean_abs_diff": d.mean(),
        "median_abs_diff": d.median(),
        "p95_abs_diff": d.quantile(0.95),
        "max_abs_diff": d.max(),
        "mean_net_flips": m["net_flips"].mean(),
        "median_net_flips": m["net_flips"].median(),
        "max_net_flips": m["net_flips"].max(),
        f"mean_signed_diff_{la}_minus_{lb}": ds_mean_diff.mean(),
        "signed_diff_SE": se,
        "signed_diff_CI95_low": lo,
        "signed_diff_CI95_high": hi,
        "signed_diff_wilcoxon_p": p_sign,
        f"seed_gap_{la}": seed_a,
        f"seed_gap_{lb}": seed_b,
        "machine_gap": ds["mean_abs_diff"].mean(),
        "machine_gap_over_seed_gap": ds["mean_abs_diff"].mean()
        / np.mean([seed_a, seed_b]),
    }
    if {"n_features_a", "n_features_b"} <= set(m.columns):
        summary["runs_with_different_feature_count"] = int(
            (m["n_features_a"] != m["n_features_b"]).sum()
        )
    # Audit metadata columns to verify identical hyperparameters and capture hardware strings
    for col in ["model_config", "hardware"]:
        if f"{col}_a" in m.columns:
            summary[f"{col}_identical"] = (
                bool((m[f"{col}_a"] == m[f"{col}_b"]).all())
                if col == "model_config"
                else None
            )
            summary[f"{col}_{la}"] = " | ".join(map(str, m[f"{col}_a"].unique()))
            summary[f"{col}_{lb}"] = " | ".join(map(str, m[f"{col}_b"].unique()))
    pd.Series(summary).to_csv(out / "summary.csv", header=["value"])

    # ---------------- overview figure ----------------
    fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))

    # Panel 1: Parity scatter plot of per-run accuracies
    a0 = ax[0, 0]
    sc = a0.scatter(
        m["accuracy_b"],
        m["accuracy_a"],
        c=m["abs_diff"],
        cmap="viridis_r",
        s=14,
        alpha=0.8,
    )
    lim = [min(m["accuracy_a"].min(), m["accuracy_b"].min()), 1.0]
    a0.plot(lim, lim, "k--", lw=1)
    a0.set(
        xlabel=f"accuracy on {lb}",
        ylabel=f"accuracy on {la}",
        title="Every run (colour = |difference|)",
    )
    fig.colorbar(sc, ax=a0, label="|diff|")

    # Panel 2: Distribution of signed accuracy differences across runs
    a1 = ax[0, 1]
    a1.hist(m["diff"], bins=40, edgecolor="black")
    a1.axvline(0, c="k", ls="--")
    a1.axvline(m["diff"].mean(), c="tab:red", label=f"mean {m['diff'].mean():+.4f}")
    a1.set(
        xlabel=f"accuracy {la} - {lb}", ylabel="runs", title="Signed difference per run"
    )
    a1.legend()

    # Panel 3: Log-scale histogram of minimum changed test predictions
    a2 = ax[0, 2]
    top = int(m["net_flips"].max()) + 1
    a2.hist(m["net_flips"], bins=np.arange(0, top + 1) - 0.5, edgecolor="black")
    a2.set_yscale("log")
    a2.axvline(
        m["net_flips"].median(),
        c="tab:red",
        ls="--",
        label=f"median {m['net_flips'].median():.0f}",
    )
    a2.set(
        xlabel="net flipped test predictions",
        ylabel="runs (log)",
        title="How many predictions changed",
    )
    a2.legend()

    # Panel 4: Quantization effect showing how single-sample flips scale as 1 / n_test
    a3 = ax[1, 0]
    a3.scatter(m["n_test_a"], m["abs_diff"], s=12, alpha=0.6)
    xs = np.logspace(np.log10(m["n_test_a"].min()), np.log10(m["n_test_a"].max()), 50)
    for k, ls in [(1, ":"), (5, "--"), (20, "-.")]:
        a3.plot(xs, k / xs, ls, c="gray", label=f"{k} flip{'s' if k > 1 else ''}")
    a3.set_xscale("log")
    a3.set_yscale("symlog", linthresh=1e-4)
    a3.set(
        xlabel="test set size",
        ylabel="|accuracy difference|",
        title="Small test sets magnify a few flips",
    )
    a3.legend(fontsize=8)

    # Panel 5: Boxplot comparing cross-machine noise against within-machine seed variance
    a4 = ax[1, 1]
    a4.boxplot(
        [ds["seed_gap_a"].dropna(), ds["seed_gap_b"].dropna(), ds["mean_abs_diff"]],
        tick_labels=[
            f"seed vs seed\n({la})",
            f"seed vs seed\n({lb})",
            f"{la} vs {lb}\n(same seed)",
        ],
        showfliers=False,
    )
    a4.set(
        ylabel="mean |accuracy difference| per dataset", title="Machine gap vs seed gap"
    )

    # Panel 6: Waterfall/sorted bar chart checking for directional platform bias across datasets
    a5 = ax[1, 2]
    s = ds.sort_values("mean_diff").reset_index()
    a5.bar(
        s.index,
        s["mean_diff"],
        color=np.where(s["mean_diff"] >= 0, "tab:blue", "tab:orange"),
        width=1,
    )
    a5.axhline(0, c="k", lw=1)
    a5.axhline(
        ds["mean_diff"].mean(),
        c="tab:red",
        ls="--",
        label=f"mean {ds['mean_diff'].mean():+.4f} (SE {se:.4f})",
    )
    a5.set(
        xlabel="datasets (sorted)",
        ylabel=f"mean accuracy {la} - {lb}",
        title="Any systematic bias?",
    )
    a5.legend()

    fig.suptitle(
        f"Consistency of the same method on {la} vs {lb}: {len(m)} runs, {m['dataset'].nunique()} datasets, seeds {seeds}"
    )
    fig.tight_layout()
    fig.savefig(out / "overview.png", dpi=180)
    plt.close(fig)

    # ---------------- top datasets figure ----------------
    t = ds.head(args.top).iloc[::-1]
    fig, a = plt.subplots(figsize=(8, 0.4 * len(t) + 1.5))
    a.barh(t.index, t["mean_net_flips"], color="tab:purple")
    for i, (name, row) in enumerate(t.iterrows()):
        a.text(
            row["mean_net_flips"],
            i,
            f"  n_test={int(row['n_test'])}",
            va="center",
            fontsize=8,
        )
    a.set(
        xlabel="mean net flipped predictions per run",
        title=f"{args.top} datasets with the most flips",
    )
    fig.tight_layout()
    fig.savefig(out / "top_datasets.png", dpi=180)
    plt.close(fig)

    # ---------------- text report ----------------
    # Generate a ready-to-paste narrative summary based on whether the 95% bootstrap CI spans zero
    sentence = (
        f"Rerunning the baseline on {lb} versus {la} gave identical accuracy in {summary['identical_runs_pct']:.0f}% of "
        f"{len(m)} runs. The mean absolute difference was {summary['mean_abs_diff']:.4f} (median {summary['median_net_flips']:.0f} "
        f"test series, maximum {summary['max_abs_diff']:.4f}), against {np.mean([seed_a, seed_b]):.4f} between forest seeds on one machine. "
        f"The mean signed difference across datasets was {summary[f'mean_signed_diff_{la}_minus_{lb}']:+.4f} "
        f"(95% CI [{lo:+.4f}, {hi:+.4f}]), so there is {'no' if lo <= 0 <= hi else 'a'} systematic offset."
    )
    lines = [
        "# Cross-machine consistency",
        "",
        f"- A ({la}): `{args.a}`",
        f"- B ({lb}): `{args.b}`",
        "",
    ]
    lines += [
        f"- **{k}**: {v:.5f}" if isinstance(v, float) else f"- **{k}**: {v}"
        for k, v in summary.items()
    ]
    lines += [
        "",
        "Net flips are a lower bound: two opposite flips cancel in accuracy.",
        "",
        "## Paste into the report",
        "",
        sentence,
        "",
        "## Datasets with most flips",
        "",
        "```",
        ds.head(args.top).round(4).to_string(),
        "```",
    ]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print(
        "\n".join(
            f"{k:45s} {v:.5f}" if isinstance(v, float) else f"{k:45s} {v}"
            for k, v in summary.items()
        )
    )
    print("\n" + sentence)
    print(
        f"\nWrote per_run.csv, per_dataset.csv, summary.csv, overview.png, top_datasets.png, report.md to {out.resolve()}"
    )


if __name__ == "__main__":
    main()
