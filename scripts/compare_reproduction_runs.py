"""
Compare baseline and improved QUANT reproduction runs.
Separate from scripts.verify_results.py; does not modify official-paper workflows.

Example:
python -m scripts.compare_reproduction_runs --baseline_results="results/reproduction_results.csv" --improved_results="results/reproduction_results_criterion_entropy_with_all_cores.csv" --outputfolder="results/baseline_vs_improved"

Optional single dataset:
add --dataset="Covid3MonthDiscrete"

Dependencies: pandas, numpy, matplotlib
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

KEYS = ["dataset", "resample_id"]
METRICS = ["accuracy", "balanced_accuracy", "log_loss", "auc", "fit_time_s", "predict_time_s"]
HIGHER = {"accuracy", "balanced_accuracy", "auc"}
LOWER = {"log_loss", "fit_time_s", "predict_time_s"}
REQUIRED = set(KEYS + ["accuracy"])


def arguments():
    p = argparse.ArgumentParser(description="Compare baseline and improved reproduction CSVs.")
    p.add_argument("--baseline_results", required=True)
    p.add_argument("--improved_results", required=True)
    p.add_argument("--outputfolder", required=True)
    p.add_argument("--dataset", default=None)
    p.add_argument("--tie_tolerance", type=float, default=0.001)
    return p.parse_args()


def load_csv(path, label):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} file not found: {path}")
    df = pd.read_csv(path)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"{label} CSV missing columns: {sorted(missing)}")
    df = df.copy()
    df["dataset"] = df["dataset"].astype("string").str.strip()
    ids = pd.to_numeric(df["resample_id"], errors="coerce")
    df["resample_id"] = [
        str(int(x)) if pd.notna(x) and float(x).is_integer() else str(x)
        for x in ids
    ]
    for metric in METRICS:
        if metric in df:
            df[metric] = pd.to_numeric(df[metric], errors="coerce")
    if df[KEYS].isna().any().any():
        raise ValueError(f"{label} CSV has missing dataset/resample keys.")
    dup = df.duplicated(KEYS, keep=False)
    if dup.any():
        raise ValueError(f"{label} CSV has duplicate dataset/resample keys:\n{df.loc[dup, KEYS].head(10)}")
    return df


def add_outcomes(frame, metrics, tolerance):
    out = frame.copy()
    for m in metrics:
        b, n = f"baseline_{m}", f"improved_{m}"
        if b not in out or n not in out:
            continue
        delta = out[n] - out[b]
        out[f"delta_{m}"] = delta
        outcome = pd.Series("Not comparable", index=out.index, dtype="object")
        valid = out[[b, n]].notna().all(axis=1)
        if m in HIGHER:
            outcome.loc[valid & (delta > tolerance)] = "Improved"
            outcome.loc[valid & (delta < -tolerance)] = "Regressed"
        else:
            outcome.loc[valid & (delta < -tolerance)] = "Improved"
            outcome.loc[valid & (delta > tolerance)] = "Regressed"
        outcome.loc[valid & (delta.abs() <= tolerance)] = "Draw"
        out[f"outcome_{m}"] = outcome
    return out


def summarize(frame, metrics, level, tolerance):
    rows = []
    for m in metrics:
        b, n = f"baseline_{m}", f"improved_{m}"
        if b not in frame or n not in frame:
            continue
        valid = frame[[b, n]].dropna()
        if valid.empty:
            continue
        d = valid[n] - valid[b]
        better = d > tolerance if m in HIGHER else d < -tolerance
        worse = d < -tolerance if m in HIGHER else d > tolerance
        rows.append({
            "level": level, "metric": m, "matched_count": len(valid),
            "baseline_mean": valid[b].mean(), "improved_mean": valid[n].mean(),
            "mean_delta_improved_minus_baseline": d.mean(),
            "mean_absolute_delta": d.abs().mean(),
            "rmse_delta": float(np.sqrt(np.mean(d ** 2))),
            "improved": int(better.sum()), "draws": int((d.abs() <= tolerance).sum()),
            "regressed": int(worse.sum())
        })
    return pd.DataFrame(rows)


def per_dataset(resamples, metrics, tolerance):
    rows = []
    for dataset, g in resamples.groupby("dataset"):
        row = {"dataset": dataset, "matched_resamples": len(g)}
        for m in metrics:
            b, n = f"baseline_{m}", f"improved_{m}"
            if b not in g or n not in g:
                continue
            v = g[[b, n]].dropna()
            if v.empty:
                continue
            d = v[n] - v[b]
            row[f"baseline_mean_{m}"] = v[b].mean()
            row[f"improved_mean_{m}"] = v[n].mean()
            row[f"mean_delta_{m}"] = d.mean()
            row[f"mae_{m}"] = d.abs().mean()
            better = d > tolerance if m in HIGHER else d < -tolerance
            worse = d < -tolerance if m in HIGHER else d > tolerance
            row[f"improved_{m}_count"] = int(better.sum())
            row[f"draw_{m}_count"] = int((d.abs() <= tolerance).sum())
            row[f"regressed_{m}_count"] = int(worse.sum())
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    a = arguments()
    if a.tie_tolerance < 0:
        raise ValueError("--tie_tolerance must be non-negative.")
    base = load_csv(a.baseline_results, "Baseline")
    new = load_csv(a.improved_results, "Improved")
    if a.dataset:
        base = base[base.dataset == a.dataset].copy()
        new = new[new.dataset == a.dataset].copy()
        if base.empty and new.empty:
            raise ValueError(f"Dataset not found in either CSV: {a.dataset}")

    metrics = [m for m in METRICS if m in base.columns and m in new.columns]
    if "accuracy" not in metrics:
        raise ValueError("Both files must contain accuracy.")
    b = base[KEYS + metrics].rename(columns={m: f"baseline_{m}" for m in metrics})
    n = new[KEYS + metrics].rename(columns={m: f"improved_{m}" for m in metrics})

    joined = b.merge(n, on=KEYS, how="inner", validate="one_to_one")
    if joined.empty:
        raise ValueError("No matching dataset/resample_id pairs found.")
    only_base = b[KEYS].merge(n[KEYS], on=KEYS, how="left", indicator=True)
    only_base = only_base[only_base._merge == "left_only"].drop(columns="_merge")
    only_new = n[KEYS].merge(b[KEYS], on=KEYS, how="left", indicator=True)
    only_new = only_new[only_new._merge == "left_only"].drop(columns="_merge")

    resamples = add_outcomes(joined, metrics, a.tie_tolerance)
    ds = per_dataset(resamples, metrics, a.tie_tolerance)
    resample_summary = summarize(resamples, metrics, "resample", a.tie_tolerance)

    # Average each run independently per dataset, then compare dataset means.
    bm = base.groupby("dataset", as_index=False)[metrics].mean()
    nm = new.groupby("dataset", as_index=False)[metrics].mean()
    bm = bm.rename(columns={m: f"baseline_{m}" for m in metrics})
    nm = nm.rename(columns={m: f"improved_{m}" for m in metrics})
    means = bm.merge(nm, on="dataset", how="inner", validate="one_to_one")
    means = add_outcomes(means, metrics, a.tie_tolerance)
    mean_summary = summarize(means, metrics, "dataset_mean", a.tie_tolerance)

    out = Path(a.outputfolder)
    csvdir, figdir = out / "csvs", out / "figures"
    csvdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)
    resamples.to_csv(csvdir / "resample_comparison.csv", index=False)
    ds.to_csv(csvdir / "per_dataset_summary.csv", index=False)
    means.to_csv(csvdir / "dataset_mean_comparison.csv", index=False)
    resample_summary.to_csv(csvdir / "resample_metric_summary.csv", index=False)
    mean_summary.to_csv(csvdir / "dataset_mean_metric_summary.csv", index=False)
    only_base.to_csv(csvdir / "baseline_only_resamples.csv", index=False)
    only_new.to_csv(csvdir / "improved_only_resamples.csv", index=False)

    # Record run metadata so configuration differences remain visible.
    metadata = []
    for label, frame in [("baseline", base), ("improved", new)]:
        for col in ["method", "model_config", "seed", "hardware"]:
            if col in frame:
                vals = frame[col].dropna().astype(str).unique().tolist()
                metadata.append({"run": label, "field": col, "unique_values": len(vals),
                                 "values": " | ".join(vals[:20])})
    pd.DataFrame(metadata).to_csv(csvdir / "run_metadata.csv", index=False)

    if "delta_accuracy" in resamples:
        vals = resamples.delta_accuracy.dropna()
        if not vals.empty:
            plt.figure(figsize=(8, 5))
            plt.hist(vals, bins=30)
            plt.axvline(0, linestyle="--")
            plt.xlabel("Improved accuracy - baseline accuracy")
            plt.ylabel("Resample count")
            plt.title("Resample-level accuracy changes")
            plt.tight_layout()
            plt.savefig(figdir / "resample_accuracy_delta_distribution.png", dpi=200)
            plt.close()
    if "mean_delta_accuracy" in ds and not ds.empty:
        ordered = ds.sort_values("mean_delta_accuracy")
        plt.figure(figsize=(11, max(6, len(ordered) * 0.16)))
        plt.barh(ordered.dataset, ordered.mean_delta_accuracy)
        plt.axvline(0, linestyle="--")
        plt.xlabel("Mean accuracy change (improved - baseline)")
        plt.ylabel("Dataset")
        plt.title("Mean accuracy change by dataset")
        plt.tight_layout()
        plt.savefig(figdir / "dataset_mean_accuracy_change.png", dpi=200)
        plt.close()
    valid = resamples[["baseline_accuracy", "improved_accuracy"]].dropna()
    if not valid.empty:
        lo, hi = valid.min().min(), valid.max().max()
        plt.figure(figsize=(6, 6))
        plt.scatter(valid.baseline_accuracy, valid.improved_accuracy, alpha=0.6)
        plt.plot([lo, hi], [lo, hi], linestyle="--")
        plt.xlabel("Baseline accuracy")
        plt.ylabel("Improved accuracy")
        plt.title("Baseline vs improved accuracy per resample")
        plt.tight_layout()
        plt.savefig(figdir / "baseline_vs_improved_accuracy.png", dpi=200)
        plt.close()

    report = [
        "# Baseline vs improved reproduction comparison", "",
        f"- Baseline: `{a.baseline_results}`",
        f"- Improved: `{a.improved_results}`",
        f"- Matched dataset/resample pairs: {len(joined)}",
        f"- Matched datasets: {joined.dataset.nunique()}",
        f"- Baseline-only keys: {len(only_base)}",
        f"- Improved-only keys: {len(only_new)}",
        f"- Tie tolerance: {a.tie_tolerance}", "",
        "Higher is better for accuracy, balanced accuracy and AUC. Lower is better "
        "for log loss and execution times. W/D/L labels are from the improved run's "
        "perspective. Resample IDs are matched by key; this does not independently "
        "verify identical split membership.", "",
        "## Resample-level metric summary", "",
        resample_summary.to_string(index=False), "",
        "## Dataset-mean metric summary", "",
        mean_summary.to_string(index=False), "",
    ]
    if not ds.empty and "mean_delta_accuracy" in ds:
        report += ["## Ten largest accuracy improvements", "",
                   ds.nlargest(10, "mean_delta_accuracy")[
                       ["dataset", "matched_resamples", "mean_delta_accuracy"]
                   ].to_string(index=False), "",
                   "## Ten largest accuracy regressions", "",
                   ds.nsmallest(10, "mean_delta_accuracy")[
                       ["dataset", "matched_resamples", "mean_delta_accuracy"]
                   ].to_string(index=False), ""]
    (out / "comparison_report.md").write_text("\n".join(report), encoding="utf-8")

    print("\nComparison complete")
    print(f"Baseline rows: {len(base):,}")
    print(f"Improved rows: {len(new):,}")
    print(f"Matched dataset/resample pairs: {len(joined):,}")
    print(f"Matched datasets: {joined.dataset.nunique():,}")
    print(f"Baseline-only keys: {len(only_base):,}")
    print(f"Improved-only keys: {len(only_new):,}")
    print(f"Output: {out.resolve()}")
    print("\nAccuracy: resample-level")
    print(resample_summary[resample_summary.metric == "accuracy"].to_string(index=False))
    print("\nAccuracy: dataset means")
    print(mean_summary[mean_summary.metric == "accuracy"].to_string(index=False))


if __name__ == "__main__":
    main()
