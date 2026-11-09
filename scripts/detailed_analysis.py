"""Detailed QUANT reproduction analysis. Separate from scripts.verify_results.

Run from project root:
  python -m scripts.detailed_analysis --my_results="results/run.csv"
  python -m scripts.detailed_analysis --my_results="results/run.csv" --dataset="Covid3MonthDiscrete"
  python -m scripts.detailed_analysis --my_results="results/run.csv" --profile_data

The frozen scripts.verify_results workflows are not imported or modified.
"""
import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_MEAN = Path("reference/quant/results/accuracy_mean_ucr142.csv")
DEFAULT_RESAMPLES = Path("reference/quant/results/accuracy_resamples_ucr142.csv")


def clean_name(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s)).strip("_")


def read_ours(path):
    df = pd.read_csv(path)
    required = {"dataset", "resample_id", "accuracy"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df = df.copy()
    df["dataset"] = df["dataset"].astype(str).str.strip()
    rid = pd.to_numeric(df["resample_id"], errors="coerce")
    df["resample_id"] = rid.astype("Int64").astype(str).where(rid.notna(), df["resample_id"].astype(str))
    df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce")
    df = df.dropna(subset=["dataset", "accuracy"])
    df = df[df["accuracy"].between(0, 1)]
    dup = df.duplicated(["dataset", "resample_id"], keep=False)
    if dup.any():
        raise ValueError("Duplicate dataset/resample_id rows:\n" +
                         df.loc[dup, ["dataset", "resample_id"]].head(12).to_string(index=False))
    return df


def stats_by_dataset(df):
    g = df.groupby("dataset")["accuracy"]
    out = g.agg(
        n_resamples="count", mean="mean", median="median", std="std",
        min="min", max="max",
        q25=lambda x: x.quantile(.25), q75=lambda x: x.quantile(.75)
    ).reset_index()
    out["range"] = out["max"] - out["min"]
    out["iqr"] = out["q75"] - out["q25"]
    return out


def load_paper_mean(path):
    if not path.exists():
        return None
    p = pd.read_csv(path)
    col = "accuracy" if "accuracy" in p.columns else (
        "mean_accuracy" if "mean_accuracy" in p.columns else None)
    if "dataset" not in p or col is None:
        raise ValueError(f"Unexpected paper mean CSV schema: {list(p.columns)}")
    return p[["dataset", col]].rename(columns={col: "paper_mean"})


def load_paper_resamples(path):
    if not path.exists():
        return None
    p = pd.read_csv(path)
    if "dataset" not in p:
        raise ValueError("Paper resample CSV must contain dataset column.")
    long = p.melt(id_vars="dataset", var_name="resample_id", value_name="paper_accuracy")
    long["resample_id"] = long["resample_id"].astype(str)
    long["paper_accuracy"] = pd.to_numeric(long["paper_accuracy"], errors="coerce")
    return long.dropna(subset=["paper_accuracy"])


def outcome(d, tol):
    return np.where(d > tol, "win", np.where(d < -tol, "loss", "draw"))


def plot_dataset(group, paper_group, folder, name):
    folder.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.hist(group["accuracy"], bins=15, edgecolor="black")
    ax.set(xlim=(0, 1), xlabel="Accuracy", ylabel="Resample count",
           title=f"Accuracy distribution — {name}")
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(folder / "accuracy_distribution.png", dpi=160)
    plt.close(fig)

    a = group.copy()
    a["_sort"] = pd.to_numeric(a["resample_id"], errors="coerce")
    a = a.sort_values(["_sort", "resample_id"])
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(a["resample_id"], a["accuracy"], marker="o", label="Our reproduction")
    if paper_group is not None and len(paper_group):
        b = paper_group.copy()
        b["_sort"] = pd.to_numeric(b["resample_id"], errors="coerce")
        b = b.sort_values(["_sort", "resample_id"])
        ax.plot(b["resample_id"], b["paper_accuracy"], marker="x", label="Paper")
    ax.set(ylim=(0, 1.02), xlabel="Resample ID", ylabel="Accuracy",
           title=f"Accuracy by resample ID — {name}")
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(folder / "accuracy_by_resample.png", dpi=160)
    plt.close(fig)


def profile_ts(dataset, data_dir, out_dir, resample):
    try:
        from aeon.datasets import load_from_ts_file
    except Exception as e:
        return None, f"aeon unavailable; .ts profiling skipped ({e})"
    folder = data_dir / dataset
    train = folder / f"{dataset}{resample}_TRAIN.ts"
    test = folder / f"{dataset}{resample}_TEST.ts"
    if not train.exists() or not test.exists():
        return None, f"Numbered .ts files not found: {train} / {test}"
    try:
        Xtr, ytr = load_from_ts_file(str(train))
        Xte, yte = load_from_ts_file(str(test))
        ytr, yte = np.asarray(ytr).astype(str), np.asarray(yte).astype(str)

        def shape_info(X):
            a = np.asarray(X)
            if a.dtype != object and a.ndim == 3:
                return int(a.shape[1]), [int(a.shape[2])] * len(a), a
            lengths, dims, cases = [], set(), []
            for x in X:
                z = np.asarray(x, dtype=float)
                if z.ndim == 1:
                    z = z.reshape(1, -1)
                dims.add(z.shape[0]); lengths.append(z.shape[-1]); cases.append(z)
            return (next(iter(dims)) if len(dims) == 1 else -1), lengths, cases

        dtr, ltr, ctr = shape_info(Xtr)
        dte, lte, cte = shape_info(Xte)

        def value_stats(cases):
            values, missing = [], 0
            for x in cases:
                z = np.asarray(x, dtype=float)
                missing += int(np.isnan(z).sum())
                values.extend(z[np.isfinite(z)].ravel())
            return (float(np.mean(values)), float(np.std(values)),
                    float(np.min(values)), float(np.max(values)), missing) if len(values) else (
                    np.nan, np.nan, np.nan, np.nan, missing)

        trv, tev = value_stats(ctr), value_stats(cte)
        trc, tec = pd.Series(ytr).value_counts(), pd.Series(yte).value_counts()
        classes = sorted(set(ytr) | set(yte))
        class_table = pd.DataFrame([{
            "dataset": dataset, "class": c,
            "train_count": int(trc.get(c, 0)), "test_count": int(tec.get(c, 0)),
            "train_fraction": float(trc.get(c, 0) / len(ytr)),
            "test_fraction": float(tec.get(c, 0) / len(yte))
        } for c in classes])
        out_dir.mkdir(parents=True, exist_ok=True)
        class_table.to_csv(out_dir / "class_distribution.csv", index=False)

        def entropy(counts, n):
            return -sum((v/n)*math.log2(v/n) for v in counts if v) if n else np.nan

        row = {
            "dataset": dataset, "profile_resample_id": resample,
            "n_train": len(ytr), "n_test": len(yte), "n_classes": len(classes),
            "train_dimensions": dtr, "test_dimensions": dte,
            "train_length_min": min(ltr), "train_length_max": max(ltr),
            "test_length_min": min(lte), "test_length_max": max(lte),
            "train_class_entropy_bits": entropy(trc.values, len(ytr)),
            "test_class_entropy_bits": entropy(tec.values, len(yte)),
            "train_majority_fraction": float(trc.max()/len(ytr)),
            "test_majority_fraction": float(tec.max()/len(yte)),
            "train_value_mean": trv[0], "train_value_std": trv[1],
            "train_value_min": trv[2], "train_value_max": trv[3], "train_missing": trv[4],
            "test_value_mean": tev[0], "test_value_std": tev[1],
            "test_value_min": tev[2], "test_value_max": tev[3], "test_missing": tev[4],
        }
        return pd.DataFrame([row]), "Profiled numbered .ts files with aeon."
    except Exception as e:
        return None, f"Could not profile .ts files: {type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser(description="Detailed analysis; leaves scripts.verify_results untouched.")
    ap.add_argument("--my_results", required=True)
    ap.add_argument("--outputfolder", default="results/detailed_analysis")
    ap.add_argument("--dataset", default=None, help="Optional dataset filter; omit for all datasets.")
    ap.add_argument("--paper_mean", default=str(DEFAULT_MEAN))
    ap.add_argument("--paper_resamples", default=str(DEFAULT_RESAMPLES))
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--profile_data", action="store_true", help="Profile numbered .ts files using aeon.")
    ap.add_argument("--profile_resample", default="0")
    ap.add_argument("--tie_tolerance", type=float, default=.001)
    args = ap.parse_args()

    if args.tie_tolerance < 0:
        ap.error("--tie_tolerance must be non-negative")
    ours_all = read_ours(Path(args.my_results))
    if args.dataset:
        ours = ours_all[ours_all.dataset.str.casefold() == args.dataset.casefold()].copy()
        if ours.empty:
            raise ValueError(f"Dataset '{args.dataset}' not found. Available examples: {sorted(ours_all.dataset.unique())[:25]}")
        dataset = ours.dataset.iloc[0]
    else:
        ours, dataset = ours_all.copy(), None

    root = Path(args.outputfolder) / (clean_name(dataset) if dataset else "all_datasets")
    root.mkdir(parents=True, exist_ok=True)
    stats = stats_by_dataset(ours)
    stats.to_csv(root / "resample_statistics.csv", index=False)
    ours.sort_values(["dataset", "resample_id"]).to_csv(root / "per_resample_results.csv", index=False)

    pm = load_paper_mean(Path(args.paper_mean))
    pr = load_paper_resamples(Path(args.paper_resamples))
    mc = rc = None

    if pm is not None:
        mc = stats[["dataset", "mean"]].merge(pm, on="dataset", how="inner")
        mc["difference_ours_minus_paper"] = mc["mean"] - mc["paper_mean"]
        mc["absolute_difference"] = mc["difference_ours_minus_paper"].abs()
        mc["outcome"] = outcome(mc.difference_ours_minus_paper, args.tie_tolerance)
        mc.sort_values("difference_ours_minus_paper").to_csv(root / "dataset_mean_comparison.csv", index=False)
        pd.DataFrame({"dataset": sorted(set(ours.dataset)-set(pm.dataset))}).to_csv(root / "datasets_only_in_ours.csv", index=False)
        pd.DataFrame({"dataset": sorted(set(pm.dataset)-set(ours.dataset))}).to_csv(root / "datasets_only_in_paper.csv", index=False)
        fig, ax = plt.subplots(figsize=(10, max(5, min(14, .25*len(mc)+2))))
        z = mc.sort_values("difference_ours_minus_paper")
        ax.barh(z.dataset, z.difference_ours_minus_paper)
        ax.axvline(0, linewidth=1)
        ax.set(xlabel="Our mean − paper mean (accuracy units)", title="Dataset mean accuracy differences")
        ax.grid(axis="x", alpha=.25); fig.tight_layout()
        fig.savefig(root / "dataset_mean_difference.png", dpi=160); plt.close(fig)

    if pr is not None:
        rc = ours[["dataset", "resample_id", "accuracy"]].merge(
            pr, on=["dataset", "resample_id"], how="inner")
        rc["difference_ours_minus_paper"] = rc.accuracy - rc.paper_accuracy
        rc["absolute_difference"] = rc.difference_ours_minus_paper.abs()
        rc["outcome"] = outcome(rc.difference_ours_minus_paper, args.tie_tolerance)
        rc.to_csv(root / "resample_id_comparison.csv", index=False)
        ok = set(zip(ours.dataset, ours.resample_id))
        pk = set(zip(pr.dataset, pr.resample_id))
        unmatched = ([{"dataset": d, "resample_id": r, "source": "ours_only"} for d, r in sorted(ok-pk)] +
                     [{"dataset": d, "resample_id": r, "source": "paper_only"} for d, r in sorted(pk-ok)])
        pd.DataFrame(unmatched, columns=["dataset", "resample_id", "source"]).to_csv(
            root / "unmatched_resample_ids.csv", index=False)

    # One folder per dataset, even when analysing all 142.
    for name, group in ours.groupby("dataset"):
        folder = root / "datasets" / clean_name(name)
        pg = pr[pr.dataset == name] if pr is not None else None
        plot_dataset(group, pg, folder, name)
        stats[stats.dataset == name].to_csv(folder / "resample_statistics.csv", index=False)
        if mc is not None:
            mc[mc.dataset == name].to_csv(folder / "paper_mean_comparison.csv", index=False)
        if rc is not None:
            rc[rc.dataset == name].to_csv(folder / "resample_id_comparison.csv", index=False)

    profile, profile_note = None, "Not requested. Use --profile_data to inspect numbered .ts files."
    if args.profile_data:
        names = [dataset] if dataset else sorted(ours.dataset.unique())
        rows, notes = [], []
        for name in names:
            p, note = profile_ts(name, Path(args.data_dir), root / "profiles" / clean_name(name), args.profile_resample)
            notes.append(f"- {name}: {note}")
            if p is not None:
                rows.append(p)
        profile = pd.concat(rows, ignore_index=True) if rows else None
        profile_note = "\n".join(notes)
        if profile is not None:
            profile.to_csv(root / "dataset_structure_profile.csv", index=False)

    lines = [
        f"# Detailed analysis: {dataset or 'All datasets'}", "",
        "## Scope and cautions", "",
        f"- Datasets: {ours.dataset.nunique()}; result rows: {len(ours)}.",
        f"- Resamples per dataset (min/max): {ours.groupby('dataset').size().min()}/{ours.groupby('dataset').size().max()}.",
        f"- Tie tolerance: {args.tie_tolerance}.",
        "- Matching resample IDs does not prove identical train/test instance membership or order.",
        "- Paper comparisons are included only when the reference CSVs are available.", "",
        "## Reproduction accuracy summary", "",
        f"- Macro mean of dataset means: {stats['mean'].mean():.6f}.",
        f"- Median dataset mean: {stats['mean'].median():.6f}.",
        f"- Lowest mean: {stats.loc[stats['mean'].idxmin(), 'dataset']} ({stats['mean'].min():.6f}).",
        f"- Highest mean: {stats.loc[stats['mean'].idxmax(), 'dataset']} ({stats['mean'].max():.6f}).", "",
    ]
    if mc is not None and len(mc):
        lines += [
            "## Paper dataset-mean comparison", "",
            f"- Matched datasets: {len(mc)}.",
            f"- Our macro mean: {mc['mean'].mean():.6f}; paper macro mean: {mc.paper_mean.mean():.6f}.",
            f"- Mean difference: {mc.difference_ours_minus_paper.mean():+.6f}.",
            f"- MAE: {mc.absolute_difference.mean():.6f}.",
            f"- RMSE: {np.sqrt(np.mean(mc.difference_ours_minus_paper**2)):.6f}.",
            f"- Dataset-level W/D/L: {(mc.outcome=='win').sum()}/{(mc.outcome=='draw').sum()}/{(mc.outcome=='loss').sum()}.",
            "", "### Ten largest negative gaps", ""
        ]
        lines += [f"- {r.dataset}: {r.difference_ours_minus_paper:+.6f}" for r in mc.nsmallest(10, "difference_ours_minus_paper").itertuples()]
        lines.append("")
    if rc is not None:
        lines += [
            "## Paper resample-ID comparison", "",
            f"- Matched pairs: {len(rc)}.",
            f"- Mean difference: {rc.difference_ours_minus_paper.mean():+.6f}.",
            f"- MAE: {rc.absolute_difference.mean():.6f}.",
            f"- W/D/L: {(rc.outcome=='win').sum()}/{(rc.outcome=='draw').sum()}/{(rc.outcome=='loss').sum()}.",
            "- Treat this as descriptive until exact split membership is independently checked.", ""
        ]
    lines += [
        "## Dataset structure profile", "", profile_note, "",
        "## Checks not established by this script", "",
        "- Exact official-vs-local train/test instance membership and order.",
        "- Exact equality of official and local loader outputs.",
        "- Equality of intermediate QUANT feature matrices.",
        "- Whether classifier random seeds explain the remaining gap.",
        "- Individual prediction/confusion-matrix comparison unless prediction files are separately generated.",
        "",
        "These require split-index/source identity checks, loader-to-loader tests, intermediate feature comparisons, and controlled seed experiments."
    ]
    (root / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")

    print("Detailed analysis completed.")
    print(f"Scope: {dataset or 'all datasets'}")
    print(f"Datasets: {ours.dataset.nunique()} | Rows: {len(ours)}")
    print(f"Output: {root}")
    if mc is not None:
        print("Dataset mean W/D/L:",
              (mc.outcome == "win").sum(), (mc.outcome == "draw").sum(), (mc.outcome == "loss").sum())
    if rc is not None:
        print("Resample-ID W/D/L:",
              (rc.outcome == "win").sum(), (rc.outcome == "draw").sum(), (rc.outcome == "loss").sum())
    print("Read analysis_report.md for interpretation.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
