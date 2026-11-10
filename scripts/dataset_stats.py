import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import list_datasets, load_original


def stats_for(name: str) -> dict:
    X_tr, y_tr, X_te, y_te = load_original(name)
    return {
        "dataset": name,
        "series_length": X_tr.shape[2],  # number of time points per series
        "n_train": len(y_tr),
        "n_test": len(y_te),
        "n_classes": len(set(y_tr) | set(y_te)),
        "train_test_length_match": X_tr.shape[2] == X_te.shape[2],
        "error": "",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="*", help="subset of names (default: all)")
    p.add_argument("--out", default="docs/dataset_stats.csv")
    args = p.parse_args()

    rows = []
    for name in args.datasets or list_datasets():
        try:
            rows.append(stats_for(name))
        except Exception as e:  # unequal length, missing values, bad file...
            rows.append({"dataset": name, "error": str(e)[:100]})
            print(f"PROBLEM {name}: {e}")

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    ok = df[df["error"] == ""]
    print(f"\n{len(ok)}/{len(df)} datasets read fine")
    print(ok["series_length"].describe().round(1))
    print(
        "\nLongest:\n",
        ok.nlargest(5, "series_length")[["dataset", "series_length"]].to_string(
            index=False
        ),
    )
    print(
        "\nShortest:\n",
        ok.nsmallest(5, "series_length")[["dataset", "series_length"]].to_string(
            index=False
        ),
    )


if __name__ == "__main__":
    main()
