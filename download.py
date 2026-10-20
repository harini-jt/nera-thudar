from pathlib import Path
import argparse
import re

import pandas as pd
from aeon.datasets import load_classification


DATA_DIR = Path("ucr_datasets")
REPORT_FILE = DATA_DIR / "download_report.csv"


def normalise_name(value):
    """Clean a dataset name from Excel."""
    if pd.isna(value):
        return None
    name = str(value).strip()
    if not name or name.lower() == "nan":
        return None
    return name


def read_dataset_names(file_path, column=None):
    """Read dataset names from Excel or CSV."""
    path = Path(file_path)

    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError("Input must be an .xlsx, .xls, or .csv file.")

    # Use the specified column, or automatically find 'dataset',
    # falling back to the first column.
    if column:
        if column not in df.columns:
            raise ValueError(
                f"Column '{column}' not found. Available: {list(df.columns)}"
            )
        series = df[column]
    else:
        matching = [
            col for col in df.columns
            if str(col).strip().lower() == "dataset"
        ]
        series = df[matching[0]] if matching else df.iloc[:, 0]

    names = [normalise_name(v) for v in series]
    names = [n for n in names if n is not None]

    # Remove duplicate names while preserving spreadsheet order.
    return list(dict.fromkeys(names))


def download_one(name):
    """Download a dataset and verify both predefined splits."""
    try:
        # aeon stores each dataset under DATA_DIR / dataset name.
        X_train, y_train, train_meta = load_classification(
            name=name,
            split="train",
            extract_path=str(DATA_DIR),
            return_metadata=True,
        )

        X_test, y_test, test_meta = load_classification(
            name=name,
            split="test",
            extract_path=str(DATA_DIR),
            return_metadata=True,
        )

        def shape_description(X):
            if hasattr(X, "shape"):
                return str(tuple(X.shape))
            return f"{len(X)} instances (possibly variable length)"

        return {
            "dataset": name,
            "status": "success",
            "train_cases": len(y_train),
            "test_cases": len(y_test),
            "train_shape": shape_description(X_train),
            "test_shape": shape_description(X_test),
            "classes": len(set(map(str, y_train))),
            "univariate": train_meta.get("univariate", ""),
            "equal_length": train_meta.get("equallength", ""),
            "folder": str(DATA_DIR / name),
            "error": "",
        }

    except Exception as exc:
        return {
            "dataset": name,
            "status": "failed",
            "train_cases": "",
            "test_cases": "",
            "train_shape": "",
            "test_shape": "",
            "classes": "",
            "univariate": "",
            "equal_length": "",
            "folder": str(DATA_DIR / name),
            "error": f"{type(exc).__name__}: {exc}",
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "excel_file",
        help="Path to the spreadsheet containing the dataset column",
    )
    parser.add_argument(
        "--column",
        default=None,
        help="Dataset-name column (default: 'dataset', otherwise first column)",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    names = read_dataset_names(args.excel_file, args.column)

    if not names:
        raise ValueError("No dataset names found in the input spreadsheet.")

    print(f"Found {len(names)} unique dataset names.")

    results = []
    for index, name in enumerate(names, start=1):
        print(f"[{index}/{len(names)}] {name}")
        result = download_one(name)
        results.append(result)
        print(f"    {result['status']}")
        if result["status"] == "failed":
            print(f"    {result['error']}")

    report = pd.DataFrame(results)
    report.to_csv(REPORT_FILE, index=False)

    successful = (report["status"] == "success").sum()
    failed = len(report) - successful

    print("\nFinished.")
    print(f"Successful datasets: {successful}")
    print(f"Failed datasets: {failed}")
    print(f"Report saved to: {REPORT_FILE.resolve()}")


if __name__ == "__main__":
    main()