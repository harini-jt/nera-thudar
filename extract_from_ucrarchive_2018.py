from pathlib import Path
import shutil
import pandas as pd

# Adjust these paths if your folders are elsewhere
PROJECT_DIR = Path.cwd()
REPORT_FILE = PROJECT_DIR / "ucr_datasets" / "download_report.csv"
LOCAL_ARCHIVE = PROJECT_DIR / "UCRArchive_2018"
DESTINATION_DIR = PROJECT_DIR / "ucr_datasets"

VARIANT_SUFFIXES = ("Eq", "Nmv", "Discrete")


def find_local_dataset(dataset_name):
    """Find an exact dataset folder or a possible base-dataset folder."""
    candidates = [dataset_name]

    for suffix in VARIANT_SUFFIXES:
        if dataset_name.endswith(suffix):
            candidates.append(dataset_name[:-len(suffix)])

    for candidate in candidates:
        matches = [
            path for path in LOCAL_ARCHIVE.rglob(candidate)
            if path.is_dir() and path.name == candidate
        ]
        if matches:
            return matches[0], candidate

    return None, None


def recover_failed_datasets():
    if not REPORT_FILE.exists():
        raise FileNotFoundError(f"Download report not found: {REPORT_FILE}")

    if not LOCAL_ARCHIVE.is_dir():
        raise FileNotFoundError(
            f"Local UCR archive not found: {LOCAL_ARCHIVE}"
        )

    report = pd.read_csv(REPORT_FILE)

    required_columns = {"dataset", "status"}
    if not required_columns.issubset(report.columns):
        raise ValueError(
            f"Report must contain columns: {required_columns}"
        )

    failed_names = (
        report.loc[
            report["status"].astype(str).str.lower().eq("failed"),
            "dataset",
        ]
        .dropna()
        .astype(str)
        .str.strip()
        .drop_duplicates()
        .tolist()
    )

    print(f"Failed datasets in report: {len(failed_names)}")

    results = []

    for name in failed_names:
        source, matched_name = find_local_dataset(name)

        if source is None:
            status = "not_found_locally"
            print(f"[NOT FOUND] {name}")

        else:
            destination = DESTINATION_DIR / name

            if destination.exists():
                status = "destination_exists"
                print(f"[SKIPPED] {name}: destination already exists")

            else:
                shutil.copytree(source, destination)
                status = (
                    "copied_base_variant"
                    if matched_name != name
                    else "copied_exact"
                )
                print(f"[COPIED] {name} <- {source}")

        results.append({
            "dataset": name,
            "matched_folder": matched_name or "",
            "status": status,
            "source": str(source) if source else "",
            "destination": str(DESTINATION_DIR / name),
        })

    recovery_report = DESTINATION_DIR / "local_recovery_report.csv"
    pd.DataFrame(results).to_csv(recovery_report, index=False)

    print("\nRecovery finished.")
    print(f"Exact copies: {sum(r['status'] == 'copied_exact' for r in results)}")
    print(
        "Base variant copies: "
        f"{sum(r['status'] == 'copied_base_variant' for r in results)}"
    )
    print(
        "Not found locally: "
        f"{sum(r['status'] == 'not_found_locally' for r in results)}"
    )
    print(f"Recovery report: {recovery_report}")


if __name__ == "__main__":
    recover_failed_datasets()