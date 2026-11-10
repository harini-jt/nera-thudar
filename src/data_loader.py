import os
from pathlib import Path
from aeon.datasets import load_from_ts_file
import numpy as np

ROOT_DATASET_DIR = Path(os.environ.get("DATASET_DIR", "data"))


def list_datasets(dataset_dir: Path = ROOT_DATASET_DIR) -> list[str]:
    """List only valid dataset folders."""
    datasets = []

    for dataset in dataset_dir.iterdir():
        if not dataset.is_dir():
            continue

        # Skip hidden/system folders such as .cache
        if dataset.name.startswith(".") or dataset.name.startswith("__"):
            continue

        # Confirm that the expected resample-0 files exist
        train_file = dataset / f"{dataset.name}0_TRAIN.ts"
        test_file = dataset / f"{dataset.name}0_TEST.ts"

        if train_file.is_file() and test_file.is_file():
            datasets.append(dataset.name)

    return sorted(datasets)


def _read_ts_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Loads the .ts file and returns the train/test data."""
    X, y = load_from_ts_file(path)
    return X, y


def load_dataset(
    name: str, resample_id: int, dataset_dir: Path = ROOT_DATASET_DIR
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns the train-test split for a specific resample of a specific dataset."""
    dataset_path = dataset_dir / name
    X_train, y_train = _read_ts_file(dataset_path / f"{name}{resample_id}_TRAIN.ts")
    X_test, y_test = _read_ts_file(dataset_path / f"{name}{resample_id}_TEST.ts")
    return X_train, y_train, X_test, y_test


def load_original(
    name: str, dataset_dir: Path = ROOT_DATASET_DIR
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns the train-test split for the original data of a specific dataset."""
    dataset_path = dataset_dir / name
    X_train, y_train = _read_ts_file(dataset_path / f"{name}_TRAIN.ts")
    X_test, y_test = _read_ts_file(dataset_path / f"{name}_TEST.ts")
    return X_train, y_train, X_test, y_test
