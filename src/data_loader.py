import os
from pathlib import Path
from aeon.datasets import load_from_ts_file
import numpy as np

# Allow overriding the dataset root via environment variable while defaulting to ./data
ROOT_DATASET_DIR = Path(os.environ.get("DATASET_DIR", "data"))


def list_datasets(dataset_dir: Path = ROOT_DATASET_DIR) -> list[str]:
    """List only valid dataset folders containing resample-0 train/test .ts archives."""
    datasets = []

    for dataset in dataset_dir.iterdir():
        if not dataset.is_dir():
            continue

        # Skip hidden/system folders such as .cache or __pycache__
        if dataset.name.startswith(".") or dataset.name.startswith("__"):
            continue

        # Confirm that the expected resample-0 files exist before registering the dataset
        train_file = dataset / f"{dataset.name}0_TRAIN.ts"
        test_file = dataset / f"{dataset.name}0_TEST.ts"

        if train_file.is_file() and test_file.is_file():
            datasets.append(dataset.name)

    # Return in deterministic alphabetical order for reproducible sweep ordering
    return sorted(datasets)


def _read_ts_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load an aeon/UCR .ts archive and return (X, y) arrays."""
    # For equal-length series, aeon returns a 3D ndarray of shape (n_instances, n_channels, series_length)
    X, y = load_from_ts_file(path)
    return X, y


def load_dataset(
    name: str, resample_id: int, dataset_dir: Path = ROOT_DATASET_DIR
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_train, y_train, X_test, y_test) for a specific resample split (e.g. resample_id=0)."""
    dataset_path = dataset_dir / name
    X_train, y_train = _read_ts_file(dataset_path / f"{name}{resample_id}_TRAIN.ts")
    X_test, y_test = _read_ts_file(dataset_path / f"{name}{resample_id}_TEST.ts")
    return X_train, y_train, X_test, y_test


def load_original(
    name: str, dataset_dir: Path = ROOT_DATASET_DIR
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_train, y_train, X_test, y_test) for the un-resampled default split."""
    dataset_path = dataset_dir / name
    X_train, y_train = _read_ts_file(dataset_path / f"{name}_TRAIN.ts")
    X_test, y_test = _read_ts_file(dataset_path / f"{name}_TEST.ts")
    return X_train, y_train, X_test, y_test
