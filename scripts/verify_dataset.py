import csv
from collections import Counter
import numpy as np
from pydantic import BaseModel

from src.data_loader import list_datasets, load_dataset, load_original

# Standard number of stratified resamples (0..29) in UCR/UEA time-series benchmarks
NUM_RESAMPLES = 30


class DatasetVerificationRow(BaseModel):
    """Schema for a single (dataset, resample_id) integrity check record."""

    dataset: str
    resample_id: int
    ok: bool
    # Nullable booleans allow recording failed loads without dummy values
    sizes_match: bool | None = None
    lengths_match: bool | None = None
    equals_original: bool | None = None
    differs_from_prev: bool | None = None
    note: str = ""


def class_props(y: np.ndarray, classes: list) -> np.ndarray:
    """Count class frequencies aligned to the canonical sorted `classes` order."""
    counter = Counter(y)
    return np.array([counter[k] for k in classes])


rows = []
for dataset in list_datasets():
    # Load the un-resampled split as the ground-truth shape/value reference
    X_train_base, y_train_base, X_test_base, y_test_base = load_original(dataset)
    classes = sorted(set(y_train_base) | set(y_test_base))
    prev: np.ndarray | None = None

    for resample_id in range(NUM_RESAMPLES):
        try:
            X_train_res, y_train_res, X_test_res, y_test_res = load_dataset(
                dataset, resample_id
            )
        except Exception as e:
            # Record missing/corrupt resample files without aborting the full sweep
            rows.append(
                DatasetVerificationRow(
                    dataset=dataset,
                    resample_id=resample_id,
                    ok=False,
                    note=str(e)[:100],
                ).model_dump()
            )
            continue

        rows.append(
            DatasetVerificationRow(
                dataset=dataset,
                resample_id=resample_id,
                ok=True,
                # Verify train and test sample counts are preserved across resamples
                sizes_match=(
                    y_train_base.shape[0] == y_train_res.shape[0]
                    and y_test_base.shape[0] == y_test_res.shape[0]
                ),
                # Verify channel/series dimension matches the original split
                lengths_match=(X_train_base.shape[1] == X_train_res.shape[1]),
                # Resample 0 in UCR/UEA is typically identical to the original split; 1..29 should differ
                equals_original=(
                    X_train_res.shape == X_train_base.shape
                    and np.allclose(X_train_res, X_train_base)
                ),
                # Guard against accidentally duplicated resample files across consecutive IDs
                differs_from_prev=(
                    (prev is None)
                    or not (
                        X_train_res.shape == prev.shape
                        and np.allclose(X_train_res, prev)
                    )
                ),
                note="All good",
            ).model_dump()
        )
        prev = X_train_res

    print(
        f"Checked {dataset}",
    )

# Pin `dataset` and `resample_id` as the first two CSV columns, sorting remaining fields alphabetically
fields = sorted(
    {k for row in rows for k in row},
    key=lambda k: (k != "dataset", k != "resample_id", k),
)
with open("docs/data_verification.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print("Completed writing docs/data_verification.csv.")
