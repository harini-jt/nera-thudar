import csv
from collections import Counter
import numpy as np
from pydantic import BaseModel

from src.data_loader import list_datasets, load_dataset, load_original

NUM_RESAMPLES = 30


class DatasetVerificationRow(BaseModel):
    """Class to create the rows of the dataset verification."""

    dataset: str
    resample_id: int
    ok: bool
    sizes_match: bool | None
    lengths_match: bool | None
    equals_original: bool | None
    differs_from_prev: bool | None
    note: str = ""


def class_props(y: np.ndarray, classes: list) -> np.ndarray:
    """Identifies the length of each class in the target variable."""
    counter = Counter(y)
    return np.array([counter[k] for k in classes])


rows = []
for dataset in list_datasets():
    X_train_base, y_train_base, X_test_base, y_test_base = load_original(dataset)
    classes = sorted(set(y_train_base) | set(y_test_base))
    prev: np.ndarray = None

    for resample_id in range(NUM_RESAMPLES):
        try:
            X_train_res, y_train_res, X_test_res, y_test_res = load_dataset(
                dataset, resample_id
            )
        except Exception as e:
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
                sizes_match=(
                    y_train_base.shape[0] == y_train_res.shape[0]
                    and y_test_base.shape[0] == y_test_res.shape[0]
                ),
                lengths_match=(X_train_base.shape[1] == X_train_res.shape[1]),
                equals_original=(
                    X_train_res.shape == X_train_base.shape
                    and np.allclose(X_train_res, X_train_base)
                ),
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

fields = sorted(
    {k for row in rows for k in row},
    key=lambda k: (k != "dataset", k != "resample_id", k),
)
with open("docs/data_verification.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print("Completed writing docs/data_verification.csv.")
