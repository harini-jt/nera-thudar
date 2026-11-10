import pandas as pd
from src.data_loader import ROOT_DATASET_DIR, list_datasets

sizes = {
    d: (ROOT_DATASET_DIR / d / f"{d}0_TRAIN.ts").stat().st_size / 1e6
    for d in list_datasets()
}
print(pd.Series(sizes).sort_values(ascending=False).head(20).round(1))
