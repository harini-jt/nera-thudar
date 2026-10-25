import os
import time
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import ExtraTreesClassifier

from quant import Quant


def build_pipeline(seed: int = 42):
    """Constructs the Quant + ExtraTreesClassifier pipeline per the paper's defaults."""
    return make_pipeline(
        Quant(depth=6, div=4),
        ExtraTreesClassifier(
            n_estimators=200,
            max_features=0.1,
            random_state=seed,
            n_jobs=1,  # Restricts to 1 CPU core
        ),
    )
