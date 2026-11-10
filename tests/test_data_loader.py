from pathlib import Path
import numpy as np

from aeon.datasets import load_from_ts_file
from src.data_loader import _read_ts_file

for name in [
    "ItalyPowerDemand",
    "Strawberry",
    "KeplerLightCurves",
    "AconityMINIPrinterLargeEq",
]:
    p = Path("data") / name / f"{name}3_TRAIN.ts"
    Xa, ya = load_from_ts_file(p)
    Xb, yb = _read_ts_file(p)
    assert Xa.shape == Xb.shape, (name, Xa.shape, Xb.shape)
    assert np.array_equal(Xa.astype(np.float32), Xb), name
    assert np.array_equal(ya, yb), name
print("loader matches aeon")
