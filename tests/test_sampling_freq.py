import numpy as np

from src.quant import Quant
from src.improvements.quant_with_sampling_frequency import QuantWithSamplingFrequency


def test_sampling_frequency_class():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 1, 171)).astype("float32")

    base = Quant().fit_transform(X)

    # no repeats -> identical to Quant
    same = QuantWithSamplingFrequency(depth_repeats=None).fit_transform(X)
    np.testing.assert_allclose(same, base, atol=1e-6)

    # depth-1 doubled -> exactly 4 extra copies of the depth-1 features (one per representation)
    t = QuantWithSamplingFrequency(depth_repeats={1: 2}).fit(X)
    n_depth1 = sum(int((d == 1).sum()) for d in t.interval_depths_)
    assert t.transform(X).shape[1] > base.shape[1]
    # then check the extra columns by hand for one representation
