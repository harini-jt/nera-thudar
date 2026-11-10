import numpy as np
from src.implementations.improvements import (
    QuantWithDepthDiv,
    QuantWithSamplingFrequency,
)
from src.implementations.quant import Quant


def _X():
    return np.random.default_rng(0).standard_normal((12, 1, 171)).astype("float32")


def test_defaults_match_base():
    X = _X()
    base = Quant().fit_transform(X)
    np.testing.assert_allclose(
        QuantWithSamplingFrequency().fit_transform(X), base, atol=1e-6
    )
    np.testing.assert_allclose(QuantWithDepthDiv().fit_transform(X), base, atol=1e-6)


def test_duplication_adds_exact_copies():
    X = _X()
    base = Quant().fit_transform(X)
    dup = QuantWithSamplingFrequency(depth_repeats={1: 2}).fit_transform(X)
    assert dup.shape[1] > base.shape[1]
    assert (
        QuantWithSamplingFrequency(depth_repeats={1: 0}).fit_transform(X).shape[1]
        < base.shape[1]
    )
