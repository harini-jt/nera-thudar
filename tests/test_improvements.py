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


def test_interval_stats_slope_and_std():
    import torch
    from src.implementations.improvements import QuantWithIntervalStats

    # A straight line 2*t + 1: slope 2 on every interval, std of 0..7 scaled by 2.
    line = (2 * torch.arange(8, dtype=torch.float32) + 1).reshape(1, 1, 8)
    out = QuantWithIntervalStats()._extract_quantiles(line, div=4)
    np.testing.assert_allclose(out[0, -1].item(), 2.0, atol=1e-5)
    np.testing.assert_allclose(out[0, -2].item(), 2 * np.arange(8).std(), atol=1e-5)

    X = _X()
    base = Quant().fit_transform(X)
    stats = QuantWithIntervalStats().fit_transform(X)
    assert stats.shape[1] > base.shape[1]
    assert np.isfinite(stats).all()
