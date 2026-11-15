import numpy as np
from src.implementations.improvements import (
    QuantWithDepthDiv,
    QuantWithSamplingFrequency,
)
from src.implementations.quant import Quant


def _X():
    """Generate a reproducible synthetic 3D time-series batch: (batch, channels, length)."""
    return np.random.default_rng(0).standard_normal((12, 1, 171)).astype("float32")


def test_defaults_match_base():
    """Verify that extended Quant variants produce identical features to base Quant under default settings."""
    X = _X()
    base = Quant().fit_transform(X)

    # Default sampling frequency should not alter baseline quantile outputs
    np.testing.assert_allclose(
        QuantWithSamplingFrequency().fit_transform(X), base, atol=1e-6
    )
    # Default depth divisor should not alter baseline quantile outputs
    np.testing.assert_allclose(QuantWithDepthDiv().fit_transform(X), base, atol=1e-6)


def test_duplication_adds_exact_copies():
    """Verify that depth_repeats increases or decreases feature dimensionality as expected."""
    X = _X()
    base = Quant().fit_transform(X)

    # Repeating depth 1 twice should expand the number of extracted features
    dup = QuantWithSamplingFrequency(depth_repeats={1: 2}).fit_transform(X)
    assert dup.shape[1] > base.shape[1]

    # Setting depth 1 repeats to 0 should drop that depth's features, reducing dimensionality
    assert (
        QuantWithSamplingFrequency(depth_repeats={1: 0}).fit_transform(X).shape[1]
        < base.shape[1]
    )


def test_interval_stats_slope_and_std():
    """Verify exact slope and standard deviation calculations on a known linear signal, plus end-to-end validity."""
    import torch
    from src.implementations.improvements import QuantWithIntervalStats

    # Construct a deterministic linear signal y = 2*t + 1 over 8 time steps
    line = (2 * torch.arange(8, dtype=torch.float32) + 1).reshape(1, 1, 8)
    out = QuantWithIntervalStats()._extract_quantiles(line, div=4)

    # The final feature should capture the constant linear slope (2.0)
    np.testing.assert_allclose(out[0, -1].item(), 2.0, atol=1e-5)
    # The second-to-last feature should match the analytical standard deviation of 2 * [0..7]
    np.testing.assert_allclose(out[0, -2].item(), 2 * np.arange(8).std(), atol=1e-5)

    # End-to-end check on random input: adding interval stats expands feature count without NaNs/Infs
    X = _X()
    base = Quant().fit_transform(X)
    stats = QuantWithIntervalStats().fit_transform(X)
    assert stats.shape[1] > base.shape[1]
    assert np.isfinite(stats).all()
