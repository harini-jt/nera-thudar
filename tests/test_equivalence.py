import numpy as np
import pytest
import torch

from reference.quant_base import QuantBase
from src.implementations.quant import Quant


# Test across multiple series lengths (short, non-power-of-two, and long) to catch padding/slicing edge cases
@pytest.mark.parametrize("length", [24, 171, 1024])
def test_implementation_matches_base(length: int):
    """Verify that custom Quant matches the reference QuantBase across fit_transform and transform."""
    # Generate reproducible synthetic 3D tensors: (batch, channels, series_length)
    rng = np.random.default_rng(0)
    X_train = torch.tensor(rng.standard_normal((20, 1, length)), dtype=torch.float32)
    X_test = torch.tensor(rng.standard_normal((7, 1, length)), dtype=torch.float32)

    # Fit and transform using the reference implementation (expects dummy y=None and returns a Tensor)
    base = QuantBase(depth=6, div=4)
    base_train = base.fit_transform(X_train, None)
    base_test = base.transform(X_test)

    # Fit and transform using the custom implementation (returns a NumPy array)
    imp = Quant(depth=6, div=4)
    imp_train = imp.fit_transform(X_train)
    imp_test = imp.transform(X_test)

    # Confirm output feature dimensions match and values agree within float32 tolerance
    assert imp_train.shape == base_train.shape
    np.testing.assert_allclose(imp_train, base_train.numpy(), atol=1e-6)
    np.testing.assert_allclose(imp_test, base_test.numpy(), atol=1e-6)
