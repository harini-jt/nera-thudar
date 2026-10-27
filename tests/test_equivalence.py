import numpy as np
import pytest
import torch

from reference.quant_base import QuantBase
from src.quant import Quant


@pytest.mark.parametrize("length", [24, 171, 1024])
def test_implementation_matches_base(length: int):
    rng = np.random.default_rng(0)
    X_train = torch.tensor(rng.standard_normal((20, 1, length)), dtype=torch.float32)
    X_test = torch.tensor(rng.standard_normal((7, 1, length)), dtype=torch.float32)

    base = QuantBase(depth=6, div=4)
    base_train = base.fit_transform(X_train, None)
    base_test = base.transform(X_test)

    imp = Quant(depth=6, div=4)
    imp_train = imp.fit_transform(X_train)
    imp_test = imp.transform(X_test)

    assert imp_train.shape == base_train.shape
    np.testing.assert_allclose(imp_train, base_train.numpy(), atol=1e-6)
    np.testing.assert_allclose(imp_test, base_test.numpy(), atol=1e-6)
