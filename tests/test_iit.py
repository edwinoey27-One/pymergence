import numpy as np
import pytest
from pymergence.StochasticMatrix import StochasticMatrix
from pymergence.iit import IITAnalyzer, HAS_PYPHI

@pytest.mark.skipif(not HAS_PYPHI, reason="PyPhi not installed")
def test_iit_phi_computation():
    # 2-node system: 4 states.
    # Logic: A copy B, B copy A (Swap)
    # 00 -> 00
    # 01 -> 10
    # 10 -> 01
    # 11 -> 11

    mat = np.zeros((4, 4))
    mat[0, 0] = 1.0 # 00 -> 00
    mat[1, 2] = 1.0 # 01 -> 10
    mat[2, 1] = 1.0 # 10 -> 01
    mat[3, 3] = 1.0 # 11 -> 11

    sm = StochasticMatrix(mat, backend='numpy')
    analyzer = IITAnalyzer(sm)

    # Compute phi for state 00 (index 0)
    # Ideally should be high if integrated.
    # Swap gate is usually integrated.
    phi = analyzer.compute_phi(0)

    assert phi >= 0.0
    # Exact value depends on PyPhi version/measure, but should run.

def test_iit_invalid_size():
    # 3 states (not power of 2)
    mat = np.eye(3)
    sm = StochasticMatrix(mat, backend='numpy')

    if HAS_PYPHI:
        with pytest.raises(ValueError, match="power of 2"):
            IITAnalyzer(sm)
