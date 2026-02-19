import pytest
import numpy as np
import jax
import jax.numpy as jnp
from pymergence.core.StochasticMatrix import StochasticMatrix

def test_backend_detection():
    mat = np.array([[0.5, 0.5], [0.5, 0.5]])
    sm = StochasticMatrix(mat)
    assert sm.backend == 'numpy'

    mat_jax = jnp.array(mat)
    sm_jax = StochasticMatrix(mat_jax)
    assert sm_jax.backend == 'jax'

def test_measures_consistency():
    np.random.seed(42)
    mat = np.random.rand(5, 5)
    mat = mat / mat.sum(axis=1, keepdims=True)

    sm_np = StochasticMatrix(mat, backend='numpy')
    sm_jax = StochasticMatrix(mat, backend='jax')

    # Check all measures
    assert np.isclose(sm_np.determinism(), sm_jax.determinism(), atol=1e-5)
    assert np.isclose(sm_np.degeneracy(), sm_jax.degeneracy(), atol=1e-5)
    assert np.isclose(sm_np.effectiveness(), sm_jax.effectiveness(), atol=1e-5)
    assert np.isclose(sm_np.effective_information(), sm_jax.effective_information(), atol=1e-5)

    assert np.isclose(sm_np.average_sufficiency(), sm_jax.average_sufficiency(), atol=1e-5)
    assert np.isclose(sm_np.average_necessity(), sm_jax.average_necessity(), atol=1e-5)

def test_coarse_grain_consistency():
    mat = np.array([
        [0.5, 0.5, 0.0, 0.0],
        [0.5, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.5, 0.5],
        [0.0, 0.0, 0.5, 0.5]
    ])
    sm_np = StochasticMatrix(mat, backend='numpy')
    sm_jax = StochasticMatrix(mat, backend='jax')

    cg = sm_np.finest_graining() # {{0}, {1}, {2}, {3}}

    # Coarse grain
    # Manually create a 2-block CG
    from pymergence.core.CoarseGraining import CoarseGraining
    cg2 = CoarseGraining(((0, 1), (2, 3)))

    macro_np = sm_np.coarse_grain(cg2)
    macro_jax = sm_jax.coarse_grain(cg2)

    assert macro_np == macro_jax

    # Check inconsistency
    inc_np = sm_np.inconsistency_sum_over_starts(cg2)
    inc_jax = sm_jax.inconsistency_sum_over_starts(cg2)
    assert np.isclose(inc_np, inc_jax, atol=1e-5)

def test_optimization():
    # Only test if JAX works
    mat = np.array([
        [0.9, 0.1, 0.0],
        [0.8, 0.2, 0.0],
        [0.0, 0.0, 1.0]
    ])
    # Ideally blocks: {0,1}, {2}
    sm = StochasticMatrix(mat).to_jax()

    assign, ei = sm.optimize_coarse_graining(n_macro=2, steps=200)

    # Check if {0,1} are grouped
    # assign is (3, 2)
    b0 = jnp.argmax(assign[0])
    b1 = jnp.argmax(assign[1])
    b2 = jnp.argmax(assign[2])

    assert b0 == b1
    assert b0 != b2
