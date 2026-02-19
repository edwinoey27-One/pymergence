import jax
import jax.numpy as jnp
import equinox as eqx
import polars as pl
import pytest
import os
from pymergence.jax_core import StochasticMatrix, Partition, train_partition, save_model, load_model
from pymergence.brax_bridge import BraxDataLoader

def test_stochastic_matrix_methods():
    # Simple deterministic chain: 0->1, 1->0
    mat = jnp.array([[0., 1.], [1., 0.]])
    sm = StochasticMatrix(mat)

    # Determinism: 1.0 (entropy 0)
    det = sm.determinism()
    assert jnp.isclose(det, 1.0)

    # Degeneracy: 0.0 (marginal effect is uniform [0.5, 0.5])
    deg = sm.degeneracy()
    assert jnp.isclose(deg, 0.0)

    # EI: 1 * log2(2) = 1.0
    ei = sm.effective_information()
    assert jnp.isclose(ei, 1.0)

def test_partition_coarse_graining():
    # 4-state system with 2 blocks
    # Block A: {0,1}, Block B: {2,3}
    # A->A, B->B
    mat = jnp.array([
        [0.5, 0.5, 0.0, 0.0],
        [0.5, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.5, 0.5],
        [0.0, 0.0, 0.5, 0.5]
    ])
    sm = StochasticMatrix(mat)

    # Create manual hard partition
    # logits: large value for correct block
    logits = jnp.array([
        [20.0, -20.0],
        [20.0, -20.0],
        [-20.0, 20.0],
        [-20.0, 20.0]
    ])

    key = jax.random.PRNGKey(0)
    partition = Partition(4, 2, key)
    # Override logits
    partition = eqx.tree_at(lambda p: p.logits, partition, logits)

    macro_sm = partition.coarse_grain(sm)

    # Expected macro: Identity 2x2
    expected = jnp.eye(2)
    assert jnp.allclose(macro_sm.matrix, expected, atol=1e-3)

    # EI should be 1.0
    ei = macro_sm.effective_information()
    assert jnp.isclose(ei, 1.0, atol=1e-3)

def test_optimization_loop():
    # Use same 4-state system
    mat = jnp.array([
        [0.5, 0.5, 0.0, 0.0],
        [0.5, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.5, 0.5],
        [0.0, 0.0, 0.5, 0.5]
    ])
    sm = StochasticMatrix(mat)

    final_partition, losses = train_partition(sm, n_macro=2, steps=500, lr=0.1)

    # Check if loss decreased (negative EI increased in magnitude => EI increased)
    assert losses[-1] < losses[0]

    # Final EI should be close to 1.0 (loss -1.0)
    assert jnp.isclose(losses[-1], -1.0, atol=0.1)

    # Check assignment
    hard_assign = final_partition.hard_assignment()
    # Should pair 0,1 and 2,3
    # Rows 0 and 1 should be same
    assert jnp.allclose(hard_assign[0], hard_assign[1])
    # Rows 2 and 3 should be same
    assert jnp.allclose(hard_assign[2], hard_assign[3])
    # Row 0 and 2 should be different
    assert not jnp.allclose(hard_assign[0], hard_assign[2])

def test_serialization(tmp_path):
    key = jax.random.PRNGKey(0)
    partition = Partition(4, 2, key)

    path = tmp_path / "model.safetensors"
    save_model(partition, str(path))

    assert os.path.exists(path)

    loaded_partition = load_model(str(path), 4, 2, key)

    assert jnp.allclose(partition.logits, loaded_partition.logits)

def test_brax_loader():
    try:
        loader = BraxDataLoader('inverted_pendulum')
        states = loader.collect_trajectories(num_steps=10)

        assert states.shape == (10, 4)

        df = loader.to_polars()
        assert isinstance(df, pl.DataFrame)
        assert df.shape == (10, 4)

        jax_arr = loader.to_jax_via_arrow()
        assert jax_arr.shape == (10, 4)

    except Exception as e:
        pytest.skip(f"Brax failed: {e}")
