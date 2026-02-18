import jax
import jax.numpy as jnp
import pytest
from pymergence.equivariant import GeometricPartition

def test_geometric_partition_shapes():
    key = jax.random.PRNGKey(0)
    n_particles = 10
    n_macro = 3

    # Init
    gp = GeometricPartition(n_particles, n_macro, key)

    # Input: (N, D)
    pos = jax.random.normal(key, (n_particles, 3))

    probs = gp(pos)

    assert probs.shape == (n_macro,)
    assert jnp.allclose(jnp.sum(probs), 1.0)

def test_invariances():
    key = jax.random.PRNGKey(42)
    n_particles = 5
    n_macro = 2
    gp = GeometricPartition(n_particles, n_macro, key)

    pos = jax.random.normal(key, (n_particles, 2))

    # 1. Translation
    pos_shifted = pos + jnp.array([10.0, -5.0])
    out1 = gp(pos)
    out2 = gp(pos_shifted)
    assert jnp.allclose(out1, out2, atol=1e-5), "Failed Translation Invariance"

    # 2. Rotation (90 deg)
    # R = [[0, -1], [1, 0]]
    # Apply to pos centered at COM (since rotation is about COM usually for invariance test)
    # Actually our module centers at COM first.
    # So if we rotate the whole cloud, it should be invariant.
    theta = jnp.pi / 2
    R = jnp.array([[jnp.cos(theta), -jnp.sin(theta)],
                   [jnp.sin(theta), jnp.cos(theta)]])

    # Rotate around origin
    pos_rotated = pos @ R.T
    out3 = gp(pos_rotated)
    assert jnp.allclose(out1, out3, atol=1e-5), "Failed Rotation Invariance"

    # 3. Permutation
    # Swap first two particles
    idx = jnp.array([1, 0, 2, 3, 4])
    pos_permuted = pos[idx]
    out4 = gp(pos_permuted)
    assert jnp.allclose(out1, out4, atol=1e-5), "Failed Permutation Invariance"
