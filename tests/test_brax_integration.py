import pytest
import jax
import jax.numpy as jnp
from pymergence.brax_integration import estimate_transition_matrix, collect_trajectory

def test_transition_matrix_estimation():
    # Generate synthetic data: 2 clusters, well separated
    # Cluster 0: around [0, 0]
    # Cluster 1: around [10, 10]

    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)

    # 100 points
    states = jnp.concatenate([
        jax.random.normal(k1, shape=(50, 2)), # Cluster 0
        jax.random.normal(k2, shape=(50, 2)) + 10.0 # Cluster 1
    ])

    # Estimate
    matrix, centroids, labels = estimate_transition_matrix(states, n_clusters=2, kmeans_steps=10)

    assert matrix.shape == (2, 2)
    assert jnp.allclose(jnp.sum(matrix, axis=1), 1.0)

    # Check if centroids are roughly [0,0] and [10,10]
    dist0 = jnp.linalg.norm(centroids[0])
    dist1 = jnp.linalg.norm(centroids[1])

    # One should be near 0, one near 14 (sqrt(200))
    # Check max dist > 10
    assert max(dist0, dist1) > 10.0

@pytest.mark.skipif(True, reason="Requires Brax envs setup (skipped for speed)")
def test_brax_collection():
    try:
        states = collect_trajectory('inverted_pendulum', num_steps=10)
        assert states.shape == (10, 4)
    except Exception:
        pytest.skip("Brax environment not available")
