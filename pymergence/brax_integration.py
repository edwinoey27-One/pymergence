import jax
import jax.numpy as jnp
from brax import envs

def simple_kmeans(data, k, steps=10, key=None):
    """
    Simple K-Means clustering in JAX.

    Parameters
    ----------
    data : (N, D) array
        Data points.
    k : int
        Number of clusters.
    steps : int
        Number of iterations.
    key : jax.random.PRNGKey
        Random key.

    Returns
    -------
    centroids : (k, D) array
        Cluster centroids.
    labels : (N,) array
        Cluster labels for each data point.
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    # Initialize centroids randomly from data
    N, D = data.shape
    idx = jax.random.choice(key, N, shape=(k,), replace=False)
    centroids = data[idx]

    def step(centroids):
        # Compute distances
        # data: (N, 1, D), centroids: (1, k, D)
        dists = jnp.sum((data[:, None, :] - centroids[None, :, :])**2, axis=2)
        # Assign labels
        labels = jnp.argmin(dists, axis=1)

        # Update centroids
        # Sum data points for each cluster
        # mask: (N, k)
        mask = jax.nn.one_hot(labels, k)
        sum_data = jnp.dot(mask.T, data) # (k, D)
        counts = jnp.sum(mask, axis=0)[:, None] # (k, 1)

        # Avoid division by zero
        counts = jnp.where(counts > 0, counts, 1.0)
        new_centroids = sum_data / counts

        return new_centroids, labels

    # Iterate
    # We can use jax.lax.scan but python loop is easier to read and steps is small
    for _ in range(steps):
        centroids, labels = step(centroids)

    return centroids, labels

def collect_trajectory(env_name, num_steps=1000, seed=0):
    """
    Collect a trajectory from a Brax environment using a random policy.

    Parameters
    ----------
    env_name : str
        Name of the Brax environment (e.g. 'ant', 'fetch').
    num_steps : int
        Number of steps to run.
    seed : int
        Random seed.

    Returns
    -------
    states : (num_steps, observation_size) array
        Trajectory of observations.
    """
    env = envs.create(env_name=env_name)
    key = jax.random.PRNGKey(seed)
    reset_key, run_key = jax.random.split(key)

    state = env.reset(rng=reset_key)

    # JIT the step function
    jit_step = jax.jit(env.step)

    trajectory = []

    # Split keys for actions
    keys = jax.random.split(run_key, num_steps)

    current_state = state

    for i in range(num_steps):
        # Record observation
        # Ensure observation is on CPU or JAX array
        trajectory.append(current_state.obs)

        # Random action
        action = jax.random.uniform(keys[i], shape=(env.action_size,), minval=-1.0, maxval=1.0)

        current_state = jit_step(current_state, action)

    return jnp.stack(trajectory)

def estimate_transition_matrix(states, n_clusters=10, kmeans_steps=20, key=None):
    """
    Estimate a transition matrix from a continuous trajectory using K-Means clustering.

    Parameters
    ----------
    states : (T, D) array
        Trajectory of states.
    n_clusters : int
        Number of discrete states to map to.

    Returns
    -------
    transition_matrix : (n_clusters, n_clusters) array
        Row-stochastic transition matrix.
    centroids : (n_clusters, D) array
        The cluster centers.
    labels : (T,) array
        The discrete state labels.
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    # 1. Discretize using K-Means
    centroids, labels = simple_kmeans(states, n_clusters, steps=kmeans_steps, key=key)

    # 2. Count transitions
    n = n_clusters
    counts = jnp.zeros((n, n))

    # Count (label[t], label[t+1])
    # Can use add_at
    from_states = labels[:-1]
    to_states = labels[1:]

    # Linear indices
    # idx = from * n + to
    # But JAX way:
    # We can use index_add

    # For simplicity, loop or scan? Or just numpy histogram2d equivalent?
    # jax.ops.segment_sum is good but indices are pairs.

    # Let's use a for loop over time for simplicity if T is large? No, slow.
    # Use index_add with tuple index.

    counts = counts.at[from_states, to_states].add(1.0)

    # 3. Normalize
    row_sums = jnp.sum(counts, axis=1, keepdims=True)

    # Handle empty rows (unvisited states) -> identity or uniform?
    # Identity makes sense (absorbing).
    # Or uniform.
    # Let's use uniform to be safe for ergodic assumptions?
    # Or Identity.

    # Use where to handle 0 sums
    transition_matrix = jnp.where(
        row_sums > 0,
        counts / row_sums,
        jnp.eye(n) # Default to self-loop if state never visited (or no outgoing)
    )

    return transition_matrix, centroids, labels
