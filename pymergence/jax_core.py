import jax
import jax.numpy as jnp
from functools import partial

# --- Information Theoretic Primitives ---

@jax.jit
def entropy(p, epsilon=1e-15):
    """
    Compute the Shannon entropy H(p) in bits (base 2).
    """
    p = jnp.asarray(p)
    # Avoid log(0) by ensuring everything is at least epsilon
    p = jnp.maximum(p, epsilon)
    # Use log2 for bits
    return -jnp.sum(p * jnp.log2(p))

@jax.jit
def kl_divergence(p, q, epsilon=1e-15):
    """
    Compute Kullback-Leibler divergence D_KL(p || q) in nats (base e).
    """
    p = jnp.asarray(p)
    q = jnp.asarray(q)
    p = jnp.maximum(p, epsilon)
    q = jnp.maximum(q, epsilon)
    return jnp.sum(p * jnp.log(p / q))

@jax.jit
def kl_divergence_base2(p, q, epsilon=1e-15):
    """
    Compute Kullback-Leibler divergence D_KL(p || q) in bits (base 2).
    """
    p = jnp.asarray(p)
    q = jnp.asarray(q)
    p = jnp.maximum(p, epsilon)
    q = jnp.maximum(q, epsilon)
    return jnp.sum(p * jnp.log2(p / q))

# --- Matrix Operations ---

@partial(jax.jit, static_argnames=['n'])
def transition_matrix_power(matrix, n):
    """
    Compute the n-th power of the transition matrix.
    Uses binary exponentiation via jax.lax.fori_loop or simple matrix_power.
    Since n is usually small integer, jnp.linalg.matrix_power works well.
    """
    return jnp.linalg.matrix_power(matrix, n)

@partial(jax.jit, static_argnames=['steps'])
def evolve_distribution(matrix, initial_distribution, steps=1):
    """
    Evolve an initial distribution through the stochastic matrix for a given number of steps.
    Returns the distribution after 'steps' transitions.
    """
    M_n = jnp.linalg.matrix_power(matrix, steps)
    return jnp.dot(initial_distribution, M_n)

@partial(jax.jit, static_argnames=['steps'])
def evolve_distribution_trajectory(matrix, initial_distribution, steps=1):
    """
    Evolve an initial distribution and return the full trajectory.
    Returns (steps+1, N) array.
    """
    def step_fn(carry, _):
        new_dist = jnp.dot(carry, matrix)
        return new_dist, new_dist

    # scan returns (final, stacked_outputs)
    # output contains steps 1..N
    _, trajectory = jax.lax.scan(step_fn, initial_distribution, None, length=steps)

    # Prepend initial
    full_trajectory = jnp.concatenate([initial_distribution[None, :], trajectory], axis=0)
    return full_trajectory

# --- Causal Emergence Measures ---

@jax.jit
def determinism(matrix, intervention_distribution=None):
    """
    Compute the determinism coefficient of a row-stochastic matrix.
    """
    n = matrix.shape[0]

    if intervention_distribution is None:
        intervention_distribution = jnp.ones(n) / n

    # Calculate entropy for each row
    # vmap over the rows (axis 0)
    row_entropies = jax.vmap(entropy)(matrix)

    # det_i = 1 - H(row_i) / log2(n)
    # Handle the case where n=1 to avoid division by zero (log2(1)=0)
    # If n=1, determinism is 1.0
    log_n = jnp.log2(n)

    # Use where to handle n=1 case safely
    denom = jnp.where(log_n > 0, log_n, 1.0)
    row_dets = 1.0 - row_entropies / denom

    # If n=1, row_det should be 1.0.
    # With the logic above: 1 - 0/1 = 1. Correct.

    return jnp.dot(row_dets, intervention_distribution)

@jax.jit
def degeneracy(matrix, intervention_distribution=None):
    """
    Compute the degeneracy coefficient.
    """
    n = matrix.shape[0]

    if intervention_distribution is None:
        intervention_distribution = jnp.ones(n) / n

    # Marginal effect distribution: w * G
    marginal_effect = jnp.dot(intervention_distribution, matrix)

    effect_ent = entropy(marginal_effect)

    log_n = jnp.log2(n)
    denom = jnp.where(log_n > 0, log_n, 1.0)

    return 1.0 - effect_ent / denom

@jax.jit
def effectiveness(matrix, intervention_distribution=None):
    """
    Compute effectiveness = determinism - degeneracy.
    """
    det = determinism(matrix, intervention_distribution)
    deg = degeneracy(matrix, intervention_distribution)
    return det - deg

@jax.jit
def effective_information(matrix, intervention_distribution=None):
    """
    Compute effective information = effectiveness * log2(n).
    """
    n = matrix.shape[0]
    eff = effectiveness(matrix, intervention_distribution)
    return eff * jnp.log2(n)

# --- Sufficiency and Necessity ---

@jax.jit
def sufficiency(matrix, cause, effect):
    return matrix[cause, effect]

@jax.jit
def necessity(matrix, cause, effect, intervention_distribution=None):
    n = matrix.shape[0]
    if intervention_distribution is None:
        weights = jnp.ones(n) / (n - 1)
        weights = weights.at[cause].set(0.0)
    else:
        weights = intervention_distribution

    # Vectorized:
    col = matrix[:, effect]
    weighted_sum = jnp.dot(weights, col)
    return 1.0 - weighted_sum

@jax.jit
def average_sufficiency(matrix, intervention_distribution=None):
    n = matrix.shape[0]
    if intervention_distribution is None:
        intervention_distribution = jnp.ones(n) / n

    # avg_suff = sum_{c, e} P(c) * G[c, e] * suff(c, e)
    # suff(c, e) = G[c, e]
    # avg_suff = sum_{c, e} P(c) * G[c, e]^2

    # Weighted rows
    weighted_matrix = matrix * intervention_distribution[:, None] # broadcast multiply rows by weights

    # Element-wise multiply by matrix (since sufficiency is the matrix value itself)
    return jnp.sum(weighted_matrix * matrix)

@jax.jit
def average_necessity(matrix, intervention_distribution=None):
    """
    Compute average necessity.
    Consistent with original implementation: necessity(c,e) uses uniform weights over other causes,
    while averaging uses intervention_distribution.
    """
    n = matrix.shape[0]
    if intervention_distribution is None:
        intervention_distribution = jnp.ones(n) / n

    # Calculate necessity matrix N where N[c, e] is necessity of c for e
    # nec(c, e) = 1 - sum_{c' != c} (1/(n-1)) * G[c', e]
    #           = 1 - (1/(n-1)) * ( sum_{c'} G[c', e] - G[c, e] )

    # sum_{c'} G[c', e] is the column sum of G
    col_sums = jnp.sum(matrix, axis=0) # shape (n,)

    # Broadcast to (n, n)
    col_sums_broad = col_sums[None, :] # (1, n)

    # G[c, e] is matrix itself

    # Handle n=1 case (avoid div by zero)
    # If n=1, necessity is undefined or trivial. Original code would div by 0.
    # We'll safeguard.
    denom = jnp.where(n > 1, n - 1, 1.0)

    term = (col_sums_broad - matrix) / denom
    nec_matrix = 1.0 - term

    # Weighted average using intervention_distribution
    # W[c, e] = P(c) * G[c, e]
    P = intervention_distribution[:, None] # (n, 1)
    W = P * matrix

    return jnp.sum(W * nec_matrix)
