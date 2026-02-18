import jax
import jax.numpy as jnp
import optax
from pymergence import jax_core

@jax.jit
def soft_coarse_grain(micro_matrix, assignment_matrix):
    """
    Compute the coarse-grained matrix using a soft assignment matrix.

    Parameters
    ----------
    micro_matrix : (N, N) array
        Micro-scale stochastic matrix.
    assignment_matrix : (N, M) array
        Soft assignment matrix where P[i, j] is prob of micro state i belonging to macro state j.
        Rows sum to 1.

    Returns
    -------
    macro_matrix : (M, M) array
        Row-stochastic macro matrix.
    """
    # Numerator: Sum of transitions between blocks
    # T_macro = P^T @ T_micro @ P (if P was hard indicator)
    # Here we use soft P.
    # We want to aggregate transitions from block I to block J.
    # sum_{i, j} P(i \in I) * T(i -> j) * P(j \in J)
    numerator = jnp.dot(assignment_matrix.T, jnp.dot(micro_matrix, assignment_matrix))

    # Denominator: Size of block I (effective size)
    # Each row I should be normalized by sum of outgoing transitions from I.
    # Sum of row I in numerator = sum_J (sum_{i,j} P(i \in I) T(i->j) P(j \in J))
    # = sum_i P(i \in I) * sum_j T(i->j) * sum_J P(j \in J)
    # sum_J P(j \in J) = 1 (since rows of P sum to 1)
    # sum_j T(i->j) = 1 (stochastic)
    # So row sum = sum_i P(i \in I) = column sum of P for column I.

    block_sizes = jnp.sum(assignment_matrix, axis=0)

    # Avoid division by zero for empty blocks
    block_sizes = jnp.where(block_sizes > 1e-10, block_sizes, 1.0)

    # Normalize rows
    macro_matrix = numerator / block_sizes[:, None]

    return macro_matrix

def loss_fn(logits, micro_matrix, temperature=1.0):
    """
    Loss function to minimize: -EffectiveInformation(macro_matrix).

    Parameters
    ----------
    logits : (N, M) array
        Unnormalized logits for assignment.
    micro_matrix : (N, N) array
        Fixed micro matrix.
    temperature : float
        Softmax temperature.
    """
    # Softmax to get assignment probabilities
    assignment_matrix = jax.nn.softmax(logits / temperature, axis=1)

    # Compute macro matrix
    macro_matrix = soft_coarse_grain(micro_matrix, assignment_matrix)

    # Calculate EI
    ei = jax_core.effective_information(macro_matrix)

    # We want to maximize EI, so minimize negative EI
    return -ei

def optimize_coarse_graining(micro_matrix, n_macro, steps=1000, learning_rate=0.1, key=None):
    """
    Find optimal coarse-graining with n_macro states using gradient descent.

    Parameters
    ----------
    micro_matrix : (N, N) array
        Micro-scale transition matrix.
    n_macro : int
        Number of macro states to search for.
    steps : int
        Number of optimization steps.
    learning_rate : float
        Learning rate.
    key : jax.random.PRNGKey
        Random key.

    Returns
    -------
    final_assignment : (N, M) array
        Hard assignment matrix (one-hot).
    final_ei : float
        Effective Information of the result.
    """
    N = micro_matrix.shape[0]
    if key is None:
        key = jax.random.PRNGKey(0)

    # Initialize logits randomly
    logits = jax.random.normal(key, (N, n_macro))

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(logits)

    @jax.jit
    def step(logits, opt_state):
        # Use a temperature schedule? For now fixed low temp for "hard" approximation
        temp = 0.5
        loss, grads = jax.value_and_grad(loss_fn)(logits, micro_matrix, temp)
        updates, opt_state = optimizer.update(grads, opt_state)
        logits = optax.apply_updates(logits, updates)
        return logits, opt_state, loss

    # Optimization loop
    # We can use lax.scan or python loop. Python loop is fine for top-level.
    for i in range(steps):
        logits, opt_state, loss = step(logits, opt_state)

    # Get final hard assignment
    final_assignment_indices = jnp.argmax(logits, axis=1)
    final_assignment = jax.nn.one_hot(final_assignment_indices, n_macro)

    # Compute final EI on the hard assignment
    final_macro_matrix = soft_coarse_grain(micro_matrix, final_assignment)
    final_ei = jax_core.effective_information(final_macro_matrix)

    return final_assignment, final_ei
