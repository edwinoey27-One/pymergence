import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import einops
from safetensors.numpy import save_file, load_file
import numpy as np

class StochasticMatrix(eqx.Module):
    """
    A differentiable, JIT-compilable Stochastic Matrix using Equinox.
    """
    matrix: jax.Array

    def __init__(self, matrix):
        self.matrix = jnp.asarray(matrix)

    @property
    def n_states(self):
        return self.matrix.shape[0]

    def entropy(self, p):
        # H(p) = - sum p log2 p
        p = jnp.maximum(p, 1e-15)
        return -jnp.sum(p * jnp.log2(p))

    def determinism(self, intervention_distribution=None):
        n = self.n_states
        if intervention_distribution is None:
            intervention_distribution = jnp.ones(n) / n

        row_entropies = jax.vmap(self.entropy)(self.matrix)

        log_n = jnp.log2(n)
        denom = jnp.where(log_n > 0, log_n, 1.0)

        row_dets = 1.0 - row_entropies / denom
        return jnp.dot(row_dets, intervention_distribution)

    def degeneracy(self, intervention_distribution=None):
        n = self.n_states
        if intervention_distribution is None:
            intervention_distribution = jnp.ones(n) / n

        marginal_effect = jnp.dot(intervention_distribution, self.matrix)
        effect_ent = self.entropy(marginal_effect)

        log_n = jnp.log2(n)
        denom = jnp.where(log_n > 0, log_n, 1.0)

        return 1.0 - effect_ent / denom

    def effectiveness(self, intervention_distribution=None):
        return self.determinism(intervention_distribution) - self.degeneracy(intervention_distribution)

    def effective_information(self, intervention_distribution=None):
        n = self.n_states
        eff = self.effectiveness(intervention_distribution)
        return eff * jnp.log2(n)

    def sufficiency(self, cause, effect):
        return self.matrix[cause, effect]

    def necessity(self, cause, effect, intervention_distribution=None):
        n = self.n_states
        if intervention_distribution is None:
            # Default: uniform over all OTHER causes
            # Standard def: weights = 1/(n-1) for c' != c, else 0
            weights = jnp.ones(n) / (n - 1)
            weights = weights.at[cause].set(0.0)
        else:
            weights = intervention_distribution

        # Sum over alt_cause != cause of w[alt] * P(effect | alt)
        # Vectorized: dot product
        # Ensure weights[cause] is 0 if intended, but let caller handle dist

        col = self.matrix[:, effect]
        weighted_sum = jnp.dot(weights, col)
        return 1.0 - weighted_sum

    def average_sufficiency(self, intervention_distribution=None):
        n = self.n_states
        if intervention_distribution is None:
            intervention_distribution = jnp.ones(n) / n

        # avg_suff = sum_{c, e} P(c) * G[c, e] * suff(c, e)
        # suff(c, e) = G[c, e]
        # avg_suff = sum_{c, e} P(c) * G[c, e]^2

        weighted_matrix = self.matrix * intervention_distribution[:, None]
        return jnp.sum(weighted_matrix * self.matrix)

    def average_necessity(self, intervention_distribution=None):
        n = self.n_states
        if intervention_distribution is None:
            intervention_distribution = jnp.ones(n) / n

        # nec(c, e) = 1 - sum_{c' != c} (1/(n-1)) * G[c', e]
        col_sums = jnp.sum(self.matrix, axis=0) # (n,)
        col_sums_broad = col_sums[None, :] # (1, n)

        denom = jnp.where(n > 1, n - 1, 1.0)

        # sum_{c' != c} G[c', e] = col_sum[e] - G[c, e]
        term = (col_sums_broad - self.matrix) / denom
        nec_matrix = 1.0 - term

        # Weight W_{ce} = P(c) * G[c, e]
        P = intervention_distribution[:, None]
        W = P * self.matrix

        return jnp.sum(W * nec_matrix)

    def optimize_coarse_graining(self, n_macro, steps=1000, learning_rate=0.1, key=None):
        """
        Convenience method to optimize coarse graining on this matrix.
        Returns (partition, ei).
        """
        partition, losses = train_partition(self, n_macro, steps, learning_rate, key)
        macro_sm = partition.coarse_grain(self)
        ei = macro_sm.effective_information()
        return partition.hard_assignment(), ei

# Functional Wrappers
def determinism(matrix, intervention_distribution=None):
    return StochasticMatrix(matrix).determinism(intervention_distribution)

def degeneracy(matrix, intervention_distribution=None):
    return StochasticMatrix(matrix).degeneracy(intervention_distribution)

def effectiveness(matrix, intervention_distribution=None):
    return StochasticMatrix(matrix).effectiveness(intervention_distribution)

def effective_information(matrix, intervention_distribution=None):
    return StochasticMatrix(matrix).effective_information(intervention_distribution)

class Partition(eqx.Module):
    logits: jax.Array
    n_macro: int = eqx.field(static=True)
    n_micro: int = eqx.field(static=True)
    temperature: float

    def __init__(self, n_micro, n_macro, key, temperature=1.0):
        self.logits = jax.random.normal(key, (n_micro, n_macro))
        self.n_macro = n_macro
        self.n_micro = n_micro
        self.temperature = temperature

    def __call__(self):
        return jax.nn.softmax(self.logits / self.temperature, axis=1)

    def hard_assignment(self):
        indices = jnp.argmax(self.logits, axis=1)
        return jax.nn.one_hot(indices, self.n_macro)

    def coarse_grain(self, micro_matrix: StochasticMatrix):
        P = self()
        block_mass = einops.reduce(P, 'n m -> m', 'sum')
        block_mass = jnp.where(block_mass > 1e-10, block_mass, 1.0)
        W = P / block_mass[None, :]
        macro_matrix = jnp.einsum('im, ij, jn -> mn', W, micro_matrix.matrix, P)
        return StochasticMatrix(macro_matrix)


@eqx.filter_jit
def _loss_fn(partition, micro_matrix):
    macro_sm = partition.coarse_grain(micro_matrix)
    return -macro_sm.effective_information()


@eqx.filter_jit
def _step(partition, opt_state, micro_matrix, optimizer):
    loss, grads = eqx.filter_value_and_grad(_loss_fn)(partition, micro_matrix)
    updates, opt_state = optimizer.update(grads, opt_state, partition)
    partition = eqx.apply_updates(partition, updates)
    return partition, opt_state, loss


@eqx.filter_jit
def train_partition(
    micro_matrix: StochasticMatrix,
    n_macro: int,
    steps: int = 1000,
    lr: float = 0.1,
    key=None,
):
    if key is None:
        key = jax.random.PRNGKey(0)

    n_micro = micro_matrix.n_states
    partition = Partition(n_micro, n_macro, key)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(partition)

    def scan_step(carry, _):
        partition, opt_state = carry
        partition, opt_state, loss = _step(partition, opt_state, micro_matrix, optimizer)
        return (partition, opt_state), loss

    (final_partition, _), losses = jax.lax.scan(
        scan_step, (partition, opt_state), None, length=steps
    )

    return final_partition, losses

def save_model(partition: Partition, filepath: str):
    data = {"logits": np.array(partition.logits)}
    save_file(data, filepath)

def load_model(filepath: str, n_micro: int, n_macro: int, key):
    data = load_file(filepath)
    partition = Partition(n_micro, n_macro, key)
    partition = eqx.tree_at(lambda p: p.logits, partition, jnp.array(data["logits"]))
    return partition
