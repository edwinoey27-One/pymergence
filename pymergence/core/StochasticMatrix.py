import numpy as np
import matplotlib.pyplot as plt
from pymergence.core.utils import kl_divergence, entropy
from pymergence.core.CoarseGraining import CoarseGraining, generate_all_coarse_grainings

# Try importing JAX and Equinox
try:
    import jax
    import jax.numpy as jnp
    # Correct import for the new structure
    from pymergence.accel import jax_core
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jnp = None

class StochasticMatrix:
    """
    A class for working with (row) stochastic matrices.
    Supports 'numpy' and 'jax' (Equinox) backends.
    """

    def __init__(self, matrix, validate=True, tolerance=1e-10, backend=None):
        self.tolerance = tolerance

        # Determine backend
        if backend is None:
            if HAS_JAX and (isinstance(matrix, jax.Array) or hasattr(matrix, 'device_buffer')):
                backend = 'jax'
            else:
                backend = 'numpy'

        self.backend = backend

        if self.backend == 'jax':
            if not HAS_JAX:
                raise ImportError("JAX backend requested but JAX not available.")
            # Create the Equinox module instance
            self._jax_obj = jax_core.StochasticMatrix(matrix)
            self.matrix = self._jax_obj.matrix # Expose matrix for property access
        else:
            self.matrix = np.asarray(matrix, dtype=float)
            if validate:
                self._validate_stochastic()

    def _validate_stochastic(self):
        if self.matrix.ndim != 2:
            raise ValueError("Matrix must be 2-dimensional")
        if self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("Matrix must be square")
        if np.any(self.matrix < -self.tolerance):
            raise ValueError("All matrix entries must be non-negative")
        row_sums = np.sum(self.matrix, axis=1)
        if not np.allclose(row_sums, 1.0, atol=self.tolerance):
            raise ValueError("Each row must sum to 1 (stochastic property)")
        return True

    def to_jax(self):
        """Convert to JAX backend (Equinox module)."""
        if not HAS_JAX:
            raise ImportError("JAX not available.")
        if self.backend == 'jax':
            return self._jax_obj
        # Create Equinox module directly
        return jax_core.StochasticMatrix(jnp.array(self.matrix))

    def optimize_coarse_graining(self, n_macro, steps=1000, learning_rate=0.1, key=None):
        """
        Find optimal coarse-graining using gradient descent (requires JAX).
        Delegates to jax_core.train_partition.
        """
        if not HAS_JAX:
            raise ImportError("Optimization requires JAX.")

        # Convert to Equinox module
        jax_sm = self.to_jax()

        partition, losses = jax_core.train_partition(jax_sm, n_macro, steps, learning_rate, key)

        # Compute final EI using coarse_grain method of partition
        macro_sm = partition.coarse_grain(jax_sm)
        final_ei = macro_sm.effective_information()

        return partition, float(final_ei)

    @property
    def n_states(self):
        return self.matrix.shape[0]

    def effective_information(self, intervention_distribution=None):
        if self.backend == 'jax':
            # Delegate to Equinox module method
            return float(self._jax_obj.effective_information(intervention_distribution))

        return self.effectiveness(intervention_distribution) * np.log2(self.n_states)

    def effectiveness(self, intervention_distribution=None):
        if self.backend == 'jax':
            return float(self._jax_obj.effectiveness(intervention_distribution))

        det = self.determinism(intervention_distribution)
        deg = self.degeneracy(intervention_distribution)
        return det - deg

    def determinism(self, intervention_distribution=None):
        if self.backend == 'jax':
            return float(self._jax_obj.determinism(intervention_distribution))

        # NumPy impl
        if self.n_states <= 1: return 1.0
        if intervention_distribution is None:
            intervention_distribution = np.ones(self.n_states) / self.n_states

        row_det = np.zeros(self.n_states)
        for i in range(self.n_states):
            # entropy here is from utils (NumPy)
            row_det[i] = 1 - entropy(self.matrix[i]) / np.log2(self.n_states)
        return np.sum(row_det * intervention_distribution)

    def degeneracy(self, intervention_distribution=None):
        if self.backend == 'jax':
            return float(self._jax_obj.degeneracy(intervention_distribution))

        # NumPy impl
        if self.n_states <= 1: return 1.0
        if intervention_distribution is None:
            intervention_distribution = np.ones(self.n_states) / self.n_states

        marginal = intervention_distribution @ self.matrix
        return 1 - entropy(marginal) / np.log2(self.n_states)

    def average_sufficiency(self, intervention_distribution=None):
        if self.backend == 'jax':
             return float(self._jax_obj.average_sufficiency(intervention_distribution))
        if intervention_distribution is None:
            intervention_distribution = np.ones(self.n_states) / self.n_states
        prob_of_transitions = np.diag(intervention_distribution) @ self.matrix
        return np.sum(prob_of_transitions * self.matrix)

    def average_necessity(self, intervention_distribution=None):
        if self.backend == 'jax':
             return float(self._jax_obj.average_necessity(intervention_distribution))
        if intervention_distribution is None:
            intervention_distribution = np.ones(self.n_states) / self.n_states
        prob_of_transitions = np.diag(intervention_distribution) @ self.matrix
        avg_nec = 0.0
        for cause in range(self.n_states):
            for effect in range(self.n_states):
                # Using local method or explicit formula if necessity method missing
                avg_nec += prob_of_transitions[cause, effect] * self.necessity(cause, effect)
        return avg_nec

    def sufficiency(self, cause, effect):
        if self.backend == 'jax':
            return float(self._jax_obj.sufficiency(cause, effect))
        return self.matrix[cause, effect]

    def necessity(self, cause, effect, intervention_distribution=None):
        if self.backend == 'jax':
             return float(self._jax_obj.necessity(cause, effect, intervention_distribution))
        if intervention_distribution is not None:
            weights = np.asarray(intervention_distribution, dtype=float)
        else:
            weights = np.ones(self.n_states) / (self.n_states-1)
            weights[cause] = 0.0
        weighted_transitions = [weights[alt] * self.matrix[alt, effect] for alt in range(self.n_states) if alt != cause]
        return 1 - sum(weighted_transitions)

    def coarse_grain(self, coarse_graining):
        # Handle backend logic. For JAX, we might need a workaround or implement logic.
        # CoarseGraining object is Python tuples.

        if self.backend == 'jax':
             # Fallback: Convert to numpy, do op, convert back to JAX wrapper.
             # This avoids reimplementing complex slicing logic in JAX for now.
             mat_np = np.array(self.matrix)
             temp_sm = StochasticMatrix(mat_np, backend='numpy')
             res = temp_sm.coarse_grain(coarse_graining)
             # Return JAX wrapper
             return StochasticMatrix(jnp.array(res.matrix), backend='jax')

        # NumPy logic
        n_blocks = coarse_graining.n_blocks
        coarse_matrix = np.zeros((n_blocks, n_blocks))
        for i, block_i in enumerate(coarse_graining.partition):
            for j, block_j in enumerate(coarse_graining.partition):
                coarse_matrix[i, j] = np.sum(self.matrix[np.ix_(block_i, block_j)])

        # Normalize
        row_sums = coarse_matrix.sum(axis=1)
        for i in range(n_blocks):
            if row_sums[i] > 0:
                coarse_matrix[i] /= row_sums[i]

        return StochasticMatrix(coarse_matrix, backend='numpy')

    def finest_graining(self):
        return CoarseGraining(tuple((i,) for i in range(self.n_states)))

    def inconsistency_sum_over_starts(self, coarse_graining, steps=5):
        if self.backend == 'jax':
             # Fallback to NumPy logic
             mat_np = np.array(self.matrix)
             sm_np = StochasticMatrix(mat_np, backend='numpy')
             return sm_np.inconsistency_sum_over_starts(coarse_graining, steps)

        # NumPy impl from previous
        n = self.n_states
        total_kl = 0.0
        for start_node in range(n):
            init_micro = np.zeros(n); init_micro[start_node] = 1.0
            block_index = None
            for b_idx, block in enumerate(coarse_graining.partition):
                if start_node in block: block_index = b_idx; break
            init_macro = np.zeros(len(coarse_graining.partition)); init_macro[block_index] = 1.0

            macro_matrix = self.coarse_grain(coarse_graining)
            micro_curr = init_micro; macro_curr = init_macro

            # Initial KL
            micro_agg = coarse_graining.coarse_grain_distribution(micro_curr)
            total_kl += kl_divergence(micro_agg, macro_curr)

            for _ in range(steps):
                micro_curr = micro_curr @ self.matrix
                macro_curr = macro_curr @ macro_matrix.matrix
                micro_agg = coarse_graining.coarse_grain_distribution(micro_curr)
                total_kl += kl_divergence(micro_agg, macro_curr)
        return total_kl

    def __eq__(self, other):
        if not isinstance(other, StochasticMatrix): return False
        xp = jnp if self.backend == 'jax' else np
        other_mat = other.matrix
        if other.backend != self.backend:
             other_mat = xp.array(other_mat)
        return xp.allclose(self.matrix, other_mat, atol=self.tolerance)

    def __str__(self):
        return f"StochasticMatrix({self.n_states}x{self.n_states}, backend={self.backend}):\n{self.matrix}"
    def __repr__(self):
        return f"StochasticMatrix(shape={self.matrix.shape}, backend={self.backend})"
