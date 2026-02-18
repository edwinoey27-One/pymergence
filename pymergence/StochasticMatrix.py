import numpy as np
import matplotlib.pyplot as plt
from pymergence.utils import kl_divergence, kl_divergence_base2, entropy
from pymergence.CoarseGraining import CoarseGraining, generate_all_coarse_grainings

# Try importing JAX
try:
    import jax
    import jax.numpy as jnp
    from pymergence import jax_core, optimization
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jnp = None

class StochasticMatrix:
    """
    A class for working with (row) stochastic matrices.
    
    A row stochastic matrix is a square matrix where each row sums to 1 and
    all entries are non-negative, representing transition probabilities
    between states.
    """
    
    def __init__(self, matrix, validate=True, tolerance=1e-10, backend=None):
        """
        Initialize a stochastic matrix.
        
        Parameters
        -----------
        matrix : array-like
            Square matrix with non-negative entries where rows sum to 1
        validate : bool, default=True
            Whether to validate that the matrix is stochastic
        tolerance : float, default=1e-10
            Numerical tolerance for validation checks
        backend : str, optional
            'numpy' or 'jax'. If None, auto-detects based on input type.
        """
        self.tolerance = tolerance
        
        if backend is None:
            if HAS_JAX and (isinstance(matrix, jax.Array) or (hasattr(matrix, 'device_buffer'))):
                backend = 'jax'
            else:
                backend = 'numpy'

        self.backend = backend

        if self.backend == 'jax':
            if not HAS_JAX:
                raise ImportError("JAX backend requested but JAX not available.")
            self.matrix = jnp.asarray(matrix, dtype=float)
        else:
            self.matrix = np.asarray(matrix, dtype=float)

        if validate:
            self._validate_stochastic()
    
    def _validate_stochastic(self):
        """Validate that the matrix is stochastic."""
        if self.matrix.ndim != 2:
            raise ValueError("Matrix must be 2-dimensional")
        
        if self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("Matrix must be square")
        
        if self.backend == 'jax':
            xp = jnp
        else:
            xp = np

        # Check non-negative entries
        if xp.any(self.matrix < -self.tolerance):
            raise ValueError("All matrix entries must be non-negative")
        
        # Check row sums
        row_sums = xp.sum(self.matrix, axis=1)
        if not xp.allclose(row_sums, 1.0, atol=self.tolerance):
            raise ValueError("Each row must sum to 1 (stochastic property)")
    
        return True
    
    def to_jax(self):
        """Convert to JAX backend."""
        if not HAS_JAX:
            raise ImportError("JAX not available.")
        if self.backend == 'jax':
            return self
        return StochasticMatrix(jnp.array(self.matrix), validate=False, backend='jax')

    def optimize_coarse_graining(self, n_macro, steps=1000, learning_rate=0.1, key=None):
        """
        Find optimal coarse-graining using gradient descent (requires JAX).

        Parameters
        ----------
        n_macro : int
            Number of macro states.
        steps : int
        learning_rate : float
        key : jax.random.PRNGKey

        Returns
        -------
        assignment : array
            Hard assignment matrix.
        ei : float
            Effective Information.
        """
        if not HAS_JAX:
            raise ImportError("Optimization requires JAX.")

        mat = self.matrix
        if self.backend != 'jax':
            mat = jnp.array(mat)

        assignment, ei = optimization.optimize_coarse_graining(mat, n_macro, steps, learning_rate, key)
        return assignment, float(ei)

    @property
    def n_states(self):
        """Number of states in the Markov chain."""
        return self.matrix.shape[0]
    
    
    def power(self, n):
        """
        Compute the n-th power of the transition matrix.
        """
        if n < 0:
            raise ValueError("Power must be non-negative")
        
        if self.backend == 'jax':
            if n == 0:
                return StochasticMatrix(jnp.eye(self.n_states), validate=False, backend='jax')
            result = jax_core.transition_matrix_power(self.matrix, n)
            return StochasticMatrix(result, validate=False, backend='jax')
        else:
            if n == 0:
                return StochasticMatrix(np.eye(self.n_states), validate=False)
            result = np.linalg.matrix_power(self.matrix, n)
            return StochasticMatrix(result, validate=False)

    
    def transition_probability(self, from_state, to_state, steps=1):
        """
        Get transition probability from one state to another in given steps.
        """
        if steps == 1:
            return self.matrix[from_state, to_state]
        else:
            power_matrix = self.power(steps)
            return power_matrix.matrix[from_state, to_state]
    
    def get_maxent_distribution(self):
        """
        Get the maximum entropy distribution for the stochastic matrix.
        """
        if self.backend == 'jax':
            return jnp.ones(self.n_states) / self.n_states
        return np.ones(self.n_states) / self.n_states
    
    def all_coarse_grainings(self):
        """
        Generate all possible coarse-grainings of the stochastic matrix.
        (NumPy/Python only for now as generator yields objects)
        """
        return generate_all_coarse_grainings(self.n_states)

    def finest_graining(self):
        partition = tuple((i,) for i in range(self.n_states))
        return CoarseGraining(partition)
    
    def coarsest_graining(self):
        partition = (tuple(range(self.n_states)),)
        return CoarseGraining(partition)
    
    def coarse_grain(self, coarse_graining):
        """
        Coarse grain the stochastic matrix according to a given coarse-graining.
        """
        if not isinstance(coarse_graining, CoarseGraining):
            raise ValueError("coarse_graining must be an instance of CoarseGraining")
        
        xp = jnp if self.backend == 'jax' else np

        n_blocks = coarse_graining.n_blocks
        coarse_matrix = xp.zeros((n_blocks, n_blocks), dtype=float)
        
        # JAX-friendly logic using simple loop (assuming n_blocks small)
        if self.backend == 'jax':
            # Construct projection matrix P (N, M)
            # P[i, b] = 1 if i in block b
            N = self.n_states
            M = n_blocks
            P = jnp.zeros((N, M))
            for b_idx, block in enumerate(coarse_graining.partition):
                # block is a tuple of indices
                indices = jnp.array(list(block))
                P = P.at[indices, b_idx].set(1.0)

            # coarse_matrix[I, J] = sum_{i in I, j in J} G[i, j]
            # = sum_{i, j} P[i, I] * G[i, j] * P[j, J]
            # = (P.T @ G @ P)[I, J]
            # Wait, this is unnormalized sum. Correct.

            numerator = P.T @ self.matrix @ P

            # Row normalization
            row_sums = jnp.sum(numerator, axis=1)
            row_sums = jnp.where(row_sums > 0, row_sums, 1.0)

            coarse_matrix = numerator / row_sums[:, None]

            return StochasticMatrix(coarse_matrix, validate=True, backend='jax')
        
        else:
            # NumPy logic
            for i, block_i in enumerate(coarse_graining.partition):
                for j, block_j in enumerate(coarse_graining.partition):
                    coarse_matrix[i, j] = np.sum(self.matrix[np.ix_(block_i, block_j)])

            # Normalize each row
            for i in range(n_blocks):
                row_sum = coarse_matrix[i].sum()
                if row_sum > 0:
                    coarse_matrix[i] /= row_sum

            return StochasticMatrix(coarse_matrix, validate=True, backend='numpy')

    def evolve_distribution(self, initial_distribution, steps=1):
        if self.backend == 'jax':
            return jax_core.evolve_distribution_trajectory(self.matrix, initial_distribution, steps)
        else:
            if not np.isclose(np.sum(initial_distribution), 1.0, atol=self.tolerance):
                raise ValueError("Initial distribution must sum to 1")
            
            current_distribution = np.asarray(initial_distribution, dtype=float)
            distributions = [current_distribution]
            for _ in range(steps):
                current_distribution = current_distribution.dot(self.matrix)
                current_distribution = np.clip(current_distribution, 0, None)
                distributions.append(current_distribution)
            return np.array(distributions)

    def inconsistency_sum_over_starts(self, coarse_graining, steps=5):
        n = self.n_states
        xp = jnp if self.backend == 'jax' else np
        total_kl = 0.0
        
        # Helper for KL
        kl_func = jax_core.kl_divergence if self.backend == 'jax' else kl_divergence

        # Prepare Coarse Graining Projection if JAX
        if self.backend == 'jax':
            N = n
            M = coarse_graining.n_blocks
            P = jnp.zeros((N, M))
            for b_idx, block in enumerate(coarse_graining.partition):
                indices = jnp.array(list(block))
                P = P.at[indices, b_idx].set(1.0)
            
            # Macro matrix
            macro_matrix = self.coarse_grain(coarse_graining)

            # Loop over start nodes
            # We can vectorize this too?
            # Evolving all start nodes is just evolving the identity matrix!
            # Identity (N, N) where row i is delta_i.
            # evolve_distribution_trajectory(Identity, steps) -> (steps+1, N, N)
            # This is efficient!

            micro_evolution = jax_core.evolve_distribution_trajectory(self.matrix, jnp.eye(n), steps)
            # shape (steps+1, N, N). Dim 1 is start node, Dim 2 is state.

            # Macro starts: depends on block of start node.
            # Start node i is in block b. So macro start is delta_b.
            # Macro evolution for all possible blocks:
            macro_evolution_blocks = jax_core.evolve_distribution_trajectory(macro_matrix.matrix, jnp.eye(M), steps)
            # shape (steps+1, M, M).

            # Map micro_evolution to coarse:
            # For each t, each start node i: dist is micro_evolution[t, i, :]
            # Coarse grain it: dist @ P
            # So micro_agg[t] = micro_evolution[t] @ P  -> shape (N, M)

            micro_agg = jnp.einsum('tij,jk->tik', micro_evolution, P)

            # Map start node to block index
            # node_to_block[i] = b
            node_to_block = jnp.zeros(n, dtype=int)
            for b_idx, block in enumerate(coarse_graining.partition):
                 indices = jnp.array(list(block))
                 node_to_block = node_to_block.at[indices].set(b_idx)

            # Macro dist for start node i:
            # It's macro_evolution_blocks[t, node_to_block[i], :]
            # shape (steps+1, N, M)

            macro_dist = macro_evolution_blocks[:, node_to_block, :]

            # Compute KL(micro_agg || macro_dist)
            # sum over t, i

            kl_vals = jax.vmap(lambda p, q: kl_func(p, q))(
                micro_agg.reshape(-1, M),
                macro_dist.reshape(-1, M)
            )
            total_kl = jnp.sum(kl_vals)
            return total_kl

        else:
            # Generic NumPy implementation
            for start_node in range(n):
                init_micro = np.zeros(n, dtype=float)
                init_micro[start_node] = 1.0

                block_index = None
                for b_idx, block in enumerate(coarse_graining.partition):
                    if start_node in block:
                        block_index = b_idx
                        break

                init_macro = np.zeros(len(coarse_graining.partition), dtype=float)
                init_macro[block_index] = 1.0

                macro_matrix = self.coarse_grain(coarse_graining)

                micro_dist_evolution = self.evolve_distribution(init_micro, steps)
                macro_dist_evolution = macro_matrix.evolve_distribution(init_macro, steps)

                for t in range(steps+1):
                    micro_agg = coarse_graining.coarse_grain_distribution(micro_dist_evolution[t])
                    macro_dist = macro_dist_evolution[t]
                    total_kl += kl_divergence(micro_agg, macro_dist)

            return total_kl

    def is_consistent_with(self, coarse_graining, kl_threshold=1e-2, steps=5):
        return self.inconsistency_sum_over_starts(coarse_graining, steps=steps) < kl_threshold

    def find_consistent_coarse_grainings(self, coarse_grainings=None, kl_threshold=1e-2, steps=5):
        consistent_grainings = []
        if coarse_grainings is None:
            coarse_grainings = self.all_coarse_grainings()

        for cg in coarse_grainings:
            if self.is_consistent_with(cg, kl_threshold=kl_threshold, steps=steps):
                consistent_grainings.append(cg)
        return consistent_grainings
    
    def sufficiency(self, cause, effect):
        if self.backend == 'jax':
             return jax_core.sufficiency(self.matrix, cause, effect)
        return self.matrix[cause, effect]

    def necessity(self, cause, effect, intervention_distribution=None):
        if self.backend == 'jax':
             return jax_core.necessity(self.matrix, cause, effect, intervention_distribution)

        if intervention_distribution is not None:
            weights = np.asarray(intervention_distribution, dtype=float)
            if weights.size != self.n_states - 1:
                raise ValueError(f"intervention_distribution must be length {self.n_states - 1}.")
            if not np.isclose(weights.sum(), 1.0, atol=1e-12):
                raise ValueError("intervention_distribution must sum to 1.")
        else:
            weights = np.ones(self.n_states) / (self.n_states-1)
            weights[cause] = 0.0
        weighted_transitions = [weights[alt_cause] * self.matrix[alt_cause, effect] for alt_cause in range(self.n_states) if alt_cause != cause]
        return 1 - sum(weighted_transitions)
        
    def average_sufficiency(self, intervention_distribution=None):
        if self.backend == 'jax':
            return jax_core.average_sufficiency(self.matrix, intervention_distribution)

        if intervention_distribution is None:
            intervention_distribution = np.ones(self.n_states) / self.n_states
        prob_of_transitions = np.diag(intervention_distribution) @ self.matrix
        avg_sufficiency = np.sum(prob_of_transitions * self.matrix)
        return avg_sufficiency

    def average_necessity(self, intervention_distribution=None):
        if self.backend == 'jax':
            return jax_core.average_necessity(self.matrix, intervention_distribution)

        if intervention_distribution is None:
            intervention_distribution = np.ones(self.n_states) / self.n_states
        prob_of_transitions = np.diag(intervention_distribution) @ self.matrix
        avg_necessity = 0.0
        for cause in range(self.n_states):
            for effect in range(self.n_states):
                avg_necessity += prob_of_transitions[cause, effect] * self.necessity(cause, effect)
        return avg_necessity

    def determinism(self, intervention_distribution=None):
        if self.backend == 'jax':
            return jax_core.determinism(self.matrix, intervention_distribution)

        if self.n_states == 0: raise ValueError("Empty")
        if self.n_states == 1: return 1.0
        if intervention_distribution is None:
            intervention_distribution = np.ones(self.n_states) / self.n_states
        n = self.n_states
        weights = np.asarray(intervention_distribution, dtype=float)
        row_det = np.zeros(n, dtype=float)
        for i in range(n):
            row_i = self.matrix[i,:]
            row_entropy = entropy(row_i)
            row_det[i] = 1 - row_entropy / np.log2(n)
        return np.sum(row_det * weights)

    def degeneracy(self, intervention_distribution=None):
        if self.backend == 'jax':
            return jax_core.degeneracy(self.matrix, intervention_distribution)

        if self.n_states <= 1: return 1.0
        if intervention_distribution is None:
            intervention_distribution = np.ones(self.n_states) / self.n_states
        n = self.n_states
        w = np.asarray(intervention_distribution, dtype=float)
        marginal_effect_dist = np.zeros(n, dtype=float)
        for i in range(n):
            marginal_effect_dist += w[i] * self.matrix[i,:]
        effect_entropy = entropy(marginal_effect_dist)
        return 1 - effect_entropy / np.log2(n)

    def effectiveness(self, intervention_distribution=None):
        if self.backend == 'jax':
            return jax_core.effectiveness(self.matrix, intervention_distribution)

        det = self.determinism(intervention_distribution)
        deg = self.degeneracy(intervention_distribution)
        return det - deg
    
    def effective_information(self, intervention_distribution=None):
        if self.backend == 'jax':
            return jax_core.effective_information(self.matrix, intervention_distribution)

        return self.effectiveness(intervention_distribution=intervention_distribution) * np.log2(self.n_states)

    def suff_plus_nec(self, normalize=False, intervention_distribution=None):
        if self.backend == 'jax':
            avg_suff = self.average_sufficiency(intervention_distribution)
            avg_nec = self.average_necessity(intervention_distribution)
            if normalize: return avg_suff + avg_nec - 1
            return avg_suff + avg_nec

        avg_suff = self.average_sufficiency(intervention_distribution=intervention_distribution)
        avg_nec = self.average_necessity(intervention_distribution=intervention_distribution)
        if normalize:
            return avg_suff + avg_nec - 1
        else:
            return avg_suff + avg_nec        

    def __str__(self):
        return f"StochasticMatrix({self.n_states}x{self.n_states}, backend={self.backend}):\n{self.matrix}"
    
    def __repr__(self):
        return f"StochasticMatrix(shape={self.matrix.shape}, backend={self.backend})"
    
    def __eq__(self, other):
        if not isinstance(other, StochasticMatrix):
            return False
        xp = jnp if self.backend == 'jax' else np
        other_mat = other.matrix
        if other.backend != self.backend:
             other_mat = xp.array(other_mat)
        return xp.allclose(self.matrix, other_mat, atol=self.tolerance)
    
    def __matmul__(self, other):
        if isinstance(other, StochasticMatrix):
            if self.backend == 'jax' or other.backend == 'jax':
                m1 = jnp.array(self.matrix)
                m2 = jnp.array(other.matrix)
                return StochasticMatrix(m1 @ m2, validate=False, backend='jax')
            result = self.matrix @ other.matrix
            return StochasticMatrix(result, validate=False, backend='numpy')
        else:
            return self.matrix @ other
        
    def plot(self, ax=None, **kwargs):
        if ax is None:
            _, ax = plt.subplots()
        
        # Convert to numpy for plotting
        mat = np.array(self.matrix)
        cax = ax.imshow(mat, cmap='viridis', aspect='auto', **kwargs)
        ax.set_title("Stochastic Matrix Heatmap")
        ax.set_xlabel("To State")
        ax.set_ylabel("From State")
        
        plt.colorbar(cax, ax=ax)
        return ax
