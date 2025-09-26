import numpy as np
import matplotlib.pyplot as plt
from pymergence.utils import kl_divergence, kl_divergence_base2, entropy
from pymergence.CoarseGraining import CoarseGraining, generate_all_coarse_grainings

class StochasticMatrix:
    """
    A class for working with (row) stochastic matrices.
    
    A row stochastic matrix is a square matrix where each row sums to 1 and
    all entries are non-negative, representing transition probabilities
    between states.
    """
    
    def __init__(self, matrix, validate=True, tolerance=1e-10):
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
        """
        self.matrix = np.asarray(matrix, dtype=float)
        self.tolerance = tolerance
        
        if validate:
            self._validate_stochastic()
    
    def _validate_stochastic(self):
        """Validate that the matrix is stochastic."""
        if self.matrix.ndim != 2:
            raise ValueError("Matrix must be 2-dimensional")
        
        if self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("Matrix must be square")
        
        # Check non-negative entries
        if np.any(self.matrix < -self.tolerance):
            raise ValueError("All matrix entries must be non-negative")
        
        # Check row sums
        row_sums = np.sum(self.matrix, axis=1)
        if not np.allclose(row_sums, 1.0, atol=self.tolerance):
            raise ValueError("Each row must sum to 1 (stochastic property)")
    
        return True
    
    @property
    def n_states(self):
        """Number of states in the Markov chain."""
        return self.matrix.shape[0]
    
    
    def power(self, n):
        """
        Compute the n-th power of the transition matrix.
        
        Parameters
        -----------
        n : int
            Power to raise the matrix to
            
        Returns
        --------
        StochasticMatrix
            The n-th power of the matrix
        """
        if n < 0:
            raise ValueError("Power must be non-negative")
        if n == 0:
            return StochasticMatrix(np.eye(self.n_states), validate=False)
        
        result = np.linalg.matrix_power(self.matrix, n)
        return StochasticMatrix(result, validate=False)

    
    def transition_probability(self, from_state, to_state, steps=1):
        """
        Get transition probability from one state to another in given steps.
        
        Parameters
        -----------
        from_state : int
            Starting state index
        to_state : int
            Ending state index
        steps : int, default=1
            Number of steps
            
        Returns
        --------
        float
            Transition probability
        """
        if steps == 1:
            return self.matrix[from_state, to_state]
        else:
            power_matrix = self.power(steps)
            return power_matrix.matrix[from_state, to_state]
    
    def get_maxent_distribution(self):
        """
        Get the maximum entropy distribution for the stochastic matrix.
        
        The maximum entropy distribution is uniform over the states.
        
        Returns
        --------
        np.ndarray
            Uniform distribution over states
        """
        return np.ones(self.n_states) / self.n_states
    
    def all_coarse_grainings(self):
        """
        Generate all possible coarse-grainings of the stochastic matrix.
        
        Returns
        --------
        list of CoarseGraining
            List of all possible coarse-grainings
        """
        return generate_all_coarse_grainings(self.n_states)

    def finest_graining(self):
        """
        Get the finest graining of the stochastic matrix.
        
        Returns
        --------
        CoarseGraining
            CoarseGraining object representing the finest graining,
        """
        partition = tuple((i,) for i in range(self.n_states))
        return CoarseGraining(partition)
    
    def coarsest_graining(self):
        """
        Get the coarsest graining of the stochastic matrix.

        Returns
        --------
        CoarseGraining
            CoarseGraining object representing the coarsest graining,
            which groups all states into a single block
        """
        partition = (tuple(range(self.n_states)),)
        return CoarseGraining(partition)
    
    def coarse_grain(self, coarse_graining):
        """
        Coarse grain the stochastic matrix according to a given coarse-graining.
        
        Parameters
        -----------
        coarse_graining : CoarseGraining object
            The coarse-graining to apply, which defines how states are grouped
            
        Returns
        --------
        StochasticMatrix
            The coarse-grained stochastic matrix
        """
        if not isinstance(coarse_graining, CoarseGraining):
            raise ValueError("coarse_graining must be an instance of CoarseGraining")
        
        n_blocks = coarse_graining.n_blocks
        coarse_matrix = np.zeros((n_blocks, n_blocks), dtype=float)
        
        for i, block_i in enumerate(coarse_graining.partition):
            for j, block_j in enumerate(coarse_graining.partition):
                # Sum probabilities from all states in block_i to all states in block_j
                coarse_matrix[i, j] = np.sum(self.matrix[np.ix_(block_i, block_j)])
        
        # Normalize each row
        for i in range(n_blocks):
            row_sum = coarse_matrix[i].sum()
            if row_sum > 0:
                coarse_matrix[i] /= row_sum

        return StochasticMatrix(coarse_matrix, validate=True)

    def evolve_distribution(self, initial_distribution, steps=1):
        """
        Evolve an initial distribution through the stochastic matrix for a given number of steps. 

        Mainly used to verify consistency with a coarse-graining.
        
        Parameters
        -----------
        initial_distribution : array-like
            Initial state distribution (must sum to 1)
        steps : int, default=1
            Number of steps to evolve the distribution
            
        Returns
        --------
        np.ndarray
            List of distributions at each step, including the initial state
        """
        if not np.isclose(np.sum(initial_distribution), 1.0, atol=self.tolerance):
            raise ValueError("Initial distribution must sum to 1")
        
        if steps < 0:
            raise ValueError("Steps must be non-negative")
        
        current_distribution = np.asarray(initial_distribution, dtype=float)
        
        # Collect distributions at each step
        distributions = [current_distribution]
        for _ in range(steps):
            current_distribution = current_distribution.dot(self.matrix)
            # Ensure numerical stability
            current_distribution = np.clip(current_distribution, 0, None)
            distributions.append(current_distribution)
            
        return np.array(distributions)

    def inconsistency_sum_over_starts(self, coarse_graining, steps=5):
        """
        For each node i in the micro chain, start a delta distribution on i,
        and in the macro chain, start a delta on the block containing i.
        Evolve each for 'steps' timesteps and measure KL at each step.
        Return the sum of KL divergences across all nodes i.
        """
        n = self.n_states
        total_kl = 0.0
        
        for start_node in range(n):
            # 1. Micro initial distribution = delta on 'start_node'
            init_micro = np.zeros(n, dtype=float)
            init_micro[start_node] = 1.0
            
            # 2. Macro initial distribution = delta on the block containing 'start_node'
            block_index = None
            for b_idx, block in enumerate(coarse_graining.partition):
                if start_node in block:
                    block_index = b_idx
                    break
            init_macro = np.zeros(len(coarse_graining.partition), dtype=float)
            init_macro[block_index] = 1.0
            
            macro_matrix = self.coarse_grain(coarse_graining)
            # Evolve both
            micro_dist_evolution = self.evolve_distribution(init_micro, steps)
            macro_dist_evolution = macro_matrix.evolve_distribution(init_macro, steps)
            # Sum KL for each timestep
            for t in range(steps+1):
                # Lump the micro dist onto coarse blocks
                micro_agg = coarse_graining.coarse_grain_distribution(micro_dist_evolution[t])
                macro_dist = macro_dist_evolution[t]
                total_kl += kl_divergence(micro_agg, macro_dist)
        
        return total_kl

    def is_consistent_with(self, coarse_graining, kl_threshold=1e-2, steps=5):
        """Check if the stochastic matrix is consistent with a given coarse-graining."""
        return self.inconsistency_sum_over_starts(coarse_graining, steps=steps) < kl_threshold

    def find_consistent_coarse_grainings(self, coarse_grainings=None, kl_threshold=1e-2, steps=5):
        """
        Find coarse-grainings that are consistent with the stochastic matrix.
        
        Parameters
        -----------
        coarse_grainings : list of CoarseGraining objects
            List of coarse-grainings to check for consistency
            
        Returns
        --------
        list of CoarseGraining
            List of coarse-grainings that are consistent with the matrix
        """
        consistent_grainings = []
        
        if coarse_grainings is None:
            coarse_grainings = self.all_coarse_grainings()

        for cg in coarse_grainings:
            if self.is_consistent_with(cg, kl_threshold=kl_threshold, steps=steps):
                consistent_grainings.append(cg)
        
        return consistent_grainings
    
    def sufficiency(self, cause, effect):
        """
        Compute the sufficiency of a cause for an effect.
        
        Parameters
        -----------
        cause : int
            Index of the cause state
        effect : int
            Index of the effect state

        Returns
        --------
        float
            The sufficiency value for the cause-effect pair
        """
        return self.matrix[cause, effect]

    def necessity(self, cause, effect, intervention_distribution=None):
        """
        TODO: check if we should indeed use a uniform prior over all but one state. 
        Compute the necessity of a cause for an effect.
        
        Parameters
        -----------
        cause : int
            Index of the cause state
        effect : int
            Index of the effect state

        Returns
        --------
        float
            The necessity value for the cause-effect pair
        """
        if intervention_distribution is not None:
            weights = np.asarray(intervention_distribution, dtype=float)
            if weights.size != self.n_states - 1:
                raise ValueError(f"intervention_distribution must be length {self.n_states - 1}.")
            if not np.isclose(weights.sum(), 1.0, atol=1e-12):
                raise ValueError("intervention_distribution must sum to 1.")
        else:
            # Default to uniform distribution over states
            weights = np.ones(self.n_states) / (self.n_states-1)
            weights[cause] = 0.0  # Exclude the cause state
        weighted_transitions = [weights[alt_cause] * self.matrix[alt_cause, effect] for alt_cause in range(self.n_states) if alt_cause != cause]
        return 1 - sum(weighted_transitions)
        
        
    def average_sufficiency(self, intervention_distribution=None):
        """
        Compute the average sufficiency of a stochastic matrix under a given intervention distribution.

        Parameters
        -----------
        intervention_distribution : array-like
            Probability distribution over rows, i.e. sums to 1.
        
        Returns
        --------
        avg_sufficiency : float
            The average sufficiency value for the matrix under the given intervention distribution.
        """
        if intervention_distribution is None:
            # Default to uniform distribution over states
            intervention_distribution = np.ones(self.n_states) / self.n_states
            
        prob_of_transitions = np.diag(intervention_distribution) @ self.matrix  # Weighted average of rows
        avg_sufficiency = np.sum(prob_of_transitions * self.matrix)
        return avg_sufficiency

    def average_necessity(self, intervention_distribution=None):
        """
        Compute the average necessity of a stochastic matrix under a given intervention distribution.

        Parameters
        -----------
        intervention_distribution : array-like
            Probability distribution over rows, i.e. sums to 1.
        
        Returns
        --------
        avg_necessity : float
            The average necessity value for the matrix under the given intervention distribution.
        """
        # Default to uniform distribution over states
        if intervention_distribution is None:
            # Default to uniform distribution over states
            intervention_distribution = np.ones(self.n_states) / self.n_states
        prob_of_transitions = np.diag(intervention_distribution) @ self.matrix  # Weighted average of rows
        avg_necessity = 0.0
        for cause in range(self.n_states):
            for effect in range(self.n_states):
                avg_necessity += prob_of_transitions[cause, effect] * self.necessity(cause, effect)
        return avg_necessity


    def determinism(self, intervention_distribution=None):
        """
        Compute the determinism coefficient of a row-stochastic matrix, 
        using the given intervention_distribution as row weights.
        For each row i, compute:
            det_i = KL( G[i,:] || Uniform(n) ) / log2(n)
        Then return the weighted average, weighting row i by 
        intervention_distribution[i].
        
        Parameters
        ----------
        intervention_distribution : array-like
            Probability distribution over rows, i.e. sums to 1.
       
        
        Returns
        -------
        det_coefficient : float
            The weighted-average determinism coefficient.
        """
        if self.n_states == 0:
            raise ValueError("Cannot compute determinism for an empty matrix.")
        if self.n_states == 1:
            return 1.0
        if intervention_distribution is None:
            # Default to uniform distribution over states
            intervention_distribution = np.ones(self.n_states) / self.n_states

        n = self.n_states
        weights = np.asarray(intervention_distribution, dtype=float)
        if weights.size != n:
            raise ValueError(f"intervention_distribution must be length {n}.")
        if not np.isclose(weights.sum(), 1.0, atol=1e-12):
            raise ValueError("intervention_distribution must sum to 1.")


        row_det = np.zeros(n, dtype=float)
        for i in range(n):
            row_i = self.matrix[i,:]
            row_entropy = entropy(row_i)
            row_det[i] = 1 - row_entropy / np.log2(n)

        # Weighted average
        det_coefficient = np.sum(row_det * weights)

        if det_coefficient > 1.0:
            raise ValueError("Degeneracy coefficient exceeds 1.0, which is unexpected.")
        
        return det_coefficient
    

    def degeneracy(self, intervention_distribution=None):
        """
        Compute the 'degeneracy coefficient' of a row-stochastic matrix
        under a specified intervention distribution.

        1) Let w = intervention_distribution (length n, sums to 1).
        2) Form U^E = sum_i [ w_i * G[i,:] ] (a single distribution in R^n).
        3) Compare U^E to the n-dim uniform distribution U = [1/n,...,1/n].
        4) The degeneracy coefficient is the normalised inverse entropy of the marginal effect distribution

        Parameters
        ----------
        intervention_distribution : array-like
            Probability distribution over rows, i.e. sums to 1.

        Returns
        -------
        deg_coeff : float
            The degeneracy coefficient for G under that intervention distribution.
        """
        if self.n_states == 0:
            raise ValueError("Cannot compute degeneracy for an empty matrix.")
        if self.n_states == 1:
            return 1.0
        if intervention_distribution is None:
            # Default to uniform distribution over states
            intervention_distribution = np.ones(self.n_states) / self.n_states

        n = self.n_states
        w = np.asarray(intervention_distribution, dtype=float)
        if w.size != n:
            raise ValueError(f"intervention_distribution must be length {n}.")
        if not np.isclose(w.sum(), 1.0, atol=1e-12):
            raise ValueError("intervention_distribution must sum to 1.")

        # 1) Weighted average of rows is the marginal effect distribution
        marginal_effect_dist = np.zeros(n, dtype=float)
        for i in range(n):
            marginal_effect_dist += w[i] * self.matrix[i,:]

        effect_entropy = entropy(marginal_effect_dist)

        # 4) Divide by log2(n) to normalize
        deg_coefficient = 1 - effect_entropy / np.log2(n)

        if deg_coefficient > 1.0:
            raise ValueError("Degeneracy coefficient exceeds 1.0, which is unexpected.")
        return deg_coefficient

    def effectiveness(self, intervention_distribution=None):
        """
        Compute the effectiveness of a stochastic matrix under a given intervention distribution.
        
        Effectiveness is defined as the determinism coefficient minus the degeneracy coefficient.
        
        Parameters
        -----------
        intervention_distribution : array-like
            Probability distribution over rows, i.e. sums to 1.
        
        Returns
        --------
        effectiveness : float
            The effectiveness value for the matrix under the given intervention distribution.
        """
    
        if intervention_distribution is None:
            # Default to uniform distribution over states
            intervention_distribution = np.ones(self.n_states) / self.n_states
        if not np.isclose(np.sum(intervention_distribution), 1.0, atol=1e-12):
            raise ValueError("intervention_distribution must sum to 1.")
        
        det_val = self.determinism(intervention_distribution)
        deg_val = self.degeneracy(intervention_distribution)
        if det_val - deg_val > 1.0:
            raise ValueError("effectiveness exceeds 1.0, which is unexpected.")

        return det_val - deg_val
    
    def effective_information(self, intervention_distribution=None):
        """
        Compute the effective information of a stochastic matrix under a given intervention distribution.
        
        Parameters
        -----------
        intervention_distribution : array-like, optional
            Probability distribution over rows, i.e. sums to 1.
        
        Returns
        --------
        effective_information : float
            The effective information value for the matrix under the given intervention distribution.
        """
        
        return self.effectiveness(intervention_distribution=intervention_distribution) * np.log2(self.n_states)

    def suff_plus_nec(self, normalize=False, intervention_distribution=None):
        """
        Compute the sum of average sufficiency and average necessity.
        
        Parameters
        -----------
        intervention_distribution : array-like, optional
            Probability distribution over rows, i.e. sums to 1.
        
        Returns
        --------
        float
            The sum of average sufficiency and average necessity values for the matrix
        """
        
        avg_suff = self.average_sufficiency(intervention_distribution=intervention_distribution)
        avg_nec = self.average_necessity(intervention_distribution=intervention_distribution)
        
        if normalize:
            return avg_suff + avg_nec - 1
        else:
            return avg_suff + avg_nec        




    def __str__(self):
        """String representation of the matrix."""
        return f"StochasticMatrix({self.n_states}x{self.n_states}):\n{self.matrix}"
    
    def __repr__(self):
        """Representation of the matrix."""
        return f"StochasticMatrix(shape={self.matrix.shape})"
    
    def __eq__(self, other):
        """Check equality with another StochasticMatrix."""
        if not isinstance(other, StochasticMatrix):
            return False
        return np.allclose(self.matrix, other.matrix, atol=self.tolerance)
    
    def __matmul__(self, other):
        """Matrix multiplication with another StochasticMatrix."""
        if isinstance(other, StochasticMatrix):
            result = self.matrix @ other.matrix
            return StochasticMatrix(result, validate=False)
        else:
            return self.matrix @ other
        
    def plot(self, ax=None, **kwargs):
        """
        Plot the stochastic matrix as a heatmap.
        
        Parameters
        -----------
        ax : matplotlib Axes, optional
            Axes to plot on. If None, creates a new figure and axes.
        **kwargs : dict
            Additional keyword arguments for matplotlib's imshow function
            
        Returns
        --------
        matplotlib Axes
            The axes with the plotted heatmap
        """
        
        if ax is None:
            _, ax = plt.subplots()
        
        cax = ax.imshow(self.matrix, cmap='viridis', aspect='auto', **kwargs)
        ax.set_title("Stochastic Matrix Heatmap")
        ax.set_xlabel("To State")
        ax.set_ylabel("From State")
        
        plt.colorbar(cax, ax=ax)
        
        return ax