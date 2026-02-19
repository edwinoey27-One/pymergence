import numpy as np
try:
    import pyphi
    HAS_PYPHI = True
except ImportError:
    HAS_PYPHI = False

class IITAnalyzer:
    """
    Bridge between PyMergence StochasticMatrix and PyPhi for Integrated Information Theory (IIT).
    Calculates Phi (integrated information) to quantify causal irreducibility.
    """

    def __init__(self, stochastic_matrix):
        """
        Args:
            stochastic_matrix: PyMergence StochasticMatrix object.
        """
        if not HAS_PYPHI:
            raise ImportError("pyphi is required for IIT analysis.")

        self.sm = stochastic_matrix
        # PyPhi usually requires a TPM (Transition Probability Matrix) in state-by-node format or state-by-state.
        # PyPhi works with discrete systems of binary (or n-ary) nodes.
        # Our StochasticMatrix is (N, N) where N = num_states.
        # If N = 2^k, we can map to k binary nodes.

        # Check if N is a power of 2
        n = self.sm.n_states
        self.num_nodes = int(np.log2(n))
        if 2**self.num_nodes != n:
            raise ValueError(f"State space size {n} is not a power of 2. PyPhi node mapping ambiguous.")

        # Convert to PyPhi TPM format
        # PyPhi expects a TPM where rows are states at t, cols are states at t+1.
        # Format: State-by-state is supported for 'Network'.
        # However, typically PyPhi converts it to a Node-wise mechanism.

        # We need the numpy matrix
        if hasattr(self.sm, 'backend') and self.sm.backend == 'jax':
            # Convert JAX array to numpy
            import jax
            self.tpm = np.array(self.sm.matrix)
        else:
            self.tpm = np.array(self.sm.matrix)

        # PyPhi expects states in Little-Endian or Big-Endian?
        # Default is usually Big-Endian (node 0 is MSB or LSB).
        # We assume standard lexicographic order 00, 01, 10, 11.

        self.network = pyphi.Network(self.tpm)

    def compute_phi(self, state_index=None):
        """
        Compute Big Phi (System Integrated Information) for a given state.

        Args:
            state_index: int, index of the current state.

        Returns:
            phi_value: float
        """
        if state_index is None:
            # Assume state 0 or uniform?
            # Usually phi is state-dependent.
            # Let's pick a state or average?
            # IIT 3.0/4.0 is state-dependent.
            # Let's use the maxent distribution state or just state 0.
            state_index = 0

        # Convert index to tuple of node states
        state_tuple = pyphi.convert.le_index2state(state_index, self.num_nodes)

        # Define subsystem (whole system)
        subsystem = pyphi.Subsystem(self.network, state_tuple)

        # Compute Phi (Big Phi)
        # using 'sia' (System Irreducibility Analysis)
        sia = pyphi.compute.sia(subsystem)

        return sia.phi

    def compute_major_complex(self, state_index=0):
        """
        Find the major complex (subset of nodes with max Phi).
        """
        state_tuple = pyphi.convert.le_index2state(state_index, self.num_nodes)
        subsystem = pyphi.Subsystem(self.network, state_tuple)

        # Find major complex
        # This is computationally expensive
        return pyphi.compute.major_complex(self.network, state_tuple)
