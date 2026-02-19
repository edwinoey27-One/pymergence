import numpy as np
import pandas as pd
try:
    from causallearn.search.ConstraintBased.PC import pc
    # Fix import: cit is lowercase module, not Cit
    from causallearn.utils.cit import chisq, fisherz
    HAS_CAUSAL_LEARN = True
except ImportError:
    HAS_CAUSAL_LEARN = False

class CausalDiscovery:
    """
    Learns causal graphs from observational data using the PC algorithm.
    """
    def __init__(self):
        if not HAS_CAUSAL_LEARN:
            raise ImportError("causal-learn is required for causal discovery.")

    def learn_dag(self, data, alpha=0.05):
        """
        Learn the DAG from data.
        """
        if isinstance(data, pd.DataFrame):
            data_np = data.to_numpy()
        else:
            data_np = np.asarray(data)

        cg = pc(data_np, alpha, fisherz)
        return cg

    def to_adjacency_matrix(self, cg):
        """
        Convert learned graph to adjacency matrix.
        """
        return cg.G.graph

    def estimate_transition_matrix(self, data, adjacency_matrix, n_bins=5):
        """
        Placeholder.
        """
        pass
