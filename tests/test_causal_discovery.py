import numpy as np
import pytest
from pymergence.causal_discovery import CausalDiscovery

def test_causal_discovery():
    # Generate simple causal structure: X -> Y
    n_samples = 200
    x = np.random.normal(0, 1, n_samples)
    y = 0.8 * x + np.random.normal(0, 0.2, n_samples)

    data = np.stack([x, y], axis=1) # (N, 2)

    cd = CausalDiscovery()
    cg = cd.learn_dag(data)

    adj = cd.to_adjacency_matrix(cg)

    # Check graph. X is node 0, Y is node 1.
    # Adjacency matrix in causal-learn:
    # 0 = no edge
    # 1 = X --> Y
    # -1 = X -- Y (undirected)
    # 2 = X <-- Y

    # We expect X -> Y or X -- Y (since they are correlated)
    # The direction might be ambiguous without more variables or non-gaussianity.
    # But there should be *some* edge.

    edge = adj[0, 1]
    assert edge != 0, "Should detect relationship between X and Y"

def test_conditional_independence():
    # X -> Y -> Z
    n_samples = 200
    x = np.random.normal(0, 1, n_samples)
    y = 0.8 * x + np.random.normal(0, 0.1, n_samples)
    z = 0.8 * y + np.random.normal(0, 0.1, n_samples)

    data = np.stack([x, y, z], axis=1)

    cd = CausalDiscovery()
    cg = cd.learn_dag(data)
    adj = cd.to_adjacency_matrix(cg)

    # Should detect edge X-Y and Y-Z
    # Should NOT detect edge X-Z (conditionally independent given Y)

    assert adj[0, 1] != 0
    assert adj[1, 2] != 0
    assert adj[0, 2] == 0 # No direct edge
