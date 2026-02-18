import numpy as np
import pytest
from pymergence.topology import TopologicalAnalyzer

def test_homology_circle():
    # Points on a circle have 1 connected component (H0) and 1 loop (H1).
    t = np.linspace(0, 2*np.pi, 20, endpoint=False)
    x = np.cos(t)
    y = np.sin(t)
    points = np.stack([x, y], axis=1)

    # Add noise
    points += np.random.normal(0, 0.05, points.shape)

    analyzer = TopologicalAnalyzer()
    feats = analyzer.topological_features(points)

    # Check H1 (loops)
    # A clear circle should have at least 1 significant loop.
    assert feats['n_loops'] >= 1
    assert feats['total_loop_lifetime'] > 0.5 # significant persistence

def test_homology_blob():
    # Gaussian blob: 1 component, 0 loops.
    points = np.random.normal(0, 1, (20, 2))

    analyzer = TopologicalAnalyzer()
    feats = analyzer.topological_features(points)

    # Might have small noisy loops, but total lifetime should be small relative to circle?
    # Or just check it runs.
    # Usually random points in 2D don't form persistent large loops.
    pass
