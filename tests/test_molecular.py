import jax
import jax.numpy as jnp
import pytest
from pymergence.molecular import MolecularDataLoader

def test_molecular_simulation():
    # Small simulation
    loader = MolecularDataLoader(n_particles=10, dimension=2, box_size=5.0)

    # Run
    traj = loader.simulate(steps=20, seed=0)

    # Check shape: (n_outer, N, D)
    # 20 steps, record interval 10 -> 2 frames
    assert traj.shape == (2, 10, 2)

    # Check movement
    assert not jnp.allclose(traj[0], traj[1]), "Particles should move"

def test_feature_extraction():
    loader = MolecularDataLoader(n_particles=10, dimension=2, box_size=5.0)

    # Mock trajectory: (2, 10, 2)
    traj = jnp.zeros((2, 10, 2))
    # Frame 0: all at 0
    # Frame 1: all at 2.5 (center)
    traj = traj.at[1].set(2.5)

    features = loader.get_grid_features(traj, grid_bins=2)
    # Grid 2x2. Cell size 2.5.
    # Frame 0: all in bin (0,0) (index 0)
    # Frame 1: all in bin (1,1) (index 3)

    assert features.shape == (2, 4) # (T, cells)
    assert features[0, 0] == 10
    assert features[1, 3] == 10
