import jax
import jax.numpy as jnp
from jax_md import space, energy, simulate
import equinox as eqx

class MolecularDataLoader:
    """
    Simulates a molecular system and extracts trajectories for causal analysis.
    Uses Lennard-Jones potential.
    """

    def __init__(self, n_particles=64, dimension=2, box_size=10.0, dt=1e-3):
        self.n_particles = n_particles
        self.dimension = dimension
        self.box_size = box_size
        self.dt = dt

        self.displacement, self.shift = space.periodic(box_size)

        # Lennard-Jones potential
        self.energy_fn = energy.lennard_jones_pair(self.displacement, sigma=1.0, epsilon=1.0)

        # NVT Langevin dynamics
        self.init_fn, self.apply_fn = simulate.nvt_langevin(
            self.energy_fn,
            shift_fn=self.shift,
            dt=dt,
            kT=1.0,
            gamma=0.1
        )

    def simulate(self, steps=1000, seed=0):
        key = jax.random.PRNGKey(seed)
        pos_key, sim_key = jax.random.split(key)

        R_init = jax.random.uniform(pos_key, (self.n_particles, self.dimension), maxval=self.box_size)
        state = self.init_fn(sim_key, R_init)

        record_interval = 10
        n_outer = steps // record_interval

        def outer_step(state, _):
            def inner_step(i, s):
                return self.apply_fn(s)

            state = jax.lax.fori_loop(0, record_interval, inner_step, state)
            # Must wrap position in array to ensure it's saved correctly in scan
            return state, state.position

        final_state, trajectory = jax.lax.scan(outer_step, state, None, length=n_outer)

        return trajectory

    def get_grid_features(self, trajectory, grid_bins=5):
        """
        Discretize trajectory into grid occupancy counts.
        """
        # trajectory: (T, N, D)
        # We compute histogram for each frame

        def frame_hist(pos):
            # pos: (N, D)
            # Use jnp.histogramdd logic manually for D dimensions to be robust or simple binning
            # Quantize positions
            indices = jnp.floor(pos / (self.box_size / grid_bins)).astype(jnp.int32)
            indices = jnp.clip(indices, 0, grid_bins - 1)

            # Linear index
            if self.dimension == 2:
                linear_idx = indices[:, 0] * grid_bins + indices[:, 1]
            else:
                # Generalize if needed
                linear_idx = indices[:, 0] # simplistic

            # Count occurrences of each linear index
            # bins: 0 .. grid_bins^D - 1
            n_cells = grid_bins ** self.dimension
            counts = jnp.bincount(linear_idx, length=n_cells)
            return counts

        features = jax.vmap(frame_hist)(trajectory)
        return features # (T, grid_cells)
