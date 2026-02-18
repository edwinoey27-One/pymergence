import jax
import jax.numpy as jnp
import polars as pl
from brax import envs
import numpy as np

class BraxDataLoader:
    """
    A data loader for Brax environments using Polars and PyArrow.
    """
    env_name: str
    states: jax.Array

    def __init__(self, env_name: str):
        self.env_name = env_name
        self.states = None

    def collect_trajectories(self, num_steps: int = 1000, seed: int = 0):
        """
        Collect trajectories from the Brax environment.
        """
        env = envs.create(env_name=self.env_name)
        key = jax.random.PRNGKey(seed)

        # Reset and run
        reset_key, run_key = jax.random.split(key)
        state = env.reset(rng=reset_key)

        # Wrap step for scannability
        jit_step = jax.jit(env.step)

        # Split keys for actions
        keys = jax.random.split(run_key, num_steps)

        def step_fn(state, key):
            # Random action
            action = jax.random.uniform(key, shape=(env.action_size,), minval=-1.0, maxval=1.0)
            next_state = jit_step(state, action)
            return next_state, next_state.obs

        # Run scan loop
        final_state, obs_trajectory = jax.lax.scan(step_fn, state, keys)

        self.states = obs_trajectory
        return self.states

    def to_polars(self):
        """
        Convert collected states to a Polars DataFrame.
        """
        if self.states is None:
            raise ValueError("No states collected. Run collect_trajectories first.")

        # Move to CPU for Polars
        np_states = jax.device_get(self.states)

        # Create column names
        dim = np_states.shape[1]
        cols = [f"obs_{i}" for i in range(dim)]

        df = pl.DataFrame(np_states, schema=cols, orient="row")
        return df

    def to_jax_via_arrow(self):
        """
        Demonstrate zero-copy transfer: Polars -> Arrow -> JAX.
        """
        df = self.to_polars()

        # Polars -> Arrow
        arrow_table = df.to_arrow()

        # Arrow -> Numpy (Zero Copy)
        # Using to_pandas() is usually safest/fastest path to numpy from Arrow in Python
        np_arr = arrow_table.to_pandas().to_numpy()

        return jnp.array(np_arr)
