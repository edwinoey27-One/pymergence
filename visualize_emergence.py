import jax
import jax.numpy as jnp
import numpy as np
import imageio
from brax import envs
from brax.io import image
from pymergence.brax_bridge import BraxDataLoader
from pymergence.brax_integration import estimate_transition_matrix
from pymergence.StochasticMatrix import StochasticMatrix
from pymergence.jax_core import train_partition

def visualize_emergence(env_name='inverted_pendulum', output_filename='emergence.gif'):
    print(f"1. Loading Brax Environment: {env_name}")
    loader = BraxDataLoader(env_name)

    # Collect trajectory
    print("2. Collecting Trajectory...")
    n_steps = 200
    states = loader.collect_trajectories(num_steps=n_steps, seed=0)
    states_cpu = jax.device_get(states)

    print("3. Discretizing Micro-States (K-Means)...")
    n_micro = 20
    micro_matrix, centroids, labels = estimate_transition_matrix(
        jnp.array(states_cpu), n_clusters=n_micro, kmeans_steps=20
    )

    sm = StochasticMatrix(micro_matrix, backend='jax')

    print("4. Optimizing Macro-States (Causal Emergence)...")
    n_macro = 3
    partition, ei = sm.optimize_coarse_graining(n_macro=n_macro, steps=500)
    print(f"   Effective Information: {ei:.4f} bits")

    assignment = partition.hard_assignment()
    micro_to_macro = jnp.argmax(assignment, axis=1)
    macro_trajectory = micro_to_macro[labels]

    # Visualization Loop
    print("5. Generating Visualization...")
    env = envs.create(env_name=env_name)

    key = jax.random.PRNGKey(0)
    reset_key, run_key = jax.random.split(key)
    state = env.reset(rng=reset_key)
    step_keys = jax.random.split(run_key, n_steps)
    jit_step = jax.jit(env.step)

    frames = []
    colors = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0])
    ]

    for i in range(n_steps):
        # Brax state object structure depends on backend.
        # Usually pipeline_state is the physics state.
        # But brax.io.image.render usually expects [state.pipeline_state] for v2 (MJX)
        # OR [state.qp] for v1 (Legacy).
        # The installed brax 0.14 is legacy/v1 compatible but might use pipeline_state.

        # Let's try pipeline_state if qp fails.
        try:
            qp_or_state = state.pipeline_state
        except AttributeError:
            qp_or_state = state.qp # Legacy fallback

        # Render
        frame_bytes = image.render(env.sys, [qp_or_state], width=320, height=240, fmt='png')
        frame_img = imageio.v3.imread(frame_bytes)

        if i < len(macro_trajectory):
            macro_idx = int(macro_trajectory[i])
            color = colors[macro_idx % len(colors)]

            H, W, C = frame_img.shape
            bar_height = 20
            col_u8 = (color * 255).astype(np.uint8)
            frame_img[-bar_height:, :, 0] = col_u8[0]
            frame_img[-bar_height:, :, 1] = col_u8[1]
            frame_img[-bar_height:, :, 2] = col_u8[2]

        frames.append(frame_img)

        action = jax.random.uniform(step_keys[i], shape=(env.action_size,), minval=-1.0, maxval=1.0)
        state = jit_step(state, action)

    print(f"6. Saving GIF to {output_filename}...")
    imageio.mimsave(output_filename, frames, fps=30)
    print("Done.")

if __name__ == "__main__":
    visualize_emergence()
