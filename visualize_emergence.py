import jax
import jax.numpy as jnp
import numpy as np
import imageio
import matplotlib.pyplot as plt
from brax import envs
from brax.io import image
from pymergence.integration.brax_bridge import BraxDataLoader
from pymergence.integration.brax_integration import estimate_transition_matrix
from pymergence.core.StochasticMatrix import StochasticMatrix
from pymergence.core.backend import selector
from pymergence.accel import jax_core

# Set backend to JAX for speed
selector.backend = 'jax'

def visualize_emergence(env_name='inverted_pendulum', n_steps=2000, output_filename='deep_emergence.gif'):
    print(f"=== Deep Emergence Scan: {env_name} ===")
    print(f"Steps: {n_steps}")

    # 1. Load & Collect
    print("1. Collecting High-Fidelity Trajectory...")
    loader = BraxDataLoader(env_name)
    # Collect more steps
    states = loader.collect_trajectories(num_steps=n_steps, seed=42)
    states_cpu = jax.device_get(states)

    # 2. Micro-Discretization
    # Increase resolution to capture fine dynamics
    n_micro = 100
    print(f"2. Discretizing into {n_micro} Micro-States...")
    micro_matrix, centroids, labels = estimate_transition_matrix(
        jnp.array(states_cpu), n_clusters=n_micro, kmeans_steps=50
    )

    sm = StochasticMatrix(micro_matrix, backend='jax')

    # 3. Macro-Scale Sweep (Finding True Emergence)
    print("3. Sweeping Macro-Scales (Finding Optimal Coarse-Graining)...")
    best_ei = -1.0
    best_n_macro = 0
    best_partition = None

    # Sweep from 2 to 8 macro states
    for n_macro in range(2, 9):
        print(f"   Testing n_macro={n_macro}...", end=" ")
        # Optimize
        partition, ei = sm.optimize_coarse_graining(n_macro=n_macro, steps=1000, learning_rate=0.05)
        print(f"EI: {ei:.4f} bits")

        if ei > best_ei:
            best_ei = ei
            best_n_macro = n_macro
            best_partition = partition

    print(f"-> Optimal Emergence found at N_Macro={best_n_macro} with EI={best_ei:.4f} bits")

    # 4. Map Trajectory
    assignment = best_partition.hard_assignment()
    micro_to_macro = jnp.argmax(assignment, axis=1)
    macro_trajectory = micro_to_macro[labels]

    # 5. Visualization
    print("5. Generating Deep Visualization...")
    env = envs.create(env_name=env_name)

    # Re-run for rendering
    key = jax.random.PRNGKey(42)
    reset_key, run_key = jax.random.split(key)
    state = env.reset(rng=reset_key)
    step_keys = jax.random.split(run_key, n_steps)
    jit_step = jax.jit(env.step)

    frames = []
    # Generate distinct colors for n_macro
    import matplotlib.cm as cm
    cmap = cm.get_cmap('tab10', best_n_macro)
    colors = [np.array(cmap(i)[:3]) for i in range(best_n_macro)]

    # Render every kth frame to keep GIF size reasonable?
    # Or render all. 2000 frames @ 60fps = 33s. That's fine.
    # Maybe render every 2nd frame.
    render_skip = 2

    for i in range(n_steps):
        if i % render_skip == 0:
            try:
                qp_or_state = state.pipeline_state
            except AttributeError:
                qp_or_state = state.qp

            frame_bytes = image.render(env.sys, [qp_or_state], width=320, height=240, fmt='png')
            frame_img = imageio.v3.imread(frame_bytes)

            # Overlay
            if i < len(macro_trajectory):
                macro_idx = int(macro_trajectory[i])
                color = colors[macro_idx]

                # Bar overlay
                H, W, C = frame_img.shape
                bar_height = 20
                col_u8 = (color * 255).astype(np.uint8)
                frame_img[-bar_height:, :, 0] = col_u8[0]
                frame_img[-bar_height:, :, 1] = col_u8[1]
                frame_img[-bar_height:, :, 2] = col_u8[2]

                # Add text? (Requires PIL/OpenCV, sticking to pixel manipulation for simplicity)

            frames.append(frame_img)

        # Step
        action = jax.random.uniform(step_keys[i], shape=(env.action_size,), minval=-1.0, maxval=1.0)
        state = jit_step(state, action)

    print(f"6. Saving High-Res GIF to {output_filename}...")
    imageio.mimsave(output_filename, frames, fps=30)
    print("Done.")

if __name__ == "__main__":
    # Ensure headless rendering
    import os
    os.environ['MUJOCO_GL'] = 'egl'
    visualize_emergence(n_steps=1000) # Run 1000 for this demo
