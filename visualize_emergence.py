import jax
import jax.numpy as jnp
import numpy as np
import imageio
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from brax import envs
from pymergence.integration.brax_bridge import BraxDataLoader
from pymergence.integration.brax_integration import estimate_transition_matrix
from pymergence.core.StochasticMatrix import StochasticMatrix
from pymergence.core.backend import selector

# Set backend to JAX for speed
selector.backend = 'jax'

def visualize_phase_space_art(env_name='inverted_pendulum', n_steps=2000, output_filename='emergence_art.gif'):
    print(f"=== Deep Emergence Art: {env_name} ===")

    print("1. Collecting Trajectory...")
    loader = BraxDataLoader(env_name)
    states = loader.collect_trajectories(num_steps=n_steps, seed=42)
    states_cpu = np.array(jax.device_get(states))

    # Phase space projection
    X = states_cpu[:, 0]
    Y = states_cpu[:, 1]

    print("2. Computing Causal Macro-States...")
    n_micro = 100
    micro_matrix, _, labels = estimate_transition_matrix(
        jnp.array(states_cpu), n_clusters=n_micro, kmeans_steps=50
    )

    sm = StochasticMatrix(micro_matrix, backend='jax')

    print("   Optimizing Partition...")
    best_ei = -1
    best_partition = None
    best_n = 3
    for n in range(2, 6):
        p, ei = sm.optimize_coarse_graining(n_macro=n, steps=500, learning_rate=0.1)
        if ei > best_ei:
            best_ei = ei
            best_partition = p
            best_n = n

    print(f"   Optimal N_Macro: {best_n} (EI={best_ei:.3f})")

    assignment = best_partition.hard_assignment()
    micro_to_macro = np.array(jnp.argmax(assignment, axis=1))
    macro_labels = micro_to_macro[labels]

    print("3. Rendering Phase Space Fractal...")
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)

    neon_colors = ['#00FFFF', '#FF00FF', '#FFFF00', '#00FF00', '#FFA500', '#FFFFFF']

    points = np.array([X, Y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    frames = []
    window = 200

    from matplotlib.colors import to_rgba

    for t in range(0, n_steps, 5):
        ax.clear()
        ax.set_xlim(X.min()*1.1, X.max()*1.1)
        ax.set_ylim(Y.min()*1.1, Y.max()*1.1)
        ax.axis('off')

        start = max(0, t - window)
        end = t

        if end > start:
            current_segs = segments[start:end]
            seg_colors = [neon_colors[macro_labels[i] % len(neon_colors)] for i in range(start, end)]
            alphas = np.linspace(0.1, 1.0, len(current_segs))

            rgba_colors = [to_rgba(c, alpha=a) for c, a in zip(seg_colors, alphas)]
            lc = LineCollection(current_segs, colors=rgba_colors, linewidths=2.0)

            rgba_glow = [to_rgba(c, alpha=a*0.3) for c, a in zip(seg_colors, alphas)]
            lc_glow = LineCollection(current_segs, colors=rgba_glow, linewidths=6.0)

            ax.add_collection(lc_glow)
            ax.add_collection(lc)

            head_x, head_y = X[end], Y[end]
            head_color = neon_colors[macro_labels[end] % len(neon_colors)]
            ax.scatter(head_x, head_y, color='white', s=50, zorder=10)
            ax.scatter(head_x, head_y, color=head_color, s=150, alpha=0.5, zorder=9)

        fig.canvas.draw()
        # Fix for Matplotlib 3.8+ buffer access
        image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        # Convert RGBA to RGB for GIF compatibility/size
        image = image[:, :, :3]
        frames.append(image)

    print(f"4. Saving GIF to {output_filename}...")
    imageio.mimsave(output_filename, frames, fps=30)
    plt.close()
    print("Done.")

if __name__ == "__main__":
    visualize_phase_space_art(n_steps=1000)
