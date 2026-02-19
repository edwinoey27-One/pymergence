import jax
import jax.numpy as jnp
import equinox as eqx

class GeometricPartition(eqx.Module):
    """
    Maps continuous geometric state (positions) to macro-state probabilities.
    Ensures E(n) (Euclidean) and S_n (Permutation) invariance.
    """
    mlp: eqx.nn.MLP
    n_macro: int

    def __init__(self, n_particles, n_macro, key):
        self.n_macro = n_macro

        # Simple invariant architecture:
        # Input features: Sorted radii from Center of Mass.

        self.mlp = eqx.nn.MLP(
            in_size=n_particles,
            out_size=n_macro,
            width_size=64,
            depth=2,
            key=key
        )

    def __call__(self, pos):
        """
        Args:
            pos: (N, D) array of particle positions.
        Returns:
            probs: (n_macro,) probability distribution over macro states.
        """
        # 1. Translation Invariance: Center at COM
        # Handle PBC? For simplicity, assume unwrapped coordinates or centered box.
        pos_centered = pos - jnp.mean(pos, axis=0, keepdims=True)

        # 2. Rotation Invariance: Distances from COM
        radii = jnp.linalg.norm(pos_centered, axis=1) # (N,)

        # 3. Permutation Invariance: Sort radii
        radii_sorted = jnp.sort(radii)

        # 4. MLP -> Logits
        logits = self.mlp(radii_sorted)

        # Softmax
        return jax.nn.softmax(logits)
