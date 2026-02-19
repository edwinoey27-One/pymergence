# PyMergence API Surface V1

## Core
- `pymergence.StochasticMatrix`: Main entry point. Supports `backend='numpy'|'jax'`.
- `pymergence.CoarseGraining`: Partition logic.
- `pymergence.save_causal_model`: Save partition + metadata.
- `pymergence.load_causal_model`: Load partition + metadata.

## Integration (Optional)
- `pymergence.integration.BraxDataLoader`: Load data from Brax.
- `pymergence.integration.maximize_quantum_emergence`: Optimize quantum circuits.
- `pymergence.integration.CausalDiscovery`: Learn DAGs.
- `pymergence.integration.TopologicalAnalyzer`: Compute persistent homology.
- `pymergence.integration.IITAnalyzer`: Compute Phi.

## Acceleration (Experimental)
- `pymergence.accel.jax_core`: Equinox modules.
- `pymergence.accel.equivariant`: Geometric partitions.
