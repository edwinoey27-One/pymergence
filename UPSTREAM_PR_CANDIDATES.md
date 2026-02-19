# Upstream PR Candidates

These components are stable, generic, and add significant value to the upstream `pymergence` repository without introducing heavy dependencies or breaking changes.

## 1. JAX Backend & Optimization
- **Files:** `pymergence/accel/jax_core.py`, `pymergence/core/StochasticMatrix.py` (partial)
- **Description:** Adds an optional JAX backend for `StochasticMatrix` and gradient-based optimization for coarse-graining.
- **Dependency:** `jax`, `equinox`, `optax` (Optional).
- **Justification:** Solves the scalability limit of brute-force search.

## 2. Numba Acceleration
- **Files:** `pymergence/accel/numba_utils.py`
- **Description:** Fast CPU kernels for entropy and spectral gap.
- **Dependency:** `numba` (Optional).
- **Justification:** Accelerates analysis on standard hardware.

## 3. Safe IO
- **Files:** `pymergence/core/io.py`
- **Description:** `safetensors` support for saving models.
- **Dependency:** `safetensors`, `orjson`.
- **Justification:** Security and performance for large artifacts.

## 4. Topology Bridge
- **Files:** `pymergence/integration/topology.py`
- **Description:** Persistent homology wrapper.
- **Dependency:** `ripser`.
- **Justification:** Adds topological perspective to causal analysis.
