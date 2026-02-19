# SODC Integration Contract V1

**Target:** `SODC_EXPERIMENT_GENESIS_PULSE_V1`
**Library:** `pymergence` (Version 0.2.0)

## 1. Input Schema
The SODC runtime must provide state data in one of the following formats:
- **Brax Trajectory:** `(T, Obs_Dim)` numpy array or Polars DataFrame.
- **Micro-State Matrix:** `(N, N)` row-stochastic transition matrix.
- **Quantum Circuit:** PennyLane circuit function + parameters.

## 2. Output Schema
All `pymergence` analysis tools return a standard dictionary (JSON-serializable via `orjson`) + optional large tensor sidecars (`safetensors`).

### Standard Analysis Result
```json
{
  "meta": {
    "version": "0.2.0",
    "backend": "jax",
    "timestamp": 171892...,
    "parity_check": true
  },
  "metrics": {
    "effective_information": 1.25,
    "determinism": 0.9,
    "degeneracy": 0.1,
    "integrated_information_phi": 0.45
  },
  "structure": {
    "n_micro": 100,
    "n_macro": 3,
    "partition_file": "path/to/partition.safetensors"
  }
}
```

## 3. Function Signatures
- **`analyze_emergence(data, backend='auto')`** -> `ResultDict`
- **`optimize_causal_structure(data, n_macro)`** -> `ResultDict` (updates structure)

## 4. Feature Flags
Control behavior via `os.environ`:
- `PYMERGENCE_BACKEND`: `jax`, `numpy`, `auto`.
- `PYMERGENCE_PARITY_CHECK`: `1` (enforce checks), `0` (skip).

## 5. Error Handling
- `ImportError`: If optional backend (Ray, Brax) missing.
- `ParityError`: If numeric deviation > tolerance.
