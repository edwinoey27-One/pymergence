import time
import json
import numpy as np
import jax
import jax.numpy as jnp
from pymergence.core.StochasticMatrix import StochasticMatrix
from pymergence.accel.numba_utils import fast_shannon_entropy
from pymergence.distributed.manager import DistributedManager

def run_benchmarks(output_file="benchmark_results.json"):
    results = {
        "metadata": {
            "device": str(jax.devices()[0]),
            "timestamp": time.time()
        },
        "benchmarks": []
    }

    # Test cases: (N_states, label)
    cases = [
        (100, "Small"),
        (500, "Medium"),
        # (2000, "Large") # Skip large for speed in this environment
    ]

    for n, label in cases:
        print(f"Benchmarking {label} (N={n})...")
        # Generate random stochastic matrix
        key = jax.random.PRNGKey(0)
        mat = jax.random.uniform(key, (n, n))
        mat = mat / jnp.sum(mat, axis=1, keepdims=True)
        mat_np = np.array(mat)

        # 1. Baseline (NumPy) EI
        start = time.time()
        sm_np = StochasticMatrix(mat_np, backend='numpy')
        ei_np = sm_np.effective_information()
        time_np = time.time() - start

        results["benchmarks"].append({
            "task": "EffectiveInformation",
            "size": n,
            "backend": "numpy",
            "time": time_np,
            "result": float(ei_np)
        })

        # 2. Accel (JAX) EI
        # First run (compile)
        sm_jax = StochasticMatrix(mat_np, backend='jax')
        _ = sm_jax.effective_information()

        start = time.time()
        ei_jax = sm_jax.effective_information()
        # Block until ready
        _ = ei_jax + 1.0
        time_jax = time.time() - start

        results["benchmarks"].append({
            "task": "EffectiveInformation",
            "size": n,
            "backend": "jax",
            "time": time_jax,
            "result": float(ei_jax)
        })

        # Parity Check
        diff = abs(ei_np - ei_jax)
        print(f"  Parity Diff: {diff:.6f}")
        if diff > 1e-4:
            print("  WARNING: Parity check failed!")

        # 3. Numba Entropy
        # Test just the entropy part on a row
        row = mat_np[0]
        start = time.time()
        _ = fast_shannon_entropy(row) # compile
        time_numba_compile = time.time() - start

        start = time.time()
        ent_numba = fast_shannon_entropy(row)
        time_numba = time.time() - start

        results["benchmarks"].append({
            "task": "ShannonEntropy_Row",
            "size": n,
            "backend": "numba",
            "time": time_numba,
            "result": float(ent_numba)
        })

        # 4. Ray Optimization
        # Only for Medium
        if n >= 500:
            print("  Benchmarking Ray Distributed Optimization...")
            dm = DistributedManager()
            start = time.time()
            # Run 2 optimizations in parallel
            res = dm.run_parallel_optimizations(mat_np, [2, 3], backend='jax')
            time_ray = time.time() - start

            results["benchmarks"].append({
                "task": "Optimize_Parallel",
                "size": n,
                "backend": "ray",
                "time": time_ray,
                "note": "2 concurrent tasks"
            })

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    return results

if __name__ == "__main__":
    run_benchmarks()
