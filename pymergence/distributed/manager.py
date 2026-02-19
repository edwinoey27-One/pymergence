import ray
import numpy as np
try:
    from pymergence.core.StochasticMatrix import StochasticMatrix
except ImportError:
    pass

@ray.remote(num_cpus=1)
class CoarseGrainingActor:
    """
    Ray actor for distributed coarse-graining optimization.
    """
    def __init__(self, matrix_data, backend='jax'):
        self.backend = backend
        # JAX on actors: By default uses CPU unless num_gpus specified.
        # If we have GPUs, we should manage them.
        self.sm = StochasticMatrix(matrix_data, backend=backend)

    def optimize(self, n_macro, steps=1000, lr=0.1, seed=0):
        import jax
        key = jax.random.PRNGKey(seed)
        partition, ei = self.sm.optimize_coarse_graining(n_macro, steps, lr, key)

        # Return serializable results
        logits = np.array(partition.logits)
        return logits, float(ei)

class DistributedManager:
    def __init__(self, num_cpus=None, num_gpus=None):
        if not ray.is_initialized():
            ray.init(num_cpus=num_cpus, num_gpus=num_gpus, ignore_reinit_error=True)

    def run_parallel_optimizations(self, matrix_data, n_macros_list, backend='jax'):
        """
        Run optimizations for different n_macro in parallel.
        matrix_data is put into object store once.
        """
        # Efficiently share large matrix
        matrix_ref = ray.put(matrix_data)

        futures = []
        for n_macro in n_macros_list:
            # Pass reference
            actor = CoarseGrainingActor.remote(matrix_ref, backend)
            futures.append(actor.optimize.remote(n_macro))

        results = ray.get(futures)
        return results
