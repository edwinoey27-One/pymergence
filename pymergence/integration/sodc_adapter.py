from pymergence.core.StochasticMatrix import StochasticMatrix
from pymergence.core.backend import selector
from pymergence.lab.parity import ParityGuard
import os
import orjson

class SODCAdapter:
    """
    Adapter for SODC Genesis Pulse.
    """
    def __init__(self, backend='auto'):
        selector.backend = backend
        self.guard = ParityGuard()

    def analyze_matrix(self, matrix_data):
        """
        Full analysis pipeline.
        """
        # Parity Check
        if os.environ.get("PYMERGENCE_PARITY_CHECK") == "1":
            report = self.guard.check_parity(matrix_data)
            if not report['passed']:
                raise RuntimeError(f"Parity check failed: {report}")

        sm = StochasticMatrix(matrix_data, backend=selector.backend)

        result = {
            "metrics": {
                "effective_information": sm.effective_information(),
                "determinism": sm.determinism(),
                "degeneracy": sm.degeneracy()
            },
            "meta": {
                "backend": sm.backend
            }
        }
        return result

    def optimize(self, matrix_data, n_macro=3):
        sm = StochasticMatrix(matrix_data, backend='jax') # Enforce JAX for opt
        partition, ei = sm.optimize_coarse_graining(n_macro=n_macro)

        # We assume partition is an Equinox module.
        # SODC expects serializable outputs.
        # We can't return the module directly.
        # Return logits/assignment.

        return {
            "metrics": {"effective_information": ei},
            "partition_logits": partition.logits # JAX Array
        }
