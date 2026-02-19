import numpy as np
from pymergence.core.StochasticMatrix import StochasticMatrix

class ParityGuard:
    """
    Ensures scientific parity between Accel (JAX) and Baseline (NumPy) backends.
    """
    def __init__(self, tolerance=1e-5):
        self.tolerance = tolerance

    def check_parity(self, matrix_data):
        """
        Compare EI calculation.
        """
        # Baseline
        sm_np = StochasticMatrix(matrix_data, backend='numpy')
        ei_np = sm_np.effective_information()

        # Accel
        sm_jax = StochasticMatrix(matrix_data, backend='jax')
        ei_jax = sm_jax.effective_information()

        diff = abs(ei_np - ei_jax)
        passed = diff < self.tolerance

        report = {
            "metric": "EffectiveInformation",
            "baseline": float(ei_np),
            "accel": float(ei_jax),
            "diff": float(diff),
            "passed": passed
        }

        if not passed:
            print(f"PARITY CHECK FAILED: {report}")
            # In production, might raise Error or trigger rollback

        return report

    def check_optimization_parity(self, matrix_data, n_macro=2):
        """
        Check if JAX optimization yields EI >= Baseline Brute Force?
        Brute force is only feasible for small systems.
        We assume JAX optimization is 'correct' if it improves EI.
        """
        pass
