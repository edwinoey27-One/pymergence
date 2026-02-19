import os

class BackendSelector:
    """
    Manages backend selection (numpy, jax).
    """
    def __init__(self):
        self._backend = os.environ.get("PYMERGENCE_BACKEND", "auto")
        self._available = {}

        self._check_jax()

    def _check_jax(self):
        try:
            import jax
            self._available['jax'] = True
        except ImportError:
            self._available['jax'] = False

    @property
    def backend(self):
        if self._backend == "auto":
            if self._available.get('jax'):
                return "jax"
            return "numpy"
        return self._backend

    @backend.setter
    def backend(self, value):
        if value not in ["auto", "numpy", "jax"]:
            raise ValueError(f"Unknown backend: {value}")
        self._backend = value

selector = BackendSelector()
