from pymergence.core.StochasticMatrix import StochasticMatrix
from pymergence.core.CoarseGraining import CoarseGraining
from pymergence.core.io import save_causal_model, load_causal_model

# Optional integrations
try:
    from pymergence.integration.brax_bridge import BraxDataLoader
except ImportError:
    pass

try:
    from pymergence.integration.quantum import maximize_quantum_emergence
except ImportError:
    pass

__version__ = "0.2.0"
