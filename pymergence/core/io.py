import orjson
import numpy as np
import jax.numpy as jnp
from safetensors.numpy import save_file, load_file
import equinox as eqx
from pymergence.accel.jax_core import Partition

def save_causal_model(partition: Partition, metadata: dict, filepath_prefix: str):
    """
    Save a causal model (partition weights) and associated metadata.

    Args:
        partition: The learned Partition module.
        metadata: Dict of metadata (hyperparams, EI score, etc.).
        filepath_prefix: Path prefix (e.g. "models/run1").
                         Will create "models/run1.safetensors" and "models/run1.json".
    """
    # 1. Save Tensors via Safetensors
    tensor_path = f"{filepath_prefix}.safetensors"
    # Extract logits
    tensors = {"logits": np.array(partition.logits)}
    save_file(tensors, tensor_path)

    # 2. Save Metadata via Orjson
    meta_path = f"{filepath_prefix}.json"
    # Ensure metadata is serializable
    # Convert numpy types to python types
    def default(obj):
        if isinstance(obj, (np.integer, jnp.integer)):
            return int(obj)
        if isinstance(obj, (np.floating, jnp.floating)):
            return float(obj)
        if isinstance(obj, (np.ndarray, jnp.ndarray)):
            return obj.tolist()
        raise TypeError

    with open(meta_path, "wb") as f:
        f.write(orjson.dumps(metadata, default=default, option=orjson.OPT_INDENT_2))

def load_causal_model(filepath_prefix: str, n_micro: int, n_macro: int, key):
    """
    Load a causal model.
    """
    tensor_path = f"{filepath_prefix}.safetensors"
    meta_path = f"{filepath_prefix}.json"

    # Load Metadata
    with open(meta_path, "rb") as f:
        metadata = orjson.loads(f.read())

    # Load Tensors
    tensors = load_file(tensor_path)

    # Reconstruct Partition
    partition = Partition(n_micro, n_macro, key)
    partition = eqx.tree_at(lambda p: p.logits, partition, jnp.array(tensors["logits"]))

    return partition, metadata
