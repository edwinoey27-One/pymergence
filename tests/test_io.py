import numpy as np
import pytest
import os
import equinox as eqx
import jax
from pymergence.io import save_causal_model, load_causal_model
from pymergence.jax_core import Partition

def test_io_roundtrip(tmp_path):
    key = jax.random.PRNGKey(0)
    partition = Partition(n_micro=4, n_macro=2, key=key)

    metadata = {
        "ei_score": 1.23,
        "algorithm": "optax_adam",
        "notes": "Test run"
    }

    prefix = str(tmp_path / "test_model")
    save_causal_model(partition, metadata, prefix)

    # Check files exist
    assert os.path.exists(prefix + ".safetensors")
    assert os.path.exists(prefix + ".json")

    loaded_part, loaded_meta = load_causal_model(prefix, 4, 2, key)

    # Check partition logits
    assert np.allclose(partition.logits, loaded_part.logits)

    # Check metadata
    assert loaded_meta["ei_score"] == 1.23
    assert loaded_meta["notes"] == "Test run"
