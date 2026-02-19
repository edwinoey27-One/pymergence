import sys
from unittest.mock import MagicMock

# Mock all dependencies to allow importing pymergence modules in a restricted environment
for module in [
    'numpy', 'matplotlib', 'matplotlib.pyplot', 'networkx', 'pygraphviz',
    'jax', 'jax.numpy', 'equinox', 'optax', 'einops', 'safetensors',
    'safetensors.numpy', 'orjson', 'pymergence.core.utils'
]:
    sys.modules[module] = MagicMock()

import pytest
# Now we can safely import the functions
from pymergence.core.CoarseGraining import (
    generate_partition_tuples,
    generate_all_coarse_grainings,
    MAX_PARTITION_SIZE
)

def test_generate_partition_tuples_limit():
    """Test that generate_partition_tuples raises ValueError when n exceeds the limit."""
    with pytest.raises(ValueError) as excinfo:
        # It's a generator, so we need to start iterating to trigger the check
        next(generate_partition_tuples(MAX_PARTITION_SIZE + 1))
    assert f"exceeds MAX_PARTITION_SIZE ({MAX_PARTITION_SIZE})" in str(excinfo.value)
    assert "Denial of Service" in str(excinfo.value)

def test_generate_all_coarse_grainings_limit():
    """Test that generate_all_coarse_grainings raises ValueError when n exceeds the limit."""
    with pytest.raises(ValueError) as excinfo:
        generate_all_coarse_grainings(MAX_PARTITION_SIZE + 1)
    assert f"exceeds MAX_PARTITION_SIZE ({MAX_PARTITION_SIZE})" in str(excinfo.value)
    assert "Denial of Service" in str(excinfo.value)

def test_generate_partition_tuples_valid():
    """Test that generate_partition_tuples works correctly for small n."""
    # Bell number for n=3 is 5
    partitions = list(generate_partition_tuples(3))
    assert len(partitions) == 5

def test_generate_all_coarse_grainings_valid():
    """Test that generate_all_coarse_grainings works correctly for small n."""
    # Bell number for n=2 is 2
    grainings = generate_all_coarse_grainings(2)
    assert len(grainings) == 2
