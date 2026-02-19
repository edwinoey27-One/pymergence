import pytest
import numpy as np
from pymergence.core.utils import (
    kl_divergence,
    kl_divergence_base2,
    entropy,
    powerset,
    str_to_tuplet_partition
)

def test_kl_divergence_identical():
    p = np.array([0.2, 0.3, 0.5])
    assert np.isclose(kl_divergence(p, p), 0.0)
    assert np.isclose(kl_divergence_base2(p, p), 0.0)

def test_kl_divergence_known_values():
    p = np.array([0.5, 0.5])
    q = np.array([0.1, 0.9])

    # D_KL(p || q) = 0.5 * log(0.5/0.1) + 0.5 * log(0.5/0.9)
    expected_nats = 0.5 * np.log(5) + 0.5 * np.log(5/9)
    assert np.isclose(kl_divergence(p, q), expected_nats)

    expected_bits = 0.5 * np.log2(5) + 0.5 * np.log2(5/9)
    assert np.isclose(kl_divergence_base2(p, q), expected_bits)

def test_kl_divergence_zeros():
    p = np.array([1.0, 0.0])
    q = np.array([0.5, 0.5])
    # p will become [1.0, eps], q remains [0.5, 0.5]
    # D_KL = 1.0 * log(1.0/0.5) + eps * log(eps/0.5)
    expected = np.log(2.0)
    assert np.isclose(kl_divergence(p, q), expected, atol=1e-10)

def test_entropy_uniform():
    p = np.array([0.5, 0.5])
    assert np.isclose(entropy(p), 1.0)

def test_entropy_deterministic():
    p = np.array([1.0, 0.0])
    assert np.isclose(entropy(p), 0.0, atol=1e-10)

def test_entropy_known_value():
    p = np.array([0.25, 0.75])
    expected = - (0.25 * np.log2(0.25) + 0.75 * np.log2(0.75))
    assert np.isclose(entropy(p), expected)

def test_powerset():
    input_list = [1, 2]
    result = list(powerset(input_list))
    expected = [(), (1,), (2,), (1, 2)]
    assert len(result) == 4
    for item in expected:
        assert item in result

def test_str_to_tuplet_partition():
    data = '0|1|23'
    expected = ((0,), (1,), (2, 3))
    assert str_to_tuplet_partition(data) == expected

    data2 = '0|1|234|5|67'
    expected2 = ((0,), (1,), (2, 3, 4), (5,), (6, 7))
    assert str_to_tuplet_partition(data2) == expected2
