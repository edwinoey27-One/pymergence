import pytest
# Run these tests from the root directory of the project with `python -m pytest tests`

from pymergence.core.StochasticMatrix import *
from pymergence.core.CoarseGraining import *
import numpy as np


########## Test matrix construction ##########

# validate that the identity matrix is a valid stochastic matrix
def test_identity_matrix():
    identity_matrix = np.eye(3)
    stoch_matrix = StochasticMatrix(identity_matrix, validate=True)
    assert np.all(stoch_matrix.matrix == identity_matrix), "Matrix should remain unchanged for identity matrix"

# validate that a matrix with all zeros is not a valid stochastic matrix
def test_zero_matrix():
    zero_matrix = np.zeros((3, 3))
    with pytest.raises(ValueError):
        StochasticMatrix(zero_matrix, validate=True)

# validate that a matrix with negative entries is not a valid stochastic matrix
def test_negative_entries():
    negative_matrix = np.array([[0.75, -0.5, 0.75], [0.5, 0.5, 0], [0, 0, 1]])
    with pytest.raises(ValueError):
        StochasticMatrix(negative_matrix, validate=True)

# validate that a matrix with rows not summing to 1 is not a valid stochastic matrix
def test_rows_not_summing_to_one():
    non_stochastic_matrix = np.array([[0.6, 0.5, 0], [0.5, 0.5, 0], [0, 0, 1]])
    with pytest.raises(ValueError):
        StochasticMatrix(non_stochastic_matrix, validate=True)

# validate that a non-square matrix raises an error
def test_non_square_matrix():
    non_square_matrix = np.array([[0.5, 0.5], [0.5, 0.5], [0, 1]])
    with pytest.raises(ValueError):
        StochasticMatrix(non_square_matrix, validate=True)

# validate that a valid stochastic matrix passes validation
def test_valid_stochastic_matrix():
    valid_matrix = np.array([[0.5, 0.5, 0], [0.2, 0.8, 0], [0, 0.1, 0.9]])
    stoch_matrix = StochasticMatrix(valid_matrix, validate=True)
    assert np.all(stoch_matrix.matrix == valid_matrix), "Matrix should remain unchanged for valid stochastic matrix"

# validate that a matrix with small numerical errors is still considered valid
def test_numerical_errors():
    nearly_stochastic_matrix = np.array([[0.3333, 0.3333, 0.3334], [0.5, 0.5, 0], [0, 1, 0]])
    stoch_matrix = StochasticMatrix(nearly_stochastic_matrix, validate=True, tolerance=1e-3)
    assert np.allclose(stoch_matrix.matrix, nearly_stochastic_matrix), "Matrix should remain unchanged for nearly stochastic matrix"

######## Test matrix operations ##########

suff_nec_test_cases = [
    (np.array([[1/3, 1/3, 1/3],
                    [1, 0, 0],
                    [0, 0, 1]]), 7/9, 7/9),
    (np.array([[0, 0, 0, 1],
                    [0, 0, 1, 0],
                    [0, 1, 0, 0],
                    [1, 0, 0, 0]]), 1, 1),
    (np.array([[1/2, 1/2],
                [1/2, 1/2]]), 1/2, 1/2),
    (np.array([[0, 1],
                [0, 1]]), 1, 0),
    (np.array([[1/4, 1/4, 1/4, 1/4],
                    [1/4, 1/4, 1/4, 1/4],
                    [1/4, 1/4, 1/4, 1/4],
                    [1/4, 1/4, 1/4, 1/4]]), 1/4, 3/4)]


@pytest.mark.parametrize("input_matrix, expected_suff, expected_nec", suff_nec_test_cases)
def test_suffnec(input_matrix, expected_suff, expected_nec):
    # Test determinism calculation
    test_matrix = StochasticMatrix(input_matrix, validate=True)
    n = test_matrix.n_states
    intervention_distribution = np.ones(n) / n  # Uniform distribution over states
    calculated_suff = test_matrix.average_sufficiency()
    calculated_nec = test_matrix.average_necessity()
    assert np.isclose(calculated_suff, expected_suff)
    assert np.isclose(calculated_nec, expected_nec)


determinism_test_cases = [ 
    (np.array([
    [1/3, 1/3, 1/3, 0],
    [1/3, 1/3, 1/3, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 1]]), 1 - np.log2(3)/4),

    (np.array([
    [0, 0, 0, 1],
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0]]), 1),

    (np.array([
    [2/14, 0, 0, 0, 3/14, 3/14, 3/14, 3/14],
    [0, 1/21, 1/21, 1/21, 3/14, 3/14, 3/14, 3/14],
    [0, 1/21, 1/21, 1/21, 3/14, 3/14, 3/14, 3/14],
    [0, 1/21, 1/21, 1/21, 3/14, 3/14, 3/14, 3/14],
    [3/14, 3/14, 3/14, 3/14, 2/14, 0, 0, 0],
    [3/14, 3/14, 3/14, 3/14, 0, 1/21, 1/21, 1/21],
    [3/14, 3/14, 3/14, 3/14, 0, 1/21, 1/21, 1/21],
    [3/14, 3/14, 3/14, 3/14, 0, 1/21, 1/21, 1/21]]), 6/8*(1 + 1/3*(3/21*np.log2(1/21) + 12/14 * np.log2(3/14))) + 2/8*(1 + 1/3*(12/14*np.log2(3/14) + 2/14 * np.log2(2/14))))
]

degeneracy_test_cases = [ 
    (np.array([
    [1/3, 1/3, 1/3, 0],
    [1/3, 1/3, 1/3, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 1]]), 1 - 1/4 * (np.log2(6) + np.log2(2))),

    (np.array([
    [0, 0, 0, 1],
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0]]), 0),

    (np.array([
    [2/14, 0, 0, 0, 3/14, 3/14, 3/14, 3/14],
    [0, 1/21, 1/21, 1/21, 3/14, 3/14, 3/14, 3/14],
    [0, 1/21, 1/21, 1/21, 3/14, 3/14, 3/14, 3/14],
    [0, 1/21, 1/21, 1/21, 3/14, 3/14, 3/14, 3/14],
    [3/14, 3/14, 3/14, 3/14, 2/14, 0, 0, 0],
    [3/14, 3/14, 3/14, 3/14, 0, 1/21, 1/21, 1/21],
    [3/14, 3/14, 3/14, 3/14, 0, 1/21, 1/21, 1/21],
    [3/14, 3/14, 3/14, 3/14, 0, 1/21, 1/21, 1/21]]), 0)
]


@pytest.mark.parametrize("input_matrix, expected_determinism", determinism_test_cases)
def test_determinism(input_matrix, expected_determinism):
    # Test determinism calculation
    test_matrix = StochasticMatrix(input_matrix, validate=True)
    n = test_matrix.n_states
    intervention_distribution = np.ones(n) / n
    calculated_determinism = test_matrix.determinism(intervention_distribution)
    assert np.isclose(calculated_determinism, expected_determinism)

@pytest.mark.parametrize("input_matrix, expected_degeneracy", degeneracy_test_cases)
def test_degeneracy(input_matrix, expected_degeneracy):
    # Test determinism calculation
    test_matrix = StochasticMatrix(input_matrix, validate=True)
    n = test_matrix.n_states
    intervention_distribution = np.ones(n) / n
    calculated_degeneracy = test_matrix.degeneracy(intervention_distribution)
    assert np.isclose(calculated_degeneracy, expected_degeneracy)