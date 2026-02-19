import pytest
import jax.numpy as jnp
import pennylane as qml
from pymergence.integration.quantum import quantum_channel_to_stochastic_matrix, maximize_quantum_emergence

def test_quantum_channel():
    # 1 qubit circuit: Identity
    def circuit(params, wires):
        # params unused
        pass

    num_qubits = 1
    matrix = quantum_channel_to_stochastic_matrix(circuit, num_qubits, jnp.array([0.0]))

    # Expect Identity 2x2
    expected = jnp.eye(2)
    assert jnp.allclose(matrix, expected, atol=1e-5)

    # 1 qubit circuit: Hadamard
    def circuit_h(params, wires):
        qml.Hadamard(wires=0)

    matrix_h = quantum_channel_to_stochastic_matrix(circuit_h, num_qubits, jnp.array([0.0]))

    # H|0> = |+>. |<0|H|0>|^2 = 0.5. |<1|H|0>|^2 = 0.5.
    # Matrix:
    # 0 -> 0.5, 0.5
    # 1 -> 0.5, 0.5
    expected_h = jnp.array([[0.5, 0.5], [0.5, 0.5]])
    assert jnp.allclose(matrix_h, expected_h, atol=1e-5)

def test_optimization():
    # 1 qubit optimization: Start at H, end at Identity (or X)
    def circuit(params, wires):
        qml.RY(params[0], wires=0)

    init_params = jnp.array([jnp.pi/2 + 0.1])

    # H-like -> low EI (approx 0)
    # Identity -> high EI (1 bit)

    final_params, final_ei, _ = maximize_quantum_emergence(circuit, 1, init_params, steps=100)

    assert final_ei > 0.99
    # Params should be close to k*pi
    dist_to_pi = jnp.abs(jnp.remainder(final_params[0], jnp.pi))
    # It might be 0 or pi
    # If 0, dist is 0. If pi, remainder is 0.
    # remainder(x, pi) gives range [0, pi).
    # If close to pi, remainder is close to 0 (if modulo worked nicely) or close to pi.
    # sin(params) should be close to 0.
    assert jnp.abs(jnp.sin(final_params[0])) < 0.1
