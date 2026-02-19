import jax
import jax.numpy as jnp
import pennylane as qml
import optax
from pymergence.accel import jax_core

# Try to detect GPU device for PennyLane
try:
    PREFERRED_DEVICE = 'default.qubit'
except:
    PREFERRED_DEVICE = 'default.qubit'

def quantum_channel_to_stochastic_matrix(circuit_fn, num_qubits, params, device_name=None):
    """
    Compute the classical stochastic matrix for a quantum circuit in the computational basis.
    Assumes unitary evolution.
    """
    def wrapper(p):
        circuit_fn(p, wires=range(num_qubits))

    # JAX compatible matrix calculation
    # qml.matrix returns a function that returns the matrix
    U = qml.matrix(wrapper, wire_order=range(num_qubits))(params)

    # Stochastic Matrix M = |U.T|^2
    M = jnp.abs(U.T)**2

    return M

def maximize_quantum_emergence(circuit_fn, num_qubits, init_params, steps=100, learning_rate=0.1, use_gpu=True):
    """
    Optimize quantum circuit parameters to maximize Effective Information.
    """
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(init_params)

    @jax.jit
    def loss_fn(p):
        M = quantum_channel_to_stochastic_matrix(circuit_fn, num_qubits, p)
        # Minimize negative EI
        sm = jax_core.StochasticMatrix(M)
        return -sm.effective_information()

    @jax.jit
    def step(p, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(p)
        updates, opt_state = optimizer.update(grads, opt_state, p)
        p = optax.apply_updates(p, updates)
        return p, opt_state, loss

    params = init_params
    loss_history = []

    for i in range(steps):
        params, opt_state, loss = step(params, opt_state)
        loss_history.append(float(loss))

    final_loss = loss_history[-1]
    final_ei = -final_loss

    return params, final_ei, loss_history

def get_simple_ansatz(num_qubits, depth=1):
    def circuit(params, wires):
        k = 0
        for d in range(depth):
            for w in wires:
                if k < len(params):
                    qml.RY(params[k], wires=w)
                    k += 1
            for w in range(len(wires)-1):
                qml.CNOT(wires=[wires[w], wires[w+1]])

    num_params = num_qubits * depth
    return circuit, (num_params,)
