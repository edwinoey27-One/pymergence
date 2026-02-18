import jax
import jax.numpy as jnp
import pennylane as qml
import optax
from pymergence import jax_core

def quantum_channel_to_stochastic_matrix(circuit_fn, num_qubits, params):
    """
    Compute the classical stochastic matrix for a quantum circuit in the computational basis.
    Assumes unitary evolution.

    Parameters
    ----------
    circuit_fn : callable
        A function that takes (params, wires) and applies quantum operations.
    num_qubits : int
        Number of qubits.
    params : array
        Parameters for the circuit.

    Returns
    -------
    matrix : (2^N, 2^N) array
        Row-stochastic transition matrix.
    """
    # Wrap circuit_fn to match qml.matrix signature
    def wrapper(p):
        circuit_fn(p, wires=range(num_qubits))

    # Get Unitary U
    # U[i, j] = <i|U|j> (amplitude to go from state j to state i)
    # Using 'jax' interface for differentiability
    # qml.matrix returns a function that computes the matrix for given args.
    U = qml.matrix(wrapper, wire_order=range(num_qubits))(params)

    # Stochastic Matrix M where M[j, i] is prob of transition j -> i
    # M[j, i] = |<i|U|j>|^2 = |U[i, j]|^2
    # So M is the transpose of element-wise squared U.
    # U is complex, so abs(U)**2 gives real probabilities.
    M = jnp.abs(U.T)**2

    return M

def maximize_quantum_emergence(circuit_fn, num_qubits, init_params, steps=100, learning_rate=0.1):
    """
    Optimize quantum circuit parameters to maximize Effective Information of the induced classical channel.

    Parameters
    ----------
    circuit_fn : callable
        Circuit function taking (params, wires).
    num_qubits : int
        Number of qubits.
    init_params : array
        Initial parameters.
    steps : int
        Number of optimization steps.
    learning_rate : float
        Learning rate.

    Returns
    -------
    final_params : array
        Optimized parameters.
    final_ei : float
        Final Effective Information value.
    loss_history : list
        History of negative EI values (loss).
    """
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(init_params)

    # We define step function inside to capture optimizer and circuit_fn
    @jax.jit
    def step(p, opt_state):
        def loss_fn(p_val):
            M = quantum_channel_to_stochastic_matrix(circuit_fn, num_qubits, p_val)
            # Minimize negative EI (maximize EI)
            return -jax_core.effective_information(M)

        loss, grads = jax.value_and_grad(loss_fn)(p)
        updates, opt_state = optimizer.update(grads, opt_state)
        p = optax.apply_updates(p, updates)
        return p, opt_state, loss

    params = init_params
    loss_history = []

    # Training loop
    # Convert range to list or use simple loop
    for i in range(steps):
        params, opt_state, loss = step(params, opt_state)
        # Store loss (scalar)
        loss_history.append(float(loss))

    final_loss = loss_history[-1]
    final_ei = -final_loss

    return params, final_ei, loss_history

def get_simple_ansatz(num_qubits, depth=1):
    """
    Returns a simple ansatz circuit function and parameter shape.
    Circuit: Layers of RY rotations and CNOTs.
    """
    def circuit(params, wires):
        # Flatten params if needed or assume flat
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
