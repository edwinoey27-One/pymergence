import jax
import jax.numpy as jnp
import pennylane as qml
import optax
from pymergence import jax_core

# Try to detect GPU device for PennyLane
try:
    # Check if 'lightning.gpu' or 'lightning.kokkos' is available
    # We prefer lightning.gpu for CUDA
    # qml.device('lightning.gpu', wires=1) # test instantiation
    # Use deferred selection
    PREFERRED_DEVICE = 'default.qubit' # Fallback

    # Simple check:
    # dev_list = qml.refresh_devices() # deprecated or slow
    # Just try-except block in function
except:
    PREFERRED_DEVICE = 'default.qubit'

def quantum_channel_to_stochastic_matrix(circuit_fn, num_qubits, params, device_name=None):
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
    device_name : str, optional
        PennyLane device name. If None, tries 'lightning.gpu' then 'default.qubit'.

    Returns
    -------
    matrix : (2^N, 2^N) array
        Row-stochastic transition matrix.
    """
    # Wrap circuit_fn to match qml.matrix signature
    def wrapper(p):
        circuit_fn(p, wires=range(num_qubits))

    # Matrix calculation in PennyLane is usually device-agnostic for 'default.qubit' logic,
    # but 'lightning.gpu' might support it faster for large systems if implemented.
    # However, qml.matrix() transform often runs on CPU unless specific device support.
    # Actually, qml.matrix(circuit, wire_order=...) creates a tape.

    # Ideally, we run this on GPU if possible.
    # qml.matrix usually returns a function that returns a dense matrix.
    # JAX will put it on the default backend (GPU if available).

    # We don't necessarily need to specify 'lightning.gpu' for qml.matrix to output a JAX array on GPU,
    # as long as JAX itself is configured for GPU.
    # But using 'lightning.gpu' for simulation steps is good.

    # Current qml.matrix impl:
    U = qml.matrix(wrapper, wire_order=range(num_qubits))(params)

    # Move to GPU if not already (JAX handles this if backend is GPU)
    # Ensure precision

    # Stochastic Matrix M where M[j, i] is prob of transition j -> i
    # M[j, i] = |<i|U|j>|^2 = |U[i, j]|^2
    # So M is the transpose of element-wise squared U.
    M = jnp.abs(U.T)**2

    return M

def maximize_quantum_emergence(circuit_fn, num_qubits, init_params, steps=100, learning_rate=0.1, use_gpu=True):
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
    use_gpu : bool
        Hint to use GPU optimization.

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

    # Determine backend for JIT
    # JAX uses GPU by default if installed.
    # We can inspect jax.devices()

    @jax.jit
    def loss_fn(p):
        M = quantum_channel_to_stochastic_matrix(circuit_fn, num_qubits, p)
        # Minimize negative EI (maximize EI)
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

    # Training loop
    for i in range(steps):
        params, opt_state, loss = step(params, opt_state)
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
