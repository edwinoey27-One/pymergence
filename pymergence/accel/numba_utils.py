import numpy as np
from numba import jit, prange

@jit(nopython=True, parallel=True)
def fast_shannon_entropy(probs):
    """
    Compute Shannon entropy of a 1D probability array in bits using Numba with parallel reduction.
    """
    s = 0.0
    # prange allows parallel execution
    for i in prange(probs.shape[0]):
        p = probs[i]
        if p > 1e-15:
            s -= p * np.log2(p)
    return s

@jit(nopython=True, parallel=True)
def fast_conditional_entropy(joint_probs, marginal_y):
    """
    H(X|Y) = sum_{x,y} p(x,y) log(p(y)/p(x,y))
    """
    s = 0.0
    rows = joint_probs.shape[0]
    cols = joint_probs.shape[1]

    # Flatten loops for better parallelism or parallelize outer
    for i in prange(rows):
        for j in range(cols):
            p_xy = joint_probs[i, j]
            p_y = marginal_y[j]
            if p_xy > 1e-15:
                s += p_xy * np.log2(p_y / p_xy)
    return s

@jit(nopython=True)
def fast_mutual_information(joint_probs):
    """
    Compute Mutual Information I(X;Y).
    Note: Parallelizing the marginal calculation might be overhead for small matrices,
    but we can try.
    """
    rows = joint_probs.shape[0]
    cols = joint_probs.shape[1]

    marg_x = np.zeros(rows)
    marg_y = np.zeros(cols)

    # Manual marginalization
    for i in range(rows):
        for j in range(cols):
            val = joint_probs[i, j]
            marg_x[i] += val
            marg_y[j] += val

    h_x = fast_shannon_entropy(marg_x)
    h_y = fast_shannon_entropy(marg_y)

    # Joint entropy
    h_xy = 0.0
    # Re-loop for joint H
    # Cannot reuse loop easily without storing intermediates if we want clean H(X)+H(Y)-H(XY)

    # Let's compute H(XY) in parallel
    for i in range(rows):
        for j in range(cols):
            p = joint_probs[i, j]
            if p > 1e-15:
                h_xy -= p * np.log2(p)

    return h_x + h_y - h_xy

@jit(nopython=True)
def fast_spectral_gap(adj_matrix):
    """
    Compute spectral gap.
    """
    rows = adj_matrix.shape[0]
    degrees = np.zeros(rows)
    for i in range(rows):
        d = 0.0
        for j in range(rows):
            d += adj_matrix[i, j]
        degrees[i] = d

    L = np.zeros((rows, rows))
    for i in range(rows):
        for j in range(rows):
            if i == j:
                L[i, j] = degrees[i] - adj_matrix[i, j]
            else:
                L[i, j] = -adj_matrix[i, j]

    evals = np.linalg.eigvalsh(L)

    if rows > 1:
        return evals[1]
    return 0.0
