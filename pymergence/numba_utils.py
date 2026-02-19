import numpy as np
from numba import jit

@jit(nopython=True)
def fast_shannon_entropy(probs):
    """
    Compute Shannon entropy of a 1D probability array in bits using Numba.
    Input must be normalized.
    """
    s = 0.0
    for i in range(probs.shape[0]):
        p = probs[i]
        if p > 1e-15:
            s -= p * np.log2(p)
    return s

@jit(nopython=True)
def fast_conditional_entropy(joint_probs, marginal_y):
    """
    H(X|Y) = sum_{x,y} p(x,y) log(p(y)/p(x,y))
    """
    s = 0.0
    rows = joint_probs.shape[0]
    cols = joint_probs.shape[1]

    for i in range(rows):
        for j in range(cols):
            p_xy = joint_probs[i, j]
            p_y = marginal_y[j]
            if p_xy > 1e-15:
                s += p_xy * np.log2(p_y / p_xy)
    return s

@jit(nopython=True)
def fast_mutual_information(joint_probs):
    """
    Compute Mutual Information I(X;Y) from joint distribution matrix.
    I(X;Y) = H(X) + H(Y) - H(X,Y)
    """
    rows = joint_probs.shape[0]
    cols = joint_probs.shape[1]

    marg_x = np.zeros(rows)
    marg_y = np.zeros(cols)

    for i in range(rows):
        for j in range(cols):
            val = joint_probs[i, j]
            marg_x[i] += val
            marg_y[j] += val

    h_x = fast_shannon_entropy(marg_x)
    h_y = fast_shannon_entropy(marg_y)

    # Joint entropy
    h_xy = 0.0
    for i in range(rows):
        for j in range(cols):
            p = joint_probs[i, j]
            if p > 1e-15:
                h_xy -= p * np.log2(p)

    return h_x + h_y - h_xy

@jit(nopython=True)
def fast_spectral_gap(adj_matrix):
    """
    Compute spectral gap (algebraic connectivity) using simple power iteration
    or just return Eigenvalues if supported (Numba supports numpy.linalg.eigvalsh).
    """
    # Laplacian
    rows = adj_matrix.shape[0]
    degrees = np.zeros(rows)
    for i in range(rows):
        d = 0.0
        for j in range(rows):
            d += adj_matrix[i, j]
        degrees[i] = d

    # L = D - A
    L = np.zeros((rows, rows))
    for i in range(rows):
        for j in range(rows):
            if i == j:
                L[i, j] = degrees[i] - adj_matrix[i, j]
            else:
                L[i, j] = -adj_matrix[i, j]

    # Eigenvalues
    # Note: np.linalg.eigvalsh is supported in newer Numba versions.
    # If not, we might fail.
    # Assuming standard Numba installation.
    evals = np.linalg.eigvalsh(L)

    # Sort and pick second smallest
    # evals are usually sorted by eigvalsh
    if rows > 1:
        return evals[1]
    return 0.0
