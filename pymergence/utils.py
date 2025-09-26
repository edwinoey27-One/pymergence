import numpy as np
import networkx as nx
from itertools import chain, combinations


def kl_divergence(p, q, epsilon=1e-15):
    """
    Compute Kullback-Leibler divergence D_KL(p || q).
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    # Avoid log(0) by ensuring everything is at least epsilon
    p = np.maximum(p, epsilon)
    q = np.maximum(q, epsilon)
    return np.sum(p * np.log(p / q))


def kl_divergence_base2(p, q, epsilon=1e-15):
    """
    Compute Kullback-Leibler divergence D_KL(p || q) with base 2.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    # Avoid log(0) by ensuring everything is at least epsilon
    p = np.maximum(p, epsilon)
    q = np.maximum(q, epsilon)
    return np.sum(p * np.log2(p / q))

def entropy(p, epsilon=1e-15):
    """
    Compute the Shannon entropy H(p).
    """
    p = np.asarray(p, dtype=float)
    # Avoid log(0) by ensuring everything is at least epsilon
    p = np.maximum(p, epsilon)
    return -np.sum(p * np.log2(p))

def powerset(iterable):
    """
    Return the power set of an iterable.
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

def str_to_tuplet_partition(data_string):
  """
  Converts a pipe-separated string of numbers into a nested tuple of digits.

  Args:
    data_string: A string like '0|1|234|5|67'.

  Returns:
    A nested tuple like ((0,), (1,), (2, 3, 4), (5,), (6, 7)).
  """
  return tuple(tuple(int(char) for char in part) for part in data_string.split('|'))