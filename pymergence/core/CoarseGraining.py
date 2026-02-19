import numpy as np

class CoarseGraining:
    """
    A class to describe coarse-grainings, i.e. partitions of variables.
    """

    def __init__(self, coarse_graining):
        """
        Initialize a coarse-graining.

        Parameters
        -----------
        coarse_graining : tuple of tuples
            A tuple where each inner tuple represents a partition of variables.
            For example, ((0, 1), (2, 3)) means variables 0 and 1 are in one group,
            and variables 2 and 3 are in another group.
        """
        if not isinstance(coarse_graining, tuple) or not all(isinstance(partition, tuple) for partition in coarse_graining):
            raise ValueError("coarse_graining must be a tuple of tuples")
        
        self.partition = tuple(sorted(tuple(sorted(block)) for block in coarse_graining))
        self.n_blocks = len(self.partition)
        self.block_sizes = [len(block) for block in self.partition]
        self.n_variables = sum(self.block_sizes)
        
    def is_refinement_of(self, other):
        """
        Check if this coarse-graining is a refinement of another.

        A coarse-graining A is a refinement of B if every block in A is a subset
        of some block in B.

        Parameters
        -----------
        other : CoarseGraining
            The coarse-graining to check against.

        Returns
        --------
        bool
            True if this coarse-graining is a refinement of the other, False otherwise.
        """
        for block in self.partition:
            if not any(set(block).issubset(set(other_block)) for other_block in other.partition):
                return False
        return True

    def coarse_grain_distribution(self, distribution):
        """
        Coarse grain a distribution according to this coarse-graining.

        Parameters
        -----------
        distribution : np.ndarray
            A 1D array representing the distribution over the original variables.

        Returns
        --------
        np.ndarray
            A 1D array representing the coarse-grained distribution.
        """
        if len(distribution) != self.n_variables:
            raise ValueError("Distribution length must match the number of variables in the coarse-graining.")
        
        coarse_distribution = np.zeros(self.n_blocks)
        for i, block in enumerate(self.partition):
            coarse_distribution[i] = np.sum(distribution[list(block)])
        
        return coarse_distribution
    

    def __repr__(self):
        """
        String representation of the coarse-graining.
        """
        return f"CoarseGraining(coarse_graining={self.partition}, n_blocks={self.n_blocks}, block_sizes={self.block_sizes})"
    
    def __str__(self):
        """
        String representation of the coarse-graining. 
        """
        blocks_str = "|".join("".join(map(str, block)) for block in self.partition)
        return blocks_str
        

def generate_partition_tuples(n):
    """
    Generate all partitions of n elements as tuples.

    This recursive function generates all possible partitions of a set of n elements,
    where each partition is represented as a tuple of tuples. Returns a generator
    that yields each partition.

    Parameters
    -----------
    n : int
        The number of elements to partition (0 means the empty set).
    Yields:
    --------
    tuple of tuples
        Each partition is a tuple where each inner tuple represents a block of the partition.
    """ 
    if n == 0:
        yield ()
        return
    for partition in generate_partition_tuples(n-1):
        # Option 1: put the new element (n-1) in its own block
        yield partition + ((n-1,),)
        # Option 2: add the new element (n-1) to each existing block
        for i in range(len(partition)):
            yield partition[:i] + (partition[i] + (n-1,),) + partition[i+1:]


def generate_all_coarse_grainings(n):
    """
    Generate all possible coarse-grainings of n variables.

    Parameters
    -----------
    n : int
        The number of variables to coarse grain.

    Returns
    --------
    list of CoarseGraining
        A list of all possible coarse-grainings.
    """
    from itertools import combinations
    
    partition_tuples = generate_partition_tuples(n)
    return [CoarseGraining(partition) for partition in partition_tuples]