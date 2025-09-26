from pymergence.utils import *
from pymergence.StochasticMatrix import *
from pymergence.CoarseGraining import *
import networkx as nx

def calc_CP_along_CGs(stochastic_matrix, coarse_grainings=None, mode='effectiveness'):
    """
    Calculate the CP values for a given network, across a range of coarse-grainings.
    
    Parameters
    ----------
    stochastic_matrix : StochasticMatrix
        The stochastic matrix representing the network.
    coarse_grainings : list of CoarseGraining objects, optional
        List of coarse-graining partitions to consider. If None, all partitions will be generated.
    mode : str, optional
        The mode of calculation for CP values. Default is 'effectiveness', alternatively 'suff_plus_nec'.

    Returns
    -------
    all_partitions : list
        List of CoarseGraining objects representing the partitions.
    cp_values : dict
        Dictionary mapping each partition (as a string) to its CP value.
    """

    coarse_grained_matrices = {}
    considered_coarse_grainings = []

    if coarse_grainings is not None:
        for cg in coarse_grainings:
            if not isinstance(cg, CoarseGraining):
                raise TypeError("Partitions must be instances of CoarseGraining.")
            if cg.partition not in coarse_grained_matrices:
                cg_matrix = stochastic_matrix.coarse_grain(cg)
                coarse_grained_matrices[cg.partition] = cg_matrix    # Use partition tuple as key
                considered_coarse_grainings.append(cg)
    else:
        for cg in generate_all_coarse_grainings(stochastic_matrix.n_states):
            if cg.partition not in coarse_grained_matrices:
                cg_matrix = stochastic_matrix.coarse_grain(cg)
                coarse_grained_matrices[cg.partition] = cg_matrix    # Use partition tuple as key
                considered_coarse_grainings.append(cg)

    # Compute the ded_combined_normalized for each partition
    cp_values = {}
    for cg in considered_coarse_grainings:
        cg_matrix = coarse_grained_matrices[cg.partition]
        if mode == 'suff_plus_nec':
            cp_values[str(cg)] = cg_matrix.suff_plus_nec()-1
        elif mode == 'effectiveness':
            cp_values[str(cg)] = cg_matrix.effectiveness()
        else:
            raise ValueError("Invalid mode. Use 'suff_plus_nec' or 'effectiveness'.")

    return cp_values, considered_coarse_grainings

def min_delta_ancestors(coarse_graining, lattice, cp_dict, allAncestors=False):
    """
    NOTE: this function will probably be removed in the future. Better to use `delta_CP_ancestors` directly.
    Calculate the minimum difference between the CP value of a coarse-graining and its direct ancestors in the lattice.

    Parameters
    ----------
    coarse_graining : CoarseGraining
        The coarse-graining for which to calculate the minimum difference.
    lattice : nx.DiGraph
        The lattice of coarse-grainings, ordered by refinement.
    cp_dict : dict
        Dictionary mapping each partition to its CP value.
    allAncestors : bool, optional
        If True, consider all ancestors in the lattice. If False, only consider direct ancestors.
        Default is False.
    Returns
    -------
    float
        The minimum difference between the CP value of the coarse-graining and its direct ancestors.
    """
    if not isinstance(coarse_graining, CoarseGraining):
        raise TypeError("coarse_graining must be an instance of CoarseGraining.")
    if not isinstance(lattice, nx.DiGraph):
        raise TypeError("lattice must be an instance of nx.DiGraph.")
    if not isinstance(cp_dict, dict):
        raise TypeError("cp_dict must be a dictionary.")


    partition_val = cp_dict[coarse_graining.partition]

    # We need ID of the partition to query for predecessors
    partition_ID = None
    for node_ID, node in lattice.nodes(data=True):
        if node['partition'] == coarse_graining.partition:
            partition_ID = node_ID
            break

    if partition_ID is None:
        raise ValueError("Coarse graining not found in the lattice.")
    if allAncestors:
        ancestors = list(nx.ancestors(lattice, partition_ID))
    else:
        ancestors = list(lattice.predecessors(partition_ID))

    ancestor_vals = [cp_dict[lattice.nodes[anc]['partition']] for anc in ancestors]
    minDiff = partition_val - max(ancestor_vals, default=0)

    return minDiff

def delta_CP_ancestors(lattice, cp_dict, allAncestors=False):
    """
    Calculate the minimum difference between the CP value of a coarse-graining 
    and its ancestors in the lattice, across all coarse-grainings.

    TODO: this could be sped up and made networkx independent by using the adjacency matrix of the lattice

    Parameters
    ----------
    lattice : nx.DiGraph
        The lattice of coarse-grainings, ordered by refinement.
    cp_dict : dict
        Dictionary mapping each partition to its CP value.
    allAncestors : bool, optional
        If True, calculate min difference with the max across ancestors in the lattice. If False, only consider direct ancestors.
        Default is False.
    Returns
    -------
    float
        The minimum difference between the CP value of the coarse-graining and its direct ancestors.
    """
    if not isinstance(lattice, nx.DiGraph):
        raise TypeError("lattice must be an instance of nx.DiGraph.")
    if not isinstance(cp_dict, dict):
        raise TypeError("cp_dict must be a dictionary.")

    delta_cp_values = {}
    
    for node_ID, node in lattice.nodes(data=True):
        partition_val = cp_dict[node_ID]
        if allAncestors:
            ancestors = list(nx.ancestors(lattice, node_ID))
        else:
            ancestors = list(lattice.predecessors(node_ID))

        ancestor_vals = [cp_dict[anc] for anc in ancestors]
        minDiff = partition_val - max(ancestor_vals, default=0)
        delta_cp_values[node['partition']] = minDiff

    return delta_cp_values

def greedy_path_on_lattice(lattice, CP_values, start_partition=None, end_partition=None):
    """
    Find a greedy path on the lattice of coarse-grainings based on CP values.
    
    Parameters
    ----------
    lattice : nx.DiGraph
        The lattice of coarse-grainings, ordered by refinement.
    CP_values : dict
        Dictionary mapping each partition to its CP value.
    start_partition : tuple, optional
        The starting partition for the path. If None, starts from the finest partition
    end_partition : tuple, optional
        The ending partition for the path. If None, ends at the coarsest partition

    Returns
    -------
    list [(CoarseGraining, float)]
        A list of tuples (partition, CP value) representing the partitions and CP values along the greedy path.
    """
    if not isinstance(lattice, nx.DiGraph):
        raise TypeError("lattice must be an instance of nx.DiGraph.")
    if not isinstance(CP_values, dict):
        raise TypeError("CP_values must be a dictionary.")

    if start_partition is None:
        # Start from the finest partition
        start_partition = max(lattice.nodes(), key=lambda x: x.count('|'))
    if end_partition is None:
        # End at the coarsest partition
        end_partition = min(lattice.nodes(), key=lambda x: x.count('|'))

    current_partition = start_partition
    path = [(current_partition, CP_values[current_partition])]
    
    while current_partition != end_partition:
        neighbors = list(lattice.successors(current_partition))
        if not neighbors:
            break  # No more neighbors to explore

        next_partition = max(neighbors, key=lambda x: CP_values[x])
        path.append((next_partition, CP_values[next_partition]))
        current_partition = next_partition

    return path
    
def find_single_path(lattice, CP_values, start_partition=None, end_partition=None, demand_nonzero_delta=True):
    """
    Find a single path on the lattice of coarse-grainings with the method from the CE2.0 paper. 
    steps:
    1. Start from the finest partition to the *best* (highest CP) coarsest partition with more than 1 block. 
    2. Find all paths on this lattice with nonnegative delta CP values.
    2. Pick the longest path. (when tied: just pick the first one). 

    Parameters
    ----------
    lattice : nx.DiGraph
        The lattice of coarse-grainings, ordered by refinement. Possibly filtered by consistency.
    CP_values : dict
        Dictionary mapping each partition to its CP value.
    start_partition : tuple, optional
        The starting partition for the path. If None, starts from the finest partition.
    end_partition : tuple, optional
        The ending partition for the path. If None, ends at the coarsest partition with more than 1 block.
    
    Returns
    -------
    list [(CoarseGraining, float, float)]
        A list of tuples (partition, CP value, delta CP) representing the partitions, CP values, and delta CP values along the path.
    """
    if not isinstance(lattice, nx.DiGraph):
        raise TypeError("lattice must be an instance of nx.DiGraph.")
    if not isinstance(CP_values, dict):
        raise TypeError("CP_values must be a dictionary.")


    if start_partition is None:
        # Start from the finest partition
        start_partition = max(lattice.nodes(), key=lambda x: x.count('|'))
    if end_partition is None:
        # End at the best partition with n - 1 blocks
        end_partition = max(( n for n in lattice.nodes() if lattice.nodes[n]['label'].count('|') == 1), key=lambda x: CP_values[x])

    valid_paths = []
    for path in nx.all_simple_paths(lattice, source=start_partition, target=end_partition):
        # Check if the path has nonnegative delta CP values
        valid_path = True
        if demand_nonzero_delta:
            for i in range(len(path) - 1):
                current_partition = path[i]
                next_partition = path[i + 1]
                delta_cp = CP_values[next_partition] - CP_values[current_partition]
                if delta_cp < 0:
                    valid_path = False
                    break
        
        if valid_path:
            valid_paths.append(path)

    if len(valid_paths) == 0:
        raise ValueError("No valid path found with nonnegative delta CP values.")
    # Take the longest path (first one in case of ties)
    valid_paths.sort(key=len, reverse=True)
    path = valid_paths[0]

    path_values = []
    for i in range(len(path)):
        current_partition = path[i]
        cp_value = CP_values[current_partition]
        if i == 0:
            delta_cp = 0  # No delta CP for the first partition
        else:
            delta_cp = CP_values[path[i]] - CP_values[path[i - 1]]
        path_values.append((current_partition, cp_value, delta_cp))

    return path_values

def causal_emergence_single_path(stochastic_matrix, lattice, mode='effectiveness'):
    """
    Calculate the causal emergence in a stochastic matrix. 
    Find the best coarse-graining path with the CE2.0 method, then sum the delta CPs along this path. 
    Parameters
    ----------
    stochastic_matrix : StochasticMatrix
        The stochastic matrix representing the network.
    lattice : nx.DiGraph
        The lattice of coarse-grainings, ordered by refinement. Possibly filtered by consistency.
    mode : str, optional
        The mode of calculation for CP values. Default is 'effectiveness', alternatively 'suff_plus_nec'.   
    Returns
    -------
    float
        The causal emergence value, which is the sum of delta CP values along the best path.
    """
    if not isinstance(stochastic_matrix, StochasticMatrix):
        raise TypeError("stochastic_matrix must be an instance of StochasticMatrix.")
    
    if not isinstance(lattice, nx.DiGraph):
        raise TypeError("lattice must be an instance of nx.DiGraph.")
    if mode not in ['effectiveness', 'suff_plus_nec']:
        raise ValueError("Invalid mode. Use 'effectiveness' or 'suff_plus_nec'.")
    
    # Calculate CP values for the lattice
    cp_values, _ = calc_CP_along_CGs(stochastic_matrix, coarse_grainings=None, mode=mode)
    # Find the best path on the lattice
    path = find_single_path(lattice, cp_values, start_partition=None, end_partition=None, demand_nonzero_delta=True)
    # Calculate the causal emergence value as the sum of delta CP values along the path
    causal_emergence_value = sum(delta_cp for _, _, delta_cp in path)
    return causal_emergence_value, path
    
def refinement_graph(coarse_grainings):
    """
    Create a Hasse diagram for the given partitions under refinement
    Parameters
    ----------
    coarse_grainings : list of CoarseGraining objects
        List of partitions, where each partition is a tuple of sets.
    Returns
    -------
    hasse_graph : networkx.DiGraph
        Directed graph representing the Hasse diagram of the partitions.
    """
    incidence_graph = nx.DiGraph()
    # Sort partitions by number of blocks (descending) to help with layout
    coarse_grainings = sorted(coarse_grainings, key=lambda x: x.n_blocks, reverse=True)
    
    # Add nodes. Node IDs need to be strings for networkx compatibility
    for cg in coarse_grainings:
        incidence_graph.add_node(str(cg), partition=cg.partition, label=str(cg))
    
    # Add edges based on refinement relation
    for cg1 in coarse_grainings:
        for cg2 in coarse_grainings:
            if str(cg1) != str(cg2) and cg1.is_refinement_of(cg2):
                incidence_graph.add_edge(str(cg1), str(cg2))
    hasse_graph = nx.transitive_reduction(incidence_graph)
    
    # Copy node attributes (transitive_reduction doesn't preserve them)
    for node in hasse_graph.nodes():
        hasse_graph.nodes[node].update(incidence_graph.nodes[node])
    
    return hasse_graph

def draw_refinement_graph(hasse_graph, value_dict, ax=None, title="Refinement Graph"):
    """
    Draw the refinement graph with nodes colored by the value_dict.
    
    Parameters
    ----------
    hasse_graph : networkx.DiGraph
        Directed graph representing the Hasse diagram of the partitions.
    value_dict : dict
        Dictionary mapping each partition to its value.
    title : str, optional
        Title of the plot. Default is "Refinement Graph".
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    # Get node values for coloring
    values = [value_dict[hasse_graph.nodes[n]['partition']] for n in hasse_graph.nodes()]
    try:
        norm = Normalize(vmin=-max(values[1:])-0.05, vmax=max(values[1:])+0.05)
    except ValueError:
        norm = Normalize(vmin=-0.5, vmax=0.5)  # Fallback if all values are the same
    # Define a colormap: blue for low values to red for high values
    cmap = cm.coolwarm
    # Draw the graph
    pos = nx.nx_agraph.graphviz_layout(hasse_graph, prog='dot', args='-Grankdir=BT') # Is this better?

    # nx.draw(hasse_graph, pos, ax=ax, with_labels=True, node_color=values, cmap=cmap,
    #         node_size=500, font_size=10, font_color='black', edge_color='gray', label=True)
    
    node_colors = [cmap(norm(value_dict[n[1]])) for n in nx.get_node_attributes(hasse_graph, 'partition').items()]
    nx.draw_networkx_nodes(hasse_graph, pos, node_color=node_colors, node_size=700, alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(hasse_graph, pos, edge_color='gray', width=1.0, arrows=True, arrowsize=15)
    
    # Draw labels
    labels = {i: hasse_graph.nodes[i]['label'] for i in hasse_graph.nodes()}
    nx.draw_networkx_labels(hasse_graph, pos, labels=labels, font_size=8)


    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Values')
    ax.set_title(title)
    plt.tight_layout()
    return ax   