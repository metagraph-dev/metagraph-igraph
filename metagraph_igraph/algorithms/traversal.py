from typing import Tuple
from metagraph import concrete_algorithm
from metagraph.plugins.numpy.types import NumpyNodes, NumpyVector
from ..types import IGraph
import numpy as np


@concrete_algorithm("traversal.bellman_ford")
def igraph_bellman_ford(graph: IGraph, source_node: int) -> Tuple[NumpyNodes, NumpyNodes]:
    nn = graph.num_nodes
    weights = "weight" if graph.value.is_weighted() else None
    # Calculate path lengths
    if graph.value.is_weighted():
        lengths = graph.value.shortest_paths(source_node, weights=weights)
    else:
        lengths = graph.value.shortest_paths(source_node)
    lengths = np.array(lengths[0])
    missing = lengths == np.inf
    if missing.any():
        lengths[missing] = -1
    else:
        missing = None
    lengths = lengths.astype(int)
    # Calculate parents (very inefficient, but it works)
    parents = np.empty(nn, dtype=int)
    paths = graph.value.get_all_shortest_paths(source_node, weights=weights)
    for path in paths:
        node = path[-1]
        if node == source_node:
            parents[node] = node
        else:
            parents[node] = path[-2]

    print(parents)
    print(lengths)
    return (
        NumpyNodes(parents, missing_mask=missing, node_index=graph.node_index),
        NumpyNodes(lengths, missing_mask=missing, node_index=graph.node_index)
    )


@concrete_algorithm("traversal.breadth_first_search")
def igraph_breadth_first_search(graph: IGraph, source_node: int) -> NumpyVector:
    nodes = [node.index for node in graph.value.bfsiter(source_node)]
    return NumpyVector(np.array(nodes))
