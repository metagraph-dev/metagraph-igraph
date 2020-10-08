from typing import Tuple
from metagraph import concrete_algorithm, NodeID
from metagraph.plugins.numpy.types import NumpyNodeMap, NumpyVector
from ..types import IGraph
import numpy as np


@concrete_algorithm("traversal.bellman_ford")
def igraph_bellman_ford(graph: IGraph, source_node: NodeID) -> Tuple[NumpyNodeMap, NumpyNodeMap]:
    nn = graph.value.vcount()
    # Calculate path lengths
    lengths = graph.value.shortest_paths(source_node, weights=graph.edge_weight_label)
    lengths = np.array(lengths[0])
    mask = lengths != np.inf
    if not mask.all():
        lengths[~mask] = -1
        reachable = np.arange(nn)[mask]
    else:
        mask = None
        reachable = np.arange(nn)
    lengths = lengths.astype(int)
    # Calculate parents
    parents = np.empty(nn, dtype=int)
    paths = graph.value.get_shortest_paths(source_node, to=list(reachable), weights=graph.edge_weight_label)
    for dest_node, path in zip(reachable, paths):
        # Each path goes from source_node to dest_node
        if dest_node == source_node:
            parents[dest_node] = dest_node
        else:
            parents[dest_node] = path[-2]

    return (
        NumpyNodeMap(parents, mask=mask),
        NumpyNodeMap(lengths, mask=mask)
    )


@concrete_algorithm("traversal.bfs_iter")
def igraph_breadth_first_search(graph: IGraph, source_node: NodeID, depth_limit: int) -> NumpyVector:
    nodes = [node.index for node in graph.value.bfsiter(source_node)]
    return NumpyVector(np.array(nodes))
