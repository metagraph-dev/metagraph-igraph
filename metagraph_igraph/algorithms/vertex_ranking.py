from metagraph import concrete_algorithm
from metagraph.plugins.numpy.types import NumpyNodes
from ..types import IGraph
import numpy as np


@concrete_algorithm("vertex_ranking.betweenness_centrality")
def igraph_betweenness_centrality(graph: IGraph, k: int, enable_normalization: bool, include_endpoints: bool) -> NumpyNodes:
    if graph.value.is_weighted():
        bc = graph.value.betweenness(weights='weight')
    else:
        bc = graph.value.betweenness()
    return NumpyNodes(np.array(bc), node_index=graph._node_index)
