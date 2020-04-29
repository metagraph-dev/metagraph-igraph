from metagraph import concrete_algorithm
from metagraph.plugins.numpy.types import NumpyNodes
from ..types import IGraph
import igraph
import numpy as np


@concrete_algorithm("cluster.triangle_count")
def igraph_triangle_count(graph: IGraph) -> int:
    return len(graph.value.cliques(3, 3))


@concrete_algorithm("clustering.connected_components")
def igraph_connected_components(graph: IGraph) -> NumpyNodes:
    cc = graph.value.components(igraph.WEAK).membership
    return NumpyNodes(np.array(cc), node_index=graph._node_index)


@concrete_algorithm("clustering.strongly_connected_components")
def igraph_strongly_connected_components(graph: IGraph) -> NumpyNodes:
    cc = graph.value.components(igraph.STRONG).membership
    return NumpyNodes(np.array(cc), node_index=graph._node_index)
