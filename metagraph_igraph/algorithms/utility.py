from metagraph import concrete_algorithm
from metagraph.plugins.numpy.types import NumpyNodeMap, NumpyNodeSet
from ..types import IGraph
import numpy as np
import metagraph as mg


@concrete_algorithm("util.graph.degree")
def igraph_degree(graph: IGraph, in_edges: bool, out_edges: bool) -> NumpyNodeMap:
    if in_edges and out_edges:
        degrees = graph.value.degree(mode='all')
    elif in_edges:
        degrees = graph.value.degree(mode='in')
    elif out_edges:
        degrees = graph.value.degree(mode='out')
    else:
        degrees = [0] * len(graph.value.vs)
    node_ids = None if graph.is_sequential() else graph.value.vs["NodeId"]
    return NumpyNodeMap(np.array(degrees), node_ids)


@concrete_algorithm("util.graph.isomorphic")
def igraph_isomorphic(g1: IGraph, g2: IGraph) -> bool:
    return g1.value.isomorphic(g2.value)
