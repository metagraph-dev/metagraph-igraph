from metagraph import concrete_algorithm
from metagraph.plugins.numpy.types import NumpyNodeMap, NumpyNodeSet
from ..types import IGraph
import numpy as np
import metagraph as mg


@concrete_algorithm("centrality.pagerank")
def igraph_pagerank(graph: IGraph, damping: float, maxiter: int, tolerance: float) -> NumpyNodeMap:
    directed = graph.value.is_directed()
    weights = "weight" if graph.value.is_weighted() else None
    pr = graph.value.pagerank(directed=directed, weights=weights, damping=damping,
                              niter=maxiter, eps=tolerance, implementation="power")
    return NumpyNodeMap(np.array(pr))


@concrete_algorithm("centrality.betweenness")
def igraph_betweenness_centrality(graph: IGraph, nodes: mg.Optional[NumpyNodeSet], normalize: bool) -> NumpyNodeMap:
    if graph.value.is_weighted():
        bc = graph.value.betweenness(weights='weight')
    else:
        bc = graph.value.betweenness()
    return NumpyNodeMap(np.array(bc))