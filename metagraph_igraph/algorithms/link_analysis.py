from metagraph import concrete_algorithm
from metagraph.plugins.numpy.types import NumpyNodes
from ..types import IGraph
import numpy as np


@concrete_algorithm("link_analysis.pagerank")
def igraph_pagerank(graph: IGraph, damping: float=0.85, maxiter: int=50, tolerance: float=1e-5) -> NumpyNodes:
    directed = graph.value.is_directed()
    weights = "weight" if graph.value.is_weighted() else None
    pr = graph.value.pagerank(directed=directed, weights=weights, damping=damping,
                              niter=maxiter, eps=tolerance, implementation="power")
    return NumpyNodes(np.array(pr))