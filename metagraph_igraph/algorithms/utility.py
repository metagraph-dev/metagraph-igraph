from metagraph import concrete_algorithm
from metagraph.plugins.numpy.types import NumpyNodeMap, NumpyNodeSet
from ..types import IGraph
import numpy as np
import metagraph as mg


@concrete_algorithm("util.graph.isomorphic")
def igraph_isomorphic(g1: IGraph, g2: IGraph) -> bool:
    return g1.value.isomorphic(g2.value)
