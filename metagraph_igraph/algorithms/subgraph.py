from metagraph import concrete_algorithm
from metagraph.plugins.numpy.types import NumpyNodeMap, NumpyNodeSet
from ..types import IGraph
import numpy as np
import metagraph as mg


@concrete_algorithm("subgraph.subisomorphic")
def igraph_isomorphic(graph: IGraph, subgraph: IGraph) -> bool:
    return graph.value.subisomorphic_lad(subgraph.value)