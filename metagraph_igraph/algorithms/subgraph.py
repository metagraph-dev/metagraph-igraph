from metagraph import concrete_algorithm
from metagraph.plugins.numpy.types import NumpyNodeMap, NumpyNodeSet
from ..types import IGraph
import metagraph as mg


@concrete_algorithm("subgraph.extract_subgraph")
def extract_subgraph(graph: IGraph, nodes: NumpyNodeSet) -> IGraph:
    node_set = set(nodes.nodes())
    node_list = list(sorted(node_set))
    if graph.is_sequential():
        g = graph.value.copy()
        g.vs["NodeId"] = range(g.vcount())
    else:
        g = graph.value
    subg = g.subgraph(node_list)
    return IGraph(subg, node_weight_label=graph.node_weight_label, edge_weight_label=graph.edge_weight_label)


@concrete_algorithm("subgraph.k_core")
def k_core(graph: IGraph, k: int) -> IGraph:
    if graph.is_sequential():
        g = graph.value.copy()
        g.vs["NodeId"] = range(g.vcount())
    else:
        g = graph.value
    kcore = g.k_core(k)
    return IGraph(kcore, node_weight_label=graph.node_weight_label, edge_weight_label=graph.edge_weight_label)


@concrete_algorithm("subgraph.subisomorphic")
def igraph_isomorphic(graph: IGraph, subgraph: IGraph) -> bool:
    return graph.value.subisomorphic_lad(subgraph.value)


@concrete_algorithm("subgraph.maximal_independent_set")
def maximal_independent_set(graph: IGraph) -> NumpyNodeSet:
    node_sets = graph.value.largest_independent_vertex_sets()
    return NumpyNodeSet(set(node_sets[0]))
