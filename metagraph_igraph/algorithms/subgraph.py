from metagraph import concrete_algorithm
from metagraph.plugins.numpy.types import NumpyNodeMap, NumpyNodeSet
from ..types import IGraph
import metagraph as mg
import numpy as np
import random


@concrete_algorithm("subgraph.extract_subgraph")
def extract_subgraph(graph: IGraph, nodes: NumpyNodeSet) -> IGraph:
    node_list = nodes.value
    if graph.is_sequential():
        g = graph.value.copy()
        g.vs["NodeId"] = range(g.vcount())
    else:
        g = graph.value
        real_node_ids = np.array(g.vs["NodeId"])
        node_list = real_node_ids[node_list]
    subg = g.subgraph(node_list.tolist())
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
    return NumpyNodeSet(node_sets[0])


@concrete_algorithm("traversal.minimum_spanning_tree")
def minimum_spanning_tree(
    graph: IGraph
) -> IGraph:
    mst = graph.value.spanning_tree(graph.edge_weight_label)
    return IGraph(mst, node_weight_label=graph.node_weight_label, edge_weight_label=graph.edge_weight_label)


@concrete_algorithm("subgraph.sample.node_sampling")
def node_sampling(graph: IGraph, p: float) -> IGraph:
    if p <= 0 or p > 1:
        raise ValueError(f"Probability `p` must be between 0 and 1, found {p}")
    chosen_indices = np.random.random(len(graph.value.vs)) < p
    chosen_nodes = np.array(graph.value.vs.indices)[chosen_indices]
    nodes = NumpyNodeSet(chosen_nodes)
    return extract_subgraph(graph, nodes)


@concrete_algorithm("subgraph.sample.ties")
def totally_induced_edge_sampling(graph: IGraph, p: float) -> IGraph:
    """
    Totally Induced Edge Sampling method
    https://docs.lib.purdue.edu/cgi/viewcontent.cgi?article=2743&context=cstech
    """
    if p <= 0 or p > 1:
        raise ValueError(f"Probability `p` must be between 0 and 1, found {p}")
    chosen_edges = [edge.tuple for edge in graph.value.es if random.random() < p]
    chosen_nodes = set(np.array(chosen_edges).flatten())
    nodes = NumpyNodeSet(chosen_nodes)
    return extract_subgraph(graph, nodes)
