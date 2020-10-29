from metagraph import concrete_algorithm, NodeID
from metagraph.plugins.numpy.types import NumpyNodeMap
from ..types import IGraph
import igraph
import numpy as np
from typing import Tuple


@concrete_algorithm("cluster.triangle_count")
def igraph_triangle_count(graph: IGraph) -> int:
    return len(graph.value.cliques(3, 3))


@concrete_algorithm("clustering.connected_components")
def igraph_connected_components(graph: IGraph) -> NumpyNodeMap:
    cc = graph.value.components(igraph.WEAK).membership
    return NumpyNodeMap(np.array(cc))


@concrete_algorithm("clustering.strongly_connected_components")
def igraph_strongly_connected_components(graph: IGraph) -> NumpyNodeMap:
    cc = graph.value.components(igraph.STRONG).membership
    return NumpyNodeMap(np.array(cc))


@concrete_algorithm("flow.max_flow")
def max_flow(graph: IGraph, source_node: NodeID, target_node: NodeID) -> Tuple[float, IGraph]:
    aprops = IGraph.Type.compute_abstract_properties(graph, {"edge_dtype"})
    g = graph.value
    flow = g.maxflow(source_node, target_node, graph.edge_weight_label)
    out = g.copy()
    flow_vals = map(int, flow.flow) if aprops["edge_dtype"] == "int" else flow.flow
    out.es[graph.edge_weight_label] = list(flow_vals)
    return flow.value, IGraph(out, node_weight_label=graph.node_weight_label, edge_weight_label=graph.edge_weight_label)


@concrete_algorithm("flow.min_cut")
def min_cut(
    graph: IGraph,
    source_node: NodeID,
    target_node: NodeID,
) -> Tuple[float, IGraph]:
    """
    Returns the sum of the minimum cut weights and a graph containing only those edges
    which are part of the minimum cut.
    """
    g = graph.value
    cut = g.mincut(source_node, target_node, graph.edge_weight_label)
    out = igraph.Graph(len(g.vs), directed=g.is_directed())
    if graph.node_weight_label in g.vs.attributes():
        out.vs[graph.node_weight_label] = g.vs[graph.node_weight_label]
    for edge in g.es[cut.cut]:
        out.add_edge(edge.source, edge.target, **edge.attributes())
    return cut.value, IGraph(out, node_weight_label=graph.node_weight_label, edge_weight_label=graph.edge_weight_label)
