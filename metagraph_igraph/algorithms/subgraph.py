from metagraph import concrete_algorithm
from metagraph.plugins.numpy.types import NumpyNodeMap, NumpyNodeSet
from ..types import IGraph
import numpy as np
import metagraph as mg


@concrete_algorithm("subgraph.extract_subgraph")
def extract_subgraph(graph: IGraph, nodes: NumpyNodeSet) -> IGraph:
    # There is a method in igraph called `subgraph` which seems like it would do exactly what we want here,
    # but it doesn't. It collapses the node numbering, effectively erasing the previous meaning of the node ids.
    # Insteam, we find the edges to keep, remove all edges, re-add the edges to keep, and flag the remaining
    # nodes as active while leaving the others inactive. This preserves the node ids numbering.
    aprops = IGraph.Type.compute_abstract_properties(graph, {"edge_type"})
    node_set = set(nodes.nodes())
    node_list = list(sorted(node_set))
    subg = graph.value.copy()
    keep_view = subg.es.select(_within=node_list)
    edges = [kv.tuple for kv in keep_view]
    if aprops["edge_type"] == "map":
        weights = keep_view[graph.edge_weight_label]
    # Once edges are deleted, the view above is invalid
    subg.delete_edges(None)
    subg.add_edges(edges)
    if aprops["edge_type"] == "map":
        subg.es[graph.edge_weight_label] = weights
    # This sets the mask for active/inactive nodes
    subg.vs['active'] = [i in node_set for i in range(subg.vcount())]
    return IGraph(subg, node_weight_label=graph.node_weight_label, edge_weight_label=graph.edge_weight_label)


@concrete_algorithm("subgraph.subisomorphic")
def igraph_isomorphic(graph: IGraph, subgraph: IGraph) -> bool:
    return graph.value.subisomorphic_lad(subgraph.value)