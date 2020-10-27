from metagraph import translator
from metagraph.plugins import has_grblas
from .types import IGraph
import igraph
import numpy as np


if has_grblas:
    import grblas
    from metagraph.plugins.graphblas.types import (
        GrblasGraph, GrblasEdgeMap, GrblasEdgeSet, GrblasNodeMap, GrblasNodeSet
    )

    @translator
    def graph_from_graphblas(x: GrblasGraph, **props) -> IGraph:
        xprops = GrblasGraph.Type.compute_abstract_properties(x, ["is_directed", "node_type", "node_dtype",
                                                                  "edge_type", "edge_dtype"])

        size = x.edges.value.nrows
        if x.nodes is not None:
            size = x.nodes.value.nvals
        graph = igraph.Graph(size, directed=xprops["is_directed"])
        rows, cols, weights = x.edges.value.to_values()
        # TODO: get the indexes from x.nodes.value; check if sequential, sort, and add NodeId or not to graph.vs
        # TODO: if remap is needed, do that now for rows, cols
        graph.add_edges(zip(rows, cols))
        if xprops["node_type"] == "map":
            # Assumes that idx is sorted
            idx, vals = x.nodes.value.to_values()
            graph.vs["weight"] = vals
        if xprops["edge_type"] == "map":
            graph.es["weight"] = weights
        if not xprops["is_directed"]:
            graph.simplify(multiple=True, loops=True, combine_edges="first")
        ret = IGraph(graph)

        info = IGraph.Type.get_typeinfo(ret)
        info.known_abstract_props.update(xprops)

        return ret

    @translator
    def graph_to_graphblas(x: IGraph, **props) -> GrblasGraph:
        xprops = IGraph.Type.compute_abstract_properties(x, ["is_directed", "node_type", "node_dtype",
                                                             "edge_type", "edge_dtype"])

        nn = x.value.vcount()
        dmap = {
            "bool": grblas.dtypes.BOOL,
            "int": grblas.dtypes.INT64,
            "float": grblas.dtypes.FP64,
            None: grblas.dtypes.UINT8,  # unweighted graph
        }
        rows, cols = list(zip(*x.value.get_edgelist()))
        if xprops["edge_type"] == "map":
            vals = x.value.es[x.edge_weight_label]
            eclass = GrblasEdgeMap
        else:
            vals = [1]*len(rows)
            eclass = GrblasEdgeSet

        # Handle non-sequential graph
        if not x.is_sequential():
            node_ids = np.array(x.value.vs["NodeId"])
            rows = node_ids[list(rows)].tolist()
            cols = node_ids[list(cols)].tolist()
            nn = node_ids.max() + 1
        else:
            node_ids = x.value.vs.indices

        # Undirected graph must add reversed edges
        if not xprops["is_directed"]:
            rows, cols = rows + cols, cols + rows
            vals *= 2  # duplicate, not element-wise multiply

        m = grblas.Matrix.from_values(rows, cols, vals, nrows=nn, ncols=nn, dtype=dmap[xprops["edge_dtype"]],
                                      dup_op=grblas.binary.max)
        edges = eclass(m)
        if xprops["node_type"] == "map":
            v = grblas.Vector.from_values(list(node_ids), x.value.vs[x.node_weight_label])
            nodes = GrblasNodeMap(v)
        else:
            v = grblas.Vector.from_values(list(node_ids), [1]*x.value.vcount())
            nodes = GrblasNodeSet(v)
        ret = GrblasGraph(edges, nodes)

        info = GrblasGraph.Type.get_typeinfo(ret)
        info.known_abstract_props.update(xprops)

        return ret
