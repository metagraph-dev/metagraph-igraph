from metagraph import translator
from metagraph.plugins import has_grblas
from .types import IGraph
import igraph
import numpy as np


if has_grblas:
    import grblas
    from metagraph.plugins.graphblas.types import (
        GrblasGraph,
        GrblasEdgeMap,
        GrblasEdgeSet,
        GrblasNodeMap,
        GrblasNodeSet,
    )

    @translator
    def graph_from_graphblas(x: GrblasGraph, **props) -> IGraph:
        xprops = GrblasGraph.Type.compute_abstract_properties(
            x, ["is_directed", "node_type", "node_dtype", "edge_type", "edge_dtype"]
        )

        vcount = x.nodes.nvals
        is_sequential = x.nodes.size == vcount
        idx, node_weights = x.nodes.to_values()
        graph = igraph.Graph(vcount, directed=xprops["is_directed"])
        if is_sequential:
            rows, cols, edge_weights = x.value.to_values()
        else:
            # Compress node ids as required by IGraph and add NodeId attributes to `vs`
            compressed = x.value[idx, idx].new()
            rows, cols, edge_weights = compressed.to_values()
            graph.vs["NodeId"] = idx.tolist()
        graph.add_edges(zip(rows.tolist(), cols.tolist()))

        if xprops["node_type"] == "map":
            graph.vs["weight"] = node_weights.tolist()
        if xprops["edge_type"] == "map":
            graph.es["weight"] = edge_weights.tolist()
        if not xprops["is_directed"]:
            graph.simplify(multiple=True, loops=False, combine_edges="first")
        return IGraph(graph, aprops=xprops)

    @translator
    def graph_to_graphblas(x: IGraph, **props) -> GrblasGraph:
        xprops = IGraph.Type.compute_abstract_properties(
            x, ["is_directed", "node_type", "node_dtype", "edge_type", "edge_dtype"]
        )

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
        else:
            vals = np.ones_like(rows)

        # Handle non-sequential graph
        if not x.is_sequential():
            node_ids = np.array(x.value.vs["NodeId"])
            rows = node_ids[list(rows)]
            cols = node_ids[list(cols)]
            nn = node_ids.max() + 1
        else:
            node_ids = x.value.vs.indices

        # Undirected graph must add reversed edges
        if not xprops["is_directed"]:
            rows, cols = np.concatenate([rows, cols]), np.concatenate([cols, rows])
            vals = np.concatenate([vals, vals])

        m = grblas.Matrix.from_values(
            rows,
            cols,
            vals,
            nrows=nn,
            ncols=nn,
            dtype=dmap[xprops["edge_dtype"]],
            dup_op=grblas.binary.max,
        )

        if xprops["node_type"] == "map":
            nodes = grblas.Vector.from_values(node_ids, x.value.vs[x.node_weight_label])
        else:
            nodes = grblas.Vector.from_values(node_ids, np.ones(x.value.vcount(), bool))

        return GrblasGraph(m, nodes, aprops=xprops)
