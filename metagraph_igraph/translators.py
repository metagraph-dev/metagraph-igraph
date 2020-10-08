from metagraph import translator
from metagraph.plugins import has_grblas
from .types import IGraph
import igraph


if has_grblas:
    import grblas
    from metagraph.plugins.graphblas.types import GrblasGraph, GrblasEdgeMap, GrblasEdgeSet, GrblasNodeMap

    @translator
    def graph_from_graphblas(x: GrblasGraph, **props) -> IGraph:
        xprops = GrblasGraph.Type.compute_abstract_properties(x, ["is_directed", "node_type", "node_dtype",
                                                                  "edge_type", "edge_dtype"])

        graph = igraph.Graph(x.edges.value.nrows, directed=xprops["is_directed"])
        rows, cols, weights = x.edges.value.to_values()
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
        }
        rows, cols = list(zip(*x.value.get_edgelist()))
        if xprops["edge_type"] == "map":
            vals = x.value.es[x.edge_weight_label]
            eclass = GrblasEdgeMap
        else:
            vals = [1]*len(rows)
            eclass = GrblasEdgeSet
        m = grblas.Matrix.from_values(rows, cols, vals, nrows=nn, ncols=nn, dtype=dmap[xprops["edge_dtype"]])
        edges = eclass(m)
        if xprops["node_type"] == "map":
            v = grblas.Vector.from_values(range(nn), x.value.vs[x.node_weight_label])
            nodes = GrblasNodeMap(v)
        else:
            nodes = None
        ret = GrblasGraph(edges, nodes)

        info = GrblasGraph.Type.get_typeinfo(ret)
        info.known_abstract_props.update(xprops)

        return ret
