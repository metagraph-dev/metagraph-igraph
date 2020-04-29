from metagraph import translator
from metagraph.plugins import has_grblas
from .types import IGraph
import igraph


if has_grblas:
    import grblas
    from metagraph.plugins.graphblas.types import GrblasAdjacencyMatrix

    @translator
    def graph_from_graphblas(x: GrblasAdjacencyMatrix, **props) -> IGraph:
        graph = igraph.Graph(x.num_nodes, directed=x._is_directed)
        if x._weights == "unweighted":
            row_index, col_index, _ = x.value.to_values()
            for ridx, cidx in zip(row_index, col_index):
                graph.add_edge(ridx, cidx)
        else:
            row_index, col_index, weights = x.value.to_values()
            for ridx, cidx, wgt in zip(row_index, col_index, weights):
                graph.add_edge(ridx, cidx, weight=wgt)
        return IGraph(graph, weights=x._weights, dtype=x._dtype, node_index=x.node_index)

    @translator
    def graph_to_graphblas(x: IGraph, **props) -> GrblasAdjacencyMatrix:
        dtype = {
            "bool": grblas.dtypes.BOOL,
            "int": grblas.dtypes.INT64,
            "float": grblas.dtypes.FP64,
        }[x._dtype]
        rows, cols = [], []
        for edge in x.value.es:
            rows.append(edge.source)
            cols.append(edge.target)
        if x.value.is_weighted():
            vals = x.value.es["weight"]
        else:
            vals = [1]*len(rows)
        m = grblas.Matrix.new(dtype, nrows=x.num_nodes, ncols=x.num_nodes)
        m.build(rows, cols, vals)
        return GrblasAdjacencyMatrix(m, is_directed=x.value.is_directed(), node_index=x.node_index)
