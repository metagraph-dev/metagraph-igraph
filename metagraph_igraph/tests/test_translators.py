import pytest
import metagraph as mg
from metagraph import IndexedNodes
from metagraph.plugins import has_grblas
import igraph
from ..types import IGraph


def test_graphblas_2_igraph(default_plugin_resolver):
    if not has_grblas:
        pytest.skip('needs grblas')

    import grblas
    from metagraph.plugins.graphblas.types import GrblasAdjacencyMatrix
    dpr = default_plugin_resolver
    nidx = IndexedNodes("ABC")
    #    A B C
    # A [1 2  ]
    # B [  0 3]
    # C [  3  ]
    m = grblas.Matrix.from_values(
        [0, 0, 1, 1, 2], [0, 1, 1, 2, 1], [1, 2, 0, 3, 3], dtype=grblas.dtypes.INT64
    )
    x = GrblasAdjacencyMatrix(m, node_index=nidx)
    # Convert graphblas -> igraph
    g = igraph.Graph(3, directed=True,
                     edges=[(0, 0), (0, 1), (1, 1), (1, 2), (2, 1)],
                     edge_attrs={"weight": [1, 2, 0, 3, 3]})
    intermediate = IGraph(g, node_index=nidx)
    y = dpr.translate(x, IGraph)
    assert IGraph.Type.compare_objects(y, intermediate)
    # Convert graphblas <- igraph
    x2 = dpr.translate(y, GrblasAdjacencyMatrix)
    assert GrblasAdjacencyMatrix.Type.compare_objects(x, x2)
