import pytest
import metagraph as mg
from metagraph.plugins import has_grblas
from metagraph.tests.util import default_plugin_resolver
import igraph
from ..types import IGraph


def test_graphblas_2_igraph(default_plugin_resolver):
    if not has_grblas:
        pytest.skip('needs grblas')

    import grblas
    from metagraph.plugins.graphblas.types import GrblasGraph
    dpr = default_plugin_resolver
    #    0 1 2
    # 0 [1 2  ]
    # 1 [  0 3]
    # 2 [  3  ]
    m = grblas.Matrix.from_values(
        [0, 0, 1, 1, 2], [0, 1, 1, 2, 1], [1, 2, 0, 3, 3], dtype=grblas.dtypes.INT64
    )
    x = GrblasGraph(m)
    # Convert graphblas -> igraph
    g = igraph.Graph(3, directed=True,
                     edges=[(0, 0), (0, 1), (1, 1), (1, 2), (2, 1)],
                     edge_attrs={"weight": [1, 2, 0, 3, 3]})
    intermediate = IGraph(g)
    y = dpr.translate(x, IGraph)
    dpr.assert_equal(y, intermediate)
    # Convert graphblas <- igraph
    x2 = dpr.translate(y, GrblasGraph)
    dpr.assert_equal(x, x2)
