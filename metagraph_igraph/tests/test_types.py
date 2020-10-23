import pytest
from metagraph_igraph.types import IGraph
from metagraph import NodeLabels
from igraph import Graph


def test_igraph():
    #    0 1 2
    # 0 [1 2  ]
    # 1 [  0 3]
    # 2 [  3  ]
    aprops = {
        "is_directed": True,
        "node_type": "set",
        "edge_type": "map",
        "edge_dtype": "int",
    }
    g = Graph(3, directed=True, edges=[(0, 0), (0, 1), (1, 1), (1, 2), (2, 1)], edge_attrs={"weight": [1, 2, 0, 3, 3]})
    IGraph.Type.assert_equal(
        IGraph(g), IGraph(g.copy()), aprops, aprops, {}, {}
    )
    g_close = g.copy()
    g_close.es[g_close.get_eid(0, 0)]["weight"] = 1.0000000000001
    IGraph.Type.assert_equal(
        IGraph(g_close),
        IGraph(g),
        {**aprops, "edge_dtype": "float"},
        {**aprops, "edge_dtype": "float"},
        {},
        {},
    )
    g_diff = Graph(3, directed=True,
                   edges=[(0, 0), (0, 1), (1, 1), (1, 2), (2, 1)],
                   edge_attrs={"weight": [1, 3, 0, 3, 3]})
    with pytest.raises(AssertionError):
        IGraph.Type.assert_equal(
            IGraph(g), IGraph(g_diff), aprops, aprops, {}, {}
        )
    # Ignore weights if unweighted
    IGraph.Type.assert_equal(
        IGraph(g),
        IGraph(g_diff),
        {**aprops, "edge_type": "set"},
        {**aprops, "edge_type": "set"},
        {},
        {},
    )
    with pytest.raises(AssertionError):
        IGraph.Type.assert_equal(
            IGraph(g),
            IGraph(
                Graph(3, directed=True,
                      edges=[(0, 0), (0, 1), (1, 1), (1, 2), (2, 0)], edge_attrs={"weight": [1, 2, 0, 3, 3]}
                )  # change is here                             ^^^
            ),
            aprops,
            aprops,
            {},
            {},
        )
    with pytest.raises(AssertionError):
        IGraph.Type.assert_equal(
            IGraph(g),
            IGraph(
                Graph(3, directed=True,
                      edges=[(0, 0), (0, 1), (1, 1), (1, 2), (2, 1), (2, 0)], edge_attrs={"weight": [1, 2, 0, 3, 3, 0]}
                )  # extra element                                   ^^^^^^                                        ^^^
            ),
            aprops,
            aprops,
            {},
            {},
        )
    # weights don't match, so we take the fast path and declare them not equal
    with pytest.raises(AssertionError):
        IGraph.Type.assert_equal(
            IGraph(g),
            IGraph(g),
            aprops,
            {**aprops, "edge_dtype": "float"},
            {},
            {},
        )
    # mask doesn't match
    with pytest.raises(AssertionError):
        IGraph.Type.assert_equal(
            IGraph(g),
            IGraph(g, mask=[True, False, True]),
            aprops,
            aprops,
            {},
            {},
        )
    # mask does match
    IGraph.Type.assert_equal(
        IGraph(g, mask=[True, True, True]),
        IGraph(g, mask=[True, True, True]),
        aprops,
        aprops,
        {},
        {},
    )
    # mask wrong size
    with pytest.raises(TypeError):
        IGraph(g, mask=[True, True, False, True, False])
