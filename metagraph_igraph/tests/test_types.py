import pytest
from metagraph_igraph.types import IGraph
from metagraph import IndexedNodes
from igraph import Graph


def test_igraph():
    #    A B C
    # A [1 2  ]
    # B [  0 3]
    # C [  3  ]
    g = Graph(3, directed=True, edges=[(0, 0), (0, 1), (1, 1), (1, 2), (2, 1)], edge_attrs={"weight": [1, 2, 0, 3, 3]})
    IGraph.Type.assert_equal(
        IGraph(g), IGraph(g.copy()),
    )
    g_close = g.copy()
    g_close.es[g_close.get_eid(0, 0)]["weight"] = 1.0000000000001
    IGraph.Type.assert_equal(
        IGraph(g_close, dtype='float'), IGraph(g, dtype='float'),
    )
    g_diff = Graph(3, directed=True,
                   edges=[(0, 0), (0, 1), (1, 1), (1, 2), (2, 1)],
                   edge_attrs={"weight": [1, 3, 0, 3, 3]})
    with pytest.raises(AssertionError):
        IGraph.Type.assert_equal(
            IGraph(g), IGraph(g_diff),
        )
    # Ignore weights if unweighted
    IGraph.Type.assert_equal(
        IGraph(g, weights="unweighted"),
        IGraph(g_diff, weights="unweighted"),
    )
    with pytest.raises(AssertionError):
        IGraph.Type.assert_equal(
            IGraph(g),
            IGraph(
                Graph(3, directed=True,
                      edges=[(0, 0), (0, 1), (1, 1), (1, 2), (2, 0)], edge_attrs={"weight": [1, 2, 0, 3, 3]}
                )  # change is here                             ^^^
            ),
        )
    with pytest.raises(AssertionError):
        IGraph.Type.assert_equal(
            IGraph(g),
            IGraph(
                Graph(3, directed=True,
                      edges=[(0, 0), (0, 1), (1, 1), (1, 2), (2, 1), (2, 0)], edge_attrs={"weight": [1, 2, 0, 3, 3, 0]}
                )  # extra element                                   ^^^^^^                                        ^^^
            ),
        )
    # weights don't match, so we take the fast path and declare them not equal
    with pytest.raises(AssertionError):
        IGraph.Type.assert_equal(
            IGraph(g), IGraph(g, weights="any"),
        )
    # Node index affects comparison
    #    B C A
    # B [0 3  ]
    # C [3    ]
    # A [2   1]
    IGraph.Type.assert_equal(
        IGraph(g, node_index=IndexedNodes("ABC")),
        IGraph(
            Graph(3, directed=True,
                  edges=[(0, 0), (0, 1), (1, 0), (2, 0), (2, 2)],
                  edge_attrs={"weight": [0, 3, 3, 2, 1]}
            ),
            node_index=IndexedNodes("BCA"),
        ),
    )
    with pytest.raises(AssertionError):
        IGraph.Type.assert_equal(5, 5)
