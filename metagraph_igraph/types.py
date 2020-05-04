from metagraph import Wrapper, SequentialNodes
from metagraph.types import Graph, DTYPE_CHOICES, WEIGHT_CHOICES
import igraph
import math
import operator
from functools import partial


class IGraph(Wrapper, abstract=Graph):
    def __init__(self, graph, *, weights=None, dtype=None, node_index=None):
        self.value = graph
        self._node_index = node_index
        if graph.is_weighted():
            all_values = set(graph.es["weight"])
            self._dtype = self._determine_dtype(dtype, all_values)
            self._weights = self._determine_weights(weights, all_values)
        else:
            self._dtype = "bool"
            self._weights = "unweighted"

    def _determine_dtype(self, dtype, all_values):
        if dtype is not None:
            if dtype not in DTYPE_CHOICES:
                raise ValueError(f"Illegal dtype: {dtype}")
            return dtype

        all_types = {type(v) for v in all_values}
        if not all_types or (all_types - {float, int, bool}):
            return "str"
        for type_ in (float, int, bool):
            if type_ in all_types:
                return str(type_.__name__)

    def _determine_weights(self, weights, all_values):
        if weights is not None:
            if weights not in WEIGHT_CHOICES:
                raise ValueError(f"Illegal weights: {weights}")
            return weights

        if self._dtype == "str":
            return "any"
        if self._dtype == "bool":
            if all_values == {True}:
                return "unweighted"
            return "non-negative"
        else:
            min_val = min(all_values)
            if min_val < 0:
                return "any"
            elif min_val == 0:
                return "non-negative"
            else:
                if self._dtype == "int" and all_values == {1}:
                    return "unweighted"
                return "positive"

    @property
    def num_nodes(self):
        return self.value.vcount()

    @property
    def node_index(self):
        if self._node_index is None:
            self._node_index = SequentialNodes(self.num_nodes)
        return self._node_index

    def rebuild_for_node_index(self, node_index):
        """
        Returns a new instance based on `node_index`
        """
        if self.num_nodes != len(node_index):
            raise ValueError(
                f"Size of node_index ({len(node_index)}) must match num_nodes ({self.num_nodes})"
            )

        data = self.value
        if node_index != self.node_index:
            my_node_index = self.node_index
            my_node_index._verify_valid_conversion(node_index)
            index_converter = [node_index.bylabel(label) for label in my_node_index]
            g = igraph.Graph(data.vcount(), directed=data.is_directed())
            if data.is_weighted():
                for edge in data.es:
                    g.add_edge(
                        index_converter[edge.source],
                        index_converter[edge.target],
                        weight=edge["weight"],
                    )
            else:
                for edge in data.es:
                    g.add_edge(
                        index_converter[edge.source],
                        index_converter[edge.target],
                    )
            data = g
        return IGraph(
            data,
            dtype=self._dtype,
            weights=self._weights,
            node_index=node_index,
        )

    @classmethod
    def get_type(cls, obj):
        """Get an instance of this type class that describes obj"""
        if isinstance(obj, cls.value_type):
            ret_val = cls()
            ret_val.abstract_instance = cls.abstract(
                is_directed=obj.value.is_directed(),
                dtype=obj._dtype,
                weights=obj._weights,
            )
            return ret_val
        else:
            raise TypeError(f"object not of type {cls.__name__}")

    @classmethod
    def assert_equal(cls, obj1, obj2, *, rel_tol=1e-9, abs_tol=0.0, check_values=True):
        assert (
                type(obj1) is cls.value_type
        ), f"obj1 must be IGraph, not {type(obj1)}"
        assert (
                type(obj2) is cls.value_type
        ), f"obj2 must be IGraph, not {type(obj2)}"

        if check_values:
            assert obj1._dtype == obj2._dtype, f"{obj1._dtype} != {obj2._dtype}"
            assert obj1._weights == obj2._weights, f"{obj1._weights} != {obj2._weights}"
        g1 = obj1.value
        g2 = obj2.value
        assert g1.is_directed() == g2.is_directed(), f"{g1.is_directed()} != {g2.is_directed()}"
        assert g1.vcount() == g2.vcount(), f"{g1.vcount()} != {g2.vcount()}"
        assert g1.ecount() == g2.ecount(), f"{g1.ecount()} != {g2.ecount()}"
        # Convert to a common node indexing scheme
        obj2 = obj2.rebuild_for_node_index(obj1.node_index)
        g2 = obj2.value
        # Compare
        if check_values and obj1._weights != "unweighted":
            if obj1._dtype == "float":
                comp = partial(math.isclose, rel_tol=rel_tol, abs_tol=abs_tol)
                compstr = "close to"
            else:
                comp = operator.eq
                compstr = "equal to"
            for e1 in g1.es:
                try:
                    e2 = g2.es[g2.get_eid(*e1.tuple)]
                except igraph.InternalError:
                    raise AssertionError(f'Mismatched edge: {e1.tuple}')
                w1 = e1["weight"]
                w2 = e2["weight"]
                assert comp(w1, w2), f"{w1} not {compstr} {w2}"
        else:
            for e1 in g1.es:
                try:
                    g2.es[g2.get_eid(*e1.tuple)]
                except igraph.InternalError:
                    raise AssertionError(f'Mismatched edge: {e1.tuple}')
