from metagraph.wrappers import GraphWrapper
from metagraph.types import Graph
from metagraph.core.dtypes import dtypes_simplified
import igraph
import math
import operator
from functools import partial
from typing import Set, Dict, Any
import numpy as np


class IGraph(GraphWrapper, abstract=Graph):
    def __init__(
        self,
        graph,
        node_ids=None,
        node_weight_label="weight",
        edge_weight_label="weight",
        *,
        aprops=None,
    ):
        """
        :param graph: an igraph.Graph object
        :param node_ids: list of NodeIDs corresponding to the graph's vertex ids
        :param node_weight_label: default is "weight"
        :param edge_weight_label: default is "weight"

        The node_ids will be used to add a "NodeId" vertex attribute to a copy of the graph.
        Manually adding the "NodeId" vertex attribute will avoid the copy and achieves the same result.
        """
        super().__init__(aprops=aprops)
        self._assert_instance(graph, igraph.Graph)
        self.value = graph
        self.node_weight_label = node_weight_label
        self.edge_weight_label = edge_weight_label

        if node_ids is not None:
            self._assert(
                graph.vcount() == len(node_ids),
                f"node_list size ({len(node_ids)}) and # of nodes in graph ({graph.vcount()}) don't match.",
            )
            # Make a copy to avoid mutating the input
            self.value = graph.copy()
            self.value.vs["NodeId"] = node_ids

        if "NodeId" in self.value.vs.attributes():
            self._is_sequential = False
            self._nodeid_lookup = {v["NodeId"]: v.index for v in self.value.vs}
            self._assert(
                len(self._nodeid_lookup) == self.value.vcount(),
                "node_ids are not unique",
            )
        else:
            self._is_sequential = True

    def is_sequential(self):
        return self._is_sequential

    class TypeMixin:
        @classmethod
        def _compute_abstract_properties(
            cls, obj, props: Set[str], known_props: Dict[str, Any]
        ) -> Dict[str, Any]:
            ret = known_props.copy()

            # fast properties
            for prop in {"is_directed", "node_type", "edge_type"} - ret.keys():
                if prop == "is_directed":
                    ret[prop] = obj.value.is_directed()
                elif prop == "node_type":
                    if obj.node_weight_label in obj.value.vs.attribute_names():
                        ret[prop] = "map"
                    else:
                        ret[prop] = "set"
                        ret["node_dtype"] = None
                elif prop == "edge_type":
                    if obj.edge_weight_label in obj.value.es.attribute_names():
                        ret[prop] = "map"
                    else:
                        ret[prop] = "set"
                        ret["edge_dtype"] = None
                        ret["edge_has_negative_weights"] = None

            # slow properties, only compute if needed
            slow_props = props - ret.keys()
            if {"edge_dtype", "edge_has_negative_weights"} & slow_props:
                evals = obj.value.es[obj.edge_weight_label]
                earr = np.array(evals)
                ret["edge_dtype"] = dtypes_simplified.get(earr.dtype, "str")
                if "edge_has_negative_weights" in slow_props:
                    if ret["edge_dtype"] in {"bool", "str"}:
                        neg_weights = None
                    else:
                        neg_weights = earr.min() < 0
                    ret["edge_has_negative_weights"] = neg_weights
            if "node_dtype" in slow_props:
                vvals = obj.value.vs[obj.node_weight_label]
                varr = np.array(vvals)
                ret["node_dtype"] = dtypes_simplified.get(varr.dtype, "str")

            return ret

        @classmethod
        def assert_equal(
            cls,
            obj1,
            obj2,
            aprops1,
            aprops2,
            cprops1,
            cprops2,
            *,
            rel_tol=1e-9,
            abs_tol=0.0,
        ):
            assert aprops1 == aprops2, f"property mismatch: {aprops1} != {aprops2}"
            g1 = obj1.value
            g2 = obj2.value
            seq1 = obj1.is_sequential()
            seq2 = obj2.is_sequential()
            nid1 = set(g1.vs.indices) if seq1 else set(obj1._nodeid_lookup)
            nid2 = set(g2.vs.indices) if seq2 else set(obj2._nodeid_lookup)
            v1 = g1.vs
            if not seq1:
                sorter = np.array(v1["NodeId"]).argsort()
                v1 = v1[sorter.tolist()]
            v2 = g2.vs
            if not seq2:
                sorter = np.array(g2.vs["NodeId"]).argsort()
                v2 = v2[sorter.tolist()]
            # Compare
            assert (
                g1.ecount() == g2.ecount()
            ), f"num edges mismatch: {g1.ecount()} != {g2.ecount()}"
            assert (
                g1.vcount() == g2.vcount()
            ), f"num node mismatch: {g1.vcount()} != {g2.vcount()}"
            assert nid1 == nid2, f"node id mismatch: {nid1 ^ nid2}"

            if aprops1.get("node_type") == "map":
                v1vals = np.array(
                    v1[obj1.node_weight_label], dtype=aprops1["node_dtype"]
                )
                v2vals = np.array(
                    v2[obj2.node_weight_label], dtype=aprops2["node_dtype"]
                )
                if aprops1["node_dtype"] == "float":
                    assert np.isclose(v1vals, v2vals, rtol=rel_tol, atol=abs_tol)
                else:
                    assert (v1vals == v2vals).all()

            if aprops1.get("edge_type") == "map":
                if aprops1["edge_dtype"] == "float":
                    comp = partial(math.isclose, rel_tol=rel_tol, abs_tol=abs_tol)
                    compstr = "close to"
                else:
                    comp = operator.eq
                    compstr = "equal to"
                for e1 in g1.es:
                    sid = e1.source if seq1 else e1.source_vertex["NodeId"]
                    tid = e1.target if seq1 else e1.target_vertex["NodeId"]
                    s2 = sid if seq2 else obj2._nodeid_lookup[sid]
                    t2 = tid if seq2 else obj2._nodeid_lookup[tid]
                    try:
                        e2 = g2.es[g2.get_eid(s2, t2)]
                    except igraph.InternalError:
                        raise AssertionError(f"Mismatched edge: {sid, tid}")
                    w1 = e1[obj1.edge_weight_label]
                    w2 = e2[obj2.edge_weight_label]
                    assert comp(w1, w2), f"{w1} not {compstr} {w2}"
