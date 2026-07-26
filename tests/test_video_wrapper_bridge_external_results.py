"""B1 -- ``run_graph(external_results=..., on_result=...)``.

The beat-scoped loader hoist: a multi-segment beat loads MODEL/CLIP/VAE ONCE in
``prepare()`` and every segment graph OMITS its loader nodes while keeping the
wires to them. Three things had to become true for that to work, and two of them
were silent killers that only surface on segment 1:

  * a wire may name a source the graph does not contain (it was a dangling-source
    error, and seeding Kahn's ``satisfied`` set is what stops it becoming a
    phantom cycle once the source is legal);
  * ``free_after_use`` must NOT evict a caller-owned handle -- segment 0's
    cleanup would otherwise drop it and segment 1 dies "output slot unavailable";
  * a caller needs each handle the MOMENT it exists, so a partial-load failure
    can unwind what actually landed.

Every test here has a CONTROL: the honest failures (a genuinely dangling wire, a
real cycle) must still fail, or the loosening would be a hole rather than a seam.
"""

from __future__ import annotations

import pytest

from nodes._otr_video_engines import wrapper_bridge as wb

W = wb.Wire


class _Add:
    FUNCTION = "go"

    def go(self, a, b):
        return (a + b,)


class _Double:
    FUNCTION = "run"

    def run(self, x):
        return (x * 2,)


class _Boom:
    FUNCTION = "run"

    def run(self, x):
        raise RuntimeError("segment blew up")


# --- the source is legal, and never executed ------------------------------- #
def test_external_result_is_a_legal_wire_source_and_is_never_executed():
    # "ckpt" is NOT in the graph and has no class -- if the executor tried to
    # run it this would raise "has no class".
    graph = {"d": {"class": _Double, "inputs": {"x": W("ckpt", 0)}}}
    out = wb.run_graph(graph, external_results={"ckpt": (21,)}, terminal="d")
    assert out == (42,)


def test_external_result_slot_selection_matches_a_real_node_output():
    graph = {"a": {"class": _Add,
                   "inputs": {"a": W("handles", 0), "b": W("handles", 2)}}}
    res = wb.run_graph(graph, external_results={"handles": (1, 99, 5)})
    assert res["a"] == (6,)


def test_external_result_is_returned_alongside_executed_nodes():
    graph = {"d": {"class": _Double, "inputs": {"x": W("ckpt", 0)}}}
    res = wb.run_graph(graph, external_results={"ckpt": (3,)})
    assert res["ckpt"] == (3,)
    assert res["d"] == (6,)


def test_a_bare_external_value_is_normalised_to_a_one_tuple():
    graph = {"d": {"class": _Double, "inputs": {"x": W("ckpt", 0)}}}
    res = wb.run_graph(graph, external_results={"ckpt": 7})
    assert res["ckpt"] == (7,)
    assert res["d"] == (14,)


# --- THE SEGMENT-1 KILLER -------------------------------------------------- #
def test_free_after_use_may_not_evict_an_external_handle():
    """Segment 0 must not free what segment 1 renders from.

    ``d`` is the only consumer of the external, so its remaining-consumer count
    hits zero the moment ``d`` runs. Before the fix that deleted the caller's
    MODEL handle mid-beat.
    """
    graph = {"d": {"class": _Double, "inputs": {"x": W("ckpt", 0)}},
             "t": {"class": _Double, "inputs": {"x": W("d", 0)}}}
    res = wb.run_graph(graph, external_results={"ckpt": (4,)},
                       free_after_use=True)
    assert res["ckpt"] == (4,), "the caller's handle was evicted mid-beat"
    assert res["t"] == (16,)
    assert "d" not in res               # the ordinary intermediate DID free


def test_the_same_external_handle_survives_repeated_graph_runs():
    """The real multi-segment shape: ONE handle dict, three segment graphs."""
    handles = {"ckpt": (5,)}
    seen = []
    for _ in range(3):
        graph = {"d": {"class": _Double, "inputs": {"x": W("ckpt", 0)}}}
        seen.append(wb.run_graph(graph, external_results=handles,
                                 free_after_use=True, terminal="d"))
    assert seen == [(10,), (10,), (10,)]
    assert handles == {"ckpt": (5,)}    # caller's dict never mutated


# --- collisions ------------------------------------------------------------ #
def test_an_id_in_both_graph_and_external_results_is_named_not_silent():
    graph = {"ckpt": {"class": _Double, "inputs": {"x": 1}}}
    with pytest.raises(wb.GraphExecutionError) as e:
        wb.run_graph(graph, external_results={"ckpt": (1,)})
    assert "ckpt" in str(e.value) and "collides" in str(e.value)


# --- on_result ------------------------------------------------------------- #
def test_on_result_fires_per_executed_node_with_the_normalised_tuple():
    graph = {"a": {"class": _Add, "inputs": {"a": 1, "b": 2}},
             "d": {"class": _Double, "inputs": {"x": W("a", 0)}}}
    seen = []
    wb.run_graph(graph, on_result=lambda nid, out: seen.append((nid, out)))
    assert seen == [("a", (3,)), ("d", (6,))]


def test_on_result_does_NOT_fire_for_an_external():
    graph = {"d": {"class": _Double, "inputs": {"x": W("ckpt", 0)}}}
    seen = []
    wb.run_graph(graph, external_results={"ckpt": (2,)},
                 on_result=lambda nid, out: seen.append(nid))
    assert seen == ["d"]                # never "ckpt"


def test_on_result_fires_before_the_graph_aborts_so_a_partial_load_can_unwind():
    """The transaction hook: what landed before the failure must be knowable."""
    graph = {"a": {"class": _Add, "inputs": {"a": 1, "b": 2}},
             "boom": {"class": _Boom, "inputs": {"x": W("a", 0)}}}
    seen = []
    with pytest.raises(wb.GraphExecutionError):
        wb.run_graph(graph, on_result=lambda nid, out: seen.append(nid))
    assert seen == ["a"]                # "a" landed and the caller knows it


def test_a_raising_on_result_aborts_the_graph_NAMED():
    def _refuse(nid, out):
        raise ValueError("cannot take ownership")

    graph = {"a": {"class": _Add, "inputs": {"a": 1, "b": 2}}}
    with pytest.raises(wb.GraphExecutionError) as e:
        wb.run_graph(graph, on_result=_refuse)
    msg = str(e.value)
    assert "on_result" in msg and "a" in msg and "cannot take ownership" in msg


# --- CONTROLS: the honest failures must still fail ------------------------- #
def test_CONTROL_a_genuinely_dangling_wire_still_raises():
    graph = {"d": {"class": _Double, "inputs": {"x": W("nope", 0)}}}
    with pytest.raises(wb.GraphExecutionError) as e:
        wb.run_graph(graph)
    assert "unknown source" in str(e.value)


def test_CONTROL_a_dangling_wire_still_raises_even_WITH_externals_declared():
    """Declaring one external must not wave every unknown source through."""
    graph = {"d": {"class": _Double, "inputs": {"x": W("typo", 0)}}}
    with pytest.raises(wb.GraphExecutionError) as e:
        wb.run_graph(graph, external_results={"ckpt": (1,)})
    assert "unknown source" in str(e.value) and "typo" in str(e.value)


def test_CONTROL_a_real_cycle_still_raises_a_cycle():
    graph = {"a": {"class": _Double, "inputs": {"x": W("b", 0)}},
             "b": {"class": _Double, "inputs": {"x": W("a", 0)}}}
    with pytest.raises(wb.GraphExecutionError) as e:
        wb.run_graph(graph)
    assert "cycle" in str(e.value)


def test_CONTROL_free_after_use_still_frees_an_ordinary_intermediate():
    """The loosening is scoped to externals -- normal freeing is untouched."""
    graph = {"a": {"class": _Add, "inputs": {"a": 1, "b": 2}},
             "d": {"class": _Double, "inputs": {"x": W("a", 0)}}}
    res = wb.run_graph(graph, free_after_use=True)
    assert "a" not in res               # freed once its only consumer ran
    assert res["d"] == (6,)


def test_CONTROL_no_externals_behaves_exactly_as_before():
    graph = {"a": {"class": _Add, "inputs": {"a": 2, "b": 3}},
             "d": {"class": _Double, "inputs": {"x": W("a", 0)}}}
    assert wb.run_graph(graph, terminal="d") == (10,)
    assert wb.run_graph(graph, external_results=None, on_result=None,
                        terminal="d") == (10,)
