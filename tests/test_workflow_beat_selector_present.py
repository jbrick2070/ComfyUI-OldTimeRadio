"""Sprint 4.2 (2026-05-28) -- workflow JSON carries OTR_BeatSelector.

The node was registered in Sprint 4.1. Sprint 4.2 inserts it into
the canonical workflow JSON in detached mode -- no upstream/
downstream links so live behaviour is unchanged. The operator
wires it in once the Sprint 4.3 Stage 1 fan-out lands.

Pinned invariants:
  * Workflow JSON parses.
  * Exactly one OTR_BeatSelector node is present.
  * It is detached: every input link is None and every output
    links list is empty.
  * last_node_id is >= the BeatSelector's id (no id collisions).
"""
from __future__ import annotations

import json
import pathlib


WORKFLOW = pathlib.Path("workflows/otr_scifi_16gb_full.json")


def _load() -> dict:
    return json.loads(WORKFLOW.read_text(encoding="utf-8"))


def test_workflow_parses():
    """Sanity: the canonical workflow JSON is still well-formed."""
    d = _load()
    assert isinstance(d, dict)
    assert "nodes" in d
    assert "last_node_id" in d


def test_beat_selector_node_present():
    """Exactly one OTR_BeatSelector entry exists."""
    d = _load()
    matches = [n for n in d["nodes"] if n.get("type") == "OTR_BeatSelector"]
    assert len(matches) == 1


def test_beat_selector_inputs_wired_to_stage1_fanout():
    """Sprint 4.7 (2026-05-28) -- the BeatSelector's three
    candidate inputs are now wired to OTR_Stage1FanOut's three
    candidate outputs. Outputs remain detached so the operator
    decides where the winner / audit go.

    (Sprint 4.2 originally shipped the node detached; Sprint 4.7
    upgraded the wiring once Sprint 4.8's OTR_Stage1FanOut node
    landed.)
    """
    d = _load()
    bs = next(n for n in d["nodes"] if n.get("type") == "OTR_BeatSelector")
    bs_inputs_by_name = {inp["name"]: inp for inp in bs.get("inputs", [])}
    for cand in ("candidate_a_json", "candidate_b_json", "candidate_c_json"):
        assert bs_inputs_by_name[cand]["link"] is not None, (
            f"Sprint 4.7: OTR_BeatSelector input {cand!r} should "
            f"carry a link from OTR_Stage1FanOut; got "
            f"{bs_inputs_by_name[cand]['link']!r}"
        )
    for out in bs.get("outputs", []):
        links = out.get("links") or []
        assert links == [], (
            f"OTR_BeatSelector output {out.get('name')!r} should be "
            f"detached (links empty); got {links!r}"
        )


def test_beat_selector_has_three_string_inputs():
    """candidate_a_json + candidate_b_json + candidate_c_json."""
    d = _load()
    bs = next(n for n in d["nodes"] if n.get("type") == "OTR_BeatSelector")
    input_names = [inp.get("name") for inp in bs.get("inputs", [])]
    assert input_names == [
        "candidate_a_json", "candidate_b_json", "candidate_c_json",
    ]
    assert all(
        inp.get("type") == "STRING" for inp in bs.get("inputs", [])
    )


def test_beat_selector_has_two_string_outputs():
    """winning_plan_json + selector_audit."""
    d = _load()
    bs = next(n for n in d["nodes"] if n.get("type") == "OTR_BeatSelector")
    output_names = [out.get("name") for out in bs.get("outputs", [])]
    assert output_names == ["winning_plan_json", "selector_audit"]
    assert all(
        out.get("type") == "STRING" for out in bs.get("outputs", [])
    )


def test_beat_selector_id_within_last_node_id():
    """No id collisions."""
    d = _load()
    bs = next(n for n in d["nodes"] if n.get("type") == "OTR_BeatSelector")
    assert int(bs["id"]) <= int(d["last_node_id"])
    # The id must be unique across all nodes.
    all_ids = [n["id"] for n in d["nodes"]]
    assert len(all_ids) == len(set(all_ids))
