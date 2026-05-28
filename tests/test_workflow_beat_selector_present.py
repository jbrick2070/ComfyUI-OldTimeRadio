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


def test_beat_selector_is_detached():
    """Sprint 4.2 invariant: the node is registered but NOT wired
    to any upstream / downstream. Operator wires it manually once
    Sprint 4.3 (Stage 1 fan-out) lands."""
    d = _load()
    bs = next(n for n in d["nodes"] if n.get("type") == "OTR_BeatSelector")
    for inp in bs.get("inputs", []):
        assert inp.get("link") is None, (
            f"OTR_BeatSelector input {inp.get('name')!r} should be "
            "detached (link == None); got "
            f"{inp.get('link')!r}"
        )
    for out in bs.get("outputs", []):
        # Empty list OR null is acceptable for detached outputs.
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
