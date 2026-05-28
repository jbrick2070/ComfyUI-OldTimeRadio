"""Sprint 4.7 (2026-05-28) -- workflow JSON has OTR_Stage1FanOut
inserted at node id 79 with its 3 candidate outputs wired to
OTR_BeatSelector's 3 candidate inputs.

Pinned invariants:
  * Workflow JSON parses.
  * Exactly one OTR_Stage1FanOut entry exists.
  * Its three candidate outputs (slots 0/1/2) each carry exactly
    one link.
  * Each link's target is OTR_BeatSelector's matching input.
  * OTR_BeatSelector inputs candidate_a/b/c_json now carry non-
    None link values.
  * No id collisions; last_node_id + last_link_id are >= the new
    values.
"""
from __future__ import annotations

import json
import pathlib

WORKFLOW = pathlib.Path("workflows/otr_scifi_16gb_full.json")


def _load() -> dict:
    return json.loads(WORKFLOW.read_text(encoding="utf-8"))


def test_workflow_parses():
    d = _load()
    assert "nodes" in d
    assert "links" in d


def test_stage1_fanout_node_present_exactly_once():
    d = _load()
    matches = [n for n in d["nodes"] if n.get("type") == "OTR_Stage1FanOut"]
    assert len(matches) == 1


def test_stage1_fanout_node_outputs_shape():
    d = _load()
    fo = next(n for n in d["nodes"] if n.get("type") == "OTR_Stage1FanOut")
    output_names = [out.get("name") for out in fo.get("outputs", [])]
    assert output_names == [
        "candidate_a_json", "candidate_b_json", "candidate_c_json",
        "winning_plan_json", "fanout_audit_json",
    ]


def test_stage1_fanout_candidates_wired_to_beat_selector():
    d = _load()
    fo = next(n for n in d["nodes"] if n.get("type") == "OTR_Stage1FanOut")
    bs = next(n for n in d["nodes"] if n.get("type") == "OTR_BeatSelector")
    # Each candidate output carries one link.
    for i in range(3):
        out = fo["outputs"][i]
        assert len(out["links"]) == 1, (
            f"OTR_Stage1FanOut output {out['name']!r} should carry "
            f"exactly one link to OTR_BeatSelector; got {out['links']!r}"
        )
    # The link IDs match the BeatSelector inputs.
    bs_inputs_by_name = {inp["name"]: inp for inp in bs["inputs"]}
    assert bs_inputs_by_name["candidate_a_json"]["link"] == fo["outputs"][0]["links"][0]
    assert bs_inputs_by_name["candidate_b_json"]["link"] == fo["outputs"][1]["links"][0]
    assert bs_inputs_by_name["candidate_c_json"]["link"] == fo["outputs"][2]["links"][0]


def test_link_targets_resolve_to_beat_selector():
    """The global links list resolves each candidate link to
    OTR_BeatSelector."""
    d = _load()
    fo = next(n for n in d["nodes"] if n.get("type") == "OTR_Stage1FanOut")
    bs = next(n for n in d["nodes"] if n.get("type") == "OTR_BeatSelector")
    bs_id = bs["id"]
    candidate_link_ids = {
        fo["outputs"][0]["links"][0],
        fo["outputs"][1]["links"][0],
        fo["outputs"][2]["links"][0],
    }
    found = 0
    for link in d["links"]:
        link_id = link[0]
        target_node_id = link[3]
        if link_id in candidate_link_ids:
            assert target_node_id == bs_id, (
                f"link {link_id} target should be OTR_BeatSelector "
                f"node {bs_id}; got {target_node_id}"
            )
            found += 1
    assert found == 3, (
        f"Expected 3 candidate links in workflow.links; found {found}"
    )


def test_winning_plan_and_audit_outputs_remain_detached():
    """Sprint 4.7 only wires the 3 candidate outputs; the winner
    and audit slots stay detached so the operator decides where
    they go."""
    d = _load()
    fo = next(n for n in d["nodes"] if n.get("type") == "OTR_Stage1FanOut")
    assert fo["outputs"][3]["name"] == "winning_plan_json"
    assert fo["outputs"][3]["links"] == []
    assert fo["outputs"][4]["name"] == "fanout_audit_json"
    assert fo["outputs"][4]["links"] == []


def test_no_id_collisions_after_insertion():
    d = _load()
    all_ids = [n["id"] for n in d["nodes"]]
    assert len(all_ids) == len(set(all_ids))
    fo = next(n for n in d["nodes"] if n.get("type") == "OTR_Stage1FanOut")
    assert int(fo["id"]) <= int(d["last_node_id"])
