"""S16.4: OTR_BatchFluxPortraitRender.ledger_json must be wired
from FreezeCascade's script_json fanout. Pins the wire-in so a
future hand-edit cannot regress it back to ``link: null``.
"""
from __future__ import annotations

import json
import pathlib


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CANONICAL_WORKFLOW = REPO_ROOT / "workflows" / "otr_scifi_16gb_full.json"


def test_flux_portrait_ledger_wired():
    wf = json.loads(CANONICAL_WORKFLOW.read_text(encoding="utf-8"))
    portrait = [
        n for n in wf["nodes"]
        if n.get("type") == "OTR_BatchFluxPortraitRender"
    ]
    assert portrait, "no OTR_BatchFluxPortraitRender node in workflow"
    node = portrait[0]
    ledger_input = next(
        (i for i in (node.get("inputs") or [])
         if i.get("name") == "ledger_json"),
        None,
    )
    assert ledger_input is not None, (
        "OTR_BatchFluxPortraitRender has no ledger_json input -- "
        "either the input was removed from INPUT_TYPES() or the "
        "workflow JSON dropped the socket."
    )
    assert ledger_input.get("link") is not None, (
        "OTR_BatchFluxPortraitRender.ledger_json link is None. "
        "S16.4 wired this from OTR_LedgerFreezeCascade.script_json "
        "(slot 1). Re-wire before merge."
    )


def test_flux_portrait_link_source_is_freeze_cascade():
    """The link that feeds ledger_json must originate at the
    OTR_LedgerFreezeCascade node's ``script_json`` output (slot 1).
    Defensive: catches the case where someone wires a foreign source
    (e.g. directly from ScriptWriter, skipping the freeze gate).
    """
    wf = json.loads(CANONICAL_WORKFLOW.read_text(encoding="utf-8"))
    nodes_by_id = {n.get("id"): n for n in wf["nodes"]}
    portrait = next(
        n for n in wf["nodes"]
        if n.get("type") == "OTR_BatchFluxPortraitRender"
    )
    ledger_in = next(
        i for i in portrait["inputs"] if i.get("name") == "ledger_json"
    )
    link_id = ledger_in["link"]
    link = next(L for L in wf["links"] if L[0] == link_id)
    src_node = nodes_by_id.get(link[1])
    assert src_node is not None, f"link {link_id} src node missing"
    assert src_node.get("type") == "OTR_LedgerFreezeCascade", (
        f"FluxPortrait.ledger_json must source from FreezeCascade; "
        f"got {src_node.get('type')!r}."
    )
    src_outputs = src_node.get("outputs") or []
    assert link[2] < len(src_outputs), (
        f"FreezeCascade src slot {link[2]} out of range "
        f"(node has {len(src_outputs)} outputs)."
    )
    assert src_outputs[link[2]].get("name") == "script_json", (
        f"FluxPortrait.ledger_json must source from FreezeCascade's "
        f"script_json output; got "
        f"{src_outputs[link[2]].get('name')!r}."
    )
