"""tests/test_lfc_c2_freeze_verdict_preview.py

LFC commit 12.8 -- C2 wire freeze_verdict to a ShowText preview.

The cascade's freeze_verdict OUTPUT STRING needs a downstream
consumer so operators can see the cascade's verdict on the canvas
without opening the saved ledger JSON. Wiring to ShowText|pysssss
is the de-facto ComfyUI pattern for STRING preview.

Test contract:
  - cascade(62).freeze_verdict OUTPUT has at least one link.
  - The link's target is a ShowText* node.
  - The ShowText node's `text` input wires back to cascade slot 4.
  - last_link_id >= the new link id.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO_ROOT = Path(__file__).resolve().parents[1]
WF_PATH = REPO_ROOT / "workflows" / "otr_scifi_16gb_full.json"

CASCADE_NODE_ID = 62


def _load_wf():
    return json.loads(WF_PATH.read_text(encoding="utf-8"))


def _cascade(wf):
    return next(n for n in wf["nodes"] if n.get("id") == CASCADE_NODE_ID)


class TestC2FreezeVerdictPreview:
    def test_freeze_verdict_output_has_link(self):
        wf = _load_wf()
        cascade = _cascade(wf)
        fv = next(
            o for o in cascade["outputs"]
            if o["name"] == "freeze_verdict"
        )
        assert fv.get("links"), (
            "C2: cascade.freeze_verdict output should have at least "
            "one downstream link (preview node)"
        )

    def test_freeze_verdict_target_is_show_text(self):
        wf = _load_wf()
        cascade = _cascade(wf)
        fv = next(
            o for o in cascade["outputs"]
            if o["name"] == "freeze_verdict"
        )
        fv_link_ids = fv.get("links") or []
        assert fv_link_ids

        link_objs = wf.get("links") or []
        link_by_id = {l[0]: l for l in link_objs}

        nodes_by_id = {n["id"]: n for n in wf["nodes"]}

        for link_id in fv_link_ids:
            link = link_by_id.get(link_id)
            assert link is not None, (
                f"C2: cascade.freeze_verdict references link {link_id} "
                f"but link object is missing from workflow.links"
            )
            # Link format: [id, source, source_slot, target, target_slot, type]
            target_id = link[3]
            target_node = nodes_by_id.get(target_id)
            assert target_node is not None
            node_type = target_node.get("type", "")
            # Accept any ShowText variant -- ShowText|pysssss is the
            # de-facto pattern; core ShowText also acceptable.
            assert "ShowText" in node_type, (
                f"C2: cascade.freeze_verdict wired to {node_type!r} "
                f"(node {target_id}); expected a ShowText* preview node"
            )

    def test_preview_node_text_input_wires_to_cascade_slot_4(self):
        wf = _load_wf()
        cascade = _cascade(wf)
        fv = next(
            o for o in cascade["outputs"]
            if o["name"] == "freeze_verdict"
        )
        fv_link_ids = fv.get("links") or []
        link_objs = wf.get("links") or []
        link_by_id = {l[0]: l for l in link_objs}

        for link_id in fv_link_ids:
            link = link_by_id[link_id]
            # link[1] = source node, link[2] = source slot
            assert link[1] == CASCADE_NODE_ID, (
                f"C2: link {link_id} source node should be cascade "
                f"({CASCADE_NODE_ID}), got {link[1]}"
            )
            assert link[2] == 4, (
                f"C2: link {link_id} source slot should be 4 "
                f"(freeze_verdict), got {link[2]}"
            )

    def test_last_link_id_covers_new_preview_link(self):
        wf = _load_wf()
        last = wf.get("last_link_id", 0)
        max_present = max(
            (l[0] for l in (wf.get("links") or [])),
            default=0,
        )
        assert last >= max_present, (
            f"last_link_id={last} < max link id present "
            f"({max_present})"
        )
