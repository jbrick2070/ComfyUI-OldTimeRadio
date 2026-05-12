"""tests/test_lfc_c9_estimated_minutes_preview.py

LFC commit 12.9 -- C9 decision: cascade.estimated_minutes (slot 3)
is WIRED to a ShowText preview (not removed). Useful for soak
observation of episode-length drift across runs.
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


class TestC9EstimatedMinutesPreview:
    def test_estimated_minutes_output_has_link(self):
        wf = _load_wf()
        cascade = _cascade(wf)
        em = next(
            o for o in cascade["outputs"]
            if o["name"] == "estimated_minutes"
        )
        assert em.get("links"), (
            "C9: cascade.estimated_minutes output should have at "
            "least one downstream link -- the C9 decision is WIRE "
            "(not remove)"
        )

    def test_estimated_minutes_target_is_show_text(self):
        wf = _load_wf()
        cascade = _cascade(wf)
        em = next(
            o for o in cascade["outputs"]
            if o["name"] == "estimated_minutes"
        )
        link_objs = wf.get("links") or []
        link_by_id = {l[0]: l for l in link_objs}
        nodes_by_id = {n["id"]: n for n in wf["nodes"]}

        for link_id in em["links"]:
            link = link_by_id[link_id]
            target = nodes_by_id[link[3]]
            assert "ShowText" in target.get("type", ""), (
                f"C9: estimated_minutes wired to "
                f"{target.get('type')!r}; expected ShowText* preview"
            )
            # Source confirms cascade slot 3.
            assert link[1] == CASCADE_NODE_ID
            assert link[2] == 3
