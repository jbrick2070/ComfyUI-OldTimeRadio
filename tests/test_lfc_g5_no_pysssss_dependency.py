"""tests/test_lfc_g5_no_pysssss_dependency.py

LFC commit 12.15 -- G5 fix regression. v2.0-alpha is a
foundational alpha; a third-party UI extension (`pysssss`) for
basic verdict telemetry is the wrong shape. The preview nodes
(63 + 64 in earlier commits) were removed; freeze_verdict +
estimated_minutes outputs stay on the cascade node so operators
who DO have pysssss installed can wire previews themselves -- the
core workflow no longer carries the dep.

This test pins the absence of the dep + the presence of the
cascade outputs that still expose the data.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO_ROOT = Path(__file__).resolve().parents[1]
WF_PATH = REPO_ROOT / "workflows" / "otr_canonical.json"

CASCADE_NODE_ID = 62


def _load_wf():
    return json.loads(WF_PATH.read_text(encoding="utf-8"))


class TestG5NoPysssssDependency:
    def test_no_showtext_pysssss_node_in_workflow(self):
        """The core workflow must not require pysssss for basic
        telemetry. Operators who DO have pysssss installed can
        wire their own preview on the cascade outputs; the
        canonical workflow stays portable to a clean ComfyUI core."""
        wf = _load_wf()
        pysssss_nodes = [
            n for n in wf["nodes"]
            if "pysssss" in (n.get("type") or "").lower()
        ]
        assert pysssss_nodes == [], (
            f"G5: pysssss-typed nodes present in core workflow: "
            f"{[(n['id'], n['type']) for n in pysssss_nodes]}"
        )

    def test_freeze_verdict_output_still_exists_unwired(self):
        """Output port stays on the cascade so operators can wire a
        preview themselves -- just not in the core workflow."""
        wf = _load_wf()
        cascade = next(
            n for n in wf["nodes"] if n.get("id") == CASCADE_NODE_ID
        )
        fv = next(
            o for o in cascade["outputs"] if o["name"] == "freeze_verdict"
        )
        assert fv["type"] == "STRING"
        # links may be empty (no consumer) -- that's the G5 contract.
        # The output PORT remains so operator can wire as needed.
        assert "links" in fv

    def test_estimated_minutes_output_still_exists_unwired(self):
        wf = _load_wf()
        cascade = next(
            n for n in wf["nodes"] if n.get("id") == CASCADE_NODE_ID
        )
        em = next(
            o for o in cascade["outputs"] if o["name"] == "estimated_minutes"
        )
        assert em["type"] == "INT"
        assert "links" in em

    def test_news_used_chain_unaffected_by_g5(self):
        """W2's news_used chain (writer -> cascade -> SignalLostVideo)
        was wired via link 110 in commit 12.1. G5 dropped 111 + 112
        but must NOT touch 110."""
        wf = _load_wf()
        link_ids = {l[0] for l in wf.get("links") or []}
        assert 110 in link_ids, (
            "G5 dropped pysssss links but must preserve link 110 "
            "(W2 news_used chain)"
        )
        assert 111 not in link_ids
        assert 112 not in link_ids

    def test_last_link_id_consistent_after_drop(self):
        wf = _load_wf()
        last = wf.get("last_link_id", 0)
        live_max = max(
            (l[0] for l in wf.get("links") or []),
            default=0,
        )
        assert last >= live_max, (
            f"last_link_id={last} < max live link id {live_max}"
        )

    def test_no_dropped_link_id_orphans_remain(self):
        """No socket-level link reference (input.link or
        output.links) should still point at 111 or 112."""
        wf = _load_wf()
        for node in wf["nodes"]:
            for inp in node.get("inputs", []) or []:
                link = inp.get("link")
                assert link not in (111, 112), (
                    f"node {node['id']} input {inp.get('name')!r} "
                    f"still references dropped link {link}"
                )
            for out in node.get("outputs", []) or []:
                for link_id in (out.get("links") or []):
                    assert link_id not in (111, 112), (
                        f"node {node['id']} output {out.get('name')!r} "
                        f"still references dropped link {link_id}"
                    )
