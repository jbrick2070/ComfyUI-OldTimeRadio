"""Sprint E E7 / R-8: VideoPlan freeze_done_gate rename + fail-early guard.

Closes M4. Pre-E7 the VideoPlan input was named `audio_gate` but the
post-D0d wiring sources from FreezeCascade.script_json (not from an
audio node). The misleading name was queued for rename in the Sprint
E plan §3 W-6. E7 lands the rename:

- nodes/otr_video_plan.py: INPUT_TYPES key renamed audio_gate -> freeze_done_gate;
  plan() signature renamed; em-dashes purged from comments.
- workflows/otr_scifi_16gb_full.json: node 20 input socket name updated.
- Fail-early guard: if script_json is empty/stub AND freeze_done_gate
  unwired, return sentinel error JSON instead of silently emitting
  empty plans.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE = REPO_ROOT / "nodes" / "otr_video_plan.py"
CANONICAL_JSON = REPO_ROOT / "workflows" / "otr_scifi_16gb_full.json"


def _src() -> str:
    return SOURCE.read_text(encoding="utf-8")


class TestInputRenamed:
    def test_input_types_uses_new_name(self):
        from nodes.otr_video_plan import OTRVideoPlan
        schema = OTRVideoPlan.INPUT_TYPES()
        optional = schema.get("optional", {})
        assert "freeze_done_gate" in optional, (
            "VideoPlan optional inputs must declare freeze_done_gate "
            "(Sprint E E7 rename)"
        )

    def test_input_types_drops_old_name(self):
        from nodes.otr_video_plan import OTRVideoPlan
        schema = OTRVideoPlan.INPUT_TYPES()
        optional = schema.get("optional", {})
        assert "audio_gate" not in optional, (
            "VideoPlan must NOT carry the legacy audio_gate name "
            "post-E7; CLAUDE.md no-back-compat rule"
        )

    def test_plan_signature_uses_new_name(self):
        from nodes.otr_video_plan import OTRVideoPlan
        import inspect
        sig = inspect.signature(OTRVideoPlan.plan)
        assert "freeze_done_gate" in sig.parameters
        assert "audio_gate" not in sig.parameters

    def test_input_tooltip_names_cascade_source(self):
        src = _src()
        # Tooltip must describe the actual wiring source.
        assert "OTR_LedgerFreezeCascade.script_json" in src
        assert "Sprint E E7 rename" in src


class TestWorkflowJsonRewire:
    def test_workflow_json_input_renamed(self):
        wf = json.loads(CANONICAL_JSON.read_text(encoding="utf-8"))
        n20 = [n for n in wf["nodes"] if n["id"] == 20][0]
        input_names = [i.get("name") for i in n20.get("inputs", [])]
        assert "freeze_done_gate" in input_names
        assert "audio_gate" not in input_names

    def test_workflow_json_l47_still_wired(self):
        """The link itself stays at id=47; only the socket name on the
        target side changes. Source (62.1) is unchanged."""
        wf = json.loads(CANONICAL_JSON.read_text(encoding="utf-8"))
        links = {l[0]: l for l in wf["links"]}
        assert 47 in links
        l47 = links[47]
        assert l47[1] == 62  # source node still FreezeCascade
        assert l47[2] == 1   # source slot still script_json
        assert l47[3] == 20  # target node still VideoPlan


class TestFailEarlyGuard:
    def test_empty_script_json_returns_sentinel(self):
        from nodes.otr_video_plan import OTRVideoPlan
        node = OTRVideoPlan()
        outs = node.plan(
            script_json="",
            focus_character="(all)",
            shots_per_scene=3,
        )
        assert len(outs) == 5
        # All 3 prompt outputs carry the sentinel error JSON.
        for i in range(3):
            err = json.loads(outs[i])
            assert "error" in err
            assert "freeze_done_gate" in err["error"]
        # pass3_prompt_count == 0
        assert outs[3] == 0
        # debug_summary names the missing-cascade condition
        assert "missing upstream cascade" in outs[4]

    def test_stub_script_json_returns_sentinel(self):
        from nodes.otr_video_plan import OTRVideoPlan
        node = OTRVideoPlan()
        for stub in ("{}", "[]", "   "):
            outs = node.plan(
                script_json=stub,
                focus_character="(all)",
                shots_per_scene=3,
            )
            err = json.loads(outs[0])
            assert "error" in err

    def test_no_em_dashes_in_source(self):
        """CLAUDE.md project rule + feedback memory: no em-dashes in
        OTR Python source. The E7 rename touched two em-dash comments
        in this file; verify they were purged."""
        src = _src()
        assert "—" not in src, (
            "Em-dash found in nodes/otr_video_plan.py -- forbidden "
            "by CLAUDE.md project rule (cp1252 subprocess decode trap)"
        )
