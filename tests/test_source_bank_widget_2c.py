"""tests/test_source_bank_widget_2c.py

Multi-modal story schema STAGE 2 CHUNK 2C -- the `source_bank` selector
widget on OTR_LedgerScriptWriter (kibitz-converged plan,
kibitz-runs/2026-07-05-multimodal-2c/r4/final.md).

Pins:
  1. Widget surface: source_bank stays pinned at slot 23; choices come
     LIVE from the routing registry (exact list, registry order, including
     non-runnable custom banks -- the honest-error contract); default scifi_news.
  2. Registration fail-loud: a broken registry RAISES out of INPUT_TYPES
     (deliberate exception to the "INPUT_TYPES must never raise"
     convention; no baked-in fallback choice list).
  3. Gate-first ordering: a non-runnable source_bank pick raises
     StoryBankNotRunnableError as the FIRST act of run() -- before the
     story-scaffold env mutation and before _resolve_inputs (RSS fetch).
  4. Threading: resolve_creative_system_prompt(source_bank_id=...) selects
     the pack; compose_line/compose_line_draft thread it end-to-end, and
     every recursive compose_line self-call forwards it (AST pin).
  5. _resolve_inputs carries source_bank as the one authoritative value.
  6. Headless surface: source_bank is on both CREATIVE_WHITELISTs and
     patch_widget_by_name lands it at slot 23 of the canonical workflow
     (shifted -2 by the 2026-07-05 style-engine consolidation, which
     deleted the style / style_custom widgets).
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
_SCRIPTS = _REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from nodes import _otr_story_routing as routing  # noqa: E402
from nodes import OTR_LedgerScriptWriter as W_mod  # noqa: E402
from nodes.OTR_LedgerScriptWriter import (  # noqa: E402
    OTR_LedgerScriptWriter,
    _resolve_inputs,
)
from nodes._otr_creative_prompt_router import (  # noqa: E402
    resolve_creative_system_prompt,
)

_CANONICAL_WORKFLOW = _REPO / "workflows" / "otr_canonical.json"
_PUBLIC_DOMAIN_BANK = "public_domain"
_NON_RUNNABLE_BANK = "custom_source_bank"


# ---------------------------------------------------------------------------
# 1. Widget surface
# ---------------------------------------------------------------------------
class TestWidgetSurface:
    def test_source_bank_positional_pin(self):
        # Stage 3C (2026-07-06) appended visual_style after source_bank;
        # Google API (2026-07-08) appended its slot pair after visual_style;
        # Source Banks v2 appended source_ref after those.
        spec = OTR_LedgerScriptWriter.INPUT_TYPES()
        order = list(spec["required"].keys()) + list(spec["optional"].keys())
        assert order[23] == "source_bank"
        assert order[24] == "visual_style"
        assert order[25] == "google_api_slot_a_model"
        assert order[26] == "google_api_slot_b_model"
        assert order[27] == "source_ref"

    def test_choices_are_exactly_the_registry_in_order(self):
        spec = OTR_LedgerScriptWriter.INPUT_TYPES()
        choices, meta = spec["optional"]["source_bank"]
        assert choices == list(routing.list_bank_ids())
        assert meta["default"] == "scifi_news"
        # The honest-error contract: non-runnable custom banks ARE listed.
        assert _NON_RUNNABLE_BANK in choices
        assert _PUBLIC_DOMAIN_BANK in choices

    def test_default_is_a_runnable_bank(self):
        bank = routing.require_runnable_bank("scifi_news")
        assert bank.runnable is True


# ---------------------------------------------------------------------------
# 2. Registration fail-loud (no fallback choice list)
# ---------------------------------------------------------------------------
class TestRegistrationFailLoud:
    def test_broken_registry_raises_out_of_input_types(self, monkeypatch):
        def _boom():
            raise routing.RegistryValidationError(
                "test: banks.json unreadable")
        monkeypatch.setattr(W_mod._otr_story_routing, "list_bank_ids", _boom)
        with pytest.raises(routing.StoryRoutingError):
            OTR_LedgerScriptWriter.INPUT_TYPES()


# ---------------------------------------------------------------------------
# 3. Gate-first ordering
# ---------------------------------------------------------------------------
class TestGateFirst:
    def test_non_runnable_pick_raises_before_any_side_effect(
            self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            W_mod, "_apply_story_scaffold_env",
            lambda *_a, **_k: calls.append("scaffold_env") or "auto")
        monkeypatch.setattr(
            W_mod, "_resolve_inputs",
            lambda *_a, **_k: calls.append("resolve_inputs") or {})
        node = OTR_LedgerScriptWriter()
        with pytest.raises(routing.StoryBankNotRunnableError) as ei:
            node.run(source_bank=_NON_RUNNABLE_BANK)
        assert _NON_RUNNABLE_BANK in str(ei.value)
        assert calls == [], (
            f"run() performed side effects before the runnable gate: {calls}"
        )

    def test_unknown_bank_raises_unknown_bank_error(self):
        node = OTR_LedgerScriptWriter()
        with pytest.raises(routing.UnknownBankError):
            node.run(source_bank="no_such_bank")


# ---------------------------------------------------------------------------
# 4. Threading
# ---------------------------------------------------------------------------
# 5. Threading
# ---------------------------------------------------------------------------
class TestThreading:
    def test_resolver_routes_the_selected_bank(self):
        science = resolve_creative_system_prompt(
            "mistralai/Mistral-Nemo-Instruct-2407",
            phase="line_composer_system")
        other = resolve_creative_system_prompt(
            "mistralai/Mistral-Nemo-Instruct-2407",
            phase="line_composer_system",
            source_bank_id=_PUBLIC_DOMAIN_BANK)
        assert science != other, (
            "source_bank_id did not change the resolved prompt -- the "
            "widget would be dead"
        )
        # Cross-check against the lane pack on disk.
        pack_path = (_REPO / "nodes" / "story_packs" / _PUBLIC_DOMAIN_BANK /
                     "faithful_radio_adaptation.json")
        pack = json.loads(pack_path.read_text(encoding="utf-8"))
        assert other == pack["prompt_stages"]["line_composer_system"]

    def test_compose_line_threads_source_bank(self, monkeypatch):
        from nodes import _otr_line_composer as lc
        from nodes import _otr_creative_prompt_router as router
        seen = []
        real = router.resolve_creative_system_prompt

        def _spy(repo_id, phase, source_bank_id="media_archive"):
            seen.append(source_bank_id)
            return real(repo_id, phase, source_bank_id=source_bank_id)

        monkeypatch.setattr(
            router, "resolve_creative_system_prompt", _spy)
        req = lc.LineRequest(
            speaker="MARGOT",
            intent="steady the room",
            mood="calm",
            canon_header="",
            last_lines=[],
        )
        out = lc.compose_line(
            creative_fn=lambda *args, **kwargs: (
                "A quiet line about the machine."),
            req=req,
            creative_repo_id="mistralai/Mistral-Nemo-Instruct-2407",
            source_bank_id=_PUBLIC_DOMAIN_BANK,
        )
        assert out.text
        assert seen == [_PUBLIC_DOMAIN_BANK]


    def test_compose_line_forwards_bank_to_draft(self):
        src = (_REPO / "nodes" / "_otr_line_composer.py").read_text(
            encoding="utf-8")
        tree = ast.parse(src)
        fn = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "compose_line"
        )
        draft_calls = [
            call for call in ast.walk(fn)
            if isinstance(call, ast.Call)
            and getattr(call.func, "id", getattr(call.func, "attr", ""))
            == "compose_line_draft"
        ]
        assert len(draft_calls) == 1
        assert "source_bank_id" in {
            keyword.arg for keyword in draft_calls[0].keywords
        }


    def test_writer_call_sites_pass_the_resolved_bank(self):
        # AST pin: every _OTRLC.compose_line( call in the writer passes
        # source_bank_id.
        src = (_REPO / "nodes" / "OTR_LedgerScriptWriter.py").read_text(
            encoding="utf-8")
        tree = ast.parse(src)
        sites = 0
        for call in ast.walk(tree):
            if not isinstance(call, ast.Call):
                continue
            f = call.func
            if (isinstance(f, ast.Attribute) and f.attr == "compose_line"
                    and getattr(f.value, "id", "") == "_OTRLC"):
                kwarg_names = {k.arg for k in call.keywords}
                assert "source_bank_id" in kwarg_names, (
                    f"writer compose_line call at line {call.lineno} "
                    f"missing source_bank_id"
                )
                sites += 1
        assert sites == 2, f"expected 2 writer compose_line sites, {sites}"


# ---------------------------------------------------------------------------
# 6. Resolved surface
# ---------------------------------------------------------------------------
class TestResolvedSurface:
    def test_resolve_inputs_carries_source_bank(self):
        resolved = _resolve_inputs(custom_premise="test premise")
        assert resolved["source_bank"] == "scifi_news"
        resolved2 = _resolve_inputs(
            custom_premise="test premise", source_bank=_PUBLIC_DOMAIN_BANK)
        assert resolved2["source_bank"] == _PUBLIC_DOMAIN_BANK


# ---------------------------------------------------------------------------
# 7. Headless surface
# ---------------------------------------------------------------------------
class TestHeadlessSurface:
    def test_source_bank_on_both_whitelists(self):
        from nodes._otr_workflow_apply import CREATIVE_WHITELIST as pkg_wl
        import otr_api
        assert "source_bank" in pkg_wl
        assert "source_bank" in otr_api.CREATIVE_WHITELIST

    def test_patch_widget_by_name_lands_slot_23(self):
        import otr_api
        spec = OTR_LedgerScriptWriter.INPUT_TYPES()
        schemas = {
            "OTR_LedgerScriptWriter": {
                "input": {
                    "required": spec["required"],
                    "optional": spec["optional"],
                },
            },
        }
        workflow = otr_api.load_workflow(str(_CANONICAL_WORKFLOW))
        otr_api.patch_widget_by_name(
            workflow, 1, "source_bank", "scifi_news", schemas)
        node1 = next(n for n in workflow["nodes"] if n["id"] == 1)
        # The Google API selectors and source_ref were appended after
        # visual_style; source_bank stays at slot 23 (was 25 before the
        # style-engine consolidation). S5 platform-portability appended
        # the six llm runtime-policy widgets at 28-33 (vector = 34).
        assert len(node1["widgets_values"]) == 34
        assert node1["widgets_values"][23] == "scifi_news"
        assert node1["widgets_values"][25] == "(select Google API model)"
        assert node1["widgets_values"][26] == "(select Google API model)"
        assert node1["widgets_values"][27] == ""
        assert node1["widgets_values"][28] == "cuda"
        assert node1["widgets_values"][33] == "Q8_0"
