"""tests/test_source_bank_widget_2c.py

Multi-modal story schema STAGE 2 CHUNK 2C -- the `source_bank` selector
widget on OTR_LedgerScriptWriter (kibitz-converged plan,
kibitz-runs/2026-07-05-multimodal-2c/r4/final.md).

Pins:
  1. Widget surface: source_bank is the LAST optional entry; choices come
     LIVE from the routing registry (exact list, registry order, including
     non-runnable banks -- the honest-error contract); default science_news.
  2. Registration fail-loud: a broken registry RAISES out of INPUT_TYPES
     (deliberate exception to the "INPUT_TYPES must never raise"
     convention; no baked-in fallback choice list).
  3. Gate-first ordering: a non-runnable source_bank pick raises
     StoryBankNotRunnableError as the FIRST act of run() -- before the
     story-scaffold env mutation and before _resolve_inputs (RSS fetch).
  4. Refine capture regression (BUG found by kibitz r2): the refine _core
     capture must contain ONLY real run() parameters -- the old bare
     locals() capture leaked `os` + `_scaffold` and made every
     refine-enabled run a TypeError since 2026-06-24. source_bank must
     survive into the refine re-entry kwargs.
  5. Threading: resolve_creative_system_prompt(source_bank_id=...) selects
     the pack; compose_line/compose_line_draft thread it end-to-end, and
     every recursive compose_line self-call forwards it (AST pin).
  6. _resolve_inputs carries source_bank as the one authoritative value.
  7. Headless surface: source_bank is on both CREATIVE_WHITELISTs and
     patch_widget_by_name lands it at slot 25 of the canonical workflow.
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

_CANONICAL_WORKFLOW = _REPO / "workflows" / "otr_scifi_16gb_full.json"
_NON_RUNNABLE_BANK = "media_archive"  # runnable:false lane pack (2B)


# ---------------------------------------------------------------------------
# 1. Widget surface
# ---------------------------------------------------------------------------
class TestWidgetSurface:
    def test_source_bank_is_last_optional(self):
        spec = OTR_LedgerScriptWriter.INPUT_TYPES()
        order = list(spec["optional"].keys())
        assert order[-1] == "source_bank"

    def test_choices_are_exactly_the_registry_in_order(self):
        spec = OTR_LedgerScriptWriter.INPUT_TYPES()
        choices, meta = spec["optional"]["source_bank"]
        assert choices == list(routing.list_bank_ids())
        assert meta["default"] == "science_news"
        # The honest-error contract: non-runnable banks ARE listed.
        assert _NON_RUNNABLE_BANK in choices

    def test_default_is_a_runnable_bank(self):
        bank = routing.require_runnable_bank("science_news")
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
# 4. Refine capture regression (kibitz r2 bug)
# ---------------------------------------------------------------------------
class TestRefineCoreCapture:
    def test_core_kwargs_are_exactly_run_parameters(self, monkeypatch):
        from nodes import _otr_story_select as sel

        class _Rcfg:
            target_grade = "B"
            bar = 0
            effective_passes = 2

        monkeypatch.setattr(
            sel, "resolve_refine_passes", lambda *_a, **_k: _Rcfg())

        captured = {}

        def _fake_loop(self, _rcfg, _core):
            captured.update(_core)
            return ("", "", "", 0, "")

        monkeypatch.setattr(
            OTR_LedgerScriptWriter, "_refine_loop", _fake_loop)
        node = OTR_LedgerScriptWriter()
        out = node.run(source_bank="science_news",
                       refine_target_grade="B")
        assert out == ("", "", "", 0, "")
        # The regression: non-parameter locals leaked into the capture and
        # made self.run(**_core) a TypeError on every refine pass.
        assert "os" not in captured
        assert "_scaffold" not in captured
        # Excluded-by-design keys.
        for k in ("self", "refine_target_grade", "_refine_active",
                  "_refine_prior_macro", "_refine_prior_critique",
                  "_refine_forced_cast_seed"):
            assert k not in captured
        # The selection survives re-entry.
        assert captured["source_bank"] == "science_news"
        # Every captured key is a real run() parameter -> **_core is safe.
        import inspect
        params = inspect.signature(OTR_LedgerScriptWriter.run).parameters
        unknown = [k for k in captured if k not in params]
        assert not unknown, f"non-parameter keys captured: {unknown}"


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
            source_bank_id=_NON_RUNNABLE_BANK)
        assert science != other, (
            "source_bank_id did not change the resolved prompt -- the "
            "widget would be dead"
        )
        # Cross-check against the lane pack on disk.
        pack_path = (_REPO / "nodes" / "story_packs" / _NON_RUNNABLE_BANK /
                     "media_restoration_adventure.json")
        pack = json.loads(pack_path.read_text(encoding="utf-8"))
        assert other == pack["prompt_stages"]["line_composer_system"]

    def test_default_is_byte_identical_science(self):
        default = resolve_creative_system_prompt(
            "mistralai/Mistral-Nemo-Instruct-2407",
            phase="line_composer_system")
        explicit = resolve_creative_system_prompt(
            "mistralai/Mistral-Nemo-Instruct-2407",
            phase="line_composer_system", source_bank_id="science_news")
        assert default == explicit

    def test_compose_line_draft_threads_source_bank(self, monkeypatch):
        from nodes import _otr_line_composer as lc
        from nodes import _otr_creative_prompt_router as router
        seen = []
        real = router.resolve_creative_system_prompt

        def _spy(repo_id, phase, source_bank_id="science_news"):
            seen.append(source_bank_id)
            return real(repo_id, phase, source_bank_id=source_bank_id)

        monkeypatch.setattr(
            router, "resolve_creative_system_prompt", _spy)
        req = lc.LineRequest(
            speaker="MARGOT",
            intent="steady the room",
            mood="calm",
            target_words=8,
            canon_header="",
            last_lines=[],
        )
        out = lc.compose_line(
            creative_fn=lambda *a, **k: "A quiet line about the machine.",
            req=req,
            creative_repo_id="mistralai/Mistral-Nemo-Instruct-2407",
            source_bank_id=_NON_RUNNABLE_BANK,
        )
        assert out.text
        assert seen and all(s == _NON_RUNNABLE_BANK for s in seen)

    def test_every_recursive_compose_line_call_forwards_the_bank(self):
        # AST pin (kibitz r3 M2): compose_line's recursive self-calls and
        # its compose_line_draft call must ALL forward source_bank_id --
        # a missed one silently falls back to science.
        src = (_REPO / "nodes" / "_otr_line_composer.py").read_text(
            encoding="utf-8")
        tree = ast.parse(src)
        fn = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "compose_line")
        checked = 0
        for call in ast.walk(fn):
            if not isinstance(call, ast.Call):
                continue
            callee = call.func
            name = getattr(callee, "id", getattr(callee, "attr", ""))
            if name not in ("compose_line", "compose_line_draft"):
                continue
            kwarg_names = {k.arg for k in call.keywords}
            assert "source_bank_id" in kwarg_names, (
                f"{name} call at line {call.lineno} does not forward "
                f"source_bank_id"
            )
            checked += 1
        assert checked >= 4, (
            f"expected >=4 threaded calls (draft + 3 recursive); "
            f"found {checked}"
        )

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
        assert sites == 3, f"expected 3 writer compose_line sites, {sites}"


# ---------------------------------------------------------------------------
# 6. Resolved surface
# ---------------------------------------------------------------------------
class TestResolvedSurface:
    def test_resolve_inputs_carries_source_bank(self):
        resolved = _resolve_inputs(custom_premise="test premise")
        assert resolved["source_bank"] == "science_news"
        resolved2 = _resolve_inputs(
            custom_premise="test premise", source_bank=_NON_RUNNABLE_BANK)
        assert resolved2["source_bank"] == _NON_RUNNABLE_BANK


# ---------------------------------------------------------------------------
# 7. Headless surface
# ---------------------------------------------------------------------------
class TestHeadlessSurface:
    def test_source_bank_on_both_whitelists(self):
        from nodes._otr_workflow_apply import CREATIVE_WHITELIST as pkg_wl
        import otr_api
        assert "source_bank" in pkg_wl
        assert "source_bank" in otr_api.CREATIVE_WHITELIST

    def test_patch_widget_by_name_lands_slot_25(self):
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
            workflow, 1, "source_bank", "science_news", schemas)
        node1 = next(n for n in workflow["nodes"] if n["id"] == 1)
        assert len(node1["widgets_values"]) == 26
        assert node1["widgets_values"][25] == "science_news"
