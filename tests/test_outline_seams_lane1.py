"""tests/test_outline_seams_lane1.py

LANE-ENABLEMENT CHUNK 1 -- the three outline STAGE system prompts migrate
from hard-wired constants to pack routing (router repo=None lane).

Pins:
  1. BYTE IDENTITY: the science pack's outline_macro/phase/beat_system seams
     == the _otr_outline constants (which remain as the extraction fixture).
  2. Router: the three new phases resolve the pack seam; repo=None on the
     science lane returns those exact bytes; the plain "outline" phase is
     UNTOUCHED (object identity preserved for the period-overlay sentinel).
  3. FAIL-LOUD: a bank whose pack lacks the outline seams raises through
     the router (media_archive -- its lane-enablement item).
  4. generate_outline threads source_bank_id (signature + AST pin on the
     writer's two call sites).
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from nodes import _otr_outline as outline_mod
from nodes._otr_creative_prompt_router import resolve_creative_system_prompt

_REPO = Path(__file__).resolve().parent.parent

_SEAM_TO_CONST = {
    "outline_macro_system": outline_mod._MACRO_SYSTEM_PROMPT,
    "outline_phase_system": outline_mod._PHASE_SYSTEM_PROMPT,
    "outline_beat_system": outline_mod._BEAT_SYSTEM_PROMPT,
}


class TestByteIdentity:
    @pytest.mark.parametrize("phase", sorted(_SEAM_TO_CONST))
    def test_router_science_matches_constant(self, phase):
        resolved = resolve_creative_system_prompt(None, phase=phase)
        assert resolved == _SEAM_TO_CONST[phase]

    @pytest.mark.parametrize("phase", sorted(_SEAM_TO_CONST))
    def test_explicit_science_bank_identical(self, phase):
        assert resolve_creative_system_prompt(
            None, phase=phase, source_bank_id="science_news"
        ) == _SEAM_TO_CONST[phase]

    def test_plain_outline_phase_object_identity_untouched(self):
        resolved = resolve_creative_system_prompt(
            "mistralai/Mistral-Nemo-Instruct-2407", phase="outline")
        assert resolved is outline_mod._SYSTEM_PROMPT


class TestFailLoud:
    @pytest.mark.parametrize("phase", sorted(_SEAM_TO_CONST))
    def test_bank_without_outline_seams_raises(self, phase):
        # media_archive's pack ships only {line_composer_system, coda_system}
        # (2B exact-key contract) -- outline seams are its lane-enablement
        # item; resolution must fail LOUD, never fall to science.
        with pytest.raises(Exception) as ei:
            resolve_creative_system_prompt(
                None, phase=phase, source_bank_id="media_archive")
        assert "outline" in str(ei.value) or "seam" in str(ei.value).lower()


class TestThreading:
    def test_generate_outline_signature(self):
        params = inspect.signature(outline_mod.generate_outline).parameters
        assert "source_bank_id" in params
        assert params["source_bank_id"].default == "science_news"

    def test_writer_call_sites_pass_bank(self):
        src = (_REPO / "nodes" / "OTR_LedgerScriptWriter.py").read_text(
            encoding="utf-8")
        tree = ast.parse(src)
        sites = 0
        for call in ast.walk(tree):
            if not isinstance(call, ast.Call):
                continue
            f = call.func
            if (isinstance(f, ast.Attribute)
                    and f.attr == "generate_outline"):
                kwargs = {k.arg for k in call.keywords}
                assert "source_bank_id" in kwargs, (
                    f"writer generate_outline call at line {call.lineno} "
                    f"missing source_bank_id")
                sites += 1
        assert sites == 2, f"expected 2 writer call sites, found {sites}"

    def test_outline_stage_constants_still_fixture(self):
        # The constants stay in _otr_outline as the extraction fixture; the
        # router imports them for _MODERN_BY_PHASE. Non-empty sanity.
        for const in _SEAM_TO_CONST.values():
            assert isinstance(const, str) and len(const) > 50
