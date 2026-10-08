"""tests/test_outline_seams_lane1.py

LANE-ENABLEMENT CHUNK 1 -- the three outline STAGE system prompts migrate
from hard-wired constants to pack routing (router repo=None lane).

Pins:
  1. Router: the plain "outline" phase is UNTOUCHED (object identity
     preserved for the period-overlay sentinel).
  2. NON-DEFAULT ROUTING: public_domain has its own outline seams; routing
     must use them, not the default bank's.
  3. generate_outline threads source_bank_id (signature + AST pin on the
     writer's call site).
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from nodes import _otr_outline as outline_mod
from nodes._otr_creative_prompt_router import resolve_creative_system_prompt

_REPO = Path(__file__).resolve().parent.parent

_STAGE_PHASES = (
    "outline_macro_system",
    "outline_phase_system",
    "outline_beat_system",
)


class TestByteIdentity:
    def test_plain_outline_phase_object_identity_untouched(self):
        resolved = resolve_creative_system_prompt(
            "mistralai/Mistral-Nemo-Instruct-2407", phase="outline")
        assert resolved is outline_mod._SYSTEM_PROMPT


class TestPublicDomainRouting:
    @pytest.mark.parametrize("phase", sorted(_STAGE_PHASES))
    def test_public_domain_outline_seams_route(self, phase):
        resolved = resolve_creative_system_prompt(
            None, phase=phase, source_bank_id="public_domain")
        assert "public-domain" in resolved or "adaptation" in resolved
        assert resolved != resolve_creative_system_prompt(None, phase=phase)


class TestThreading:
    def test_generate_outline_signature(self):
        params = inspect.signature(outline_mod.generate_outline).parameters
        assert "source_bank_id" in params
        assert params["source_bank_id"].default == "media_archive"

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
        assert sites == 1, f"expected 1 authoritative writer call site, found {sites}"
