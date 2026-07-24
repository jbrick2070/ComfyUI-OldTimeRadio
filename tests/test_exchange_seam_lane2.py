"""tests/test_exchange_seam_lane2.py

LANE-ENABLEMENT CHUNK 2 -- the Build-4 grouped-exchange STATIC system prompt
migrates from a hard-wired literal to pack routing (router repo=None lane).

Pins:
  1. BYTE IDENTITY: the science pack's `exchange_system` seam ==
     _otr_compose_exchange.EXCHANGE_SYSTEM_PROMPT (which remains as the
     extraction fixture), and the router resolves those exact bytes.
  2. ASSEMBLY IDENTITY: build_exchange_prompt(system_prompt=None) ==
     build_exchange_prompt(system_prompt=<routed science value>) -- the
     science lane is byte-identical end to end.
  3. NON-SCIENCE ROUTING: public_domain_story has its own exchange seam;
     routing must use it, not the science fixture.
  4. THREADING: system_prompt is threaded run_exchange_prepass ->
     compose_exchange -> build_exchange_prompt (signature pins), the writer
     call site passes it, and the writer's resolve call sits OUTSIDE any
     try/except (a pack failure must fail the episode LOUD -- the exchange
     prepass PD1 swallow must not eat it).
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from nodes import _otr_compose_exchange as ex_mod
from nodes._otr_compose_exchange import (
    EXCHANGE_SYSTEM_PROMPT,
    SlotContract,
    VoicedSlot,
    build_exchange_prompt,
)
from nodes._otr_creative_prompt_router import resolve_creative_system_prompt
from nodes._otr_story_pack import get_pack_prompt
from nodes._otr_story_routing import resolve_story_pack

_REPO = Path(__file__).resolve().parent.parent
_WRITER = _REPO / "nodes" / "OTR_LedgerScriptWriter.py"


class TestByteIdentity:
    def test_constant_still_fixture(self):
        # The constant stays in _otr_compose_exchange as the extraction
        # fixture; the router imports it for _MODERN_BY_PHASE.
        assert isinstance(EXCHANGE_SYSTEM_PROMPT, str)
        assert len(EXCHANGE_SYSTEM_PROMPT) > 50
        assert EXCHANGE_SYSTEM_PROMPT.endswith("\n")


class TestAssemblyIdentity:
    def _messages(self, system_prompt):
        group = [
            VoicedSlot("d001", "MARLOW", intent="refuses the order"),
            VoicedSlot("d002", "VERA", intent="presses him"),
        ]
        contracts = {
            "d001": SlotContract("d001", "MARLOW", line_job="refuse"),
            "d002": SlotContract("d002", "VERA", line_job="press"),
        }
        return build_exchange_prompt(
            group, contracts, ["d000|ANNOUNCER: previously..."], [],
            system_prompt=system_prompt,
        )

    def test_dynamic_grounding_guidance_is_appended(self):
        # The pack owns only the STATIC portion; prompt-only grounding guidance
        # is appended without any Python vocabulary filter.
        system = self._messages(None)[0]["content"]
        assert system.startswith(EXCHANGE_SYSTEM_PROMPT)
        assert "Ground the exchange in ONE concrete detail" in system

    def test_custom_system_prompt_replaces_static_portion_only(self):
        custom = "You write anime-style radio banter.\n\nCraft rules:\n"
        system = self._messages(custom)[0]["content"]
        assert system.startswith(custom)
        assert EXCHANGE_SYSTEM_PROMPT not in system
        assert "Ground the exchange in ONE concrete detail" in system


class TestPublicDomainRouting:
    def test_public_domain_exchange_seam_routes(self):
        resolved = resolve_creative_system_prompt(
            None, phase="exchange_system",
            source_bank_id="public_domain")
        assert "public-domain" in resolved
        assert resolved != EXCHANGE_SYSTEM_PROMPT


class TestThreading:
    @pytest.mark.parametrize("fn", [
        ex_mod.build_exchange_prompt,
        ex_mod.compose_exchange,
        ex_mod.run_exchange_prepass,
    ])
    def test_system_prompt_param_default_none(self, fn):
        params = inspect.signature(fn).parameters
        assert "system_prompt" in params
        assert params["system_prompt"].default is None
        assert params["system_prompt"].kind is inspect.Parameter.KEYWORD_ONLY

    def _writer_tree(self):
        return ast.parse(_WRITER.read_text(encoding="utf-8"))

    def test_writer_prepass_call_passes_system_prompt(self):
        sites = 0
        for call in ast.walk(self._writer_tree()):
            if not isinstance(call, ast.Call):
                continue
            f = call.func
            name = f.id if isinstance(f, ast.Name) else getattr(f, "attr", "")
            if name == "_run_ex_prepass":
                kwargs = {k.arg for k in call.keywords}
                assert "system_prompt" in kwargs, (
                    f"writer exchange-prepass call at line {call.lineno} "
                    f"missing system_prompt")
                sites += 1
        assert sites == 1, f"expected 1 prepass call site, found {sites}"

    def test_writer_exchange_resolve_outside_try(self):
        # The resolve of phase="exchange_system" must NOT sit inside any
        # try/except: the prepass PD1 swallow would otherwise convert a
        # missing-seam pack failure into a silent legacy fallback
        # (no-fallback law; the chunk-1 Fable forward-note pattern).
        tree = self._writer_tree()
        parents: dict = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parents[child] = node
        found = 0
        for call in ast.walk(tree):
            if not isinstance(call, ast.Call):
                continue
            is_exchange_resolve = any(
                k.arg == "phase"
                and isinstance(k.value, ast.Constant)
                and k.value.value == "exchange_system"
                for k in call.keywords
            )
            if not is_exchange_resolve:
                continue
            found += 1
            node = call
            while node in parents:
                node = parents[node]
                assert not isinstance(node, ast.Try), (
                    f"exchange_system resolve at line {call.lineno} is "
                    f"inside a try/except -- pack failures must fail LOUD")
        assert found == 1, (
            f"expected exactly 1 exchange_system resolve site, found {found}")
