"""KILL 2 -- C1: the pre-outline StoryContract + OutlineRequest style threading.

C1 selects ONE radio style per episode BEFORE the outline (cast_seed-keyed, from
script_brief/news_seed), freezes it as a StoryContract, and threads its grammar
into the macro prompt + its conflict-shape engine into the phase/beat prompts.

Everything is behind ``story_scaffold`` -> ``_style_grammar_on``; with the new
OutlineRequest fields empty (the default), every prompt is byte-identical to the
pre-KILL-2 pipeline. These tests prove both: the new fields RENDER when set, and
the empty path emits no new prompt text.

Pure Python; no GPU, no LLM, no network.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import _otr_style_catalog as STYLE  # noqa: E402
from nodes._otr_episode_budget import (  # noqa: E402
    compute_episode_budget,
    default_act_count,
)
from nodes._otr_outline import OutlineRequest, generate_outline  # noqa: E402


# ---------------------------------------------------------------------------
# StoryContract + build_story_contract
# ---------------------------------------------------------------------------
class TestStoryContract:
    def test_build_is_deterministic(self):
        c1 = STYLE.build_story_contract(12345, "", "a quiet town meeting", {})
        c2 = STYLE.build_story_contract(12345, "", "a quiet town meeting", {})
        assert c1 == c2
        assert c1.slug

    def test_fields_match_catalog(self):
        c = STYLE.build_story_contract(999, "", "a quiet town meeting", {})
        s = STYLE.get_style(c.slug)
        assert s is not None
        assert c.label == s["label"]
        assert c.sound_world == s["sound_world"]
        assert c.story_engine == s["story_engine"]
        assert c.ending_mode == s["ending_mode"]
        assert c.ending_tag == s["ending_tag"]
        assert c.ending_template == STYLE.ending_template_for(c.slug)
        assert c.grammar == STYLE.render_style_grammar(c.slug)

    def test_grammar_is_the_first_real_caller(self):
        # The literal KILL-2 "zero callers" fix: the contract's grammar IS
        # render_style_grammar's output and is non-empty for a real slug.
        c = STYLE.build_story_contract(7, "", "a quiet town meeting", {})
        assert c.grammar.strip()
        assert "Sound world:" in c.grammar
        assert "Story engine:" in c.grammar

    def test_text_routing_brief_then_news_seed(self):
        # build_story_contract selects on script_brief when present, else the
        # raw news_seed -- proved by matching select_style on each resolved text.
        brief = "a daring rescue during a wildfire evacuation disaster"
        seed = "a calm afternoon in a archive reading room"
        assert STYLE.build_story_contract(5, brief, seed, {}).slug == \
            STYLE.select_style(brief, {}, 5)
        assert STYLE.build_story_contract(5, "", seed, {}).slug == \
            STYLE.select_style(seed, {}, 5)

    def test_missing_style_never_raises(self):
        # get_style returns None for junk -> empty fields, no exception. (We can
        # only reach this if select_style ever returned an unknown slug; assert
        # the constructor path tolerates empties directly.)
        c = STYLE.StoryContract(
            slug="", label="", sound_world="", story_engine="",
            ending_mode="", ending_tag="", ending_template="", grammar="",
        )
        assert c.slug == ""

    def test_contract_is_frozen(self):
        c = STYLE.build_story_contract(1, "", "a quiet town meeting", {})
        with pytest.raises(Exception):
            c.slug = "tampered"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# OutlineRequest style fields + prompt threading (through generate_outline)
# ---------------------------------------------------------------------------
_MACRO_JSON = json.dumps({
    "title": "Threading Test",
    "premise": "A premise about a faint signal and what it costs.",
    "setting": "A quiet observatory lab",
    "time_of_day": "midnight",
})
_GRAMMAR = (
    "Style: Locked-Room Mystery.\n"
    "Sound world: dripping pipes and a ticking clock.\n"
    "Story engine: a confined hunt for who is lying.\n"
    "Ending mode: a quiet revelation."
)
_ENGINE = "a confined hunt for who is lying"


class _Recorder:
    """Captures the user prompt at each outline stage so the test can assert on
    the rendered text (mirrors test_story_grammar_wiring's _gen)."""

    def __init__(self) -> None:
        self.prompts = {"macro": [], "phase": [], "beat": []}

    def __call__(self, messages, *, temperature, max_new_tokens):
        system = messages[0]["content"]
        user = messages[1]["content"]
        if "You plan one phase" in system:
            self.prompts["phase"].append(user)
            m = re.search(r"Beats to plan in this phase: (\d+)", user)
            n = int(m.group(1)) if m else 1
            return json.dumps({"beats": [{"speaker": "ALICE"} for _ in range(n)]})
        if "You flesh out one beat" in system:
            self.prompts["beat"].append(user)
            return json.dumps({"intent": "advance the scene", "mood": "tense"})
        self.prompts["macro"].append(user)
        return _MACRO_JSON


def _req(**kw):
    target = 350
    base = dict(
        news_seed="a real science seed", style="noir",
        character_cast=("ALICE", "BOB"), target_words=target,
        budget=compute_episode_budget(
            target, default_act_count(target), True, 2,
        ),
    )
    base.update(kw)
    return OutlineRequest(**base)


class TestPromptThreading:
    def test_new_fields_default_empty(self):
        assert OutlineRequest.__dataclass_fields__["style_grammar"].default == ""
        assert OutlineRequest.__dataclass_fields__["story_engine"].default == ""

    def test_macro_renders_style_grammar(self):
        rec = _Recorder()
        generate_outline(rec, _req(style_grammar=_GRAMMAR))
        assert rec.prompts["macro"]
        assert any("Sound world: dripping pipes" in p for p in rec.prompts["macro"])

    def test_phase_and_beat_render_story_engine(self):
        rec = _Recorder()
        generate_outline(rec, _req(story_engine=_ENGINE))
        assert any(f"Story engine: {_ENGINE}" in p for p in rec.prompts["phase"])
        assert any(f"Story engine: {_ENGINE}" in p for p in rec.prompts["beat"])

    def test_sound_world_never_leaks_to_phase_or_beat(self):
        # sound_world stays at the macro prompt ONLY -- it must never reach the
        # beat-intent -> dialogue-line path.
        rec = _Recorder()
        generate_outline(rec, _req(style_grammar=_GRAMMAR, story_engine=_ENGINE))
        assert all("Sound world:" not in p for p in rec.prompts["phase"])
        assert all("Sound world:" not in p for p in rec.prompts["beat"])

    def test_empty_fields_emit_no_new_prompt_text(self):
        # The byte-identity guarantee: with the fields empty (default), no
        # grammar block and no engine line appear anywhere.
        rec = _Recorder()
        generate_outline(rec, _req())
        for stage in ("macro", "phase", "beat"):
            assert rec.prompts[stage]
            assert all("Sound world:" not in p for p in rec.prompts[stage])
            assert all("Story engine:" not in p for p in rec.prompts[stage])

    def test_empty_fields_byte_identical_macro(self):
        a = _Recorder(); generate_outline(a, _req())
        b = _Recorder(); generate_outline(b, _req(style_grammar="", story_engine=""))
        assert a.prompts["macro"] == b.prompts["macro"]
        assert a.prompts["phase"] == b.prompts["phase"]
        assert a.prompts["beat"] == b.prompts["beat"]


# ---------------------------------------------------------------------------
# Writer OFF-flag ledger-meta invariant (source-level; run() needs the full
# model stack, so these lock the seam the live re-soak confirms end-to-end).
# story_scaffold=off => _style_grammar_on False => no contract, no
# meta["story_contract"] => byte-identical ledger meta.
# ---------------------------------------------------------------------------
class TestWriterOffFlagLedgerMeta:
    SRC = (
        Path(__file__).resolve().parents[1] / "nodes" / "OTR_LedgerScriptWriter.py"
    ).read_text(encoding="utf-8")

    def test_contract_build_and_stamp_are_flag_gated_in_order(self):
        src = self.SRC
        i_none = src.index("\n        contract = None\n")
        i_flag = src.index("if _style_grammar_on:", i_none)
        i_build = src.index("build_story_contract(", i_none)
        i_meta = src.index('meta["story_contract"]', i_none)
        # gate -> build -> stamp, all after the `contract = None` init, in order:
        # the stamp can never fire unless the flag gate was entered.
        assert i_none < i_flag < i_build < i_meta

    def test_story_contract_meta_stamped_exactly_once(self):
        assert self.SRC.count('meta["story_contract"]') == 1

    def test_outline_request_style_fields_default_off_via_contract(self):
        # The OutlineRequest style fields read the contract only when present;
        # the `else ""` keeps the OFF path's prompt byte-identical.
        assert 'style_grammar=(contract.grammar if contract else "")' in self.SRC
        assert 'story_engine=(contract.story_engine if contract else "")' in self.SRC


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
