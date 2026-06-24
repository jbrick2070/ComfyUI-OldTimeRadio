"""Story-grammar build (2026-06-24, chunks 4-6) -- the WIRING that takes the
inert chunk 1-3 catalog/role machinery live behind ``OTR_ENABLE_STYLE_GRAMMAR``.

Covers:
  C4  composer ``LineRequest.ending_template`` render + byte-identity when empty;
      ``build_sq_data(climax_role=...)`` threading (default = irreversible_choice
      => byte-identical; a style ending tag => that climax class, validator-clean).
  C5  the announcer-close gate in ``_assemble_outline`` (OFF = the exact pre-grammar
      string; ON = a non-outcome close that is NEVER removed -- still an announcer
      beat, so budget validator #7 is satisfied); ``style_grammar_enabled`` config.
  selector  ``select_style`` determinism (C7-safe) + ``ending_template_for``.

Pure Python; no GPU, no LLM, no network. The lever defaults OFF, so every
assertion here that does not set the env proves the byte-identical baseline.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import _otr_config as CFG  # noqa: E402
from nodes import _otr_line_composer as LC  # noqa: E402
from nodes import _otr_story_quality_l12 as L12  # noqa: E402
from nodes import _otr_style_catalog as STYLE  # noqa: E402
from nodes._otr_episode_budget import (  # noqa: E402
    compute_episode_budget,
    default_act_count,
)
from nodes._otr_outline import (  # noqa: E402
    OutlineRequest,
    generate_outline,
)


# ---------------------------------------------------------------------------
# C4 -- composer ending_template render + byte-identity
# ---------------------------------------------------------------------------
class TestComposerEndingTemplate:
    def _req(self, **kw):
        base = dict(
            speaker="ALEX", intent="say something", mood="tense",
            target_words=20, speaker_role="character",
            canon_header="", last_lines=[],
        )
        base.update(kw)
        return LC.LineRequest(**base)

    def test_default_has_no_ending_block(self):
        prompt = LC._build_user_prompt(self._req())
        assert "Ending:" not in prompt

    def test_ending_template_renders_when_set(self):
        tmpl = STYLE.ENDING_TEMPLATES["revelation"]
        prompt = LC._build_user_prompt(self._req(ending_template=tmpl))
        assert "Ending:" in prompt
        # The full instruction text reaches the prompt verbatim.
        assert "REVELATION" in prompt

    def test_byte_identical_empty_ending_template(self):
        a = LC._build_user_prompt(self._req())
        b = LC._build_user_prompt(self._req(ending_template=""))
        assert a == b

    def test_field_default_is_empty_string(self):
        assert LC.LineRequest.__dataclass_fields__["ending_template"].default == ""


# ---------------------------------------------------------------------------
# C4 -- build_sq_data climax_role threading
# ---------------------------------------------------------------------------
@dataclass
class _FakeBeat:
    beat_id: str
    speaker_role: str
    intent: str = "they argue over the matter"
    mood: str = "tense"
    speaker: str = "ALEX"


def _char_beats(n: int) -> list:
    return [_FakeBeat(f"b{i:03d}", "character", intent=f"beat {i} unfolds")
            for i in range(1, n + 1)]


class TestBuildSqClimaxRole:
    def test_default_climax_is_irreversible_choice(self):
        beats = _char_beats(4)
        sq = L12.build_sq_data(beats, {}, "two people in a room", 7)
        last_role = sq[beats[-1].beat_id]["beat_role"]
        assert last_role == L12.BEAT_ROLE_IRREVERSIBLE_CHOICE

    def test_style_ending_tag_becomes_climax_class(self):
        beats = _char_beats(5)
        sq = L12.build_sq_data(
            beats, {}, "two people in a room", 7, climax_role="revelation",
        )
        last_role = sq[beats[-1].beat_id]["beat_role"]
        assert last_role == "revelation"
        # exactly one climax-class role, and it is the last voiced char beat
        climax = [bid for bid, e in sq.items()
                  if e["beat_role"] in L12.CLIMAX_CLASS_ROLES]
        assert climax == [beats[-1].beat_id]

    def test_invalid_climax_role_falls_soft(self):
        beats = _char_beats(3)
        sq = L12.build_sq_data(
            beats, {}, "x", 1, climax_role="not_a_real_role",
        )
        assert sq[beats[-1].beat_id]["beat_role"] == L12.BEAT_ROLE_IRREVERSIBLE_CHOICE

    def test_all_ending_tags_validate_as_climax(self):
        # every style ending tag the writer might thread must be a real
        # climax-class role with a template (no validator surprise at runtime)
        beats = _char_beats(4)
        for tag in STYLE.ENDING_TAGS:
            sq = L12.build_sq_data(beats, {}, "x", 1, climax_role=tag)
            assert sq[beats[-1].beat_id]["beat_role"] == tag


# ---------------------------------------------------------------------------
# config -- the lever defaults OFF
# ---------------------------------------------------------------------------
class TestStyleGrammarConfig:
    def test_default_on_when_unset(self, monkeypatch):
        # 2026-06-24 operator flip: the lever is DEFAULT ON.
        monkeypatch.delenv("OTR_ENABLE_STYLE_GRAMMAR", raising=False)
        assert CFG.style_grammar_enabled() is True

    def test_empty_string_is_default_on(self, monkeypatch):
        monkeypatch.setenv("OTR_ENABLE_STYLE_GRAMMAR", "")
        assert CFG.style_grammar_enabled() is True

    @pytest.mark.parametrize("val", ["1", "true", "yes", "on", "ON", "True"])
    def test_truthy_values_enable(self, monkeypatch, val):
        monkeypatch.setenv("OTR_ENABLE_STYLE_GRAMMAR", val)
        assert CFG.style_grammar_enabled() is True

    @pytest.mark.parametrize("val", ["0", "false", "no", "off", "OFF"])
    def test_killswitch_values_disable(self, monkeypatch, val):
        monkeypatch.setenv("OTR_ENABLE_STYLE_GRAMMAR", val)
        assert CFG.style_grammar_enabled() is False


# ---------------------------------------------------------------------------
# selector -- determinism + template lookup
# ---------------------------------------------------------------------------
class TestSelector:
    def test_select_style_deterministic(self):
        a = STYLE.select_style("a quiet town meeting", {}, 12345)
        b = STYLE.select_style("a quiet town meeting", {}, 12345)
        assert a == b and a

    def test_select_style_varies_by_seed(self):
        seeds = {STYLE.select_style("a quiet town meeting", {}, s)
                 for s in range(40)}
        assert len(seeds) > 1  # different episodes get different styles

    def test_ending_template_for_nonempty(self):
        slug = STYLE.select_style("a quiet town meeting", {}, 999)
        assert STYLE.ending_template_for(slug).strip()


# ---------------------------------------------------------------------------
# C5 -- the announcer-close gate (end-to-end through generate_outline)
# ---------------------------------------------------------------------------
_OFF_CLOSE = (
    "Close on a concrete final image showing what changed (use "
    "the central object if set); no moral, thesis, or "
    "news-summary tag."
)
_MACRO_JSON = json.dumps({
    "title": "Grammar Wiring Test",
    "premise": "A premise about a faint science signal and what it costs.",
    "setting": "A quiet observatory lab",
    "time_of_day": "midnight",
})


def _stage_of(messages):
    system = messages[0]["content"]
    if "You plan one phase" in system:
        return "phase"
    if "You flesh out one beat" in system:
        return "beat"
    return "macro"


def _gen(messages, *, temperature, max_new_tokens):
    stage = _stage_of(messages)
    if stage == "phase":
        user = messages[1]["content"]
        m = re.search(r"Beats to plan in this phase: (\d+)", user)
        n = int(m.group(1)) if m else 1
        return json.dumps({"beats": [{"speaker": "ALICE"} for _ in range(n)]})
    if stage == "beat":
        return json.dumps({"intent": "advance the scene", "mood": "tense"})
    return _MACRO_JSON


def _request():
    target = 350
    return OutlineRequest(
        news_seed="a real science seed", style="noir",
        character_cast=("ALICE", "BOB"),
        target_words=target,
        budget=compute_episode_budget(
            target, default_act_count(target), True, 2,
        ),
    )


def _announcer_close(outline):
    announcer = [b for b in outline.beats if b.speaker_role == "announcer"]
    return announcer


class TestAnnouncerCloseGate:
    def test_killswitch_is_exact_pre_grammar_string(self, monkeypatch):
        # OTR_ENABLE_STYLE_GRAMMAR=0 restores the exact pre-grammar close.
        monkeypatch.setenv("OTR_ENABLE_STYLE_GRAMMAR", "0")
        outline = generate_outline(_gen, _request())
        ann = _announcer_close(outline)
        assert len(ann) == 2  # open + close (budget validator #7)
        assert ann[-1].intent == _OFF_CLOSE

    def test_default_unset_is_non_outcome(self, monkeypatch):
        # 2026-06-24: DEFAULT ON -- an unset env now gives the non-outcome close.
        monkeypatch.delenv("OTR_ENABLE_STYLE_GRAMMAR", raising=False)
        close = _announcer_close(generate_outline(_gen, _request()))[-1]
        assert close.intent != _OFF_CLOSE
        assert "outcome" in close.intent.lower()

    def test_on_is_non_outcome_and_close_not_removed(self, monkeypatch):
        monkeypatch.setenv("OTR_ENABLE_STYLE_GRAMMAR", "1")
        outline = generate_outline(_gen, _request())
        ann = _announcer_close(outline)
        # close is NEVER removed -- still exactly two announcer beats
        assert len(ann) == 2
        close = ann[-1]
        assert close.speaker_role == "announcer"
        assert close.intent != _OFF_CLOSE
        # the ON close explicitly refuses to state the outcome
        low = close.intent.lower()
        assert "do not state" in low or "do not state, summarize" in low
        assert "outcome" in low

    def test_on_close_keeps_beat_shape(self, monkeypatch):
        # target_words / mood / arc_phase are unchanged -- only the prose differs
        monkeypatch.setenv("OTR_ENABLE_STYLE_GRAMMAR", "0")
        off = _announcer_close(generate_outline(_gen, _request()))[-1]
        monkeypatch.setenv("OTR_ENABLE_STYLE_GRAMMAR", "1")
        on = _announcer_close(generate_outline(_gen, _request()))[-1]
        assert on.target_words == off.target_words
        assert on.mood == off.mood
        assert on.arc_phase == off.arc_phase


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
