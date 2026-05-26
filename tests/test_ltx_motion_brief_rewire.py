"""Sprint 8.5: LTX motion brief-rewire tests.

`_build_ltx_role_prompt` now reads `meta.tempo_hint` via the canonical
`_read_brief_field` reader and joins it to the v1 brief fragment as
a comma-suffixed pacing cue. Resolution order:

  brief + tempo -> "{brief}, {tempo}"
  brief only    -> brief                (legacy path)
  tempo only    -> tempo                (new path -- usable on a
                                          v1-brief failure if the
                                          v2 producer landed tempo)
  neither       -> template only

All four cases still go through the 240-char total + 140-char motion-
position guards. The motion-first invariant (template leads) is
preserved -- tempo lives in the trailing fragment region.
"""
from __future__ import annotations

import inspect

import pytest

from nodes import batch_ltx_render as bltx


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _ledger(
    *,
    brief: str | None = "a dim radio room under a bare bulb, rain on tin",
    brief_status: str | None = "ok",
    tempo_hint: str | None = None,
    atmosphere_terms: list | None = None,
) -> dict:
    meta: dict = {}
    if brief is not None:
        meta["story_brief"] = brief
    if brief_status is not None:
        meta["story_brief_status"] = brief_status
        meta["story_brief_terms"] = {
            "setting":    [],
            "lighting":   [],
            "atmosphere": atmosphere_terms or [],
        }
    if tempo_hint is not None:
        meta["tempo_hint"] = tempo_hint
    return {"meta": meta}


def _empty_line() -> dict:
    return {"speaker_role": "character", "char_id": "PROBE", "text": "."}


# ---------------------------------------------------------------------------
# 1. tempo_hint appears in the prompt fragment
# ---------------------------------------------------------------------------


class TestTempoHintLands:

    def test_tempo_hint_appended_with_brief(self):
        led = _ledger(tempo_hint="hurried")
        prompt = bltx._build_ltx_role_prompt(
            "sfx", _empty_line(), led,
        )
        assert "hurried" in prompt
        # Brief still threads through.
        assert "radio room" in prompt or "bare bulb" in prompt

    def test_tempo_hint_alone_when_brief_failed(self):
        """v1 brief failed (status='failed', fragment empty) but v2
        producer landed tempo_hint -- tempo alone becomes the
        fragment."""
        led = _ledger(
            brief="",
            brief_status="failed",
            tempo_hint="slow",
        )
        prompt = bltx._build_ltx_role_prompt(
            "sfx", _empty_line(), led,
        )
        assert "slow" in prompt

    def test_tempo_hint_comma_suffixed_to_brief(self):
        """Order locked: brief comes first, then tempo, comma-joined."""
        led = _ledger(
            brief="CANARY_BRIEF",
            atmosphere_terms=["tense"],
            tempo_hint="CANARY_TEMPO",
        )
        prompt = bltx._build_ltx_role_prompt(
            "sfx", _empty_line(), led,
        )
        i_brief = prompt.find("CANARY_BRIEF")
        i_tempo = prompt.find("CANARY_TEMPO")
        assert i_brief > -1
        assert i_tempo > -1
        assert i_brief < i_tempo, (
            "tempo_hint must follow brief fragment, not precede it"
        )


# ---------------------------------------------------------------------------
# 2. Motion-first invariant preserved with tempo
# ---------------------------------------------------------------------------


class TestMotionFirstInvariant:

    def test_motion_verb_stays_under_140_chars(self):
        """With tempo + brief appended, the role template's leading
        motion verb must still land at or before char 140."""
        led = _ledger(tempo_hint="slow")
        prompt = bltx._build_ltx_role_prompt(
            "sfx", _empty_line(), led,
        )
        motion_idx = bltx._first_motion_verb_index(prompt)
        assert motion_idx >= 0
        assert motion_idx <= bltx._LTX_MOTION_VERB_POSITION_CEILING, (
            f"motion verb landed at char {motion_idx}; ceiling is "
            f"{bltx._LTX_MOTION_VERB_POSITION_CEILING}"
        )

    def test_template_still_leads_each_role(self):
        """Every role's template starts with motion words -- the
        rewire must NOT have changed that."""
        led = _ledger(tempo_hint="hurried")
        for role in ("sfx", "music", "announcer"):
            prompt = bltx._build_ltx_role_prompt(
                role, _empty_line(), led,
            )
            # The template chunk comes first; tempo lives in the
            # appended fragment.
            template = bltx._PROMPT_BY_ROLE.get(
                role, bltx._PROMPT_BY_ROLE["sfx"],
            )
            assert prompt.startswith(template), (
                f"role {role!r} no longer leads with its motion "
                "template -- motion-first rule regressed"
            )


# ---------------------------------------------------------------------------
# 3. Budget guard with tempo
# ---------------------------------------------------------------------------


class TestBudgetGuardWithTempo:

    def test_tempo_dropped_when_combined_exceeds_240(self):
        """If brief + tempo pushes the total over 240 chars, the
        function drops the fragment and returns the template alone."""
        long_brief = "a " * 100  # ~200 chars
        led = _ledger(brief=long_brief, tempo_hint="slow")
        prompt = bltx._build_ltx_role_prompt(
            "sfx", _empty_line(), led,
        )
        # Template-only path on budget overflow.
        assert prompt == bltx._PROMPT_BY_ROLE["sfx"]
        assert "slow" not in prompt

    def test_total_under_240_with_typical_tempo(self):
        led = _ledger(tempo_hint="slow")
        prompt = bltx._build_ltx_role_prompt(
            "sfx", _empty_line(), led,
        )
        assert len(prompt) <= bltx._LTX_TOTAL_CHAR_BUDGET


# ---------------------------------------------------------------------------
# 4. Fallback paths
# ---------------------------------------------------------------------------


class TestFallbackPaths:

    def test_v1_era_ledger_byte_identical_to_pre_8_5(self):
        """A ledger with no tempo_hint key at all must produce the
        exact pre-Sprint-8.5 fragment shape (brief alone, no tempo
        suffix)."""
        led_pre = _ledger(tempo_hint=None)
        led_with_empty_tempo = _ledger(tempo_hint="")
        prompt_pre = bltx._build_ltx_role_prompt(
            "sfx", _empty_line(), led_pre,
        )
        prompt_with_empty = bltx._build_ltx_role_prompt(
            "sfx", _empty_line(), led_with_empty_tempo,
        )
        # Same output: empty tempo_hint string == missing tempo_hint.
        assert prompt_pre == prompt_with_empty

    def test_both_signals_empty_returns_template(self):
        led = _ledger(
            brief="",
            brief_status="failed",
            tempo_hint="",
        )
        prompt = bltx._build_ltx_role_prompt(
            "sfx", _empty_line(), led,
        )
        assert prompt == bltx._PROMPT_BY_ROLE["sfx"]

    def test_non_string_tempo_hint_treated_as_empty(self):
        led = _ledger(tempo_hint=42)  # type: ignore[arg-type]
        prompt = bltx._build_ltx_role_prompt(
            "sfx", _empty_line(), led,
        )
        # No "42" string leaked.
        assert "42" not in prompt

    def test_whitespace_tempo_filtered(self):
        led = _ledger(tempo_hint="   \t  ")
        prompt = bltx._build_ltx_role_prompt(
            "sfx", _empty_line(), led,
        )
        # Whitespace-only tempo treated as empty; same render as
        # no-tempo path.
        led_no_tempo = _ledger(tempo_hint=None)
        prompt_no_tempo = bltx._build_ltx_role_prompt(
            "sfx", _empty_line(), led_no_tempo,
        )
        assert prompt == prompt_no_tempo


# ---------------------------------------------------------------------------
# 5. Source-level wiring lock
# ---------------------------------------------------------------------------


class TestReaderHelperUsed:

    def test_function_source_uses_brief_reader(self):
        src = inspect.getsource(bltx._build_ltx_role_prompt)
        assert "_read_brief_field" in src, (
            "_build_ltx_role_prompt no longer uses _read_brief_field "
            "-- Sprint 8.5 wiring regressed"
        )

    def test_function_reads_tempo_hint_key(self):
        src = inspect.getsource(bltx._build_ltx_role_prompt)
        assert '"tempo_hint"' in src, (
            "_build_ltx_role_prompt no longer reads meta.tempo_hint "
            "-- Sprint 8.5 wiring regressed"
        )


# ---------------------------------------------------------------------------
# 6. Log observability
# ---------------------------------------------------------------------------


class TestLogObservability:

    def test_log_surfaces_tempo_hint(self, caplog):
        caplog.set_level("INFO")
        led = _ledger(tempo_hint="hurried")
        bltx._build_ltx_role_prompt("sfx", _empty_line(), led)
        log_text = "\n".join(r.getMessage() for r in caplog.records)
        assert "tempo_hint" in log_text
        assert "hurried" in log_text
