"""KILL 2 -- C2: the announcer OPEN by INPUT STARVATION (SafeOpenBrief).

Under story_scaffold, the open is built from a SafeOpenBrief (setting / time /
status-quo / cast / era) captured before build_sq_data mutates the setup beat;
the script_brief -- which can carry the outcome -- is NEVER passed, so no spoiler
can leak (the outcome is never an input). The fallback is equally starved.

With the flag off, compose_announcer_intro runs its ORIGINAL path verbatim
(real script_brief, original system prompt) -- byte-identical.

Pure Python; no GPU, no LLM, no network.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import _otr_line_composer as LC  # noqa: E402


def _make_fn(reply, capture):
    def _fn(messages, **kw):
        capture.append(messages)
        return reply
    return _fn


_SB = LC.SafeOpenBrief(
    setting="a fog-bound lighthouse",
    time_of_day="midnight",
    opening_status_quo="KEEPER waits for a ship that will not come",
    cast=("KEEPER", "MARA"),
    era="1937",
)
_SPOILER = "SPOILER: the ship sank and everyone aboard drowned at the end"
_GOOD_OPEN = (
    "The lamp turns over black water as the keeper waits alone. "
    "What is he listening for?"
)


class TestSafeOpenBrief:
    def test_frozen_and_default_era(self):
        sb = LC.SafeOpenBrief(setting="x", time_of_day="y",
                              opening_status_quo="z", cast=("A",))
        assert sb.era == ""
        with pytest.raises(Exception):
            sb.setting = "other"  # type: ignore[misc]


class TestComposeOpenFlagOn:
    def test_uses_safe_brief_never_script_brief(self):
        cap = []
        res = LC.compose_announcer_intro(
            creative_fn=_make_fn(_GOOD_OPEN, cap),
            script_brief=_SPOILER, story_scaffold=True, safe_open_brief=_SB,
        )
        assert res.compose_flags == ("announcer_intro",)
        assert res.text == _GOOD_OPEN
        sysmsg, usermsg = cap[0][0]["content"], cap[0][1]["content"]
        # the spoiler brief reached NEITHER message
        assert "SPOILER" not in sysmsg and "SPOILER" not in usermsg
        assert "sank" not in usermsg.lower() and "drowned" not in usermsg.lower()
        # the safe system prompt + the locked cast were used
        assert sysmsg == LC._ANNOUNCER_INTRO_SYSTEM_SAFE
        assert "do not reveal" in sysmsg.lower()
        assert "KEEPER" in usermsg and "MARA" in usermsg
        assert "a fog-bound lighthouse" in usermsg

    def test_safe_open_validation_fail_fails_loud(self):
        # NO-FALLBACK (2026-07-03): a too-short safe-open reply fails the char band
        # -> RAISES; no deterministic safe-open template ships.
        import pytest
        cap = []
        with pytest.raises(RuntimeError, match="no-fallback"):
            LC.compose_announcer_intro(
                creative_fn=_make_fn("Hush.", cap),
                script_brief=_SPOILER, story_scaffold=True, safe_open_brief=_SB,
            )

    def test_no_brief_falls_through_to_original_path(self):
        # story_scaffold on but no safe_open_brief -> the guard requires BOTH,
        # so the original path runs (defensive; never crashes).
        cap = []
        res = LC.compose_announcer_intro(
            creative_fn=_make_fn(_GOOD_OPEN, cap),
            script_brief="a quiet tale about a signal in the archives",
            story_scaffold=True, safe_open_brief=None,
        )
        assert cap[0][0]["content"] == LC._ANNOUNCER_INTRO_SYSTEM
        assert res.compose_flags == ("announcer_intro",)


class TestComposeOpenFlagOff:
    def test_off_runs_original_path_with_script_brief(self):
        cap = []
        res = LC.compose_announcer_intro(
            creative_fn=_make_fn(_GOOD_OPEN, cap),
            script_brief="a quiet tale about a signal in the archives",
            story_scaffold=False, safe_open_brief=_SB,
        )
        sysmsg, usermsg = cap[0][0]["content"], cap[0][1]["content"]
        assert sysmsg == LC._ANNOUNCER_INTRO_SYSTEM           # original prompt
        assert "a quiet tale about a signal" in usermsg       # real brief used
        assert res.compose_flags == ("announcer_intro",)


class TestFallbackSafeOpen:
    def test_builds_from_setting_only(self):
        out = LC.fallback_safe_open(LC.SafeOpenBrief(
            setting="an abandoned radio station", time_of_day="dawn",
            opening_status_quo="ignored here", cast=("RO",), era="",
        ))
        assert "abandoned radio station" in out
        assert out.startswith("Good evening. This is SIGNAL LOST.")

    def test_empty_brief_still_a_complete_sentence(self):
        out = LC.fallback_safe_open(LC.SafeOpenBrief(
            setting="", time_of_day="", opening_status_quo="", cast=(),
        ))
        assert out.endswith("Stay with us.")


# ---------------------------------------------------------------------------
# Writer wiring (source-level; run() needs the full model stack).
# ---------------------------------------------------------------------------
class TestWriterC2Wiring:
    SRC = (
        Path(__file__).resolve().parents[1] / "nodes" / "OTR_LedgerScriptWriter.py"
    ).read_text(encoding="utf-8")

    def test_capture_happens_before_build_sq_data(self):
        i_cap = self.SRC.index("safe_open_brief = None")
        i_build = self.SRC.index("_sq_by_beat = _OTRSQL12.build_sq_data(")
        assert i_cap < i_build

    def test_capture_is_flag_gated(self):
        i_cap = self.SRC.index("safe_open_brief = None")
        i_flag = self.SRC.index("if _style_grammar_on:", i_cap)
        i_sob = self.SRC.index("_OTRLC.SafeOpenBrief(", i_cap)
        assert i_cap < i_flag < i_sob

    def test_intro_call_starves_script_brief_under_flag(self):
        assert '"" if _style_grammar_on else script_brief' in self.SRC
        assert "safe_open_brief=safe_open_brief," in self.SRC
        assert "story_scaffold=_style_grammar_on," in self.SRC


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
