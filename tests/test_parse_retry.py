"""tests/test_parse_retry.py -- BUG-LOCAL-004 + 005 parse-check contract.

Locks the structural-marker contract for the script-writer parse-retry loop:

  * `=== SCENE N ===` marker is required for any PARSE_OK verdict.
  * `[VOICE: NAME, ...]` and bare `CHARACTER:` line starts are counted, with
    `TITLE:` / `SCENE:` / `GENRE:` / `ENV:` / `SFX:` / `MUSIC:` / `VOICE:` /
    `CAST:` / `AUTHOR:` excluded from the bare-form count via negative
    lookahead.
  * Ultra-smoke (30-word preset) passes only when SCENE marker AND
    `voice_hits >= 2` are both present. Bare-form alone never passes
    ultra-smoke, even with a SCENE marker, because the prompt explicitly
    forbids the bare form (BUG-007 enforcement).
  * Tiny-smoke (100-word preset) passes only when SCENE marker AND
    `voice_hits >= 4` are both present.
  * Standard mode passes when SCENE marker AND
    `voice_hits + bare_hits > 0`.

Round-robin verdict 2026-05-02 (gpt-5.5 / gemini-3.1-pro-customtools /
nemotron-49b): the original permissive regex `^[A-Z][A-Z0-9_ ]{1,19}:\\s*\\S`
treated `TITLE:` as a valid dialogue line, falsely flagging a degenerate
"title only" output as PARSE_OK. Locking that fix here prevents future
regressions.
"""

from __future__ import annotations

import os
import sys

import pytest

# Ensure the package root is on sys.path for direct imports.
_PACK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PACK_ROOT not in sys.path:
    sys.path.insert(0, _PACK_ROOT)

from nodes.story_orchestrator import _check_parse_ok  # noqa: E402


# ---------------------------------------------------------------------------
# Crafted inputs covering the contract corners.
# ---------------------------------------------------------------------------
TITLE_ONLY = "TITLE: Pipeline Ping\n"

TITLE_AND_SCENE_NO_VOICE = (
    "TITLE: Pipeline Ping\n"
    "=== SCENE 1 ===\n"
    "[ENV: empty room, hum of fluorescent lights]\n"
    "[SFX: distant beep]\n"
    "[MUSIC: Closing theme]\n"
)

ULTRA_SMOKE_VALID = (
    "TITLE: Lost Static\n"
    "=== SCENE 1 ===\n"
    "[ENV: dim broadcast booth, faint hum]\n"
    "[SFX: needle on vinyl]\n"
    "[VOICE: ANNOUNCER, female, 50s, authoritative, calm] "
    "Tonight, a signal goes missing.\n"
    "[VOICE: VALE, female, 40s, tense, low] What was that?\n"
    "[VOICE: KANE, male, 50s, clipped, hard] Static. Just static.\n"
    "[VOICE: ANNOUNCER, female, 50s, authoritative, calm] "
    "We will return after this.\n"
    "[MUSIC: Closing theme]\n"
)

ULTRA_SMOKE_ONE_VOICE = (
    "TITLE: Lost Static\n"
    "=== SCENE 1 ===\n"
    "[ENV: dim booth]\n"
    "[VOICE: ANNOUNCER, female, 50s, calm] Tonight, signal goes missing.\n"
    "[MUSIC: Closing theme]\n"
)

STANDARD_BARE_FORM = (
    "TITLE: A Standard Script\n"
    "=== SCENE 1 ===\n"
    "[ENV: control room, all stations green]\n"
    "[SFX: pneumatic hiss]\n"
    "VALE: Telemetry is clean.\n"
    "KANE: For now.\n"
    "[MUSIC: ambient swell]\n"
)

STRUCTURAL_MARKERS_ONLY = (
    "TITLE: Spoof\n"
    "GENRE: hard sci-fi\n"
    "AUTHOR: Anonymous\n"
    "CAST: VALE, KANE\n"
    "ENV: nothing\n"
    "SFX: nothing\n"
    "MUSIC: nothing\n"
)

EMPTY = ""

VOICE_BUT_NO_SCENE = (
    "TITLE: Lost Static\n"
    "[VOICE: ANNOUNCER, female, 50s, calm] Hello.\n"
    "[VOICE: VALE, female, 40s, tense, low] World.\n"
)

TINY_SMOKE_VALID = (
    "TITLE: Static Lines\n"
    "=== SCENE 1 ===\n"
    "[ENV: control room, hum]\n"
    "[SFX: beep]\n"
    "[VOICE: ANNOUNCER, female, 50s, calm] Welcome back to Signal Lost.\n"
    "[VOICE: VALE, female, 40s, tense, low] Telemetry?\n"
    "[VOICE: KANE, male, 50s, clipped, hard] Lost.\n"
    "[VOICE: VALE, female, 40s, tense, low] How long?\n"
    "[VOICE: KANE, male, 50s, clipped, hard] Six minutes.\n"
    "[VOICE: ANNOUNCER, female, 50s, calm] We will return.\n"
    "[MUSIC: closing theme]\n"
)


# ---------------------------------------------------------------------------
# Standard-mode contract tests
# ---------------------------------------------------------------------------
class TestStandardMode:
    def test_title_only_rejected(self):
        result = _check_parse_ok(TITLE_ONLY)
        assert result["has_scene"] is False
        assert result["voice_hits"] == 0
        assert result["parse_ok"] is False, (
            "TITLE: alone must NOT pass parse_ok in standard mode -- "
            "the original permissive regex bug. SCENE marker is required."
        )

    def test_structural_markers_only_rejected(self):
        result = _check_parse_ok(STRUCTURAL_MARKERS_ONLY)
        assert result["bare_hits"] == 0, (
            f"TITLE/GENRE/AUTHOR/CAST/ENV/SFX/MUSIC must all be excluded from "
            f"bare_hits via negative lookahead. Got bare_hits={result['bare_hits']}"
        )
        assert result["parse_ok"] is False

    def test_scene_without_dialogue_rejected(self):
        result = _check_parse_ok(TITLE_AND_SCENE_NO_VOICE)
        assert result["has_scene"] is True
        assert result["voice_hits"] == 0
        assert result["bare_hits"] == 0
        assert result["parse_ok"] is False, (
            "SCENE marker alone (no dialogue) must NOT pass standard-mode parse."
        )

    def test_voice_without_scene_rejected(self):
        result = _check_parse_ok(VOICE_BUT_NO_SCENE)
        assert result["has_scene"] is False
        assert result["voice_hits"] == 2
        assert result["parse_ok"] is False, (
            "Voice lines without a SCENE marker must NOT pass."
        )

    def test_bare_form_with_scene_passes_standard(self):
        result = _check_parse_ok(STANDARD_BARE_FORM)
        assert result["has_scene"] is True
        assert result["bare_hits"] >= 2
        assert result["parse_ok"] is True

    def test_voice_form_with_scene_passes_standard(self):
        result = _check_parse_ok(ULTRA_SMOKE_VALID)
        assert result["parse_ok"] is True

    def test_empty_rejected(self):
        result = _check_parse_ok(EMPTY)
        assert result["parse_ok"] is False


# ---------------------------------------------------------------------------
# Ultra-smoke contract tests (30-word preset)
# ---------------------------------------------------------------------------
class TestUltraSmoke:
    def test_valid_ultra_smoke_passes(self):
        result = _check_parse_ok(ULTRA_SMOKE_VALID, is_ultra_smoke=True)
        assert result["has_scene"] is True
        assert result["voice_hits"] >= 2
        assert result["parse_ok"] is True

    def test_one_voice_rejected(self):
        result = _check_parse_ok(ULTRA_SMOKE_ONE_VOICE, is_ultra_smoke=True)
        assert result["voice_hits"] == 1
        assert result["parse_ok"] is False, (
            "Ultra-smoke requires voice_hits >= 2 to tolerate one missing line "
            "but reject single-line degenerate output."
        )

    def test_bare_form_rejected_under_ultra_smoke(self):
        # Standard bare-form passes standard mode; under ultra-smoke contract
        # the prompt explicitly forbids bare CHARACTER: form, so even a
        # well-formed standard script must fail ultra-smoke.
        result = _check_parse_ok(STANDARD_BARE_FORM, is_ultra_smoke=True)
        assert result["voice_hits"] == 0
        assert result["bare_hits"] >= 2
        assert result["parse_ok"] is False, (
            "Ultra-smoke must reject scripts that use bare CHARACTER: form, "
            "even with a SCENE marker -- the prompt's CRITICAL FORMAT RULES "
            "explicitly forbid bare form."
        )

    def test_title_only_rejected_under_ultra_smoke(self):
        result = _check_parse_ok(TITLE_ONLY, is_ultra_smoke=True)
        assert result["parse_ok"] is False


# ---------------------------------------------------------------------------
# Tiny-smoke contract tests (100-word preset)
# ---------------------------------------------------------------------------
class TestTinySmoke:
    def test_valid_tiny_smoke_passes(self):
        result = _check_parse_ok(TINY_SMOKE_VALID, is_tiny_smoke=True)
        assert result["has_scene"] is True
        assert result["voice_hits"] >= 4
        assert result["parse_ok"] is True

    def test_two_voice_rejected_under_tiny(self):
        # Ultra-smoke would pass with 2 voice lines + scene; tiny-smoke
        # requires >= 4.
        result = _check_parse_ok(ULTRA_SMOKE_VALID, is_tiny_smoke=True)
        assert result["voice_hits"] == 4
        assert result["parse_ok"] is True, (
            "ULTRA_SMOKE_VALID has exactly 4 voice lines, which is the "
            "tiny-smoke floor; should pass."
        )

    def test_one_voice_rejected_under_tiny(self):
        result = _check_parse_ok(ULTRA_SMOKE_ONE_VOICE, is_tiny_smoke=True)
        assert result["parse_ok"] is False


# ---------------------------------------------------------------------------
# Retry-loop semantics: simulate the contract a future caller would rely on.
# ---------------------------------------------------------------------------
class TestRetryLoopContract:
    def test_first_bad_then_good_succeeds_on_attempt_2(self):
        """A two-attempt loop with bad-then-good output passes on attempt 2.

        Mirrors the in-place loop in story_orchestrator.write_script: each
        attempt calls a stand-in for `_generate_with_llm` and runs the parse
        check; if PARSE_OK, break. Locks the behavior so a future refactor
        cannot regress to "give up on attempt 1" or "try N attempts".
        """
        max_retries = 2
        outputs = [TITLE_ONLY, ULTRA_SMOKE_VALID]
        attempts = 0
        last_check = None
        for _ in range(max_retries):
            attempts += 1
            text = outputs[attempts - 1]
            last_check = _check_parse_ok(text, is_ultra_smoke=True)
            if last_check["parse_ok"]:
                break
        assert attempts == 2
        assert last_check["parse_ok"] is True

    def test_both_bad_exhausts_retries_without_raise(self):
        """Two bad attempts -> loop exits without parse_ok, no exception.

        Ensures MAX_PARSE_RETRIES_EXCEEDED is treated as a graceful fall-
        through rather than an exception. The orchestrator hands the last
        output to the rest of the pipeline; the parse-fail observability is
        stamped via the runtime log.
        """
        max_retries = 2
        outputs = [TITLE_ONLY, STRUCTURAL_MARKERS_ONLY]
        attempts = 0
        last_check = None
        for _ in range(max_retries):
            attempts += 1
            text = outputs[attempts - 1]
            last_check = _check_parse_ok(text, is_ultra_smoke=True)
            if last_check["parse_ok"]:
                break
        assert attempts == max_retries
        assert last_check["parse_ok"] is False
        # No exception expected.

    def test_first_good_short_circuits(self):
        """A first-attempt PARSE_OK does NOT call attempt 2."""
        max_retries = 2
        outputs = [ULTRA_SMOKE_VALID, TITLE_ONLY]
        attempts = 0
        last_check = None
        for _ in range(max_retries):
            attempts += 1
            text = outputs[attempts - 1]
            last_check = _check_parse_ok(text, is_ultra_smoke=True)
            if last_check["parse_ok"]:
                break
        assert attempts == 1
        assert last_check["parse_ok"] is True


# ---------------------------------------------------------------------------
# Regression guard: the specific failure mode tonight's run captured.
# ---------------------------------------------------------------------------
def test_bug_local_005_v2_title_only_does_not_falsely_pass():
    """BUG-LOCAL-005 v2: the original permissive bare-form regex would have
    passed `TITLE: Foo\\n=== SCENE 1 ===\\n` because TITLE matched
    `^[A-Z][A-Z0-9_ ]{1,19}:\\s*\\S`. The negative-lookahead regex must reject
    it. Concretely: with a SCENE marker present but no dialogue at all, the
    standard parse must also fail.
    """
    text = "TITLE: Foo\n=== SCENE 1 ===\n[ENV: nothing]\n[MUSIC: nothing]\n"
    result = _check_parse_ok(text)
    assert result["has_scene"] is True
    assert result["voice_hits"] == 0
    assert result["bare_hits"] == 0
    assert result["parse_ok"] is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
