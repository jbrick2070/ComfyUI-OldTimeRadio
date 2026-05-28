"""tests/test_bug_local_291_editor_token_budget.py

BUG-LOCAL-291 (2026-05-27): bump Editor max_new_tokens 1024 -> 2048.

Live evidence: pending_20260527_205321 ("Winds of Despair") -- the
Editor pass exhausted its 2-attempt retry budget with
"JSONDecodeError: Unterminated string starting at: line 14 column
24 (char 4773)". 1024-token cap was truncating mid-string on real
EditorVerdict output (per-axis notes + overall_note + JSON
overhead = 3500-4500 chars typical).

Same shape as BUG-LOCAL-285 on run_story_brief_reflection.
"""
from __future__ import annotations

from nodes import _otr_editor_pass as _ep


def test_editor_max_new_tokens_at_least_4096():
    """BUG-296 (2026-05-27): the cap must be >= 4096. 2048 still
    truncated attempt 1 at base temp 0.30 on live runs -- the
    model produces verbose JSON at base temp and only fits at
    lower retry temp. 4096 lets attempt 1 land first try."""
    assert _ep._EDITOR_MAX_NEW_TOKENS >= 4096


def test_editor_retry_temperature_strictly_below_base():
    """PD invariant -- structural retries lower entropy, never
    raise it. The cap bump must not touch this."""
    assert _ep._EDITOR_RETRY_TEMPERATURE < _ep._EDITOR_BASE_TEMPERATURE


def test_editor_max_attempts_unchanged():
    """Retry budget stays at 2 -- bumping the cap shouldn't also
    expand the retry ladder (separate concern)."""
    assert _ep._EDITOR_MAX_ATTEMPTS == 2
