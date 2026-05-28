"""tests/test_bug_local_293_extract_token_budget.py

BUG-LOCAL-293 (2026-05-27): bump Extract max_new_tokens 4096 -> 16384.

Live evidence: pending_20260527_212428 (18-line episode). Extract
attempt 1 failed on callback_to='none' (BUG-292), attempt 2 at
lower temp produced longer JSON and truncated mid-string at
char 12358. Ladder exhausted, Commit pass-through'd, Story Room
iteration didn't reach the ledger.

The schema-wrapped JSON for an 18-line episode runs ~3000-4000
tokens; the 4096 cap was starving Mistral-Nemo into truncation.
Same shape as BUG-285 (story_brief) and BUG-291 (Editor).
"""
from __future__ import annotations

from nodes import _otr_story_room_extract as _ex


def test_extract_max_new_tokens_at_least_16384():
    """Cap >= 16384 so 18-30 line episodes don't truncate."""
    assert _ex._EXTRACT_MAX_NEW_TOKENS >= 16384


def test_extract_retry_temp_strictly_below_base():
    """PD invariant -- structural retries lower entropy, never raise it."""
    assert _ex._EXTRACT_STRUCTURAL_RETRY_TEMPERATURE < \
        _ex._EXTRACT_BASE_TEMPERATURE
