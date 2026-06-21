"""tests/test_story_engine_v1.py -- story-engine v1 sprint (F1..F8).

Per-feature unit coverage for the content-only story-quality fixes. Pure
Python, no GPU, no LLM (mock generate_fns only). One file, grouped by feature.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes._otr_line_composer import (  # noqa: E402
    LineRequest,
    _build_user_prompt,
    compose_line_draft,
)


def _req(**over):
    base = dict(
        speaker="ALICE",
        intent="reveal the signal",
        mood="tense",
        target_words=15,
        canon_header="TITLE: x\nSETTING: y\nTIME: z\nPREMISE: w",
        last_lines=[("BOB", "What did you find?")],
    )
    base.update(over)
    return LineRequest(**base)


# ===========================================================================
# F1 -- length tail no longer hard-caps every line at 20-30 words
# ===========================================================================

class TestF1LengthTail:

    def test_tail_drops_literal_word_cap(self):
        prompt = _build_user_prompt(_req(target_words=120))
        assert "20-30 words" not in prompt
        assert "about 20-30" not in prompt

    def test_tail_keeps_spoken_cadence_rider(self):
        prompt = _build_user_prompt(_req())
        assert "spoken-length -- one breath" in prompt
        assert "Ground this line in the news facts" in prompt

    def test_word_count_target_still_present(self):
        # the per-line target is still communicated via WRITE LINE
        prompt = _build_user_prompt(_req(target_words=42))
        assert "Word count target: 42." in prompt

    def test_token_budget_scales_to_beat_target(self):
        captured = {}

        def mock_fn(messages, *, temperature, max_new_tokens, **kw):
            captured.setdefault("mnt", max_new_tokens)
            return "ALICE: I found a signal buried in the noise."

        # small beat target -> attempt-1 budget scales (15*4=60 < cap 200)
        compose_line_draft(creative_fn=mock_fn, req=_req(target_words=15))
        assert captured["mnt"] == 60

    def test_token_budget_capped_for_long_beat(self):
        captured = {}

        def mock_fn(messages, *, temperature, max_new_tokens, **kw):
            captured.setdefault("mnt", max_new_tokens)
            return "ALICE: I found a signal buried in the noise."

        # large beat target -> capped at 200 (864*4 would be 3456)
        compose_line_draft(creative_fn=mock_fn, req=_req(target_words=864))
        assert captured["mnt"] == 200

    def test_token_budget_zero_target_uses_full_cap(self):
        captured = {}

        def mock_fn(messages, *, temperature, max_new_tokens, **kw):
            captured.setdefault("mnt", max_new_tokens)
            return "ALICE: I found a signal buried in the noise."

        # zero/falsy target -> full cap, never the 40 floor starving it
        compose_line_draft(creative_fn=mock_fn, req=_req(target_words=0))
        assert captured["mnt"] == 200
