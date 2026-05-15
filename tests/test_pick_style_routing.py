"""S32 B2 -- pick_style per-sub-pass routing.

S31 (and S32 B1): both passes routed through `creative_fn`.
S32 B2 (this commit): pass 2 (chooser) flips to `technical_fn`.
The chooser is a structured short-output index pick; the
canonical slot for that pass is technical.

Tests:
* `test_pick_style_pass1_uses_creative` -- runtime: pass 1
  (inventor) hits `creative_fn`.
* `test_pick_style_pass2_uses_technical` -- runtime: pass 2
  (chooser) hits `technical_fn`.
* `test_pick_style_meta_records_per_pass_slot` -- returned
  StylePick has `pass1_slot == "creative"` and
  `pass2_slot == "technical"`.
* `test_audio_c7_byte_identical_b2` -- pinned in
  `tests/test_audio_byte_identical.py` (canary; default config
  creative == technical means both passes hit the same model
  either way).
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


_VALID_INVENTOR_RESPONSE = (
    "1. closed_room_suspense\n"
    "2. noir_interrogation\n"
    "3. arctic_research_horror\n"
    "4. desert_outpost_thriller\n"
    "5. jungle_expedition_mystery\n"
)


def _build_fns():
    """Construct distinct creative_fn / technical_fn instances with
    call-tracking. The inventor pass returns the canned 5-candidate
    response; the chooser pass returns "1".
    """
    creative_calls: list[dict] = []
    technical_calls: list[dict] = []

    def creative_fn(messages, *, temperature, max_new_tokens):
        creative_calls.append({"temperature": temperature, "max_new_tokens": max_new_tokens})
        return _VALID_INVENTOR_RESPONSE

    def technical_fn(messages, *, temperature, max_new_tokens):
        technical_calls.append({"temperature": temperature, "max_new_tokens": max_new_tokens})
        # Chooser returns one of the candidate strings (not an index).
        return "closed_room_suspense"

    return creative_fn, technical_fn, creative_calls, technical_calls


def test_pick_style_pass1_uses_creative():
    """Pass 1 (inventor) MUST hit creative_fn, not technical_fn."""
    from nodes import _otr_style_picker as sp

    cf, tf, c_calls, t_calls = _build_fns()
    try:
        sp.pick_style(
            creative_fn=cf,
            technical_fn=tf,
            article_text="A test article about radio drama production.",
            seed_pool=["s1", "s2", "s3", "s4", "s5", "s6"],
            rng=random.Random(42),
            model_id="test/model",
        )
    except Exception:
        # Even if the chooser returns a value not in candidates, the
        # routing assertions below still hold.
        pass

    # Pass 1 must have hit creative.
    assert len(c_calls) >= 1, (
        "Inventor pass (pass 1) must call creative_fn at least once. "
        f"Got 0 creative calls (technical: {len(t_calls)})."
    )


def test_pick_style_pass2_uses_technical():
    """Pass 2 (chooser) MUST hit technical_fn (the S32 B2 flip).
    Distinct call-tracking on each fn confirms the flip landed.
    """
    from nodes import _otr_style_picker as sp

    cf, tf, c_calls, t_calls = _build_fns()
    try:
        sp.pick_style(
            creative_fn=cf,
            technical_fn=tf,
            article_text="A test article about radio drama production.",
            seed_pool=["s1", "s2", "s3", "s4", "s5", "s6"],
            rng=random.Random(42),
            model_id="test/model",
        )
    except Exception:
        pass

    assert len(t_calls) >= 1, (
        "Chooser pass (pass 2) must call technical_fn at least once "
        "after S32 B2. Got 0 technical calls. The B1->B2 flip did "
        "not land or the per-sub-pass routing inside pick_style is "
        "still routing pass 2 through creative."
    )


def test_pick_style_meta_records_per_pass_slot():
    """StylePick.pass1_slot == "creative" and
    StylePick.pass2_slot == "technical" -- the forensic stamps that
    let downstream consumers audit slot routing without re-deriving
    from the writer's slot scheduler.
    """
    from nodes import _otr_style_picker as sp

    cf, tf, _, _ = _build_fns()
    pick = sp.pick_style(
        creative_fn=cf,
        technical_fn=tf,
        article_text="A test article about radio drama production.",
        seed_pool=["s1", "s2", "s3", "s4", "s5", "s6"],
        rng=random.Random(42),
        model_id="test/model",
    )

    assert pick.pass1_slot == "creative", (
        f"StylePick.pass1_slot must be \"creative\"; "
        f"got {pick.pass1_slot!r}."
    )
    assert pick.pass2_slot == "technical", (
        f"StylePick.pass2_slot must be \"technical\" (S32 B2 flip); "
        f"got {pick.pass2_slot!r}."
    )
