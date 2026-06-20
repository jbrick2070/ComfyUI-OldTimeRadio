"""Regression: Kokoro must offer a DEEP per-character voice pool so CastLock
assigns DISTINCT voices per character (like Bark), not the same voice twice.

Root cause fixed 2026-06-20: the voice reference bank had a single kokoro
entry (bm_george, announcer-only), so for char_voice the casting ladder
collapsed to one male voice and failed for females. We registered the 13
on-disk Kokoro voices (4 female / 9 male) as char_voice entries.
"""
from __future__ import annotations

import os

os.environ.setdefault("OTR_TEST_MODE", "1")

from nodes._otr_voice_bank import assign_voice_for_slot, load_voice_bank


def _kokoro_char_entries():
    bank = load_voice_bank()[0]
    return [e for e in bank
            if e.engine == "kokoro" and "char_voice" in e.roles]


def test_kokoro_char_pool_has_depth_per_gender():
    ko = _kokoro_char_entries()
    males = [e for e in ko if e.gender == "male"]
    females = [e for e in ko if e.gender == "female"]
    # Enough distinct voices per gender that a typical cast never collides.
    assert len(males) >= 3, f"too few male kokoro char voices: {males}"
    assert len(females) >= 2, f"too few female kokoro char voices: {females}"


def test_kokoro_casts_distinct_voices_per_character():
    bank = load_voice_bank()[0]
    used: set = set()
    assigned = []
    for cid, gender in [("c01", "male"), ("c02", "male"),
                        ("c03", "male"), ("c04", "female")]:
        ref = assign_voice_for_slot(
            char_id=cid, role="char_voice", gender=gender,
            engine="kokoro", episode_seed=7,
            used_voice_ref_ids=used, allow_voice_reuse=False, bank=bank,
        )
        used.add(ref.voice_ref_id)
        assigned.append(ref.voice_ref_id)
    assert len(set(assigned)) == len(assigned), (
        f"kokoro reused a voice across characters: {assigned}"
    )
