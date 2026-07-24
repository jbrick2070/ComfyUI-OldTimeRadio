"""Canonical-versus-TTS delivery behavior for all six banks."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import production_ledger as PL
from nodes._otr_readiness import (
    normalize_for_delivery,
    stamp_text_for_tts_delivery,
    text_for_tts_source_sha256,
)
from nodes._otr_text_delivery import (
    CONTENT_OWNED,
    LEGACY,
    TextDeliveryError,
    delivery_mode_for_meta,
    resolve_line_delivery,
)
from nodes.scene_sequencer import _verify_bus_clip_counts


@pytest.mark.parametrize(
    "bank,expected",
    [
        ("media_archive", LEGACY),
        ("original", LEGACY),
        ("public_domain", LEGACY),
        ("shakespeare", LEGACY),
        ("scifi_news", CONTENT_OWNED),
        ("scifi_news_pro", CONTENT_OWNED),
    ],
)
def test_six_bank_delivery_modes(bank, expected):
    assert delivery_mode_for_meta({"source_bank": bank}) == expected


def test_legacy_falls_back_to_canonical_without_a_fresh_stamp():
    canonical = "Dr. Vale keeps the velvet chair beside the smoking lamp."
    assert resolve_line_delivery({"line_id": "l1", "text": canonical}, LEGACY) == (
        canonical,
        canonical,
    )
    stale = {
        "line_id": "l1",
        "text": canonical,
        "text_for_tts": "old projection",
        "text_for_tts_source_sha256": text_for_tts_source_sha256("old text"),
    }
    assert resolve_line_delivery(stale, LEGACY) == (canonical, canonical)


def test_legacy_uses_a_fresh_pronunciation_projection():
    canonical = "Dr. Vale checked 2 dials."
    row = {
        "line_id": "l1",
        "text": canonical,
        "text_for_tts": "Doctor Vale checked two dials.",
        "text_for_tts_source_sha256": text_for_tts_source_sha256(canonical),
    }
    assert resolve_line_delivery(row, LEGACY) == (
        canonical,
        "Doctor Vale checked two dials.",
    )


def test_content_owned_requires_a_complete_fresh_projection():
    canonical = "Dr. Vale checked 2 dials."
    fresh = {
        "line_id": "l1",
        "text": canonical,
        "text_for_tts": "Doctor Vale checked two dials.",
        "text_for_tts_source_sha256": text_for_tts_source_sha256(canonical),
    }
    assert resolve_line_delivery(fresh, CONTENT_OWNED) == (
        canonical,
        "Doctor Vale checked two dials.",
    )
    with pytest.raises(TextDeliveryError):
        resolve_line_delivery({"line_id": "l1", "text": canonical}, CONTENT_OWNED)
    with pytest.raises(TextDeliveryError):
        resolve_line_delivery(
            {
                **fresh,
                "text_for_tts_source_sha256": text_for_tts_source_sha256(
                    "different canonical"
                ),
            },
            CONTENT_OWNED,
        )


def test_normalization_is_pronunciation_only():
    canonical = "Dr. Vale checked 3 dials & a meter."
    delivery, receipt = normalize_for_delivery(canonical)
    assert delivery.startswith("Doctor Vale")
    assert " and " in delivery
    assert receipt["changed"] is True
    assert canonical == "Dr. Vale checked 3 dials & a meter."


def _voiced(line_id, text):
    return {
        "line_id": line_id,
        "beat_id": line_id,
        "shot_id": "sh01",
        "char_id": "c01",
        "speaker_role": "character",
        "boundary": None,
        "text": text,
        "char_count": len(text),
        "word_count": len(text.split()),
    }


def test_stamp_preserves_complete_canonical_story_at_any_length(tmp_path):
    sentence = (
        "Dr. Vale sits in the velvet chair, lights a cigarette, and watches "
        "the ordinary harbor machinery turn beneath a lavender sky. "
    )
    canonical = (sentence * 60).strip()
    ledger = PL.new_ledger("delivery_long", str(tmp_path / "ep"))
    ledger.set_lines([_voiced("l1", canonical)])
    before = dict(ledger.data["lines"][0])

    summary = stamp_text_for_tts_delivery(ledger)

    row = ledger.data["lines"][0]
    assert row["text"] == canonical
    assert row["word_count"] == before["word_count"]
    assert row["char_count"] == before["char_count"]
    assert row["text_for_tts"]
    assert row["text_for_tts_source_sha256"] == text_for_tts_source_sha256(
        canonical
    )
    assert summary["lines_stamped"] == 1


def test_bus_clip_accounting_remains_structural():
    _verify_bus_clip_counts(
        announcer_consumed=2,
        announcer_provided=2,
        character_consumed=3,
        character_provided=3,
    )
    with pytest.raises(ValueError, match="MISMATCH"):
        _verify_bus_clip_counts(
            announcer_consumed=2,
            announcer_provided=1,
            character_consumed=3,
            character_provided=3,
        )
