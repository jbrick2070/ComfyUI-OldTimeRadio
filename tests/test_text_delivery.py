"""720-bakeoff C2 (S2 P1.3): TTS delivery-text routing.

Covers:
* the science_news byte-parity contract (legacy lanes speak canonical
  text unchanged -- the spine stays byte-identical before AND after
  C2/C3);
* the content-owned delivery stamp + the single resolver
  (fresh/absent/stale);
* the two-bus consumed==provided terminal check (surplus + shortfall).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes._otr_script_prep import prepare_text  # noqa: E402
from nodes._otr_readiness import (  # noqa: E402
    normalize_for_delivery,
    stamp_text_for_tts_delivery,
    text_for_tts_source_sha256,
)
from nodes._otr_text_delivery import (  # noqa: E402
    CONTENT_OWNED,
    LEGACY,
    TextDeliveryError,
    delivery_mode_for_meta,
    resolve_line_delivery,
)
from nodes.scene_sequencer import _verify_bus_clip_counts  # noqa: E402
from nodes import production_ledger as _PL  # noqa: E402


# ---------------------------------------------------------------------------
# 1. science_news byte-parity fixture (MUST hold before AND after C2/C3)
# ---------------------------------------------------------------------------

# Representative science_news-style canonical lines -> the EXACT delivery
# string the spine has always sent to TTS (prepare_text of the canonical,
# because legacy canonical is already Phase-7-normalized in place). Pinned
# literally so any drift in the legacy delivery path fails loud.
_BYTE_PARITY_GOLDEN = {
    "We're flying blind without GPS.": "We're flying blind without GPS.",
    "EDNA: The signal is gone.": "The signal is gone.",
    "(whispering) They are listening.": "They are listening.",
    "The tape *is* the tape.": "The tape is the tape.",
    "Wait… what was that?": "Wait... what was that?",
}


@pytest.mark.parametrize("canonical,expected", list(_BYTE_PARITY_GOLDEN.items()))
def test_science_news_byte_parity_prepared(canonical, expected):
    """The prepared delivery of a legacy line is byte-identical to the
    pinned golden (locks the spine text against prep drift)."""
    assert prepare_text(canonical) == expected


@pytest.mark.parametrize("canonical", list(_BYTE_PARITY_GOLDEN))
def test_legacy_resolver_is_passthrough(canonical):
    """The resolver's LEGACY branch returns delivery == canonical, so the
    prepared bytes are unchanged vs a direct prepare_text(canonical)."""
    row = {"line_id": "l1", "text": canonical}
    got_canon, got_delivery = resolve_line_delivery(row, LEGACY)
    assert got_canon == canonical
    assert got_delivery == canonical
    assert prepare_text(got_delivery) == prepare_text(canonical)


def test_legacy_passthrough_ignores_any_text_for_tts_stamp():
    """Even if a stray text_for_tts is present, LEGACY mode never uses it
    -- byte-parity depends on canonical, not a stamp."""
    row = {"line_id": "l1", "text": "hello", "text_for_tts": "GOODBYE"}
    _c, delivery = resolve_line_delivery(row, LEGACY)
    assert delivery == "hello"


# ---------------------------------------------------------------------------
# 2. delivery mode resolution
# ---------------------------------------------------------------------------

def test_delivery_mode_science_news_is_legacy():
    assert delivery_mode_for_meta({"source_bank": "science_news"}) == LEGACY


def test_delivery_mode_untagged_is_legacy():
    assert delivery_mode_for_meta({}) == LEGACY
    assert delivery_mode_for_meta(None) == LEGACY


def test_delivery_mode_scifi_fable2_is_content_owned():
    assert delivery_mode_for_meta({"source_bank": "scifi_fable2"}) == CONTENT_OWNED


# ---------------------------------------------------------------------------
# 3. normalize_for_delivery + stamp
# ---------------------------------------------------------------------------

def test_normalize_for_delivery_expands_abbrev_and_numbers():
    delivery, receipt = normalize_for_delivery("Dr. Vale checked 3 dials.")
    assert delivery.startswith("Doctor Vale")
    assert "3" not in delivery  # digits spelled out
    assert receipt["abbreviations_expanded"] == 1
    assert receipt["changed"] is True


def test_normalize_for_delivery_noop_leaves_text_and_flags_unchanged():
    delivery, receipt = normalize_for_delivery("Nothing to expand here.")
    assert delivery == "Nothing to expand here."
    assert receipt["changed"] is False


def _voiced(line_id, text, **over):
    r = {"line_id": line_id, "beat_id": line_id, "shot_id": "sh01",
         "char_id": "c01", "speaker_role": "character", "boundary": None,
         "text": text, "char_count": len(text),
         "word_count": len(text.split())}
    r.update(over)
    return r


def test_stamp_leaves_canonical_untouched_and_stamps_delivery(tmp_path):
    led = _PL.new_ledger(episode_id="stamp_ep", out_dir=str(tmp_path / "ep"))
    led.set_lines([
        _voiced("l1", "Dr. Vale saw 2 lights."),
        _voiced("l2", "", skip=True, tts_skip_reason="reviewer_skip: flat"),
        {"line_id": "m1", "beat_id": "m1", "shot_id": "sh00",
         "char_id": "music_open", "speaker_role": "music_open",
         "boundary": None, "text": ""},
    ])
    before_text = led.data["lines"][0]["text"]
    before_wc = led.data["lines"][0]["word_count"]
    summary = stamp_text_for_tts_delivery(led)
    row = led.data["lines"][0]
    # canonical untouched
    assert row["text"] == before_text
    assert row["word_count"] == before_wc
    # delivery stamped + normalized + sha correct + receipt present
    assert row["text_for_tts"].startswith("Doctor Vale")
    assert row["text_for_tts_source_sha256"] == text_for_tts_source_sha256(before_text)
    assert row["text_for_tts_receipt"]["changed"] is True
    # skip line + music line NOT stamped
    assert "text_for_tts" not in led.data["lines"][1]
    assert "text_for_tts" not in led.data["lines"][2]
    assert summary["lines_stamped"] == 1


def test_stamp_covers_line_even_when_delivery_equals_canonical(tmp_path):
    led = _PL.new_ledger(episode_id="stamp_ep2", out_dir=str(tmp_path / "ep"))
    led.set_lines([_voiced("l1", "Plain spoken words.")])
    stamp_text_for_tts_delivery(led)
    row = led.data["lines"][0]
    assert row["text_for_tts"] == "Plain spoken words."  # stamped anyway
    assert row["text_for_tts_source_sha256"]


# ---------------------------------------------------------------------------
# 4. content-owned resolver: fresh / absent / stale
# ---------------------------------------------------------------------------

def test_content_owned_resolver_returns_fresh_stamp():
    canonical = "Dr. Vale saw 2 lights."
    row = {
        "line_id": "l1", "text": canonical,
        "text_for_tts": "Doctor Vale saw two lights.",
        "text_for_tts_source_sha256": text_for_tts_source_sha256(canonical),
    }
    got_canon, delivery = resolve_line_delivery(row, CONTENT_OWNED)
    assert got_canon == canonical
    assert delivery == "Doctor Vale saw two lights."


def test_content_owned_resolver_absent_stamp_is_terminal():
    row = {"line_id": "l1", "text": "Dr. Vale saw 2 lights."}
    with pytest.raises(TextDeliveryError):
        resolve_line_delivery(row, CONTENT_OWNED)


def test_content_owned_resolver_empty_stamp_is_terminal():
    row = {"line_id": "l1", "text": "hello", "text_for_tts": "   ",
           "text_for_tts_source_sha256": text_for_tts_source_sha256("hello")}
    with pytest.raises(TextDeliveryError):
        resolve_line_delivery(row, CONTENT_OWNED)


def test_content_owned_resolver_stale_sha_is_terminal():
    # Stamp was made for OLD text; canonical has since changed.
    row = {
        "line_id": "l1", "text": "the NEW canonical text",
        "text_for_tts": "the old delivery text",
        "text_for_tts_source_sha256": text_for_tts_source_sha256("the OLD canonical text"),
    }
    with pytest.raises(TextDeliveryError):
        resolve_line_delivery(row, CONTENT_OWNED)


# ---------------------------------------------------------------------------
# 5. two-bus consumed==provided terminal check
# ---------------------------------------------------------------------------

def test_bus_counts_exact_match_passes():
    _verify_bus_clip_counts(
        announcer_consumed=2, announcer_provided=2,
        character_consumed=3, character_provided=3,
        announcer_line_ids=["a1", "a2"], character_line_ids=["c1", "c2", "c3"])


def test_bus_counts_announcer_surplus_fails():
    with pytest.raises(ValueError, match="clip-count MISMATCH"):
        _verify_bus_clip_counts(
            announcer_consumed=1, announcer_provided=3,
            character_consumed=2, character_provided=2)


def test_bus_counts_character_surplus_fails():
    with pytest.raises(ValueError, match="clip-count MISMATCH"):
        _verify_bus_clip_counts(
            announcer_consumed=1, announcer_provided=1,
            character_consumed=2, character_provided=5)


def test_bus_counts_shortfall_also_fails():
    with pytest.raises(ValueError):
        _verify_bus_clip_counts(
            announcer_consumed=3, announcer_provided=1,
            character_consumed=2, character_provided=2)
