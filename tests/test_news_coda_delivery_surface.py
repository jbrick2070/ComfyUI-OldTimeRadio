"""Live bug 5: exact complete-coda delivery and sibling isolation."""
from __future__ import annotations

import hashlib
import inspect
import json
import re

import pytest

from nodes import _otr_freeze_cascade as FC
from nodes import _otr_line_composer as LC
from nodes._otr_line_composer import (
    LineRequest,
    _news_coda_complete_sentence_prefixes,
    compose_news_coda,
    finalize_news_coda_surface,
    spoken_surface_flags_for_line,
)
from nodes._otr_readiness import normalize_for_delivery
from nodes._otr_reroll import repair_ledger_spoken_hygiene
from nodes._otr_story_rules import resolve_story_rules


LIVE_HEADER = (
    "EPISODE_TITLE: TBD\n"
    "SETTING: The Thomas Jefferson Building.\n"
    "TIME: Late evening.\n"
    "PREMISE: Two archivists find hidden production diaries.\n\n"
    "Specificity anchors (use selectively):\n"
    "- National Film Registry\n"
    "- Library of Congress\n"
    "- Summer Movies on the Lawn\n"
    "- Cultural Heritage\n"
    "- Film Preservation\n"
)
LIVE_BRIDGE = "From tonight's hidden production diaries"
LIVE_FACT = (
    "This summer, the Library of Congress invites the public to the Thomas "
    "Jefferson Building for 'Summer Movies on the Lawn.' Join us as we "
    "celebrate the National Film Registry, featuring the 1976 classic "
    "'Rocky,' a testament to American perseverance and a cornerstone of our "
    "shared cultural heritage."
)
LIVE_FIRST_SENTENCE = (
    "This summer, the Library of Congress invites the public to the Thomas "
    "Jefferson Building for 'Summer Movies on the Lawn.'"
)


def _announcer_req(header: str = LIVE_HEADER) -> LineRequest:
    return LineRequest(
        speaker="ANNOUNCER",
        intent="read the source note",
        mood="reflective",
        target_words=15,
        canon_header=header,
        last_lines=[],
        speaker_role="announcer",
        words_per_beat_range=(18, 40),
    )


def _projected_flags(text: str, req: LineRequest, bank: str):
    projected, _receipt = normalize_for_delivery(text)
    grounded_caps = tuple(re.findall(r"\b[A-Z][A-Z0-9-]{1,}\b", text))
    return spoken_surface_flags_for_line(
        projected,
        req,
        rules=resolve_story_rules(bank),
        allowed_all_caps=grounded_caps,
    )


def test_live_b009_first_pass_keeps_exact_clean_sentence_and_never_rewrites_fact():
    captured = []

    def bridge_writer(messages, **kwargs):
        captured.append((messages, kwargs))
        return LIVE_BRIDGE

    result = compose_news_coda(
        creative_fn=bridge_writer,
        news_close_brief=LIVE_FACT,
        premise="Two archivists find hidden production diaries.",
        cast_seed=17,
        source_bank_id="media_archive",
        canon_header=LIVE_HEADER,
        words_per_beat_range=(18, 40),
    )

    assert result.text.endswith(LIVE_FIRST_SENTENCE)
    assert "news_coda_fact_reduced" in result.compose_flags
    assert "news_coda_truncated" in result.compose_flags
    assert "news_coda_fact_deferred_to_credits" not in result.compose_flags
    assert not _projected_flags(result.text, _announcer_req(), "media_archive")
    # The bridge model never receives any source fact, including the part that
    # the deterministic finalizer ultimately selected.
    all_prompts = json.dumps(captured, ensure_ascii=False)
    assert "Library of Congress" not in all_prompts
    assert "National Film Registry" not in all_prompts


def test_sentence_prefix_parser_never_splits_initials_or_versions():
    fact = (
        "This episode adapts H. G. Wells's novel The Time Machine. "
        "The source license is CC BY-NC 4.0. Credits identify the edition."
    )
    prefixes = _news_coda_complete_sentence_prefixes(fact)
    assert prefixes == (
        "This episode adapts H. G. Wells's novel The Time Machine. "
        "The source license is CC BY-NC 4.0.",
        "This episode adapts H. G. Wells's novel The Time Machine.",
    )
    assert all(not value.endswith(("H.", "G.", "4.")) for value in prefixes)


def test_finalizer_never_returns_an_unscored_emergency_sentence(monkeypatch):
    monkeypatch.setattr(
        LC,
        "spoken_surface_flags_for_line",
        lambda *args, **kwargs: [
            ("forced_dirty", "forced dirty surface", "forced_dirty")
        ],
    )
    result = finalize_news_coda_surface(
        bridge=LIVE_BRIDGE,
        fact=LIVE_FACT,
        req=_announcer_req(),
        rules=resolve_story_rules("media_archive"),
    )
    assert result.text == ""
    assert "hygiene_row_failed_mechanical" in result.compose_flags
    assert (
        "hygiene_row_failed_mechanical:news_coda_no_clean_surface"
        in result.compose_flags
    )


@pytest.mark.parametrize(
    ("bank", "fact", "anchors"),
    (
        (
            "public_domain",
            (
                "This episode adapts H. G. Wells's public-domain novel The "
                "Time Machine, preserving the return, the witnesses, and the "
                "unsettling proof of the journey."
            ),
            ("H. G. Wells", "The Time Machine", "the witnesses"),
        ),
        (
            "shakespeare",
            (
                "This episode adapts Macbeth, Act 1, Scene 3 from Folger "
                "Shakespeare, a CC BY-NC scene source used here for "
                "noncommercial study and adaptation."
            ),
            ("Macbeth", "Act 1, Scene 3", "Folger Shakespeare"),
        ),
    ),
)
def test_single_sentence_source_note_defers_intact_fact_to_credits(
    bank,
    fact,
    anchors,
):
    header = (
        "TITLE: x\nSETTING: y\nTIME: z\n"
        "Specificity anchors (use selectively):\n"
        + "".join(f"- {anchor}\n" for anchor in anchors)
        + "\nPREMISE: w"
    )
    req = _announcer_req(header)
    result = finalize_news_coda_surface(
        bridge="From tonight's source record",
        fact=fact,
        req=req,
        rules=resolve_story_rules(bank),
    )
    if "news_coda_fact_deferred_to_credits" in result.compose_flags:
        assert result.text in {
            "For tonight's source note: See tonight's credits.",
            "The complete source note appears in tonight's credits.",
            "See tonight's credits for the source note.",
            "See the credits.",
        }
    else:
        # A single sentence is indivisible: it is either retained exactly or
        # deferred wholesale, never clipped into a newly punctuated fragment.
        assert result.text.endswith(fact)
    assert not _projected_flags(result.text, req, bank)


def _live_ledger(text: str = f"{LIVE_BRIDGE}: {LIVE_FACT}") -> dict:
    return {
        "cast": [{"char_id": "announcer", "name": "ANNOUNCER"}],
        "lines": [{
            "line_id": "b009",
            "beat_id": "b009",
            "char_id": "announcer",
            "speaker_role": "announcer",
            "text": text,
            "char_count": len(text),
            "word_count": len(text.split()),
            "target_words": 15,
            "compose_flags": ["news_coda_bridge", "news_coda_truncated"],
            "skip": False,
        }],
        "meta": {
            "source_bank": "media_archive",
            "canon_header": LIVE_HEADER,
            "words_per_beat_range": [18, 40],
            "news": {"news_close_brief": LIVE_FACT},
        },
    }


def test_dirty_old_coda_is_repaired_as_combined_surface_with_hash_receipt():
    # Exact live-era failure shape: the row already contains a manufactured
    # character-cut suffix, while meta.news retains the authoritative note.
    old_cut = (
        f"{LIVE_BRIDGE}: This summer, the Library of Congress invites the "
        "public to the Thomas Jefferson Building for 'Summer Movies on the "
        "Lawn.' Join us as we celebrate the National Film Registry, featuring "
        "the 1976."
    )
    ledger = _live_ledger(old_cut)
    report = repair_ledger_spoken_hygiene(
        ledger,
        creative_fn=lambda *args, **kwargs: LIVE_BRIDGE,
    )
    line = ledger["lines"][0]
    assert report.scanned == 1
    assert report.repaired == 1
    assert report.failed == 0
    assert line["text"].endswith(LIVE_FIRST_SENTENCE)
    assert "featuring the 1976" not in line["text"]
    assert "news_coda_fact_reduced" in line["compose_flags"]
    assert not _projected_flags(line["text"], _announcer_req(), "media_archive")

    receipt = ledger["meta"]["news_coda_spoken_reduction"]
    assert receipt["line_id"] == "b009"
    assert receipt["action"] == "fact_reduced"
    assert receipt["source_fact_sha256"] == hashlib.sha256(
        LIVE_FACT.encode("utf-8")
    ).hexdigest()
    assert receipt["source_fact_retained_in_meta_news"] is True
    assert ledger["meta"]["news"]["news_close_brief"] == LIVE_FACT


def test_phase7_expansion_then_combined_coda_scour_is_clean():
    fact = (
        "The source record remains available. Archive references 999999, "
        "999999, and 999999 for the complete index."
    )
    ledger = _live_ledger(f"From tonight's archive: {fact}")
    ledger["meta"]["news"]["news_close_brief"] = fact
    # Model the exact legacy order: Phase 7 expands delivery in canonical text,
    # then the shared scour verifies that expanded row.
    FC._phase_7_audio_readiness(type("L", (), {"data": ledger})(), enable=True)
    assert "nine hundred" in ledger["lines"][0]["text"]

    report = repair_ledger_spoken_hygiene(
        ledger,
        creative_fn=lambda *args, **kwargs: "From tonight's archive",
    )
    assert report.repaired == 1
    assert ledger["lines"][0]["text"].endswith(
        "The source record remains available."
    )
    assert not _projected_flags(
        ledger["lines"][0]["text"], _announcer_req(), "media_archive",
    )


def test_row_local_failure_details_are_returned_for_phase_receipts():
    ledger = {
        "cast": [{"char_id": "c01", "name": "EDNA"}],
        "lines": [{
            "line_id": "b001",
            "beat_id": "b001",
            "char_id": "c01",
            "speaker_role": "character",
            "text": "...",
            "compose_flags": [],
            "skip": False,
        }],
        "meta": {"source_bank": "media_archive"},
    }
    report = repair_ledger_spoken_hygiene(
        ledger,
        creative_fn=lambda *args, **kwargs: "...",
    )
    assert report.failed == 1
    assert len(report.failures) == 1
    failure = report.failures[0]
    assert failure["line_id"] == "b001"
    assert failure["original_sha256"] == hashlib.sha256(b"...").hexdigest()
    assert failure["failure"] == "spoken_hygiene_unspeakable_row"
    assert "hygiene_row_failed_mechanical" in failure["repair_flags"]
    assert ledger["lines"][0]["skip"] is True
    source = inspect.getsource(FC)
    assert "for failure in _p7_hygiene.failures" in source
    assert "for receipt in _p7_hygiene.receipts" not in source


@pytest.mark.parametrize("bank", ("scifi_news", "scifi_news_pro"))
def test_content_owned_direct_scour_is_byte_identical(bank):
    ledger = {
        "cast": [{"char_id": "c01", "name": "EDNA"}],
        "lines": [{
            "line_id": "l001",
            "beat_id": "b001",
            "char_id": "c01",
            "speaker_role": "character",
            "text": "You're playing with fire.",
            "text_for_tts": "You're playing with fire.",
            "compose_flags": [],
            "skip": False,
        }],
        "meta": {
            "source_bank": bank,
            "content_authorship": {"seal": "unchanged"},
        },
    }
    before = json.dumps(ledger, sort_keys=True, ensure_ascii=False)
    report = repair_ledger_spoken_hygiene(
        ledger,
        creative_fn=lambda *args, **kwargs: "A changed line.",
    )
    after = json.dumps(ledger, sort_keys=True, ensure_ascii=False)
    assert report.scanned == report.repaired == report.failed == 0
    assert before == after
