"""The announcer's frame may not name a work it is not performing tonight.

WHY THIS FILE EXISTS. Two shipped `shakespeare` episodes were framed as the
wrong play: a Twelfth Night scene announced as *"Verona ... Capulets and
Montagues"*, and a Tempest scene framed as Romeo and Juliet. The announcer was
ordered by its seam to name "the play-world place" while the prompt carried no
play at all, so the model supplied one from memory.

WHY THE OBVIOUS CHECK IS THE WRONG ONE, measured. The first proposal was
"assert no description contains another CAST ROW's name". Over the real corpus
that check returns mostly legitimate relational prose, and it would NOT have
caught this defect -- "Verona" is nobody's cast row. The second proposal was
"assert the setting contains no proper name outside the locked cast". That one
REJECTS THE CORRECT SETTING on five of the fourteen manifest rows: Hamlet is
"a cold platform at Elsinore", Romeo and Juliet is "In Capulet's garden",
Comedy of Errors is "Antipholus of Ephesus", plus Arden and the wood near
Athens. None of those places is a cast member, and vaguing them to satisfy an
assertion is anti-fidelity on the lane where fidelity outranks arc.

WHAT IS LEFT IS THE INVERSION, and it is what this file asserts: a frame may
name its OWN work freely and may not name a DIFFERENT one. That is finite,
falsifiable against the manifest, and cannot reject legitimate writing.

THE CORPUS IS BUILT FROM THE MANIFEST, never hand-listed. A hand-written
blocklist of forbidden words is the shape a QA pass walked 14 of 16 ways past
on 2026-08-17; it also goes stale the moment a scene is added.

THE LIMIT, STATED HONESTLY: this catches CROSS-PLAY leakage only. It cannot
catch a wholly invented place belonging to no play in the manifest. That
residue is what the live leg is for, and no green run here may be reported as
"the wrong-play frame is fixed".

PER THE LAW: a test/receipt assertion, never a runtime reject. Nothing in this
file is imported by the generation path.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from nodes import _otr_line_composer as LC  # noqa: E402

MANIFEST = (REPO / "config" / "source_banks" / "shakespeare"
            / "curated_scenes.sample.json")

# Words that are capitalized in a synopsis without naming a place or house.
_NOT_A_SIGNATURE = {
    "A", "An", "The", "In", "On", "At", "When", "While", "Three", "Two",
    "Act", "Scene", "Lord", "Lady", "Duke", "King", "Queen", "Sir",
}


def _rows():
    with MANIFEST.open(encoding="utf-8") as handle:
        data = json.load(handle)
    rows = data.get("scenes") or data.get("rows") or data
    assert isinstance(rows, list) and rows, f"no scene rows in {MANIFEST}"
    return rows


def _mid_sentence_capitals(text: str) -> set[str]:
    """Capitalized tokens that are NOT line- or sentence-initial.

    Verse capitalizes every line, so a newline-crossing match would harvest
    "Because", "Lest" and "Awake" as though they were place names. Requiring a
    lowercase letter or comma and then SPACES ON THE SAME LINE keeps that out
    and leaves the genuine proper nouns -- Capulet, Montague, Illyria,
    Malvolio, Elsinore.
    """
    return set(re.findall(r"[a-z,][ \t]+([A-Z][a-z]{3,})\b", text))


def _signature_terms(row) -> set[str]:
    """Proper nouns that identify THIS row's work.

    Three sources, in increasing power: the title, the row's own synopsis and
    cast hints, and the VENDORED SCENE TEXT the row points at. The text is what
    makes this detector real -- the shipped frame said "Capulets and
    Montagues", and neither word appears in the manifest's structured fields.
    """
    terms = set()
    title = str(row.get("play_title") or "").strip()
    if title:
        terms.add(title)
        # "Romeo and Juliet" -> Romeo, Juliet. The shipped defect never used
        # the title, so title-only matching would have missed the very thing
        # this file exists for.
        for tok in re.findall(r"[A-Z][a-zA-Z']+", title):
            if tok not in _NOT_A_SIGNATURE:
                terms.add(tok)
    for hint in row.get("cast_hints") or ():
        for tok in re.findall(r"[A-Z][a-zA-Z']+", str(hint)):
            terms.add(tok)
    synopsis = str(row.get("synopsis") or row.get("scene_label") or "")
    for tok in re.findall(r"[A-Z][a-zA-Z']+", synopsis):
        if tok not in _NOT_A_SIGNATURE and len(tok) > 3:
            terms.add(tok)
    text_path = str(row.get("text_path") or "")
    if text_path:
        vendored = MANIFEST.parent / text_path
        if vendored.is_file():
            terms |= _mid_sentence_capitals(
                vendored.read_text(encoding="utf-8", errors="replace"))
    return {t for t in terms if t not in _NOT_A_SIGNATURE}


def _leaks(foreign_terms, frame: str) -> set[str]:
    """Which foreign signatures appear in this frame.

    STEM MATCH, not whole word. The shipped defect wrote "Capulets" and
    "Montagues" -- plurals of the house names -- so a `\\b...\\b` match would
    have let the real failure through while looking rigorous. Anchor the start
    of the word and allow an inflected tail.
    """
    return {
        term for term in foreign_terms
        if re.search(rf"\b{re.escape(term)}(s|s'|'s|es)?\b", frame,
                     re.IGNORECASE)
    }


def test_the_manifest_is_readable_and_populated():
    """Guard the guard. If the manifest moves, every test below would pass
    vacuously on an empty corpus."""
    rows = _rows()
    assert len(rows) >= 10, f"expected the curated manifest, got {len(rows)}"
    assert all(r.get("play_title") for r in rows), "a row has no play_title"


def test_every_row_has_signature_terms_to_check_against():
    for row in _rows():
        assert _signature_terms(row), f"no signature terms for {row!r}"


#: A term appearing in this many DISTINCT works is vocabulary, not a
#: signature. Shakespeare's own habits supply plenty -- "Jove", "Heaven",
#: "Fortune" -- and flagging those would make the check a false-positive
#: machine, which is exactly why the cast-only proposal was cut.
_SHARED_TERM_WORK_THRESHOLD = 3


def _terms_by_work(rows) -> dict:
    by_work: dict = {}
    for row in rows:
        by_work.setdefault(str(row.get("play_title")), set()).update(
            _signature_terms(row))
    return by_work


def _foreign_terms(row, rows) -> set[str]:
    """Terms belonging to OTHER works only -- never to this one, and never
    shared widely enough to be period vocabulary."""
    by_work = _terms_by_work(rows)
    mine = str(row.get("play_title"))
    own = by_work.get(mine, set())

    spread: dict = {}
    for terms in by_work.values():
        for term in terms:
            spread[term] = spread.get(term, 0) + 1

    foreign: set[str] = set()
    for work, terms in by_work.items():
        if work == mine:
            continue
        foreign |= terms
    return {
        t for t in (foreign - own)
        if spread.get(t, 0) < _SHARED_TERM_WORK_THRESHOLD
    }


def test_the_detector_would_have_caught_the_shipped_defect():
    """THE REGRESSION PIN. The real failure was a Twelfth Night scene framed
    as "Verona ... Capulets and Montagues". If this assertion ever stops
    firing on that string, the detector has stopped detecting."""
    rows = _rows()
    tn = [r for r in rows if r.get("play_title") == "Twelfth Night"]
    assert tn, "Twelfth Night is not in the manifest"
    foreign = _foreign_terms(tn[0], rows)
    shipped_frame = (
        "Tonight, from Verona, where the Capulets and the Montagues keep "
        "their long quarrel."
    )
    hits = _leaks(foreign, shipped_frame)
    assert hits, (
        "the detector does not flag the frame that actually shipped; "
        f"foreign terms were {sorted(foreign)[:12]}"
    )
    # Named explicitly: the hit comes from the HOUSES, not from "Verona".
    # Verona appears nowhere in this manifest or in the vendored balcony
    # scene, so the detector catches this frame on Capulet/Montague alone.
    # That is worth knowing -- a future frame that invents ONLY a place name
    # would slip past, which is what the live leg is for.
    assert {"Capulet", "Montague"} & hits, sorted(hits)


def test_a_correct_frame_for_its_own_play_is_not_flagged():
    """THE FALSE-POSITIVE PIN, and the reason the cast-only check was cut.
    Elsinore is not a cast member and IS the right place for Hamlet 1.1."""
    rows = _rows()
    ham = [r for r in rows if r.get("play_title") == "Hamlet"]
    assert ham, "Hamlet is not in the manifest"
    foreign = _foreign_terms(ham[0], rows)
    good_frame = (
        "Tonight, a scene from Hamlet. On the cold platform at Elsinore, "
        "Horatio keeps a watch he does not believe in."
    )
    hits = {t for t in foreign if re.search(rf"\b{re.escape(t)}\b", good_frame)}
    assert not hits, (
        f"a correct Hamlet frame was flagged as cross-play: {sorted(hits)}"
    )


@pytest.mark.parametrize("title", ["Twelfth Night", "The Tempest", "Macbeth"])
def test_a_composed_frame_carries_its_own_work_and_no_other(title):
    """END TO END through the real composer: the WORK line names tonight's
    work, and no other manifest work's signature appears in the prompt."""
    rows = _rows()
    row = [r for r in rows if r.get("play_title") == title]
    assert row, f"{title} is not in the manifest"

    calls: list = []

    def fn(messages, **kwargs):
        calls.append(messages)
        return "Tonight, a quiet room and one question that will not rest."

    LC.compose_announcer_intro(
        creative_fn=fn,
        script_brief="",
        story_scaffold=True,
        safe_open_brief=LC.SafeOpenBrief(
            setting="a chamber",
            time_of_day="night",
            opening_status_quo="A letter waits on the table.",
            cast=("MARA", "DELACROIX"),
            era="1947",
            work_title=title,
        ),
        source_bank_id="shakespeare",
    )
    assert calls, "the composer never called the model"
    user = calls[0][1]["content"]

    # PRESENCE FIRST. An absence-only assertion is satisfied by an empty
    # prompt, which is how the original defect survived its coverage.
    assert f"WORK: a scene from {title}" in user, user

    foreign = _foreign_terms(row[0], rows)
    leaked = {t for t in foreign if re.search(rf"\b{re.escape(t)}\b", user)}
    assert not leaked, (
        f"the prompt for {title} carries another work's signature: "
        f"{sorted(leaked)}"
    )


# ---------------------------------------------------------------------------
# THE PUBLICATION IS NOT A PLAY.
#
# Caught by the r3 panel against a driver claim that was measurably false. The
# plan asserted that every non-adaptation lane yields an empty work_title. It
# does not: `identity_from_meta` maps media_archive's `source_label` (the FEED,
# e.g. "Now See Hear!") onto the same `work_title` field, and 56 of the 98 live
# media_archive ledgers on disk carry one. Ungated, this change would have made
# every one of those episodes announce "a scene from Now See Hear!" -- inventing
# a play, which is a worse fidelity defect than the wrong-play frame item F was
# opened to fix.
#
# The lesson is the shape, not the string: one field carrying two meanings must
# be gated on the LANE, never on whether the value is non-empty.
# ---------------------------------------------------------------------------


def test_adaptation_kinds_are_exactly_the_two_performing_lanes():
    from nodes import _otr_source_identity as SID

    assert SID.ADAPTATION_SOURCE_KINDS == frozenset(
        {"public_domain", "shakespeare"}
    ), SID.ADAPTATION_SOURCE_KINDS
    assert "media_archive" not in SID.ADAPTATION_SOURCE_KINDS


def test_a_media_archive_publication_is_never_announced_as_a_work():
    """The identity still CARRIES the publisher -- the coda and provenance
    surfaces need it -- but the announcer must not perform it."""
    from nodes import _otr_source_identity as SID

    meta = {
        "source_bank": "media_archive",
        "source_meta": {
            "source_label": "Now See Hear!",
            "post_headline": "A reel returns to the shelf",
        },
    }
    identity = SID.identity_from_meta(meta)
    assert identity.work_title == "Now See Hear!", (
        "the publisher must still reach the provenance surfaces"
    )
    assert identity.source_kind not in SID.ADAPTATION_SOURCE_KINDS, (
        "media_archive must not qualify as a performing lane"
    )


def test_a_shakespeare_scene_does_qualify_as_a_performing_lane():
    """The positive control. Without it the gate above is satisfied by a
    predicate that is false for everything."""
    from nodes import _otr_source_identity as SID

    meta = {
        "source_bank": "shakespeare",
        "source_meta": {"play_title": "Twelfth Night", "play_code": "TN"},
    }
    identity = SID.identity_from_meta(meta)
    assert identity.work_title == "Twelfth Night"
    assert identity.source_kind in SID.ADAPTATION_SOURCE_KINDS
