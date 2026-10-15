"""A salvaged script survives its own seal check (plan item 5, 2026-10-14).

The integrity check used to re-parse the accepted draft against a roster
REBUILT from the speakers in its output. Salvage adopts strangers and leaves
silent cast unspoken, so that rebuilt roster promoted an adopted label to
locked cast and dropped silent originals -- the re-parse resolved labels by a
different rung, and a finished episode was thrown away at the selected-draft
check as "parsed artifact seal is stale". The draft now carries the exact
roster it was parsed against (`FinalDraft.cast_roster`), bound into its seal.

Real parser -> build_final_draft -> _validate_selected_final_draft ->
_final_draft_ledger_payload, with a valid selected attempt receipt.
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import _otr_scifi_news_pro as F2  # noqa: E402
from nodes._otr_scifi_news_pro_markup import parse_scifi_news_pro_markup  # noqa: E402


def _script(lines):
    return "\n".join(
        ["TITLE: The Quiet Relay", "MUSIC: low hum",
         "ANNOUNCER: Tonight, a signal.", "SCENE 1: a relay station"]
        + list(lines)
        + ["ANNOUNCER: The station went quiet.", "CODA: It never stopped.",
           "MUSIC: closing tone", "END."])


def _treatment(cast):
    return F2.Treatment.model_validate({
        "title": "The Quiet Relay",
        "dramatic_question": "Will the doctor trust the relay before dawn?",
        "setting": "a mountain relay station at night",
        "cast_shapes": [{"name": n, "role": "scientist", "want": "the truth",
                         "pressure": "time", "register": "clipped"} for n in cast],
        "turn": "The silence is the message.",
        "priced_ending": {"choice": "The doctor sends the record onward.",
                          "cost_paid": "The last reserve charge is spent."},
        "news_thread": "A survey found measurable heat before tremors.",
        "news_close_read": (
            "The source survey measured heat before tremors at Mount Etna."),
    })


def _draft(cast, lines, *, salvage=True, aliases=None):
    raw = _script(lines)
    parsed, defects = parse_scifi_news_pro_markup(
        raw, list(cast), dict(aliases or {}), salvage=salvage)
    assert parsed is not None, defects
    trace = (F2.PassAttemptTrace(
        attempt=1, temperature=F2._TEMP["script"], outcome="accepted",
        selected=True, character_words=parsed.character_word_count,
        scene_character_words=F2._scene_character_word_counts(parsed)),)
    treatment = _treatment(cast)
    draft = F2.build_final_draft(raw, parsed, treatment, p3_attempts=trace,
                                 salvaged=salvage, speaker_aliases=aliases)
    return draft, treatment


def _survives(draft, treatment):
    F2._validate_selected_final_draft(draft, treatment=treatment)
    payload = F2._final_draft_ledger_payload(draft)
    assert payload["artifact_sha256"] == draft.hashes.artifact_sha256


CHEN = "Dr. Haorong Chen"


@pytest.mark.parametrize("cast,lines", [
    # Short label adopted first, then the long one (failed before the fix).
    ((CHEN,), [f"{CHEN}: The sample is not decaying.",
               "ELOTWIZ: Then it is alive.",
               "DR. MICHAEL ELOTWIZ: I said it is alive."]),
    # The reverse order (the live pass006 order, which already passed).
    ((CHEN,), [f"{CHEN}: The sample is not decaying.",
               "DR. MICHAEL ELOTWIZ: I said it is alive.",
               "ELOTWIZ: Then it is alive."]),
    # An adopted stranger that would suppress a locked alias once promoted
    # (failed before the fix).
    ((CHEN, "Eli Marsh"), ["Eli Marsh: Look at this.",
                           "PROFESSOR CHEN: I see it.",
                           "CHEN: It is warm."]),
    # Two locked Chens, one silent, an ambiguous CHEN adopted.
    ((CHEN, "Bob Chen", "Eli Marsh"), ["Eli Marsh: Look at this.",
                                       "CHEN: It is warm."]),
])
def test_a_salvaged_draft_survives_selection_and_the_ledger_receipt(cast, lines):
    draft, treatment = _draft(cast, lines)
    assert draft.cast_roster == tuple(cast)
    _survives(draft, treatment)


def test_an_authored_alias_and_a_strict_draft_still_survive():
    draft, treatment = _draft((CHEN,), ["Doc: The sample is not decaying."],
                              salvage=False, aliases={CHEN: ["Doc"]})
    _survives(draft, treatment)
    strict, treatment = _draft((CHEN, "Eli Marsh"),
                               [f"{CHEN}: Look.", "Eli Marsh: I see it."],
                               salvage=False)
    assert strict.salvaged is False
    _survives(strict, treatment)


def _short_then_long():
    return _draft((CHEN,), [f"{CHEN}: The sample is not decaying.",
                            "ELOTWIZ: Then it is alive.",
                            "DR. MICHAEL ELOTWIZ: I said it is alive."])


def test_a_draft_without_its_roster_is_refused():
    draft, treatment = _short_then_long()
    for roster in ((), ("",), [CHEN]):
        bad = dataclasses.replace(draft, cast_roster=roster)
        with pytest.raises(F2.NewsProScriptError, match="no locked cast roster"):
            F2._validate_selected_final_draft(bad, treatment=treatment)


def test_a_forged_roster_raw_or_hash_is_refused():
    draft, treatment = _short_then_long()
    # The roster is sealed: swapping it -- even for the old rebuilt-from-output
    # roster -- breaks the parse or the artifact hash.
    rebuilt = tuple(F2._speakers_in_order(draft.parsed))
    with pytest.raises(F2.NewsProScriptError):
        F2._validate_selected_final_draft(
            dataclasses.replace(draft, cast_roster=rebuilt), treatment=treatment)
    # A roster that re-parses identically but is not the treatment's cast.
    extra = draft.cast_roster + ("Eli Marsh",)
    with pytest.raises(F2.NewsProScriptError):
        F2._validate_selected_final_draft(
            dataclasses.replace(draft, cast_roster=extra), treatment=treatment)
    with pytest.raises(F2.NewsProScriptError):
        F2._validate_selected_final_draft(
            dataclasses.replace(draft, raw_source=draft.raw_source + "\n"),
            treatment=treatment)
    forged = dataclasses.replace(
        draft, hashes=dataclasses.replace(draft.hashes, artifact_sha256="0" * 64))
    with pytest.raises(F2.NewsProScriptError, match="hash seal"):
        F2._validate_selected_final_draft(forged, treatment=treatment)


def test_the_roster_is_bound_into_the_artifact_seal():
    """Where no treatment is at hand -- the ledger receipt -- only the seal
    stands between a swapped roster and a draft that re-parses identically."""
    draft, _treatment = _short_then_long()
    swapped = dataclasses.replace(draft, cast_roster=draft.cast_roster + ("Eli Marsh",))
    with pytest.raises(F2.NewsProScriptError, match="hash seal"):
        F2._final_draft_ledger_payload(swapped)


def test_the_roster_must_match_the_treatment_it_is_selected_against():
    draft, _treatment_ok = _short_then_long()
    with pytest.raises(F2.NewsProScriptError, match="roster does not match"):
        F2._validate_selected_final_draft(
            draft, treatment=_treatment((CHEN, "Someone Else")))
