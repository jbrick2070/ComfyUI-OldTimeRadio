"""P0 dossier extraction is LABELLED SECTIONS, assembled by Python.

WHY (live failure, 2026-08-25): `gemma-4-E2B-it` failed all three ladder rungs
on `scifi_news_pro`'s P0 dossier with the same error each time -- "no decodable
top-level JSON object found" -- after stopping at 503 tokens of a 700-token
budget, i.e. it believed it had finished while the JSON object was still
unclosed. Evidence: `docs/2026-08-25-leg1-dossier-failure-evidence.md`.

Grammar-constrained decoding was the obvious fix and was REJECTED at r1: under
a grammar the legal minimum for `DossierLLM` is one fact plus three empty
buckets and `facts_to_keep` is never source-checked, so a small model's
constrained output sits near that floor -- a LOUD failure would have become a
SILENT hollow dossier.

So the model is no longer asked to balance a brace. It writes bullets under
six headers; Python builds the object. One contract for small AND large models,
not a small-model fallback lane.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from nodes._otr_scifi_news_pro import parse_dossier_sections  # noqa: E402

WELL_FORMED = """FACTS:
- The system requires no expertise to operate
- The system could be used at home
NUMBERS:
- 17,000
- 2015
PEOPLE:
- Canan Dagdeviren
PLACES:
- MIT
THINGS:
- ultrasound
VECTORS:
- Fear of cancer diagnosis
"""


def test_well_formed_reply_maps_every_section():
    got = parse_dossier_sections(WELL_FORMED)
    assert got["facts_to_keep"] == [
        "The system requires no expertise to operate",
        "The system could be used at home"]
    assert got["allowed_numbers"] == ["17,000", "2015"]
    assert got["named_entities"]["people"] == ["Canan Dagdeviren"]
    assert got["named_entities"]["places"] == ["MIT"]
    assert got["named_entities"]["things"] == ["ultrasound"]
    assert got["dramatizable_vectors"] == ["Fear of cancer diagnosis"]


def test_result_validates_as_a_real_DossierLLM():
    """The parser's output must satisfy the UNCHANGED schema -- the whole point
    is that only the transport changed, not the contract."""
    from nodes._otr_scifi_news_pro import DossierLLM
    model = DossierLLM.model_validate(parse_dossier_sections(WELL_FORMED))
    assert model.facts_to_keep[0].startswith("The system requires")
    assert model.named_entities.people == ["Canan Dagdeviren"]


@pytest.mark.parametrize("noise", [
    "```\nFACTS:\n- a fact\n```",                      # fenced
    "Here is the dossier.\n\nFACTS:\n- a fact",        # leading prose
    "**FACTS:**\n* a fact",                            # emphasis + star bullet
    "## FACTS\n1. a fact",                             # heading + enumeration
    "facts:\n• a fact",                           # lowercase + real bullet
    'FACTS:\n- "a fact"',                              # quoted item
])
def test_tolerates_the_decoration_models_actually_emit(noise):
    """Every one of these is a shape a model reaches for unprompted. None of
    them may cost the extraction."""
    got = parse_dossier_sections(noise)
    assert got["facts_to_keep"] == ["a fact"], f"failed on: {noise!r}"


def test_unknown_header_ends_the_section_rather_than_absorbing_items():
    """Items after an unrecognized header must NOT be appended to whatever
    section happened to be open -- that would silently file numbers as facts."""
    got = parse_dossier_sections(
        "FACTS:\n- real fact\nSUMMARY:\n- not a fact\n")
    assert got["facts_to_keep"] == ["real fact"]


def test_duplicate_headers_accumulate():
    """A model that emits FACTS twice meant to ADD facts; dropping the first
    block would discard real extraction."""
    got = parse_dossier_sections("FACTS:\n- one\nFACTS:\n- two\n")
    assert got["facts_to_keep"] == ["one", "two"]


def test_empty_extraction_still_fails_validation_rather_than_passing_hollow():
    """THE anti-regression for the reason grammar was rejected. A reply with no
    facts must NOT sail through as a valid empty dossier -- min_length=1 has to
    still bite, so the ladder retries instead of shipping a hollow ledger."""
    from nodes._otr_scifi_news_pro import DossierLLM
    empty = parse_dossier_sections("NUMBERS:\n- 5\n")
    assert empty["facts_to_keep"] == []
    with pytest.raises(Exception):
        DossierLLM.model_validate(empty)


def test_parser_never_raises_on_hostile_input():
    """It is a tolerant reader, not a validator; emptiness is the schema's
    decision. Raising here would convert a retryable miss into a crash."""
    for junk in ["", "   ", "\n\n", "no headers at all",
                 "FACTS:", "- orphan bullet with no header",
                 "```json\n{\"facts_to_keep\": []}\n```"]:
        out = parse_dossier_sections(junk)
        assert isinstance(out, dict)
        assert set(out) == {"facts_to_keep", "allowed_numbers",
                            "named_entities", "dramatizable_vectors"}


def test_pack_prompt_asks_for_sections_and_not_json():
    """The pack seam and the parser must agree. If someone re-words the prompt
    back to JSON, this fires by name."""
    pack = json.loads(
        (REPO / "nodes" / "story_packs" / "scifi_news_pro"
         / "scifi_news_pro.json").read_text(encoding="utf-8"))
    # The seam lives INSIDE prompt_stages, not at the top level. Reading it
    # from the top level is exactly the mistake that broke the pack: a
    # top-level write added an unknown key, StoryPackValidationError fired
    # inside OTR_LedgerScriptWriter.INPUT_TYPES(), and 377 tests went red.
    sysmsg = pack["prompt_stages"]["scifi_news_pro_dossier_system"]
    assert "LABELLED SECTIONS" in sysmsg
    assert "no JSON" in sysmsg
    for header in ("FACTS:", "NUMBERS:", "PEOPLE:", "PLACES:",
                   "THINGS:", "VECTORS:"):
        assert header in sysmsg, f"pack prompt lost the {header} header"


def test_greedy_label_regression():
    """The header label must be GREEDY.

    While the pattern ended in `$` the anchor forced full consumption, so a
    lazy `{3,20}?` was harmless. Adding the inline-content capture `(.*)$`
    gave the lazy quantifier somewhere to dump the rest and `FACTS:` started
    matching as label `FAC` + inline `TS:` -- which silently unmatched EVERY
    header and emptied every section. Caught only because a direct probe was
    run after the edit; the existing tests would have caught it too, but this
    one names the cause.
    """
    got = parse_dossier_sections("FACTS:\n- one\n- two")
    assert got["facts_to_keep"] == ["one", "two"]


def test_inline_known_header_raises_a_named_defect():
    """r2 QA findings 1+2: one rule must win. Both prior guesses shipped
    and were caught (silent drop, then section hijack), so the parser
    refuses to guess: loud and retryable beats silently wrong."""
    from nodes._otr_scifi_news_pro import DossierSectionDefect
    for shape in ("FACTS: a fact",
                  "FACTS: inline one\n- bullet two",
                  "FACTS:\n- one\nPEOPLE: including Dr. Smith\n- two",
                  "PEOPLE: ROLES:\n- Ada",
                  "FACTS: - a fact"):
        with pytest.raises(DossierSectionDefect) as exc:
            parse_dossier_sections(shape)
        assert exc.value.defect_code == "ambiguous_inline_header", shape


def test_exotic_colon_header_ends_the_section_without_contaminating_it():
    got = parse_dossier_sections(
        "FACTS:\n- f\nKEY TAKEAWAYS 2024:\n- stray\n")
    assert got["facts_to_keep"] == ["f"]
    assert got["named_entities"]["people"] == []


# --------------------------------------------------------------------------- #
# Blast-radius r1 findings (agy). Every one of these was SILENT DATA LOSS --
# the parser returned a smaller dossier rather than failing, so nothing in the
# pipeline could tell a thin extraction from a thin source.
# --------------------------------------------------------------------------- #

def test_a_wrapped_fact_does_not_destroy_the_rest_of_its_section():
    """M1. A continuation line matches the permissive label pattern, so it was
    read as an unknown header -- which set current=None and discarded EVERY
    remaining item in the section. A colon is now what makes a header."""
    got = parse_dossier_sections(
        "FACTS:\n- a fact that wraps\nonto a second line\n- another fact\n")
    assert got["facts_to_keep"] == [
        "a fact that wraps", "onto a second line", "another fact"]


def test_numbered_header_is_a_header_not_a_bullet():
    """M2. `1. FACTS:` matched the bullet pattern first, lost header status,
    and every item beneath it was dropped -- a 100% silent loss."""
    got = parse_dossier_sections("1. FACTS:\n- a fact\n2. NUMBERS:\n- 42\n")
    assert got["facts_to_keep"] == ["a fact"]
    assert got["allowed_numbers"] == ["42"]


def test_an_ordinary_numbered_fact_is_still_an_item():
    """The M2 fix must not promote every numbered line to a header."""
    got = parse_dossier_sections("FACTS:\n1. the system is portable\n")
    assert got["facts_to_keep"] == ["the system is portable"]


def test_header_qualifier_is_absorbed_not_filed_as_an_entity():
    """M4. `PEOPLE / CHARACTERS:` left `/ CHARACTERS:` as inline content and
    filed it as a PERSON -- garbage injected straight into named_entities."""
    got = parse_dossier_sections("PEOPLE / CHARACTERS:\n- Ada\n")
    assert got["named_entities"]["people"] == ["Ada"]


def test_repair_prompt_carries_the_source_story():
    """M3, and the worst of the four. The repair rung asked the model to redo
    a SOURCE EXTRACTION with the source removed -- a blind last attempt that
    could only hallucinate or repeat itself before the lane died."""
    from nodes._otr_scifi_news_pro import _dossier_section_repair
    original = [
        {"role": "system", "content": "you extract dossiers"},
        {"role": "user", "content": "SCIENCE STORY:\nA portable scanner."},
    ]
    msgs = _dossier_section_repair(
        original_prompt=original,
        failed_output="garbage",
        error=ValueError("facts_to_keep must not be empty"),
    )
    body = "\n".join(m["content"] for m in msgs)
    assert "A portable scanner." in body, "repair rung lost the source story"
    assert "LABELLED SECTIONS" in body
    assert "JSON object" not in body, "repair rung steers back to JSON"


# --------------------------------------------------------------------------- #
# Blast-radius r2 findings. Same class as r1: SILENT loss or MISFILING, never a
# crash -- so nothing downstream could tell a mangled extraction from a thin
# source. The governing rule that came out of these: a colon makes a header
# ONLY when nothing follows it; a colon with content after it is content.
# --------------------------------------------------------------------------- #

def test_a_fact_containing_a_colon_does_not_close_its_section():
    """MF-1, the most destructive and the likeliest. A colon inside a fact is
    ordinary English ("Source: Nature", "The system: a scanner"). Reading it as
    an unknown header closed the section and discarded everything after it."""
    got = parse_dossier_sections(
        "FACTS:\n- one\nSource: Nature\n- two\n")
    assert got["facts_to_keep"] == ["one", "Source: Nature", "two"]


def test_dash_and_bracket_header_qualifiers_still_open_their_section():
    """MF-2. `PEOPLE - KEY ROLES:` failed to match, so PEOPLE never opened and
    Ada was appended to FACTS -- a whole section misfiled into another list."""
    for header in ("PEOPLE - KEY ROLES:", "PEOPLE [main]:",
                   "PEOPLE / CHARACTERS:"):
        got = parse_dossier_sections(f"FACTS:\n- f\n{header}\n- Ada\n")
        assert got["named_entities"]["people"] == ["Ada"], header
        assert got["facts_to_keep"] == ["f"], f"{header} contaminated FACTS"


# --------------------------------------------------------------------------- #
# r2 QA finding 5 (Fable): text mode used to skip the structural rung. A parse
# miss surfaced as a ValidationError, which the ladder routes straight to the
# typed repair -- so max_attempts=3 was really two calls. DossierSectionDefect
# subclasses json.JSONDecodeError precisely so the structural rung engages.
# These drive the REAL structured_call with a fake slot to pin the ladder.
# --------------------------------------------------------------------------- #

def _drive_ladder(replies):
    from nodes._otr_structured_call import (
        structured_call, StructuredCallFailedError)
    from nodes._otr_scifi_news_pro import (
        DossierLLM, parse_dossier_sections, _dossier_section_repair)
    it = iter(replies)
    temps = []

    def slot_fn(messages, *, temperature, max_new_tokens):
        temps.append(round(temperature, 3))
        return next(it)

    try:
        out = structured_call(
            prompt=[{"role": "system", "content": "extract"},
                    {"role": "user",
                     "content": "SCIENCE STORY:\nA portable scanner."}],
            schema=DossierLLM, slot_fn=slot_fn,
            base_temperature=0.3, structural_retry_temperature=0.15,
            repair_prompt_factory=_dossier_section_repair,
            max_new_tokens=700, helper_name="ladder_probe",
            text_parser=parse_dossier_sections)
        return temps, out.facts_to_keep
    except StructuredCallFailedError as exc:
        return temps, ("EXHAUSTED", exc.attempts)


def test_section_defect_engages_the_structural_rung():
    """A defective reply earns the SAME-prompt lower-temperature retry, not a
    straight jump to typed repair. The temperature sequence is the proof."""
    temps, facts = _drive_ladder(["FACTS: inline bad", "FACTS:\n- good fact"])
    assert facts == ["good fact"]
    assert temps == [0.3, 0.15], (
        "structural rung did not fire for a section defect")


def test_text_mode_has_a_true_three_call_budget():
    """defect -> defect -> typed repair rescues. Three real model calls."""
    temps, facts = _drive_ladder(
        ["FACTS: bad", "FACTS: bad again", "FACTS:\n- rescued"])
    assert facts == ["rescued"]
    assert len(temps) == 3


def test_text_mode_exhausts_honestly_at_three():
    temps, verdict = _drive_ladder(["FACTS: a", "FACTS: b", "FACTS: c"])
    assert verdict == ("EXHAUSTED", 3)
    assert len(temps) == 3
