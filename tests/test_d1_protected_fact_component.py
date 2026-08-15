"""D1 -- the clean stage must not erase a Python-owned fact.

THE DEFECT, FROM THE ARTIFACT (PBUG-20260815-01). The closing announcer row is
a COMPOSITE: a model-authored bridge plus a deterministic source fact that
Python appends verbatim. The clean stage (shipped 2026-08-14) had no concept of
a protected span -- it judged the row as one block and handed the whole thing to
a model to rewrite. `reel_of_mystery` b016 composed a factual Library of
Congress note naming three real films and SHIPPED *"Clarisse's gaze meets the
reel's enigmatic label"*. The fact was gone; `meta` still advertised that the
episode had spoken it. 9 of 14 voiced rows across three episodes.

WHY THE GUARANTEE IS STRUCTURAL AND NOT PERSUASIVE. The protected text is
physically prevented from reaching the model's edit surface. Asking a model
nicely to preserve a fact leaves a model deciding; checking afterwards ships the
wrong row and merely notices. So these tests assert the row is never JUDGED --
not merely that it came back similar. Similarity cannot tell "protected" from
"rewritten into something close", which is exactly why the original defect
survived a green suite.

CPU-only: plain dicts and a scripted slot. No model, no GPU.
"""

from __future__ import annotations

import ast
import inspect

import pytest

from nodes import _otr_ledger_clean as lcl
from nodes import _otr_line_composer as composer


FACT = ("In other news, the Library of Congress announces its film loans for "
        "the month, including 'None But the Lonely Heart'.")


# --------------------------------------------------------------------------- #
# the emitter: the row says it is a composite, at the moment it becomes one
# --------------------------------------------------------------------------- #
class TestTheComposerMarksTheRow:
    def test_a_fact_bearing_coda_is_marked_protected(self):
        res = composer.compose_news_coda(
            creative_fn=lambda *a, **k: "And in other news",
            news_close_brief=FACT, premise="a lighthouse")
        assert lcl.PROTECTED_FACT_COMPONENT_FLAG in res.compose_flags
        assert FACT in res.text

    def test_a_FACT_ONLY_coda_is_still_protected(self):
        """The bridge outcome and the protection are independent.

        A bridge that failed validation is dropped, and the fact is still
        there, still Python-owned. Deriving protection from
        `news_coda_bridge`/`news_coda_fact_only` would also be circular --
        those are decided after the point where a clean pass needs to already
        know to keep its hands off.
        """
        res = composer.compose_news_coda(
            creative_fn=lambda *a, **k: "",
            news_close_brief=FACT, premise="a lighthouse")
        assert "news_coda_fact_only" in res.compose_flags
        assert lcl.PROTECTED_FACT_COMPONENT_FLAG in res.compose_flags
        assert res.text == FACT

    def test_a_coda_with_NO_fact_is_not_protected(self):
        """Nothing Python-owned was appended, so the row stays fully
        judgeable. Marking it would exempt an ordinary authored line for no
        reason -- the guard must not grow wider than the defect."""
        res = composer.compose_news_coda(
            creative_fn=lambda *a, **k: "anything",
            news_close_brief="", premise="a lighthouse")
        assert res.compose_flags == ("news_coda_no_brief",)

    def test_the_flag_is_ONE_constant_not_two_spellings(self):
        """`BUG_BIBLE.yaml` 12.86: a producer and a consumer that each spell
        the same key by hand drift apart, and the guard silently stops
        guarding. Walks the AST so a mention inside this module's own
        docstrings cannot satisfy it."""
        tree = ast.parse(inspect.getsource(composer))
        literals = [
            n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and n.value == lcl.PROTECTED_FACT_COMPONENT_FLAG
        ]
        assert literals == [], (
            "the composer hardcodes the flag string instead of importing "
            "PROTECTED_FACT_COMPONENT_FLAG")


# --------------------------------------------------------------------------- #
# the consumer: a marked row is never judged, never rewritten, never counted
# --------------------------------------------------------------------------- #
def _ledger_with_protected_coda(*, protect: bool) -> dict:
    """One character row plus a closing announcer row carrying bridge + fact.

    `protect=False` builds the pre-fix shape, so a test can prove the guard is
    what makes the difference rather than the fixture being uncleanable.
    """
    coda_flags = ["news_coda_bridge"]
    if protect:
        coda_flags.append(lcl.PROTECTED_FACT_COMPONENT_FLAG)
    return {
        "cast": [
            {"char_id": "c01", "name": "Nan Reyes"},
            {"char_id": "announcer", "name": "ANNOUNCER"},
        ],
        "beats": [
            {"beat_id": "b000", "speaker": "ANNOUNCER"},
            {"beat_id": "b001", "speaker": "Nan Reyes",
             "beat_intent": "admit the lamp has been dead for days"},
            {"beat_id": "b016", "speaker": "ANNOUNCER"},
        ],
        "lines": [
            {"line_id": "L000", "beat_id": "b000", "char_id": "announcer",
             "speaker": "ANNOUNCER", "speaker_role": "announcer",
             "text": "Tonight, from the lighthouse."},
            {"line_id": "L001", "beat_id": "b001", "char_id": "c01",
             "speaker": "Nan Reyes", "speaker_role": "character",
             "text": "The lamp is dead."},
            {"line_id": "b016", "beat_id": "b016", "char_id": "announcer",
             "speaker": "ANNOUNCER", "speaker_role": "announcer",
             "text": "And in other news: " + FACT,
             "compose_flags": coda_flags},
        ],
        "meta": {"source_bank": "media_archive"},
    }


#: Reuse the slot the clean-stage suite already proves correct. It keys its
#: replies on the LINE under discussion rather than a call counter, which a
#: hand-rolled stub gets wrong: the pass interleaves brief, judge and repair
#: calls, so a positional script answers about the wrong row.
try:  # pragma: no cover - pytest rootdir puts tests/ on sys.path
    from test_ledger_clean_stage import _Slot, _dirty_judgement
except ImportError:  # pragma: no cover
    from tests.test_ledger_clean_stage import _Slot, _dirty_judgement  # type: ignore

#: What the model said instead, on the real episode.
SHIPPED_REWRITE = "Clarisse's gaze meets the reel's enigmatic label."


def _slot_that_destroys_the_coda() -> "_Slot":
    """A judge that condemns the coda and a repairer that deletes the fact.

    Modelled on the live failure rather than on a generic "dirty" case: the
    judge calls the factual note not-speech, and the repair returns the exact
    sentence `reel_of_mystery` shipped.
    """
    return _Slot(
        judgements={"Library of Congress": [_dirty_judgement(FACT)]},
        repairs={"Library of Congress": [{"text": SHIPPED_REWRITE}]},
    )


class TestTheCleanStageKeepsItsHandsOff:
    def test_a_protected_row_is_byte_identical_after_the_pass(self):
        ledger = _ledger_with_protected_coda(protect=True)
        before = ledger["lines"][2]["text"]
        lcl.run_ledger_clean(
            ledger, slot_fn=_slot_that_destroys_the_coda(),
            bank_id="media_archive")
        assert ledger["lines"][2]["text"] == before
        assert FACT in ledger["lines"][2]["text"]

    def test_the_protected_row_gets_NO_entry_in_rows(self):
        """The D1 acceptance assertion. A `rows` entry means the judge read it
        and something rewrote it -- the pass is LUCK regardless of how similar
        the output looks, because similarity cannot distinguish "protected"
        from "rewritten into something close"."""
        ledger = _ledger_with_protected_coda(protect=True)
        receipt = lcl.run_ledger_clean(
            ledger, slot_fn=_slot_that_destroys_the_coda(),
            bank_id="media_archive")
        touched = [r.get("line_id") for r in receipt["rows"]]
        assert "b016" not in touched
        assert receipt["protected_rows"] == ["b016"]

    def test_the_protected_row_is_never_THE_LINE_UNDER_JUDGEMENT(self):
        """Skipped BEFORE the judge, not after. A row that is read and then
        discarded still spends a model call and still leaves a later edit
        somewhere to put the rewrite back.

        Asserted on the SUBJECT, not on the whole prompt. The coda legitimately
        appears inside a neighbouring row's before/after context window -- the
        pass is supposed to show the judge what surrounds a line. Context is
        not jeopardy; being `THE LINE:` is. An earlier cut of this test banned
        the fact from the prompt entirely and failed for that reason, which
        would have been a false alarm about a correct behaviour.
        """
        ledger = _ledger_with_protected_coda(protect=True)
        slot = _slot_that_destroys_the_coda()
        lcl.run_ledger_clean(ledger, slot_fn=slot, bank_id="media_archive")
        subjects = [
            line[len("THE LINE: "):]
            for prompt in slot.judge_prompts
            for line in prompt.splitlines()
            if line.startswith("THE LINE: ")
        ]
        assert subjects, "the fixture judged nothing at all"
        assert not any(FACT in s for s in subjects), (
            "the protected row was handed to the judge as the line to rule on")
        assert slot.repair_calls == 0

    def test_WITHOUT_the_flag_the_same_row_is_destroyed(self):
        """The fixture is not self-fulfilling: the identical ledger without
        the marker loses its fact, which is the shipped defect reproduced."""
        ledger = _ledger_with_protected_coda(protect=False)
        lcl.run_ledger_clean(
            ledger, slot_fn=_slot_that_destroys_the_coda(),
            bank_id="media_archive")
        assert FACT not in ledger["lines"][2]["text"]
        assert ledger["lines"][2]["text"] == SHIPPED_REWRITE

    def test_ordinary_voiced_rows_are_STILL_judged(self):
        """The guard must not grow into a general exemption. The opening
        announcer row and the character row are untouched by this change --
        and announcer rows in general stay judgeable, which is why the marker
        is per-row rather than keyed on `speaker_role`."""
        ledger = _ledger_with_protected_coda(protect=True)
        slot = _slot_that_destroys_the_coda()
        receipt = lcl.run_ledger_clean(
            ledger, slot_fn=slot, bank_id="media_archive")
        assert receipt["voiced_rows"] == 3
        assert slot.judge_calls == 2, "the two unprotected rows are still read"
