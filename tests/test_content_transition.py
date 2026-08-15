"""D2 -- an authorized rewrite between acceptance and the audit.

THE DEFECT (PBUG-20260815-02a). `_otr_content_authorship` stamps a receipt
proving the ledger text is what the lane ACCEPTED, and the freeze cascade later
re-validates by hashing the live rows. That contract was written when nothing
ran in between. Two sanctioned passes were then added AFTER the receipt and
BEFORE the audit -- the model-driven `run_ledger_clean` and the blanking
`run_ledger_cleanup` -- and neither was taught to exist by the receipt. So the
audit started failing on its own legitimate output: `scifi_news` died
`CodexPreTailAuditError: line receipt mismatch for l004`, and `l004` was
provably the FIRST row the clean stage repaired, at 13.6 minutes, after the
writing was finished.

THE FIX IS NOT A WIDER TOLERANCE, and that is the assertion this file exists to
protect. Fuzzy matching would hide real corruption along with the false
positive. The receipt LEARNS that a named stage ran and proves the before AND
after states, so an unattributed mutation still fails exactly as loudly as it
did before.

CPU-only: plain dicts and hashes.
"""

from __future__ import annotations

import copy

import pytest

from nodes import _otr_content_transition as tr
from nodes._otr_content_authorship import (
    ContentAuthorshipError, receipt_sha256, stamp_receipt, validate_receipt,
)


def _ledger():
    return {
        "meta": {"source_bank": "bank_a"},
        "lines": [
            {"line_id": "l1", "text": "Exact raw text.", "skip_tts": False},
            {"line_id": "l2", "text": "Second line.", "skip_tts": False},
            {"line_id": "note", "text": "Not voiced.", "skip_tts": True},
        ],
    }


def _accepted():
    led = _ledger()
    stamp_receipt(led, owner_bank="bank_a", accepted_artifacts={"final": {"x": 1}})
    return led


def _clean_stage_rewrites(led, line_id, new_text):
    """Do what the clean stage does: capture, rewrite, capture, stamp."""
    pre = tr.capture_text_state(led)
    for row in led["lines"]:
        if row["line_id"] == line_id:
            row["text"] = new_text
    post = tr.capture_text_state(led)
    transition = tr.build_transition(
        pre, post, stages=["ledger_clean"],
        cleaner_receipt={"rows": [{"line_id": line_id}]},
    )
    tr.stamp_transition(led, transition)
    return transition


# --------------------------------------------------------------------------- #
# the state capture
# --------------------------------------------------------------------------- #
class TestTextState:
    def test_it_hashes_only_the_rows_the_AUDITOR_will_hash(self):
        """The selector is imported from the authorship module, not restated.
        A transition proved over a different row set than the audit checks
        would be worse than none -- it would look like evidence."""
        state = tr.capture_text_state(_ledger())
        assert set(state.hashes) == {"l1", "l2"}
        assert state.voiced_line_count == 2

    def test_the_digest_is_stable_across_captures(self):
        assert tr.capture_text_state(_ledger()).digest == \
            tr.capture_text_state(_ledger()).digest


# --------------------------------------------------------------------------- #
# building it
# --------------------------------------------------------------------------- #
class TestBuildTransition:
    def test_a_NO_OP_emits_NOTHING(self):
        """An episode the clean stage did not change must validate through the
        untouched v1 path. A transition in the ledger has to MEAN something
        happened, or it is useless as a signal."""
        state = tr.capture_text_state(_ledger())
        assert tr.build_transition(state, state, stages=["ledger_clean"]) is None

    def test_an_unauthorized_stage_is_refused_at_BUILD_time(self):
        state = tr.capture_text_state(_ledger())
        other = tr.capture_text_state(_ledger())
        with pytest.raises(tr.ContentTransitionError):
            tr.build_transition(state, other, stages=["some_other_pass"])

    def test_it_names_the_rows_that_moved(self):
        led = _ledger()
        t = _clean_stage_rewrites(led, "l1", "Repaired text.")
        assert t["affected_line_ids"] == ["l1"]
        assert t["authorized_stages"] == ["ledger_clean"]
        assert t["cleaner_receipt_digest"]

    def test_a_BLANKED_row_is_recorded_as_dropped_not_changed(self):
        """`run_ledger_cleanup` can blank a row out of the voiced set. "The row
        is gone" and "the row is unchanged" must be distinguishable."""
        led = _ledger()
        pre = tr.capture_text_state(led)
        led["lines"][1]["skip_tts"] = True
        post = tr.capture_text_state(led)
        t = tr.build_transition(pre, post, stages=["ledger_cleanup"])
        assert t["dropped_line_ids"] == ["l2"]
        assert t["post"]["voiced_line_count"] == 1


# --------------------------------------------------------------------------- #
# the composite validator -- the whole point
# --------------------------------------------------------------------------- #
class TestCompositeValidation:
    def test_an_UNTRANSITIONED_ledger_validates_exactly_as_before(self):
        """Every historical ledger and every episode nothing rewrote keeps its
        original proof. The new path must not become the only path."""
        led = _accepted()
        assert validate_receipt(led)["coverage"]["complete"] is True

    def test_an_AUTHORIZED_clean_stage_rewrite_now_VALIDATES(self):
        """The defect, fixed. This raised `line receipt mismatch` before."""
        led = _accepted()
        _clean_stage_rewrites(led, "l1", "The repaired line, without the action.")
        assert validate_receipt(led)["coverage"]["complete"] is True

    def test_an_AUTHORIZED_cleanup_BLANKING_validates(self):
        led = _accepted()
        pre = tr.capture_text_state(led)
        led["lines"][1]["skip_tts"] = True
        post = tr.capture_text_state(led)
        tr.stamp_transition(led, tr.build_transition(
            pre, post, stages=["ledger_cleanup"]))
        assert validate_receipt(led)["coverage"]["complete"] is True

    def test_an_UNATTRIBUTED_mutation_still_FAILS(self):
        """THE assertion that separates "taught the audit about a mutator" from
        "widened the audit's tolerance". A row that moved after the transition
        was stamped matches neither pre nor post, and that is corruption."""
        led = _accepted()
        _clean_stage_rewrites(led, "l1", "Repaired.")
        assert validate_receipt(led)  # the authorized state is fine
        led["lines"][1]["text"] = "and then something else edited me"
        with pytest.raises(ContentAuthorshipError):
            validate_receipt(led)

    def test_a_transition_FORGED_onto_a_different_acceptance_fails(self):
        """The chain must start where the accepted artifact ended, or a
        transition could be lifted from one episode onto another."""
        led = _accepted()
        _clean_stage_rewrites(led, "l1", "Repaired.")
        stolen = copy.deepcopy(led["meta"][tr.CONTENT_TRANSITION_META_KEY])
        other = _accepted()
        other["lines"][0]["text"] = "Repaired."
        other["meta"][tr.CONTENT_TRANSITION_META_KEY] = stolen
        stolen["pre"]["hashes"]["l1"] = "0" * 64
        with pytest.raises(ContentAuthorshipError):
            validate_receipt(other)

    def test_a_transition_that_hides_a_row_it_changed_fails(self):
        led = _accepted()
        _clean_stage_rewrites(led, "l1", "Repaired.")
        led["meta"][tr.CONTENT_TRANSITION_META_KEY]["affected_line_ids"] = []
        with pytest.raises(ContentAuthorshipError):
            validate_receipt(led)

    @pytest.mark.parametrize("break_it", [
        "version", "stage", "pre_shape", "coverage",
    ])
    def test_an_UNREADABLE_transition_is_refused_not_trusted(self, break_it):
        """An unreadable transition is not permission."""
        led = _accepted()
        _clean_stage_rewrites(led, "l1", "Repaired.")
        t = led["meta"][tr.CONTENT_TRANSITION_META_KEY]
        if break_it == "version":
            t["version"] = "content_transition_v99"
        elif break_it == "stage":
            t["authorized_stages"] = ["something_else"]
        elif break_it == "pre_shape":
            t["pre"] = "not a mapping"
        else:
            t["post"]["voiced_line_count"] = 99
        with pytest.raises(ContentAuthorshipError):
            validate_receipt(led)

    def test_ONE_window_covers_BOTH_authorized_stages(self):
        """The design catch from the D2 QA round, and it would have bitten the
        wiring commit.

        `run_ledger_clean` and `run_ledger_cleanup` both run between acceptance
        and the audit. One transition PER STAGE cannot work: the second one's
        `pre` is the first one's OUTPUT, so it can never equal the parent
        receipt the chain must start from. One window around both, naming both
        stages, is the only shape that validates.
        """
        led = _accepted()
        pre = tr.capture_text_state(led)
        led["lines"][0]["text"] = "The repaired line."   # ledger_clean
        led["lines"][1]["skip_tts"] = True               # ledger_cleanup
        post = tr.capture_text_state(led)
        tr.stamp_transition(led, tr.build_transition(
            pre, post, stages=["ledger_clean", "ledger_cleanup"],
            parent_authorship_digest=receipt_sha256(led),
        ))
        assert validate_receipt(led)["coverage"]["complete"] is True

    def test_a_SECOND_stamp_is_refused_rather_than_silently_overwriting(self):
        """Two emission points would leave the survivor's pre-state pointing at
        the loser's output -- failing later, at the freeze, with a message about
        a row instead of about the wiring."""
        led = _accepted()
        _clean_stage_rewrites(led, "l1", "Repaired.")
        pre = tr.capture_text_state(led)
        led["lines"][1]["text"] = "Second repair."
        post = tr.capture_text_state(led)
        second = tr.build_transition(pre, post, stages=["ledger_cleanup"])
        with pytest.raises(tr.ContentTransitionError):
            tr.stamp_transition(led, second)

    def test_restamping_the_IDENTICAL_transition_is_allowed(self):
        """An idempotent retry of the same tail must not trip the guard."""
        led = _accepted()
        t = _clean_stage_rewrites(led, "l1", "Repaired.")
        tr.stamp_transition(led, t)
        assert validate_receipt(led)

    def test_a_transition_that_OVER_declares_is_refused(self):
        """Checking only "everything that changed is declared" lets a receipt
        name innocent rows. This record exists to say what happened; one that
        lies in detail is one nobody reads."""
        led = _accepted()
        _clean_stage_rewrites(led, "l1", "Repaired.")
        led["meta"][tr.CONTENT_TRANSITION_META_KEY]["affected_line_ids"] = ["l1", "l2"]
        with pytest.raises(ContentAuthorshipError):
            validate_receipt(led)

    def test_the_parent_digest_binds_the_chain_to_THIS_acceptance(self):
        """Line ids repeat across episodes by construction and two episodes can
        carry an identical short row, so pre-state equality alone would let a
        transition be lifted between episodes -- or between LANES, since the
        receipt digest is what covers owner_bank and accepted_artifacts."""
        led = _accepted()
        pre = tr.capture_text_state(led)
        led["lines"][0]["text"] = "Repaired."
        post = tr.capture_text_state(led)
        tr.stamp_transition(led, tr.build_transition(
            pre, post, stages=["ledger_clean"],
            parent_authorship_digest="f" * 64,          # not this ledger's
        ))
        with pytest.raises(ContentAuthorshipError):
            validate_receipt(led)

    def test_a_transition_claiming_NO_parent_still_validates_on_hashes(self):
        """The digest is checked when CLAIMED. Omitting it falls back to the
        pre-state proof rather than becoming a silent bypass."""
        led = _accepted()
        _clean_stage_rewrites(led, "l1", "Repaired.")
        assert not led["meta"][tr.CONTENT_TRANSITION_META_KEY][
            "parent_authorship_digest"]
        assert validate_receipt(led)

    def test_the_v1_receipt_still_records_what_was_ACCEPTED(self):
        """The parent receipt keeps its meaning. Rewriting it to describe what
        SHIPPED would destroy the only record of what the model accepted --
        which is exactly the field a reader needs to tell an authorized clean
        from a lane that quietly rewrote its own output."""
        led = _accepted()
        before = copy.deepcopy(led["meta"]["content_authorship"])
        _clean_stage_rewrites(led, "l1", "Repaired.")
        validate_receipt(led)
        assert led["meta"]["content_authorship"] == before
