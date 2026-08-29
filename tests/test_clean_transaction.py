"""D2 chunks 1 + 3 -- the authorized clean window is a TRANSACTION.

THE DEFECT (PBUG-20260815-02a, the other half). `_otr_content_transition`
taught the acceptance receipt that a named stage may rewrite a row. That
closes the false-positive audit failure, but it says nothing about what
happens when the reseal itself cannot be proved -- and the failure being
fixed here is a render that DIED at 13.6 minutes with the script finished,
the cast built and the score written. `scifi_news` raised
`CodexPreTailAuditError: line receipt mismatch for l004`; `l004` was the
first row the clean stage repaired.

SO THE WINDOW IS A TRANSACTION AND LAW 7 IS THE ACCEPTANCE CRITERION: a
render must not die. Reseal the proof when the reseal can be proved; when it
cannot, put the ledger back exactly as the lane accepted it, say so in a
durable receipt, and let the episode ship. The repaired text is what gets
sacrificed, never the render.

TWO THINGS THIS FILE REFUSES TO LET DRIFT.

1. THE PROOF IS NOT LOOSENED. An unattributed mutation still fails exactly as
   loudly as before -- `_CodexTailFinalizer._proof` stays fail-loud, and
   the (now-deleted) codex lane test called that primitive directly with
   an unattributed mutation and must keep raising. A transaction that made
   corruption survivable would be worse than the bug.

2. THE RESTORE IS EXACT AND IN PLACE. The writer tail holds `meta` as a local
   variable, so a restore that REBOUND `led.data["meta"]` would leave every
   later stamp writing into a detached dict -- a silent, ledger-shaped hole.

CPU-only: plain dicts, hashes, and stub slot functions.
"""

from __future__ import annotations

import copy
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import _otr_clean_transaction as txn  # noqa: E402
from nodes import _otr_content_transition as tr  # noqa: E402
from nodes._otr_content_authorship import (  # noqa: E402
    ContentAuthorshipError, stamp_receipt, validate_receipt,
)

TRANSITION_KEY = tr.CONTENT_TRANSITION_META_KEY


def _ledger_data():
    return {
        "meta": {"source_bank": "bank_a"},
        "lines": [
            {"line_id": "l1", "text": "Exact raw text.", "skip_tts": False},
            {"line_id": "l2", "text": "Second line.", "skip_tts": False},
            {"line_id": "note", "text": "Not voiced.", "skip_tts": True},
        ],
    }


def _accepted_led():
    """A ledger at the instant the lane stamped its acceptance receipt."""
    data = _ledger_data()
    stamp_receipt(
        data, owner_bank="bank_a", accepted_artifacts={"final": {"x": 1}})
    return SimpleNamespace(data=data)


def _clean_stage_rewrites(led, line_id, new_text):
    """What `run_ledger_clean` does: rewrite a row and record that it did."""
    for row in led.data["lines"]:
        if row["line_id"] == line_id:
            row["text"] = new_text
    led.data["meta"]["ledger_clean"] = {"rows": [{"line_id": line_id}]}


def _cleanup_blanks(led, line_id):
    """What `run_ledger_cleanup` does to a row with nothing sayable in it."""
    for row in led.data["lines"]:
        if row["line_id"] == line_id:
            row["text"] = ""
            row["skip"] = True
    led.data["meta"]["ledger_cleanup"] = {"blanked": [line_id]}


class _LaneProof:
    """A lane proof held in memory, like `_CodexTailFinalizer.expected`.

    The point of the protocol is that this state exists NOWHERE on disk, so a
    ledger-only snapshot cannot restore it.
    """

    def __init__(self, expected, fail_reseal=False):
        self.expected = dict(expected)
        self.fail_reseal = fail_reseal
        self.reseal_calls = 0

    def snapshot_proof_state(self):
        return dict(self.expected)

    def restore_proof_state(self, state):
        self.expected = dict(state or {})

    def reseal_proof(self, ledger_data):
        self.reseal_calls += 1
        if self.fail_reseal:
            raise RuntimeError("lane proof cannot be resealed")
        live = {
            row["line_id"]: row["text"] for row in ledger_data["lines"]
        }
        self.expected = {
            line_id: live[line_id]
            for line_id in self.expected if line_id in live
        }


# --------------------------------------------------------------------------- #
# who gets a transaction
# --------------------------------------------------------------------------- #
class TestOpenTransaction:
    def test_a_lane_with_no_acceptance_receipt_gets_NO_transaction(self):
        """The receipt IS the thing at risk. A legacy lane has no proof for a
        transition to be bound to, so it pays no deep copy and takes no new
        path -- which is what keeps public_domain / shakespeare byte-identical
        through this change."""
        led = SimpleNamespace(data=_ledger_data())
        assert txn.open_transaction(led) is None

    def test_a_ledger_carrying_a_receipt_gets_one(self):
        assert txn.open_transaction(_accepted_led()) is not None

    def test_the_predicate_is_the_RECEIPT_not_the_bank_name(self):
        """A future lane that starts stamping a receipt is covered the day it
        does, without anyone remembering to add its id to a list."""
        led = _accepted_led()
        led.data["meta"]["source_bank"] = "bank_a"
        led.data["meta"]["content_authorship"]["owner_bank"] = "bank_a"
        assert txn.open_transaction(led) is not None


# --------------------------------------------------------------------------- #
# the commit path
# --------------------------------------------------------------------------- #
class TestReseal:
    def test_a_rewritten_row_is_resealed_and_the_ledger_VALIDATES(self):
        """THE REGRESSION. Before this landed, `validate_receipt` raised
        `canonical text hash mismatch for 'l1'` here -- the exact death that
        killed scifi_news after the writing was finished."""
        led = _accepted_led()
        window = txn.open_transaction(led)
        _clean_stage_rewrites(led, "l1", "Exact repaired text.")

        report = window.reconcile()

        assert report["outcome"] == "resealed"
        validate_receipt(led.data)
        assert led.data["lines"][0]["text"] == "Exact repaired text."

    def test_the_transition_names_EXACTLY_the_row_that_moved(self):
        led = _accepted_led()
        window = txn.open_transaction(led)
        _clean_stage_rewrites(led, "l1", "Exact repaired text.")

        window.reconcile()

        transition = led.data["meta"][TRANSITION_KEY]
        assert transition["affected_line_ids"] == ["l1"]
        assert transition["authorized_stages"] == list(txn.CLEAN_WINDOW_STAGES)

    def test_the_window_names_BOTH_passes_not_one(self):
        """One transition per pass is impossible -- the second pass's pre-state
        is the first pass's output, so the chain could never start at the
        acceptance. The window is captured once around both."""
        led = _accepted_led()
        window = txn.open_transaction(led)
        _clean_stage_rewrites(led, "l1", "Exact repaired text.")
        _cleanup_blanks(led, "l2")

        window.reconcile()

        transition = led.data["meta"][TRANSITION_KEY]
        assert set(transition["authorized_stages"]) == {
            "ledger_clean", "ledger_cleanup"}
        assert transition["affected_line_ids"] == ["l1"]
        assert transition["dropped_line_ids"] == ["l2"]
        validate_receipt(led.data)

    def test_the_parent_digest_binds_the_chain_to_THIS_acceptance(self):
        """Pre-state equality alone would let a transition be lifted from one
        episode onto another: line ids repeat by construction."""
        from nodes._otr_content_authorship import receipt_sha256

        led = _accepted_led()
        window = txn.open_transaction(led)
        _clean_stage_rewrites(led, "l1", "Exact repaired text.")

        window.reconcile()

        assert led.data["meta"][TRANSITION_KEY][
            "parent_authorship_digest"] == receipt_sha256(led.data)

    def test_a_NO_OP_stamps_NOTHING(self):
        """An episode the clean stage did not change validates through the
        untouched v1 path. A transition has to MEAN something happened, or it
        is useless as a signal -- and every historical ledger would otherwise
        be pushed onto the new path for nothing."""
        led = _accepted_led()
        before = copy.deepcopy(led.data)
        window = txn.open_transaction(led)

        report = window.reconcile()

        assert report["outcome"] == "noop"
        assert TRANSITION_KEY not in led.data["meta"]
        assert led.data == before

    def test_a_NO_OP_leaves_the_lane_proof_untouched(self):
        led = _accepted_led()
        proof = _LaneProof({"l1": "Exact raw text.", "l2": "Second line."})
        window = txn.open_transaction(led, finalizer=proof)

        window.reconcile()

        assert proof.reseal_calls == 0
        assert proof.expected == {
            "l1": "Exact raw text.", "l2": "Second line."}

    def test_the_lane_proof_is_resealed_to_the_authorized_text(self):
        led = _accepted_led()
        proof = _LaneProof({"l1": "Exact raw text.", "l2": "Second line."})
        window = txn.open_transaction(led, finalizer=proof)
        _clean_stage_rewrites(led, "l1", "Exact repaired text.")

        window.reconcile()

        assert proof.expected["l1"] == "Exact repaired text."


# --------------------------------------------------------------------------- #
# the undo path -- Law 7
# --------------------------------------------------------------------------- #
class TestDegradation:
    def _failed_window(self):
        led = _accepted_led()
        proof = _LaneProof(
            {"l1": "Exact raw text.", "l2": "Second line."},
            fail_reseal=True,
        )
        window = txn.open_transaction(led, finalizer=proof)
        _clean_stage_rewrites(led, "l1", "Exact repaired text.")
        return led, proof, window

    def test_an_unprovable_reseal_does_NOT_kill_the_render(self):
        """Law 7. Today's behaviour -- killing a render after 13.6 minutes of
        finished work -- is itself the violation being fixed."""
        led, _proof, window = self._failed_window()

        report = window.reconcile()  # must not raise

        assert report["outcome"] == "restored_pre_clean"

    def test_the_ledger_is_restored_to_the_ACCEPTED_text(self):
        led, _proof, window = self._failed_window()

        window.reconcile()

        assert led.data["lines"][0]["text"] == "Exact raw text."
        validate_receipt(led.data)

    def test_the_rolled_back_clean_TELEMETRY_does_not_survive_in_meta(self):
        """A `ledger_clean` receipt describing edits that no longer exist would
        tell every downstream reader the judge touched rows it did not -- and
        D1's acceptance assertion reads exactly that field. It moves into the
        degradation receipt instead, so nothing is lost and nothing lies."""
        led, _proof, window = self._failed_window()

        receipt = window.reconcile()

        assert "ledger_clean" not in led.data["meta"]
        assert receipt["attempted_receipts"]["ledger_clean"] == {
            "rows": [{"line_id": "l1"}]}

    def test_the_degradation_receipt_is_durable_and_names_the_cause(self):
        led, _proof, window = self._failed_window()

        window.reconcile()

        receipt = led.data["meta"][txn.DEGRADATION_META_KEY]
        assert receipt["outcome"] == "restored_pre_clean"
        assert receipt["error_type"] == "RuntimeError"
        assert "cannot be resealed" in receipt["error"]
        assert receipt["restored_state_proved"] is True

    def test_no_transition_is_left_behind_by_a_rollback(self):
        led, _proof, window = self._failed_window()

        window.reconcile()

        assert TRANSITION_KEY not in led.data["meta"]

    def test_the_IN_MEMORY_lane_proof_is_restored_too(self):
        """A ledger-only copy cannot restore `expected`: it lives nowhere on
        disk. A finalizer left holding post-clean text after a rollback would
        fail the pre-save gate on a ledger that is perfectly correct."""
        led, proof, window = self._failed_window()
        proof.expected["l1"] = "something the failed reseal left behind"

        window.reconcile()

        assert proof.expected == {
            "l1": "Exact raw text.", "l2": "Second line."}

    def test_restore_keeps_the_SAME_meta_and_lines_OBJECTS(self):
        """The writer tail holds `meta` as a local. Rebinding the container
        would leave every later stamp writing into a detached dict -- a silent
        hole shaped exactly like a working ledger."""
        led, _proof, window = self._failed_window()
        meta_obj = led.data["meta"]
        lines_obj = led.data["lines"]

        window.reconcile()

        assert led.data["meta"] is meta_obj
        assert led.data["lines"] is lines_obj

    def test_a_ROLLBACK_that_itself_fails_still_does_not_kill_the_render(self):
        """The Law 7 violation hiding inside the Law 7 enforcement.

        `_degrade` used to be called from a bare `except`, so anything IT raised
        -- here a finalizer whose `restore_proof_state` throws -- propagated out
        of `reconcile()` to an unguarded call site and killed the render. The
        fallback needed a fallback, and this is the test that says so.
        """
        led = _accepted_led()

        class _Hostile(_LaneProof):
            def restore_proof_state(self, state):
                raise RuntimeError("the rollback itself is broken")

        proof = _Hostile({"l1": "Exact raw text."}, fail_reseal=True)
        window = txn.open_transaction(led, finalizer=proof)
        _clean_stage_rewrites(led, "l1", "Exact repaired text.")

        receipt = window.reconcile()  # must not raise

        assert receipt["outcome"] == "rollback_failed"
        assert receipt["rollback_error_type"] == "RuntimeError"
        assert receipt["restored_state_proved"] is False
        # Still durable: whoever reads this ledger must be able to see BOTH
        # failures, or the mixed state looks like an ordinary episode.
        stamped = led.data["meta"][txn.DEGRADATION_META_KEY]
        assert stamped["outcome"] == "rollback_failed"
        assert "cannot be resealed" in stamped["error"]

    def test_a_lane_with_no_finalizer_degrades_the_same_way(self):
        """scifi_news_pro carries no tail_finalizer but DOES stamp
        content_authorship, enforced terminally at
        `_otr_freeze_cascade.py:803`. One shared contract covers both lanes."""
        led = _accepted_led()
        window = txn.open_transaction(led)
        _clean_stage_rewrites(led, "l1", "Exact repaired text.")
        # An unreadable stamped transition is the lane-independent failure.
        led.data["meta"][TRANSITION_KEY] = {"version": "from_the_future"}

        report = window.reconcile()

        assert report["outcome"] == "restored_pre_clean"
        assert led.data["lines"][0]["text"] == "Exact raw text."
        validate_receipt(led.data)

# --------------------------------------------------------------------------- #
# the lane side of the protocol -- COVERAGE LAPSED 2026-08-16 (scifi_news rip)
# --------------------------------------------------------------------------- #
# `TestFinalizerProtocol` lived here and pinned the ONE real implementer of
# both the writer TailFinalizer protocol and this module's three-method proof
# protocol: `_CodexTailFinalizer`. It died with the codex lane.
#
# The transaction machinery itself is STILL covered -- every test above drives
# a stub lane proof, and the writer-tail test below exercises the real seam.
# What lapsed is the "a renamed method would go unnoticed" guard, which needs
# a REAL implementer to bite. The protocol surface is deliberately KEPT as
# extension space, so restore this class the moment a lane implements it.
# --------------------------------------------------------------------------- #
# the wiring -- code that is not called is dead code
# --------------------------------------------------------------------------- #
def test_the_writer_tail_actually_reseals_an_authorized_clean(
        tmp_path, monkeypatch):
    """END TO END through `_run_writer_tail`. The unit tests above prove the
    transaction; this proves the writer OPENS one -- the whole failure mode
    being fixed is a pass that shipped and ran dormant.

    The harness is imported rather than copied so it cannot drift from the
    tail's real signature; if that helper is renamed this fails at import,
    loudly, which is the correct outcome.
    """
    from tests.test_scifi_news_pro_tail_context import _make_ctx
    from nodes import _otr_ledger_clean
    from nodes.OTR_LedgerScriptWriter import OTR_LedgerScriptWriter

    ctx = _make_ctx(tmp_path / "scifi_news_pro", monkeypatch)
    ctx.led.data["meta"]["source_bank"] = "scifi_news_pro"
    stamp_receipt(
        ctx.led.data, owner_bank="scifi_news_pro",
        accepted_artifacts={"final_script": {"v": 1}},
    )
    accepted = ctx.led.data["meta"]["content_authorship"]["line_proofs"]
    assert accepted, "the fixture must have voiced rows to prove"

    def _rewriting_clean(ledger_data, **_kwargs):
        for row in ledger_data["lines"]:
            if row.get("line_id") == "b003":
                row["text"] = "Get the kit out."
        ledger_data.setdefault("meta", {})["ledger_clean"] = {
            "rows": [{"line_id": "b003"}]}

    monkeypatch.setattr(
        _otr_ledger_clean, "run_ledger_clean", _rewriting_clean)

    OTR_LedgerScriptWriter()._run_writer_tail(ctx)

    meta = ctx.led.data["meta"]
    assert meta[TRANSITION_KEY]["affected_line_ids"] == ["b003"]
    # The assertion that matters: this call raised
    # `canonical text hash mismatch for 'b003'` before the transaction landed.
    validate_receipt(ctx.led.data)


def test_the_delivered_word_receipt_is_restamped_after_the_window(
        tmp_path, monkeypatch):
    """The window is the last thing that touches canonical `text`, so the
    receipt taken at `writer_final_rows` describes a draft that no longer
    exists. The post-clean stamp lands under its OWN stage name, so the
    pre-clean record survives beside it instead of being overwritten."""
    from tests.test_scifi_news_pro_tail_context import _make_ctx
    from nodes.OTR_LedgerScriptWriter import OTR_LedgerScriptWriter

    ctx = _make_ctx(tmp_path, monkeypatch)

    OTR_LedgerScriptWriter()._run_writer_tail(ctx)

    meta = ctx.led.data["meta"]
    # The top-level writer_word_delivery alias was retired 2026-08-28 (V4
    # finding 12): stamp_actual merges the LATEST receipt onto word_budget
    # itself, so the stage pointer lives there now.
    assert "writer_word_delivery" not in meta
    assert meta["word_budget"]["stage"] == "writer_final_rows_post_clean"
    receipts = meta["word_budget"]["actual_receipts"]
    assert "writer_final_rows" in receipts
    assert "writer_final_rows_post_clean" in receipts
