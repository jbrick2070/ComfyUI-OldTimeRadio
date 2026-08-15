"""The clean/cleanup window as ONE transaction, so a failed reseal costs a
repair rather than a finished render.

WHY THIS EXISTS. Two sanctioned passes run between the acceptance receipt and
the audit -- `run_ledger_clean` (a model rewrites a row it judged unclean) and
`run_ledger_cleanup` (blanks an unsayable row out of the voiced set). Their
edits are legitimate, but the receipt was written when nothing ran in between,
so the audit began failing on the pipeline's own authorized output.
`_otr_content_transition` teaches the receipt that a named stage ran. THIS
module is the other half: what happens when that proof cannot be completed.

LAW 7 IS THE WHOLE POINT. `scifi_news` died `CodexPreTailAuditError: line
receipt mismatch for l004` after 13.6 minutes -- with the script finished, the
cast built and the score written. Killing a render at that point is not a safety
property, it is the defect. So the window is a TRANSACTION: reseal the proof if
the reseal can be proved, and if it cannot, put the ledger back exactly as the
lane accepted it, say so in a durable receipt, and let the episode ship. The
repaired text is what gets sacrificed, never the render.

WHY THE SNAPSHOT IS THE WHOLE LEDGER AND NOT A FIELD LIST. The surfaces at risk
are row presence and order, `text`, the text metrics, the skip fields and their
reasons, the compose flags, the delivery fields, and every proof/coverage
surface in `meta`. A hand-maintained list of those is a list that goes stale the
first time a pass learns to write one more field -- and a restore that misses a
field leaves a ledger that is neither the old state nor the new one. Copying
`lines` and `meta` wholesale cannot miss a field by construction. It is exact
rather than clever: between the snapshot and the reconcile, the ONLY writers are
the two authorized passes, so rolling both containers back rolls back exactly
those two passes and nothing else.

AND A LEDGER-ONLY COPY IS NOT ENOUGH. `_CodexTailFinalizer` holds the accepted
text in memory as `self.expected`; nothing on disk can restore it. So the
transaction snapshots the finalizer's own state through a small protocol the
lane implements, and a lane that has no finalizer (scifi_news_pro carries none)
simply has no state to carry -- its proof surface is the shared authorship
receipt, which lives in `meta` and is covered above.

Aliasing matters here and it is easy to get wrong: the writer tail holds `meta`
as a local, so REBINDING `led.data["meta"]` would leave every later stamp
writing into a detached dict. Both containers are therefore restored IN PLACE.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, Mapping, Optional

log = logging.getLogger(__name__)

__all__ = [
    "CLEAN_WINDOW_STAGES",
    "DEGRADATION_META_KEY",
    "CleanTransaction",
    "open_transaction",
]

#: Every authorized pass inside the captured window, in the order the writer
#: runs them. ONE transition spans the whole window rather than one per pass:
#: the second pass's pre-state is the first pass's output, so a per-pass chain
#: could never start from the acceptance the receipt is bound to.
CLEAN_WINDOW_STAGES = ("ledger_clean", "ledger_cleanup")

#: Where a rolled-back window records itself. Present ONLY on a degraded run --
#: its absence is the ordinary case and carries no information, which is what
#: makes its presence worth reading.
DEGRADATION_META_KEY = "content_transition_degraded"

#: The lane-side protocol. A finalizer that implements these three carries its
#: in-memory proof through the window; one that implements none is a lane with
#: no in-memory proof. A finalizer that implements SOME of them is a rename in
#: progress, and that is the case worth shouting about -- see `_finalizer_hook`.
_FINALIZER_PROTOCOL = (
    "snapshot_proof_state",
    "restore_proof_state",
    "reseal_proof",
)


def _authorship():
    try:
        from . import _otr_content_authorship as _CA
    except ImportError:  # pragma: no cover -- flat test/standalone load
        import _otr_content_authorship as _CA  # type: ignore
    return _CA


def _transition():
    try:
        from . import _otr_content_transition as _CT
    except ImportError:  # pragma: no cover -- flat test/standalone load
        import _otr_content_transition as _CT  # type: ignore
    return _CT


def _has_authorship_receipt(ledger_data: Mapping[str, Any]) -> bool:
    meta = ledger_data.get("meta")
    if not isinstance(meta, Mapping):
        return False
    return isinstance(meta.get("content_authorship"), Mapping)


def _finalizer_hook(finalizer: Any) -> Optional[Any]:
    """The finalizer if it speaks the protocol, else None -- loudly if partial.

    A lane with no finalizer is ordinary and silent. A finalizer implementing
    every method is ordinary and silent. A finalizer implementing SOME of them
    means a method was renamed and one side was not updated -- the exact
    producer/consumer mismatch `BUG_BIBLE.yaml` 12.86 is about. That cannot pass
    quietly, and it also must not raise: this runs after the script is written,
    and Law 7 outranks a tidy failure. So it warns at ERROR and degrades to "no
    in-memory proof to carry", which is the safe reading.
    """
    if finalizer is None:
        return None
    present = [name for name in _FINALIZER_PROTOCOL if callable(
        getattr(finalizer, name, None))]
    if len(present) == len(_FINALIZER_PROTOCOL):
        return finalizer
    if present:
        log.error(
            "[clean-transaction] %s implements only %s of the finalizer proof "
            "protocol %s -- its in-memory proof will NOT be carried through "
            "the clean window. This is a rename that missed a caller, not a "
            "lane without a proof.",
            type(finalizer).__name__, present, list(_FINALIZER_PROTOCOL),
        )
    return None


class CleanTransaction:
    """One authorized rewrite window, with an exact undo."""

    def __init__(self, led: Any, finalizer: Any = None) -> None:
        self._led = led
        self._finalizer = _finalizer_hook(finalizer)
        data = led.data
        self._pre_state = _transition().capture_text_state(data)
        self._lines = copy.deepcopy(data.get("lines") or [])
        self._meta = copy.deepcopy(data.get("meta") or {})
        self._finalizer_state = (
            self._finalizer.snapshot_proof_state()
            if self._finalizer is not None else None
        )

    # -- undo ------------------------------------------------------------
    def restore(self) -> None:
        """Put the ledger back exactly as the lane accepted it.

        IN PLACE on both containers, deliberately. The writer tail holds `meta`
        as a local variable and the lane runners hold row references; rebinding
        either container would leave those aliases pointing at a detached copy,
        and every later stamp would land somewhere the ledger never sees.
        """
        data = self._led.data
        rows = data.get("lines")
        if isinstance(rows, list):
            rows[:] = copy.deepcopy(self._lines)
        else:
            data["lines"] = copy.deepcopy(self._lines)
        meta = data.get("meta")
        if isinstance(meta, dict):
            meta.clear()
            meta.update(copy.deepcopy(self._meta))
        else:
            data["meta"] = copy.deepcopy(self._meta)
        if self._finalizer is not None:
            self._finalizer.restore_proof_state(self._finalizer_state)

    # -- commit ----------------------------------------------------------
    def reconcile(self) -> Dict[str, Any]:
        """Prove the window's rewrites, or roll them back and keep going."""
        try:
            return self._reseal()
        except Exception as exc:  # noqa: BLE001 -- Law 7: never kill the render
            return self._degrade(exc)

    def _reseal(self) -> Dict[str, Any]:
        _CT = _transition()
        data = self._led.data
        meta = data.setdefault("meta", {})
        post = _CT.capture_text_state(data)
        transition = _CT.build_transition(
            self._pre_state,
            post,
            stages=CLEAN_WINDOW_STAGES,
            cleaner_receipt={
                "ledger_clean": meta.get("ledger_clean"),
                "ledger_cleanup": meta.get("ledger_cleanup"),
            },
            parent_authorship_digest=_authorship().receipt_sha256(data),
        )
        if transition is None:
            # NO-OP. Nothing moved, so the ordinary v1 proof still describes the
            # ledger exactly and must stay the one in force -- a transition here
            # would put an untouched episode on the transitioned path for
            # nothing, and would make the marker meaningless as a signal. This
            # is the ONE emission point for the key, so it also owns saying the
            # key should be absent.
            _CT.stamp_transition(data, None)
            log.info(
                "[clean-transaction] the authorized window changed nothing; "
                "the acceptance receipt stands unmodified",
            )
            return {"outcome": "noop", "changed_line_ids": []}

        _CT.stamp_transition(data, transition)
        if self._finalizer is not None:
            # The lane re-points its in-memory proof at the authorized text and
            # proves it did so. A failure here lands in `reconcile`'s handler.
            self._finalizer.reseal_proof(data)
        # The composite validator is the real gate: it re-reads the v1 receipt,
        # validates the transition against it, and compares the live rows. If
        # this passes, the freeze cascade's own call to it will pass too.
        _authorship().validate_receipt(data)

        changed = list(transition.get("affected_line_ids") or ())
        dropped = list(transition.get("dropped_line_ids") or ())
        log.info(
            "[clean-transaction] authorized window resealed: %d row(s) "
            "rewritten %s, %d dropped from the voiced set %s",
            len(changed), changed, len(dropped), dropped,
        )
        return {
            "outcome": "resealed",
            "changed_line_ids": changed,
            "dropped_line_ids": dropped,
        }

    def _degrade(self, exc: Exception) -> Dict[str, Any]:
        """Roll the window back, record why, and let the episode ship."""
        data = self._led.data
        attempted = {
            "ledger_clean": copy.deepcopy(
                (data.get("meta") or {}).get("ledger_clean")),
            "ledger_cleanup": copy.deepcopy(
                (data.get("meta") or {}).get("ledger_cleanup")),
        }
        self.restore()
        receipt: Dict[str, Any] = {
            "outcome": "restored_pre_clean",
            "authorized_stages": list(CLEAN_WINDOW_STAGES),
            "error_type": type(exc).__name__,
            "error": str(exc),
            # The rolled-back passes' own telemetry, kept HERE because the
            # restore removed it from `meta` -- and it had to: a `ledger_clean`
            # receipt describing edits that no longer exist would tell every
            # downstream reader the judge touched rows it did not.
            "attempted_receipts": attempted,
        }
        log.error(
            "[clean-transaction] the authorized clean window could not be "
            "proved (%s: %s). The ledger is RESTORED to the accepted text and "
            "the episode continues -- the repairs are lost, the render is not.",
            type(exc).__name__, exc,
        )
        try:
            # RE-PROVE THE RESTORED STATE -- and re-prove it, do not re-RESEAL
            # it. `reseal_proof` is very often the thing that just failed, so
            # calling it here would report the original failure a second time
            # and label a perfectly good rollback as corrupt. The ledger
            # snapshot and the finalizer snapshot were taken at the same
            # instant, from a state the lane had already proved, so the pair is
            # consistent by construction; what needs checking is that the
            # restore actually put it back, and that is exactly what
            # `validate_receipt` reads. The lane's own `_proof` runs again a
            # few steps later at `before_save` regardless.
            _authorship().validate_receipt(data)
            receipt["restored_state_proved"] = True
        except Exception as restore_exc:  # noqa: BLE001
            # The restored state is the state the lane itself accepted and
            # proved, so this should be unreachable. If it happens the ledger is
            # genuinely corrupt -- but the freeze cascade is the ONE terminal
            # authority on that (`_otr_freeze_cascade.py:249-252`, terminal at
            # `:803`), and refusing here too would just move the same refusal
            # earlier with a less specific message.
            receipt["restored_state_proved"] = False
            receipt["restore_proof_error"] = str(restore_exc)
            log.error(
                "[clean-transaction] the RESTORED ledger does not prove "
                "either (%s). The freeze cascade will refuse it; this is a "
                "corrupt acceptance, not a clean-stage fault.",
                restore_exc,
            )
        data.setdefault("meta", {})[DEGRADATION_META_KEY] = receipt
        return receipt


def open_transaction(led: Any, finalizer: Any = None) -> "CleanTransaction | None":
    """Open the window, or return None when there is no proof to protect.

    The predicate is the presence of the authorship receipt itself, not a bank
    name or a delivery mode: the receipt IS the thing at risk, so a ledger
    carrying one gets a transaction and a ledger without one has nothing for a
    transition to be bound to. Legacy lanes therefore pay no deep copy, and a
    future lane that starts stamping a receipt is covered the day it does.
    """
    data = getattr(led, "data", None)
    if not isinstance(data, dict) or not _has_authorship_receipt(data):
        return None
    return CleanTransaction(led, finalizer=finalizer)
