"""The status vocabulary for the required-still receipt, in ONE place.

Written 2026-08-28 for the sanctioned-gap control path. Four modules compare
these strings across package boundaries -- the image dispatcher mints them,
the render driver's still-spine validator and episode loop read them, the
clip manifest projects them, and node 92's success predicate counts them. A
typo in any one of those would not raise; it would silently classify a
refused card as an unexplained absence, which is the exact confusion this
whole control path exists to remove. So the strings live here and nobody
writes a literal.

THE RULE THE STATUSES ENCODE, because it is easy to get backwards:

* ``STATUS_OK`` -- the target was materialized THIS dispatch and its file is
  on disk. It carries a path and a content hash.
* ``STATUS_SANCTIONED_GAP`` -- the model DECLINED this card, the refusal was
  recorded with its prompt and seed, and the operator's 2026-08-22 ruling
  says the episode continues without it. It carries refusal evidence and
  NEVER a path: a gap row with a path is a contradiction, because the whole
  point is that nothing was produced.

**Absence is never a status.** A row missing from the receipt means the
receipt is incomplete, not that a target was refused -- readers must treat a
missing row as a fault rather than inferring a gap from it. That inference is
precisely the defect the 2026-08-28 panel caught in the first draft of this
work: counting every absent clip as sanctioned would report a crashed render
as a publishable degraded episode.

**NOT every absence is sanctionable, and this is STILL deliberately narrow.**
Exactly TWO reasons earn a gap row -- ``model_refusal`` (the model declined the
card, 2026-08-22) and ``cloud_job_failed`` (a partner job produced no image for
this object, 2026-09-16). Both mean the same thing about the NEXT object:
nothing. A dead path, a historical-row-only target, a no-engine skip and every
other absence still raise in the dispatcher exactly as before.

The set was widened ONCE, on evidence, and the bar for widening it again is the
same: the reason must be a POSITIVE record that something specific declined or
failed for this object, never an inference from a missing file. Read the set
through :func:`is_sanctionable_skip`, never by comparing against one constant --
that comparison is what this docstring used to describe, and a caller still
doing it would silently stop tolerating half the sanctionable cases.
"""
from __future__ import annotations

#: The target was produced this dispatch and its file exists.
STATUS_OK = "ok"

#: The model refused this card; the episode continues without it.
STATUS_SANCTIONED_GAP = "sanctioned_gap"

#: Every legal value of a receipt row's ``status`` field. A reader that sees
#: anything else is looking at a row minted by code that predates this module
#: or by a bug, and should fail loudly rather than guess.
RECEIPT_STATUSES = frozenset({STATUS_OK, STATUS_SANCTIONED_GAP})

#: The single skip reason that may be converted into a sanctioned gap
#: (operator ruling 2026-08-22). Kept here next to the statuses so the
#: narrowness travels with the vocabulary instead of living only in a comment
#: at the one site that currently enforces it.
SANCTIONABLE_SKIP_REASON = "model_refusal"

#: The second sanctionable reason (operator 2026-09-16: "any of them could
#: easily get a failed output; a failed output on a cloud video or still
#: should not break the system"). A partner still that timed out, lost its
#: job or returned an undecodable file produced no image for THIS object and
#: says nothing about the next one -- the same shape as a refusal, in a
#: different provider's words. It is recorded under its own reason rather
#: than borrowed from ``model_refusal`` because a timeout is not a refusal
#: and a receipt that says otherwise is a receipt that lies.
CLOUD_JOB_SKIP_REASON = "cloud_job_failed"

#: STILL NARROW. A dead path, a historical-row-only target, a no-engine skip
#: and every other absence remain unsanctionable and still fail the episode.
SANCTIONABLE_SKIP_REASONS = frozenset({
    SANCTIONABLE_SKIP_REASON, CLOUD_JOB_SKIP_REASON,
})


def is_sanctionable_skip(reason) -> bool:
    """True when this skip reason may be converted into a sanctioned gap."""
    return str(reason or "") in SANCTIONABLE_SKIP_REASONS


def is_sanctioned_gap(row) -> bool:
    """True when ``row`` is a receipt/clip row explicitly marked as a gap.

    Explicit by design: a row with no ``status`` is NOT a gap, however absent
    its file may be. Callers use this instead of testing for a missing path,
    so that an unexplained absence keeps failing the episode.
    """
    if not isinstance(row, dict):
        return False
    return str(row.get("status") or "") == STATUS_SANCTIONED_GAP
