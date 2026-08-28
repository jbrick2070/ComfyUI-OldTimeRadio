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

**NOT every absence is sanctionable, and this is deliberately narrow.** Only
``reason == "model_refusal"`` earns a gap row. A dead path, a historical-row-
only target, a no-engine skip and every other absence still raise in the
dispatcher exactly as before. Widening this set would re-open the failure the
gate exists to catch.
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


def is_sanctioned_gap(row) -> bool:
    """True when ``row`` is a receipt/clip row explicitly marked as a gap.

    Explicit by design: a row with no ``status`` is NOT a gap, however absent
    its file may be. Callers use this instead of testing for a missing path,
    so that an unexplained absence keeps failing the episode.
    """
    if not isinstance(row, dict):
        return False
    return str(row.get("status") or "") == STATUS_SANCTIONED_GAP
