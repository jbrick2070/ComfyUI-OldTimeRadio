"""An AUTHORIZED rewrite between the acceptance receipt and the audit.

WHY THIS EXISTS. `_otr_content_authorship` stamps a receipt proving the ledger's
canonical text is exactly what the lane ACCEPTED, and the freeze cascade later
re-validates it by hashing the live rows. That contract was written when nothing
ran in between. Then two sanctioned passes were added AFTER the receipt and
BEFORE the audit -- `run_ledger_clean` (a model rewrites a row it judged unclean)
and `run_ledger_cleanup` (blanks an unsayable row out of the voiced set) -- and
neither was ever taught to exist by the receipt.

So the audit began failing on its own legitimate output. `scifi_news` died
`CodexPreTailAuditError: line receipt mismatch for l004` -- and `l004` was
provably the FIRST row the clean stage repaired, while `l001`/`l002` shipped
untouched and passed. It died at 13.6 minutes, after the writing was finished.

THE SNAPSHOT'S VALIDITY WINDOW IS A PIPELINE FACT, NOT A CONSTANT. That is the
whole lesson (`BUG_BIBLE.yaml` 12.104). The fix is NOT to widen the audit's
tolerance -- fuzzy matching would hide real corruption along with the false
positive. The receipt has to learn that a named, authorized stage ran, and prove
the before AND after states rather than trusting either.

WHAT A TRANSITION IS. A signed statement that exactly one authorized stage
rewrote exactly these rows, from these hashes to those hashes, with the
cleaner's own receipt digest as evidence and the parent authorship receipt named
so the chain cannot be re-parented. It is DETECTIVE, not permissive: an
unattributed mutation still fails, because a row whose live hash matches neither
the pre nor the post state is corruption by definition.

WHAT IT IS NOT. Not schema v2 of the authorship receipt. The v1 receipt keeps
its meaning exactly -- "this is what the lane accepted" -- and the transition
sits BESIDE it saying what happened afterwards. Rebuilding v1 to mean "what
shipped" would destroy the only record of what the model actually accepted, and
`accepted_lines` is precisely the field a later reader needs to tell an
authorized clean from a lane that quietly rewrote its own output.

Pure and cold-import clean: stdlib only, no model call, no I/O. The voiced-row
definition and the hash function are IMPORTED from `_otr_content_authorship`
rather than restated, because two definitions of "voiced" is the
producer/consumer mismatch this module exists to survive.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Tuple

__all__ = [
    "CONTENT_TRANSITION_VERSION",
    "CONTENT_TRANSITION_META_KEY",
    "AUTHORIZED_STAGES",
    "ContentTransitionError",
    "TextState",
    "capture_text_state",
    "build_transition",
    "stamp_transition",
    "transition_of",
    "validate_transition",
]

#: Bump when the FIELD SET or the validation rules change. A validator that does
#: not recognise the version refuses rather than guessing -- an unreadable
#: transition must never be read as permission.
CONTENT_TRANSITION_VERSION = "content_transition_v1"

#: The single durable key, imported by producer and consumer alike (12.86).
CONTENT_TRANSITION_META_KEY = "content_authorship_transition"

#: The ONLY stages allowed to rewrite accepted text, named explicitly. A closed
#: set is the point: "some pass changed it" is exactly the claim the audit
#: exists to reject, so the transition has to name WHICH passes and be refused
#: if it names anything else.
#:
#: A TRANSITION SPANS A WINDOW, NOT A SINGLE PASS, and that took a QA round to
#: see. `run_ledger_clean` and `run_ledger_cleanup` BOTH run between acceptance
#: and the audit (`OTR_LedgerScriptWriter.py:6784-6798`). One transition per
#: stage cannot work: the second one's `pre` state is the FIRST one's output,
#: so it can never equal the parent receipt the chain must start from. And a
#: single transition labelled with one stage name would misattribute a rewrite
#: the other pass made. So `authorized_stages` is a LIST -- every stage that
#: was allowed to run inside the captured window -- and each member must be a
#: member of this set.
AUTHORIZED_STAGES = frozenset({"ledger_clean", "ledger_cleanup"})


class ContentTransitionError(RuntimeError):
    """The transition does not prove the rewrite it claims."""


def _authorship():
    try:
        from . import _otr_content_authorship as _CA
    except ImportError:  # pragma: no cover -- flat test/standalone load
        import _otr_content_authorship as _CA  # type: ignore
    return _CA


def _sha256_text(value: Any) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TextState:
    """The voiced surface at one instant: id -> text hash, plus the coverage.

    Coverage is carried separately rather than derived from ``hashes`` at read
    time because `run_ledger_cleanup` can BLANK a row out of the voiced set
    entirely. "The row is gone" and "the row is unchanged" must be
    distinguishable, and a bare dict of survivors cannot tell them apart.
    """

    hashes: Dict[str, str] = field(default_factory=dict)
    voiced_line_count: int = 0

    def as_receipt(self) -> Dict[str, Any]:
        return {
            "hashes": dict(sorted(self.hashes.items())),
            "voiced_line_count": self.voiced_line_count,
        }

    @property
    def digest(self) -> str:
        payload = json.dumps(
            self.as_receipt(), sort_keys=True, separators=(",", ":"),
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def capture_text_state(ledger_data: Mapping[str, Any]) -> TextState:
    """Hash every voiced row's canonical text, exactly as the auditor will.

    Uses `_otr_content_authorship`'s own voiced-row selector so the two can
    never disagree about which rows are in scope -- a transition proved over a
    different row set than the audit checks would be worse than no transition,
    because it would look like evidence.
    """
    rows = _authorship()._voiced_rows(ledger_data)
    hashes = {
        str(row.get("line_id") or ""): _sha256_text(row.get("text"))
        for row in rows
    }
    return TextState(hashes=hashes, voiced_line_count=len(hashes))


def _normalize_stages(stages: Any) -> "Tuple[str, ...]":
    """A tuple of authorized stage names, or a refusal. Accepts one or many."""
    if isinstance(stages, str):
        stages = (stages,)
    try:
        named = tuple(str(s) for s in stages)
    except TypeError as exc:  # noqa: BLE001
        raise ContentTransitionError(
            "authorized_stages must be a stage name or a list of them"
        ) from exc
    if not named:
        raise ContentTransitionError("a transition must name its stages")
    unknown = sorted(set(named) - AUTHORIZED_STAGES)
    if unknown:
        raise ContentTransitionError(
            "transition names unauthorized stage(s) %s; authorized: %s"
            % (unknown, sorted(AUTHORIZED_STAGES))
        )
    return named


def build_transition(
    pre: TextState,
    post: TextState,
    *,
    stages: Any,
    cleaner_receipt: Any = None,
    parent_authorship_digest: str = "",
) -> "Dict[str, Any] | None":
    """The receipt for one authorized rewrite WINDOW, or ``None`` on a no-op.

    ``stages`` is every authorized pass that was allowed to run between the two
    captures -- a list, because the pipeline runs `run_ledger_clean` and then
    `run_ledger_cleanup` back to back and ONE transition has to cover both. One
    per pass is impossible: the second's ``pre`` is the first's output, so it
    could never equal the parent receipt the chain must start from.

    NO TRANSITION ON A NO-OP, deliberately. An episode the clean stage did not
    change must validate through the untouched v1 path, so the ordinary case
    keeps its ordinary proof and a transition in the ledger always MEANS
    something happened. Emitting an empty one would make the marker useless as
    a signal and would put every historical ledger on the new path for nothing.
    """
    named = _normalize_stages(stages)
    changed = sorted(
        line_id for line_id, digest in post.hashes.items()
        if line_id in pre.hashes and pre.hashes[line_id] != digest
    )
    dropped = sorted(set(pre.hashes) - set(post.hashes))
    added = sorted(set(post.hashes) - set(pre.hashes))
    if not changed and not dropped and not added:
        return None
    return {
        "version": CONTENT_TRANSITION_VERSION,
        "authorized_stages": list(named),
        # NOT ENFORCED, and that is deliberate -- see `validate_transition`.
        "cleaner_receipt_digest": _receipt_digest(cleaner_receipt),
        "affected_line_ids": changed,
        "dropped_line_ids": dropped,
        "added_line_ids": added,
        "pre": pre.as_receipt(),
        "post": post.as_receipt(),
        "parent_authorship_digest": str(parent_authorship_digest or ""),
    }


def _receipt_digest(cleaner_receipt: Any) -> str:
    """Fingerprint the cleaner's own receipt, so the claim has evidence.

    Best-effort by design: a receipt that will not serialize yields "" rather
    than raising. The digest is corroboration, not the proof -- the proof is the
    before/after hashes -- and losing corroboration must not cost a render.
    """
    if cleaner_receipt is None:
        return ""
    try:
        payload = json.dumps(
            cleaner_receipt, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, default=str,
        )
    except Exception:  # noqa: BLE001
        return ""
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stamp_transition(
    ledger_data: dict, transition: "Mapping[str, Any] | None",
) -> None:
    """Write the transition, or clear a stale one when there is nothing to say.

    ONE EMISSION POINT PER LEDGER, enforced. A second stamp would overwrite the
    first, and the survivor's ``pre`` state would be the loser's OUTPUT rather
    than the acceptance -- so the ledger would fail its own audit later, with a
    hash-mismatch message pointing at a row instead of at the real fault. The
    window is captured once around ALL authorized stages and stamped once at
    the end; anything else is a wiring error, and it is cheaper to say so here
    than to debug it at the freeze.

    Re-stamping an IDENTICAL transition is allowed, so an idempotent retry of
    the same tail does not trip the guard.
    """
    meta = ledger_data.get("meta")
    if not isinstance(meta, dict):
        return
    if transition is None:
        meta.pop(CONTENT_TRANSITION_META_KEY, None)
        return
    existing = meta.get(CONTENT_TRANSITION_META_KEY)
    if isinstance(existing, Mapping) and dict(existing) != dict(transition):
        raise ContentTransitionError(
            "a different content transition is already stamped on this "
            "ledger; capture ONE window around every authorized stage and "
            "stamp it once, or the chain's pre-state stops being the "
            "acceptance"
        )
    meta[CONTENT_TRANSITION_META_KEY] = dict(transition)


def transition_of(meta: Any) -> "Dict[str, Any] | None":
    """The stamped transition, or None. Never raises on a malformed ledger."""
    if not isinstance(meta, Mapping):
        return None
    found = meta.get(CONTENT_TRANSITION_META_KEY)
    return dict(found) if isinstance(found, Mapping) else None


def validate_transition(
    ledger_data: Mapping[str, Any],
    transition: Mapping[str, Any],
    *,
    parent_line_hashes: Mapping[str, str],
) -> Tuple[Dict[str, str], int]:
    """Prove the claimed rewrite, and return the state the audit should expect.

    Three questions, in order, because a later answer is meaningless if an
    earlier one is wrong:

    1. Is this transition READABLE and authorized? An unknown version or an
       unnamed stage is refused -- an unreadable transition is not permission.
    2. Does its PRE state match what the parent v1 receipt actually proved? This
       is what stops a transition being forged onto a different acceptance: the
       chain has to start where the accepted artifact ended.
    3. Are the rows it claims to have changed EXACTLY the ones that moved? A
       row whose live hash matches neither pre nor post is corruption, and it
       still fails loud -- which is the difference between teaching the audit
       about a mutator and simply widening its tolerance.

    ``cleaner_receipt_digest`` is DELIBERATELY NOT ENFORCED. It is corroboration
    -- a fingerprint of what the clean stage said it did -- and the proof is the
    before/after hashes, which stand on their own. Requiring it would mean a
    render dies because a telemetry receipt failed to serialize, and Law 7 says
    a render must not die for that. It is diagnostic telemetry; the docstring
    says so rather than leaving a reader to assume a field named "digest" is
    load-bearing.

    Returns ``(expected_live_hashes, expected_voiced_count)`` for the caller to
    check the live ledger against.
    """
    if transition.get("version") != CONTENT_TRANSITION_VERSION:
        raise ContentTransitionError(
            "unsupported content transition version %r"
            % (transition.get("version"),)
        )
    _normalize_stages(transition.get("authorized_stages"))

    # THE CHAIN IS BOUND TO THIS EXACT ACCEPTANCE, not merely to text that
    # happens to hash the same. Line ids repeat across episodes by
    # construction, and two episodes can carry an identical short row -- so
    # pre-state equality alone would let a transition be lifted from one
    # episode onto another, or from one LANE onto another, since the receipt
    # digest is what covers `owner_bank` and `accepted_artifacts`. Verified
    # only when the transition claims one: a claim made is a claim checked.
    claimed_parent = str(transition.get("parent_authorship_digest") or "")
    if claimed_parent:
        try:
            actual_parent = _authorship().receipt_sha256(ledger_data)
        except Exception as exc:  # noqa: BLE001
            raise ContentTransitionError(
                "transition claims a parent receipt this ledger cannot "
                "produce: %s" % exc
            ) from exc
        if claimed_parent != actual_parent:
            raise ContentTransitionError(
                "transition parent digest %s does not match this ledger's "
                "acceptance receipt %s" % (claimed_parent[:12], actual_parent[:12])
            )

    pre = transition.get("pre")
    post = transition.get("post")
    if not isinstance(pre, Mapping) or not isinstance(post, Mapping):
        raise ContentTransitionError("transition pre/post states are malformed")
    pre_hashes = pre.get("hashes")
    post_hashes = post.get("hashes")
    if not isinstance(pre_hashes, Mapping) or not isinstance(post_hashes, Mapping):
        raise ContentTransitionError("transition hash maps are malformed")

    # (2) the chain must start where the accepted artifact ended.
    parent = {str(k): str(v) for k, v in dict(parent_line_hashes).items()}
    claimed_pre = {str(k): str(v) for k, v in dict(pre_hashes).items()}
    if claimed_pre != parent:
        missing = sorted(set(parent) - set(claimed_pre))
        extra = sorted(set(claimed_pre) - set(parent))
        moved = sorted(
            k for k in set(parent) & set(claimed_pre)
            if parent[k] != claimed_pre[k]
        )
        raise ContentTransitionError(
            "transition pre-state does not match the accepted receipt "
            "(missing=%s extra=%s mismatched=%s)" % (missing, extra, moved)
        )

    # (3) the declared set must be EXACTLY the set that moved -- both ways.
    # Checking only "everything that changed is declared" leaves a transition
    # free to over-declare: to list rows it never touched. That is not an
    # attack, but this record exists to say what happened, and a receipt that
    # names innocent rows is a receipt that lies. An audit trail nobody can
    # trust in detail is one nobody reads.
    declared = set(str(x) for x in (transition.get("affected_line_ids") or ()))
    dropped = set(str(x) for x in (transition.get("dropped_line_ids") or ()))
    added = set(str(x) for x in (transition.get("added_line_ids") or ()))
    expected = {str(k): str(v) for k, v in dict(post_hashes).items()}
    actually_changed = {
        line_id for line_id, digest in expected.items()
        if line_id in claimed_pre and claimed_pre[line_id] != digest
    }
    if declared != actually_changed:
        undeclared = sorted(actually_changed - declared)
        overdeclared = sorted(declared - actually_changed)
        raise ContentTransitionError(
            "transition's declared changes do not match what moved "
            "(changed but undeclared=%s; declared but unchanged=%s)"
            % (undeclared, overdeclared)
        )
    if sorted(set(claimed_pre) - set(expected)) != sorted(dropped):
        raise ContentTransitionError(
            "transition dropped rows it does not declare"
        )
    if sorted(set(expected) - set(claimed_pre)) != sorted(added):
        raise ContentTransitionError(
            "transition added rows it does not declare"
        )
    count = post.get("voiced_line_count")
    if not isinstance(count, int) or count != len(expected):
        raise ContentTransitionError(
            "transition post coverage (%r) disagrees with its own hash map (%d)"
            % (count, len(expected))
        )
    return expected, count
