"""ONE owner for the question "may this episode be published?".

WHY THIS EXISTS, AND WHY IT IS NOT A NEW GATE. The operator rule has been in
force since 2026-07-17: *a research_only source BLOCKS publish.* What was wrong
was never the rule -- it was where the rule was realised. `G14` lived inside
`run_gap_audit`, which is a READ-ONLY audit, and its only lever was
`report.errors`, which at Phase 10 means `FreezeAssertionError`. So the rule
was enforced by KILLING A FINISHED RENDER: an episode that had already been
written, cast, voiced, rendered and muxed died at the freeze, and the operator
got nothing at all -- no archival copy to study, which is the one thing a
research-only source is actually cleared for.

That is the shape the build contract calls a Law 7 violation ("a render must
not die"; structural refusal is allowed BEFORE generation, not after it). The
rule now lands where publication actually happens: `OTRMasterAudioMux` writes
the archival final either way and only the OBS copy -- the published
deliverable -- is withheld. Blocking publish and destroying the work are
different actions, and only one of them is what the operator asked for.

ONE PRODUCER. The receipt combines RIGHTS reason codes (the rights sidecar,
normalised by `_otr_provenance.normalize_provenance`) and IDENTITY reason codes
(`_otr_source_identity.identity_from_meta`) into a SINGLE durable record. It is
deliberately not two stamps: two writers on one decision is how a consumer ends
up asking one of them and getting the other one's answer, and D5's identity work
must not become a second author of publishability.

RIGHTS DECIDE. IDENTITY ONLY EXPLAINS. A degraded identity -- the coda could
not name the work -- is recorded as an informational reason code and never
blocks. Zero of the 65 public-domain units lack a title or an author (the
manifest schema enforces both), so a blocking identity rule would fire only on
a corpus fault, and the correct response to a corpus fault is a receipt that
says so, not a withheld episode.

FAIL CLOSED AT THE CONSUMER, NOT AT THE PRODUCER. `evaluate` never raises and
never blocks for a reason it cannot substantiate. The CONSUMER
(`decide_from_meta`) is the strict one: a receipt that is missing, malformed,
of an unknown version, or stamped for a DIFFERENT episode is treated as
blocked. That asymmetry is intentional. Producing a permissive default is
honest ("nothing here forbids publication"); consuming a permissive default is
a guess about a rights question, and a stale singleton from a previous episode
is exactly the way that guess goes wrong.

Pure and cold-import clean: stdlib only, no model call, no network, no I/O.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Tuple

log = logging.getLogger("OTR")

__all__ = [
    "PUBLICATION_ELIGIBILITY_VERSION",
    "PUBLICATION_ELIGIBILITY_META_KEY",
    "PublicationEligibility",
    "PublicationDecision",
    "evaluate_publication_eligibility",
    "stamp_publication_eligibility",
    "decide_from_meta",
]

#: Bump when the REASON CODES or the decision rules change, never for wording.
#: A consumer that does not recognise the version treats the receipt as
#: unreadable and therefore blocked -- an old node must never approve a
#: publication under rules it has never seen.
PUBLICATION_ELIGIBILITY_VERSION = "publication_eligibility_v1"

#: The single durable key. Both the producer and every consumer import this
#: constant rather than spelling the string twice (`BUG_BIBLE.yaml` 12.86 --
#: producer/consumer literal mismatch).
PUBLICATION_ELIGIBILITY_META_KEY = "publication_eligibility"

# --- reason codes ----------------------------------------------------------
# BLOCKING codes deny the OBS copy. INFORMATIONAL codes ride along so a reader
# knows what the receipt noticed without changing what it decided.

#: The rights sidecar normalised to `research_only`.
REASON_RIGHTS_RESEARCH_ONLY = "rights_research_only"
#: A rights record exists and clears publication.
REASON_RIGHTS_CLEARED = "rights_cleared"
#: No `meta.provenance` at all. The normaliser is opt-in per bank (the writer
#: stamps it only when the bank sets `defaults.provenance_normalize`), so its
#: ABSENCE is the ordinary state on four of six banks and cannot mean "blocked".
REASON_RIGHTS_NOT_STAMPED = "rights_not_stamped"
#: `meta.provenance` is present but not a mapping -- a shape fault worth saying.
REASON_RIGHTS_MALFORMED = "rights_malformed"
#: The identity adapter could name the work.
REASON_IDENTITY_COMPLETE = "identity_complete"
#: The identity adapter came back degraded (informational -- see the docstring).
REASON_IDENTITY_DEGRADED = "identity_degraded"
#: The lane is not one the identity adapter reads (`original`, the news lanes
#: before D6 stamps a headline). Not a fault; most banks are simply not
#: source-bearing in the bibliographic sense.
REASON_IDENTITY_NOT_APPLICABLE = "identity_not_applicable"

#: Codes that deny publication. Kept as a set so `blocking_reasons` is derived
#: from the codes rather than tracked as a second, driftable boolean.
BLOCKING_REASONS = frozenset({REASON_RIGHTS_RESEARCH_ONLY})

# --- consumer-side refusals -------------------------------------------------
# These are decisions, not receipt contents: they describe why a CONSUMER would
# not honour a receipt it was handed.
DECISION_NO_RECEIPT = "eligibility_receipt_absent"
DECISION_MALFORMED = "eligibility_receipt_malformed"
DECISION_VERSION_UNKNOWN = "eligibility_receipt_version_unknown"
DECISION_EPISODE_MISMATCH = "eligibility_receipt_episode_mismatch"

#: WHAT TO DO ABOUT IT, PER REFUSAL. A diagnostic that names the offence and
#: not the required shape is a defect class this very sprint is fixing
#: elsewhere (`BUG_BIBLE.yaml` 12.105: a markup ladder burned four rungs
#: re-emitting `END` because nothing ever said `END.`). "No eligibility
#: receipt" tells an operator what happened and leaves them to guess what to
#: do; a withheld episode with no stated remedy reads as a broken render.
#:
#: Keyed by decision reason so the remedy has ONE owner and a new refusal
#: cannot ship without one -- `tests/test_publication_eligibility.py` asserts
#: every `DECISION_*` constant appears here.
DECISION_REMEDIES = {
    DECISION_NO_RECEIPT: (
        "no publication verdict was ever stamped for this episode. The "
        "producer is OTR_LedgerFreezeCascade Phase 10 -- re-run the graph "
        "through the freeze, or re-freeze this ledger, and the mux will "
        "publish on the next pass. The archival final is already on disk and "
        "nothing was lost"
    ),
    DECISION_MALFORMED: (
        "the receipt is present but unreadable, so it was written by "
        "something other than the current producer. Re-freeze the ledger to "
        "restamp it rather than hand-editing meta.publication_eligibility"
    ),
    DECISION_VERSION_UNKNOWN: (
        "the receipt was stamped by a different version of the eligibility "
        "rules than this node reads. Re-freeze the ledger on the current "
        "build to restamp it"
    ),
    DECISION_EPISODE_MISMATCH: (
        "this receipt belongs to a DIFFERENT episode, which usually means a "
        "stale in-flight ledger singleton. Re-run the graph so the writer "
        "claims this episode, then re-freeze"
    ),
}


def _clean(value: Any) -> str:
    """Trim to a usable string. ``None`` and junk both become ``''``."""
    if value is None:
        return ""
    return str(value).strip()


@dataclass(frozen=True)
class PublicationEligibility:
    """What the ledger itself says about its own publishability.

    ``reasons`` is ORDERED and complete -- rights first, then identity -- so
    two runs over the same ledger produce the same receipt bytes and therefore
    the same digest. A digest that moves when nothing moved would defeat the
    cache key it is built for.
    """

    version: str = PUBLICATION_ELIGIBILITY_VERSION
    episode_id: str = ""
    eligible: bool = True
    reasons: Tuple[str, ...] = ()

    @property
    def blocking_reasons(self) -> Tuple[str, ...]:
        """The subset of ``reasons`` that actually denied publication."""
        return tuple(r for r in self.reasons if r in BLOCKING_REASONS)

    def as_receipt(self) -> Dict[str, Any]:
        """The durable form. This exact dict is what lands in ``meta``."""
        return {
            "version": self.version,
            "episode_id": self.episode_id,
            "eligible": self.eligible,
            "reasons": list(self.reasons),
            "blocking_reasons": list(self.blocking_reasons),
        }

    @property
    def digest(self) -> str:
        """A stable content hash of the receipt.

        `sort_keys` + a fixed separator make this reproducible across
        processes, which `hash()` on a string is NOT -- Python salts string
        hashing per interpreter, so a cache key built on it silently changes
        every server boot.
        """
        payload = json.dumps(
            self.as_receipt(), sort_keys=True, separators=(",", ":"),
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PublicationDecision:
    """A consumer's verdict on a receipt it was handed.

    Separate from :class:`PublicationEligibility` on purpose: the receipt is
    what the ledger CLAIMS, the decision is what this consumer will DO about
    it, and a missing receipt has a decision but no claim.
    """

    publishable: bool
    reason: str
    detail: str = ""
    digest: str = ""
    episode_id: str = ""
    receipt: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """The reason phrase, AND what to do about it.

        Deliberately carries no "BLOCKED"/"OK" prefix of its own: the terminal
        node prints ``obs_publish BLOCKED -- <summary>`` beside its existing
        ``obs_publish OK -> <path>``, and a second verdict word inside the
        summary would read as two verdicts about one decision.

        The REMEDY is appended for the consumer-side refusals, because an
        operator reading "no eligibility receipt" in a render log otherwise
        has to go and find out that the producer is Phase 10 of the freeze
        cascade. A rights BLOCK (``ineligible``) gets no remedy on purpose --
        that one is working as designed and there is nothing to repair.
        """
        head = "%s: %s" % (self.reason, self.detail) if self.detail else self.reason
        remedy = DECISION_REMEDIES.get(self.reason)
        return "%s. TO FIX: %s" % (head, remedy) if remedy else head


def _rights_reasons(meta: Mapping[str, Any]) -> List[str]:
    """Rights reason codes, in receipt order. Decides nothing on its own."""
    prov = meta.get("provenance")
    if prov is None:
        return [REASON_RIGHTS_NOT_STAMPED]
    if not isinstance(prov, Mapping):
        # A malformed rights record is NOT read as a block. The gap audit
        # already reports shape faults, and inventing a rights denial out of a
        # type error would withhold an episode for a reason nobody stated.
        # It is still SAID OUT LOUD: `normalize_provenance` always returns a
        # dict, so this shape means something upstream wrote the field by hand,
        # and a rights record nobody can read is worth a line in the log even
        # though it is not worth withholding an episode over.
        log.warning(
            "[OTR_PublicationEligibility] meta.provenance has type %s; "
            "expected a mapping from normalize_provenance. Rights cannot be "
            "read from it, so this episode is recorded as %s and is NOT "
            "blocked on that basis.",
            type(prov).__name__, REASON_RIGHTS_MALFORMED,
        )
        return [REASON_RIGHTS_MALFORMED]
    if prov.get("blocks_publish"):
        return [REASON_RIGHTS_RESEARCH_ONLY]
    return [REASON_RIGHTS_CLEARED]


def _identity_reasons(meta: Mapping[str, Any]) -> List[str]:
    """Identity reason codes. Always informational -- see the module docstring."""
    try:
        try:
            from ._otr_source_identity import identity_from_meta
        except ImportError:  # pragma: no cover -- flat test/standalone load
            from _otr_source_identity import identity_from_meta  # type: ignore
    except Exception:  # pragma: no cover -- never let a reason code raise
        return [REASON_IDENTITY_NOT_APPLICABLE]
    identity = identity_from_meta(meta)
    if not identity.source_kind:
        return [REASON_IDENTITY_NOT_APPLICABLE]
    if identity.is_degraded:
        return [REASON_IDENTITY_DEGRADED]
    return [REASON_IDENTITY_COMPLETE]


def evaluate_publication_eligibility(ledger_data: Any) -> PublicationEligibility:
    """Read a ledger and return its publication receipt. Never raises.

    Read-only: this computes, it does not stamp. `stamp_publication_eligibility`
    is the only writer, so a caller that merely wants to ASK cannot accidentally
    become a second author of the field.
    """
    if not isinstance(ledger_data, Mapping):
        return PublicationEligibility(
            eligible=True, reasons=(REASON_RIGHTS_NOT_STAMPED,),
        )
    meta = ledger_data.get("meta")
    if not isinstance(meta, Mapping):
        meta = {}

    reasons: List[str] = []
    reasons.extend(_rights_reasons(meta))
    reasons.extend(_identity_reasons(meta))
    eligible = not any(r in BLOCKING_REASONS for r in reasons)

    episode_id = _clean(ledger_data.get("episode_id")) or _clean(
        meta.get("episode_id")
    )
    return PublicationEligibility(
        episode_id=episode_id,
        eligible=eligible,
        reasons=tuple(reasons),
    )


def stamp_publication_eligibility(ledger_data: Any) -> PublicationEligibility:
    """THE producer. Evaluate, stamp ``meta`` and return the receipt.

    Idempotent: the receipt is derived wholly from the ledger's own rights and
    identity records and carries no timestamp, so re-stamping an unchanged
    ledger writes identical bytes and leaves the digest -- and therefore the
    mux's cache key -- exactly where it was.

    Stamping is best-effort against a malformed ledger: when ``meta`` is not a
    dict the audit has already flagged that as a critical gap, and crashing the
    stamper would replace that diagnostic with a confusing one. The receipt is
    still returned so a caller can act on it.
    """
    eligibility = evaluate_publication_eligibility(ledger_data)
    if isinstance(ledger_data, dict):
        meta = ledger_data.get("meta")
        if isinstance(meta, dict):
            meta[PUBLICATION_ELIGIBILITY_META_KEY] = eligibility.as_receipt()
    return eligibility


def decide_from_meta(
    meta: Any,
    *,
    expected_episode_id: str = "",
) -> PublicationDecision:
    """THE consumer. Read a stamped receipt and decide whether to publish.

    Strict by design (the module docstring explains the asymmetry): absent,
    malformed, unknown-version and episode-mismatched receipts all return
    ``publishable=False``.

    ``expected_episode_id`` is enforced whenever the caller supplies one, and an
    ANONYMOUS receipt fails that check rather than skipping it. An earlier cut
    compared the two only when BOTH were non-empty, reasoning that a receipt
    with no episode id "has nothing to disagree with" -- but a receipt carrying
    the current version, a valid ``eligible`` flag and an empty ``episode_id``
    clears every earlier gate and then silently skipped this one, so it would
    have answered for any episode that asked. A caller that asserts an identity
    is asking to be told when the receipt cannot match it; unprovable is not
    the same as agreed.
    """
    if not isinstance(meta, Mapping):
        return PublicationDecision(
            publishable=False,
            reason=DECISION_NO_RECEIPT,
            detail="ledger meta is not readable",
        )
    receipt = meta.get(PUBLICATION_ELIGIBILITY_META_KEY)
    if receipt is None:
        return PublicationDecision(
            publishable=False,
            reason=DECISION_NO_RECEIPT,
            detail="no %s receipt on this ledger" % PUBLICATION_ELIGIBILITY_META_KEY,
        )
    if not isinstance(receipt, Mapping):
        return PublicationDecision(
            publishable=False,
            reason=DECISION_MALFORMED,
            detail="receipt has type %s; expected a mapping"
                   % type(receipt).__name__,
        )
    version = _clean(receipt.get("version"))
    if version != PUBLICATION_ELIGIBILITY_VERSION:
        return PublicationDecision(
            publishable=False,
            reason=DECISION_VERSION_UNKNOWN,
            detail="receipt version %r; this node reads %r"
                   % (version, PUBLICATION_ELIGIBILITY_VERSION),
        )
    if not isinstance(receipt.get("eligible"), bool):
        return PublicationDecision(
            publishable=False,
            reason=DECISION_MALFORMED,
            detail="receipt 'eligible' is %r; expected a bool"
                   % (receipt.get("eligible"),),
        )

    receipt_episode = _clean(receipt.get("episode_id"))
    expected = _clean(expected_episode_id)
    if expected and receipt_episode != expected:
        detail = (
            "receipt carries no episode id, so it cannot be shown to belong to "
            "%r" % expected
            if not receipt_episode
            else "receipt is stamped for %r; this episode is %r"
                 % (receipt_episode, expected)
        )
        return PublicationDecision(
            publishable=False,
            reason=DECISION_EPISODE_MISMATCH,
            detail=detail,
            episode_id=receipt_episode,
        )

    reasons = receipt.get("reasons")
    reasons_tuple = tuple(
        str(r) for r in reasons if isinstance(r, str)
    ) if isinstance(reasons, (list, tuple)) else ()
    rebuilt = PublicationEligibility(
        version=version,
        episode_id=receipt_episode,
        eligible=bool(receipt.get("eligible")),
        reasons=reasons_tuple,
    )
    if not rebuilt.eligible:
        blocking = rebuilt.blocking_reasons or ("unspecified",)
        return PublicationDecision(
            publishable=False,
            reason="ineligible",
            detail="; ".join(blocking),
            digest=rebuilt.digest,
            episode_id=receipt_episode,
            receipt=dict(receipt),
        )
    return PublicationDecision(
        publishable=True,
        reason="eligible",
        detail="; ".join(rebuilt.reasons),
        digest=rebuilt.digest,
        episode_id=receipt_episode,
        receipt=dict(receipt),
    )
