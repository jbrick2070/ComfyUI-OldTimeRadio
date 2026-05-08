"""nodes/_otr_cast_repair.py

Phase 0+ Cast Contract Extensions §4 (adversarial classification) +
§5 (plateau-bounded repair loop).

This module is the SKELETON: data shapes + plateau-bounded loop +
custom exception. The actual LLM classifier wires up later (after the
in-flight FULL acceptance soak completes and story_orchestrator
integration lands). Today the classifier is a pluggable callable, so
the loop can be exercised end-to-end with a stub.

Design lineage: NousResearch/autonovel ``adversarial_edit.py`` pattern
plus the cheap-fast-path / expensive-fallback split established in §1
helpers (``detect_aliases`` is the cheap path; this module's classifier
is the fallback for tags that ``detect_aliases`` could not place).

Public surface
--------------
    OrphanClass         -- enum of the 5 buckets
    ClassificationResult -- per-orphan classifier output
    Classifier          -- type alias for the pluggable callable
    classify_orphans_stub -- a deterministic placeholder classifier so
                             the loop has something to call before the
                             real LLM hookup ships
    repair_orphans      -- the plateau-bounded loop
    CastContractUnreparable -- raised when the loop plateaus
    apply_classifications  -- mutate a CastContract per a batch of
                              ClassificationResult records (alias add,
                              etc.)

Module is stdlib-only -- no torch, no LLM HTTP, no ComfyUI imports --
so it imports cleanly into any node, including subprocess workers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterable, Optional

from nodes._otr_cast_contract import (
    CastContract,
    detect_aliases,
)


# ---------- §4: 5-bucket classification ----------------------------------


class OrphanClass(str, Enum):
    """5 buckets per ROADMAP "Phase 0+ candidates" -> Cast Contract §4.

    Storing as ``str`` Enum so ``ClassificationResult`` serializes via
    ``json.dumps`` without a custom encoder -- the value is the bucket
    name verbatim.
    """

    TYPO_OF_EXISTING = "TYPO_OF_EXISTING"  # auto-canonicalize -> existing character
    ALIAS_OF_EXISTING = "ALIAS_OF_EXISTING"  # add to alias map, bump version
    GENUINELY_NEW = "GENUINELY_NEW"  # hard fail, reroll cast
    NARRATIVE_LEAK = "NARRATIVE_LEAK"  # narration text -- demote, HuMo bypass
    DISCARD = "DISCARD"  # noise, drop


@dataclass(frozen=True)
class ClassificationResult:
    """Single orphan tag's classifier verdict.

    Frozen so a list of these can be hashed for plateau detection in
    ``repair_orphans``.
    """

    orphan_tag: str  # the tag as it appeared in the script
    bucket: OrphanClass
    target_character_id: str = ""  # populated for TYPO/ALIAS, empty otherwise
    confidence: float = 1.0  # 0.0 - 1.0; classifier's own confidence
    rationale: str = ""  # one-line human-readable explanation


# A classifier is any callable that takes (orphan_tag, contract,
# script_excerpt) and returns a ClassificationResult. The real
# implementation will call an LLM; the stub is deterministic.
Classifier = Callable[[str, CastContract, str], ClassificationResult]


def classify_orphans_stub(
    orphan_tag: str,
    contract: CastContract,
    script_excerpt: str,  # noqa: ARG001 -- present in real signature, unused in stub
) -> ClassificationResult:
    """Deterministic placeholder classifier so the repair loop is
    exercisable before the real LLM hookup lands.

    Heuristic: every orphan defaults to ``GENUINELY_NEW``. Conservative
    on purpose -- the real classifier will look at script context and
    pick TYPO/ALIAS/etc.; until that lands, treating every orphan as
    new prevents silent merges into the wrong character.
    """
    return ClassificationResult(
        orphan_tag=orphan_tag,
        bucket=OrphanClass.GENUINELY_NEW,
        target_character_id="",
        confidence=0.0,  # placeholder confidence -- real classifier will override
        rationale="placeholder classifier; defer to real LLM hookup",
    )


# ---------- §4: applying classifications to a contract -------------------


def apply_classifications(
    contract: CastContract,
    results: Iterable[ClassificationResult],
) -> tuple[CastContract, list[ClassificationResult]]:
    """Apply a batch of classifications.

    Mutates ``contract`` in place for the buckets that have a known
    side effect:

    - ``TYPO_OF_EXISTING`` and ``ALIAS_OF_EXISTING``: add ``orphan_tag``
      to the target character's ``aliases`` list (deduplicated,
      case-insensitive).

    For all other buckets the contract is left untouched -- those are
    the caller's responsibility (``GENUINELY_NEW`` triggers a cast
    reroll outside this module; ``NARRATIVE_LEAK`` triggers a script
    rewrite to demote the tag to narration; ``DISCARD`` is a no-op).

    Re-stamps the contract version IFF any alias was actually added.
    Returns ``(contract, applied_results)`` where ``applied_results``
    is the subset of results that produced a contract mutation (handy
    for logging "what changed this iteration").
    """
    applied: list[ClassificationResult] = []
    for r in results:
        if r.bucket not in (OrphanClass.TYPO_OF_EXISTING, OrphanClass.ALIAS_OF_EXISTING):
            continue
        if not r.target_character_id:
            continue
        target = next(
            (c for c in contract.characters if c.character_id == r.target_character_id),
            None,
        )
        if target is None:
            continue
        # Deduplicate case-insensitively.
        existing_upper = {a.upper() for a in target.aliases}
        if r.orphan_tag.upper() in existing_upper:
            continue
        if r.orphan_tag.upper() == target.canonical_name.upper():
            continue
        target.aliases.append(r.orphan_tag)
        applied.append(r)
    if applied:
        contract.stamp_version()
    return contract, applied


# ---------- §5: plateau-bounded repair loop ------------------------------


class CastContractUnreparable(RuntimeError):
    """Raised by ``repair_orphans`` when the loop plateaus before the
    orphan set goes empty.

    Attributes:
        orphans: the orphan set at the moment we gave up.
        iterations: how many classifier passes we burned.
    """

    def __init__(self, orphans: list[str], iterations: int):
        self.orphans = list(orphans)
        self.iterations = iterations
        super().__init__(
            f"cast contract repair plateaued after {iterations} iteration(s); "
            f"unresolved orphans: {sorted(self.orphans)}"
        )


@dataclass
class RepairOutcome:
    """Result of a successful ``repair_orphans`` call."""

    iterations: int
    classifications_seen: int  # total ClassificationResult records produced
    aliases_added: int
    final_orphans: list[str] = field(default_factory=list)  # always [] on success


# Hard cap on iterations. The §5 design says 3; we enforce it as the
# default while still letting tests pass a smaller value for speed.
DEFAULT_MAX_ITERATIONS = 3


def repair_orphans(
    script: str,
    contract: CastContract,
    classifier: Classifier = classify_orphans_stub,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> RepairOutcome:
    """§5: plateau-bounded repair loop.

    Algorithm:
      1. Run ``detect_aliases(script, contract)`` (the cheap §1 helper)
         to consume every orphan it can place via the 4-char prefix
         heuristic. Apply those as ALIAS_OF_EXISTING immediately.
      2. Recompute the residual orphan set by re-scanning the script
         for tags still not in canonical or alias.
      3. If empty -> return RepairOutcome (success).
      4. If non-empty: send each residual orphan to the ``classifier``
         (the real LLM in production, the stub in tests). Apply the
         results.
      5. Recompute the residual orphan set. If it equals the previous
         residual set -> raise ``CastContractUnreparable`` (the loop
         is stuck; no progress between iterations).
      6. Loop. Hard cap at ``max_iterations`` regardless.

    The "plateau" check uses set equality on the orphan tags, not
    ClassificationResult equality, because the same residuals coming
    back means *the classifier learned nothing new this round*.
    """
    if not isinstance(contract, CastContract):
        raise TypeError("contract must be a CastContract")

    # Step 1: cheap-path heuristic alias pass (one-shot, before any
    # classifier call). Mutates contract directly via apply_classifications.
    cheap_aliases = detect_aliases(script, contract)
    cheap_results = [
        ClassificationResult(
            orphan_tag=tag,
            bucket=OrphanClass.ALIAS_OF_EXISTING,
            target_character_id=cid,
            confidence=0.5,  # heuristic, not LLM-confirmed
            rationale="4-char prefix match (heuristic, pre-LLM)",
        )
        for tag, cid in cheap_aliases.items()
    ]
    _, cheap_applied = apply_classifications(contract, cheap_results)
    aliases_added_total = len(cheap_applied)
    classifications_seen_total = len(cheap_results)

    # Tags the classifier has explicitly disposed of in a non-mutating
    # bucket (DISCARD / NARRATIVE_LEAK / GENUINELY_NEW). Without this set,
    # any orphan the classifier correctly classifies as not-an-alias
    # would re-appear in residual_orphans on the next iteration, equal
    # the previous residual, and crash the loop with
    # CastContractUnreparable -- which is wrong: the classifier did its
    # job and the loop should treat those orphans as *resolved* (just
    # not via alias addition). 2026-05-08 round-robin Element 3 catch.
    decided_residuals: set[str] = set()

    prev_residual: Optional[set[str]] = None
    for iteration in range(1, max_iterations + 1):
        # Recompute residual: tags in script that are STILL not lookup-able
        # AND not yet explicitly decided by the classifier.
        live_residual = _residual_orphans(script, contract) - decided_residuals
        if not live_residual:
            return RepairOutcome(
                iterations=iteration - 1,  # everything resolved before this iter
                classifications_seen=classifications_seen_total,
                aliases_added=aliases_added_total,
                final_orphans=[],
            )

        if prev_residual is not None and live_residual == prev_residual:
            raise CastContractUnreparable(
                orphans=list(live_residual),
                iterations=iteration - 1,
            )
        prev_residual = set(live_residual)

        # Send each live residual to the classifier, collect results.
        results = [
            classifier(tag, contract, script) for tag in live_residual
        ]
        classifications_seen_total += len(results)
        _, applied = apply_classifications(contract, results)
        aliases_added_total += len(applied)

        # Mark every orphan the classifier put in a non-mutating bucket
        # as decided so the next iteration doesn't re-classify them.
        for r in results:
            if r.bucket in (
                OrphanClass.DISCARD,
                OrphanClass.NARRATIVE_LEAK,
                OrphanClass.GENUINELY_NEW,
            ):
                decided_residuals.add(r.orphan_tag)

    # Hit the iteration cap without converging.
    final = _residual_orphans(script, contract) - decided_residuals
    if not final:
        return RepairOutcome(
            iterations=max_iterations,
            classifications_seen=classifications_seen_total,
            aliases_added=aliases_added_total,
            final_orphans=[],
        )
    raise CastContractUnreparable(
        orphans=list(final),
        iterations=max_iterations,
    )


def _residual_orphans(script: str, contract: CastContract) -> set[str]:
    """Return the set of dialogue tags in ``script`` that ``contract``
    still cannot resolve via canonical name or registered alias.

    Imported lazily from ``_otr_cast_contract`` to keep this file
    self-contained against import-cycle risk if the contract module
    grows additional helpers later.
    """
    from nodes._otr_cast_contract import _extract_dialogue_tags

    residual: set[str] = set()
    for tag in _extract_dialogue_tags(script):
        if contract.lookup(tag) is None:
            residual.add(tag)
    return residual
