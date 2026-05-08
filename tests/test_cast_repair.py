"""tests/test_cast_repair.py

Phase 0+ Cast Contract Extensions §4 (5-bucket classifier) + §5
(plateau-bounded repair loop). Pure stdlib unit tests; no torch / no
LLM HTTP / no Comfy graph.

The classifier is pluggable, so every test that exercises §5 passes
its own deterministic stub instead of touching the real LLM.
"""
from __future__ import annotations

import pytest

from nodes._otr_cast_contract import (
    CastContract,
    CharacterEntry,
)
from nodes._otr_cast_repair import (
    CastContractUnreparable,
    ClassificationResult,
    DEFAULT_MAX_ITERATIONS,
    OrphanClass,
    RepairOutcome,
    apply_classifications,
    classify_orphans_stub,
    repair_orphans,
)


def _baseline_contract() -> CastContract:
    """Reusable 3-character contract for §5 tests."""
    contract = CastContract(
        characters=[
            CharacterEntry("c01", "AEGEUS", [], "bark:v2/en_speaker_5"),
            CharacterEntry("c02", "MONTY", [], "bark:v2/en_speaker_3"),
            CharacterEntry("c03", "SAILOR", [], "bark:v2/en_speaker_9"),
        ]
    )
    contract.stamp_version()
    return contract


# ---------- §4 enum + dataclass shape ------------------------------------


def test_orphan_class_has_all_5_buckets():
    expected = {
        "TYPO_OF_EXISTING",
        "ALIAS_OF_EXISTING",
        "GENUINELY_NEW",
        "NARRATIVE_LEAK",
        "DISCARD",
    }
    assert {b.value for b in OrphanClass} == expected


def test_orphan_class_serializes_as_string():
    """OrphanClass is a str-Enum so json.dumps works without a custom
    encoder."""
    import json

    payload = {"bucket": OrphanClass.ALIAS_OF_EXISTING.value}
    assert json.dumps(payload) == '{"bucket": "ALIAS_OF_EXISTING"}'


def test_classification_result_is_frozen():
    r = ClassificationResult(
        orphan_tag="MONTGOMERY",
        bucket=OrphanClass.ALIAS_OF_EXISTING,
        target_character_id="c02",
    )
    with pytest.raises(Exception):
        r.orphan_tag = "OTHER"  # type: ignore[misc]


def test_stub_classifier_always_returns_genuinely_new():
    contract = _baseline_contract()
    result = classify_orphans_stub("ZORG", contract, "ZORG: hello.\n")
    assert result.bucket == OrphanClass.GENUINELY_NEW
    assert result.orphan_tag == "ZORG"
    assert result.target_character_id == ""


# ---------- §4 apply_classifications -------------------------------------


def test_apply_classifications_adds_alias_for_alias_bucket():
    contract = _baseline_contract()
    pre_version = contract.version
    results = [
        ClassificationResult(
            orphan_tag="MONTGOMERY",
            bucket=OrphanClass.ALIAS_OF_EXISTING,
            target_character_id="c02",
            rationale="long form of MONTY",
        )
    ]
    _, applied = apply_classifications(contract, results)
    assert len(applied) == 1
    monty = next(c for c in contract.characters if c.character_id == "c02")
    assert "MONTGOMERY" in monty.aliases
    # Version must have rolled forward.
    assert contract.version != pre_version


def test_apply_classifications_typo_bucket_also_adds_alias():
    contract = _baseline_contract()
    results = [
        ClassificationResult(
            orphan_tag="MONNTY",  # double-N typo
            bucket=OrphanClass.TYPO_OF_EXISTING,
            target_character_id="c02",
        )
    ]
    _, applied = apply_classifications(contract, results)
    assert len(applied) == 1
    monty = next(c for c in contract.characters if c.character_id == "c02")
    assert "MONNTY" in monty.aliases


def test_apply_classifications_other_buckets_dont_mutate():
    contract = _baseline_contract()
    pre_version = contract.version
    results = [
        ClassificationResult(
            orphan_tag="ZORG",
            bucket=OrphanClass.GENUINELY_NEW,
        ),
        ClassificationResult(
            orphan_tag="THE LIGHTHOUSE",
            bucket=OrphanClass.NARRATIVE_LEAK,
        ),
        ClassificationResult(
            orphan_tag="STAGE DIRECTION",
            bucket=OrphanClass.DISCARD,
        ),
    ]
    _, applied = apply_classifications(contract, results)
    assert applied == []
    # Version unchanged because no aliases were added.
    assert contract.version == pre_version


def test_apply_classifications_skips_unknown_target_character_id():
    contract = _baseline_contract()
    results = [
        ClassificationResult(
            orphan_tag="MONTGOMERY",
            bucket=OrphanClass.ALIAS_OF_EXISTING,
            target_character_id="c99",  # not in the contract
        )
    ]
    _, applied = apply_classifications(contract, results)
    assert applied == []


def test_apply_classifications_dedupes_alias_case_insensitive():
    contract = CastContract(
        characters=[
            CharacterEntry("c01", "MONTY", ["MONTGOMERY"], "bark:v2/en_speaker_3"),
        ]
    )
    contract.stamp_version()
    pre_version = contract.version
    results = [
        ClassificationResult(
            orphan_tag="montgomery",  # already an alias, just lower-cased
            bucket=OrphanClass.ALIAS_OF_EXISTING,
            target_character_id="c01",
        )
    ]
    _, applied = apply_classifications(contract, results)
    assert applied == []
    assert contract.version == pre_version  # no version bump for no-op


def test_apply_classifications_skips_alias_equal_to_canonical():
    contract = _baseline_contract()
    results = [
        ClassificationResult(
            orphan_tag="MONTY",  # already canonical
            bucket=OrphanClass.ALIAS_OF_EXISTING,
            target_character_id="c02",
        )
    ]
    _, applied = apply_classifications(contract, results)
    assert applied == []


# ---------- §5 repair_orphans -------------------------------------------


def test_repair_orphans_empty_script_returns_zero_iterations():
    contract = _baseline_contract()
    outcome = repair_orphans("", contract)
    assert isinstance(outcome, RepairOutcome)
    assert outcome.iterations == 0
    assert outcome.aliases_added == 0
    assert outcome.final_orphans == []


def test_repair_orphans_resolves_via_cheap_path_alone():
    """detect_aliases (heuristic prefix match) should resolve MONTGOMERY ->
    MONTY without ever invoking the classifier."""
    contract = _baseline_contract()
    script = "AEGEUS: hello.\nMONTGOMERY: hello back.\n"

    classifier_calls: list[str] = []

    def tracking_classifier(tag, c, s):  # noqa: ARG001
        classifier_calls.append(tag)
        return classify_orphans_stub(tag, c, s)

    outcome = repair_orphans(script, contract, classifier=tracking_classifier)
    assert outcome.aliases_added == 1
    assert outcome.final_orphans == []
    assert classifier_calls == []  # cheap path resolved everything
    monty = next(c for c in contract.characters if c.character_id == "c02")
    assert "MONTGOMERY" in monty.aliases


def test_repair_orphans_invokes_classifier_for_residual():
    """A genuinely-new orphan that the heuristic cannot place (no shared
    prefix with any canonical) reaches the classifier."""
    contract = _baseline_contract()
    script = "ZORG: I am new.\n"

    seen: list[str] = []

    def tracking_classifier(tag, c, s):  # noqa: ARG001
        seen.append(tag)
        return classify_orphans_stub(tag, c, s)

    with pytest.raises(CastContractUnreparable) as exc:
        repair_orphans(script, contract, classifier=tracking_classifier)
    assert "ZORG" in exc.value.orphans
    assert seen == ["ZORG"]


def test_repair_orphans_plateau_raises_after_no_progress():
    """If the classifier keeps returning GENUINELY_NEW, the orphan set
    stays put -> plateau detection fires on iteration 2."""
    contract = _baseline_contract()
    script = "ZORG: persistent stranger.\n"

    iter_count = {"n": 0}

    def stuck_classifier(tag, c, s):  # noqa: ARG001
        iter_count["n"] += 1
        return classify_orphans_stub(tag, c, s)

    with pytest.raises(CastContractUnreparable) as exc:
        repair_orphans(script, contract, classifier=stuck_classifier)
    # Iter 1 calls classifier once, then iter 2 sees the same residual
    # and raises BEFORE another classifier call -- so iter_count["n"]
    # should be exactly 1.
    assert iter_count["n"] == 1
    assert exc.value.iterations == 1


def test_repair_orphans_classifier_bucket_alias_resolves_in_loop():
    """If the classifier returns ALIAS_OF_EXISTING, the orphan disappears
    on the next pass (no plateau)."""
    contract = _baseline_contract()
    # Orphan that DOES NOT prefix-match any canonical (ZORG -> ZORG; no
    # canonical starts with ZORG) and cheap detect_aliases will leave
    # it alone, so the classifier sees it.
    script = "ZORG: pretend I'm an alias of MONTY.\n"

    def alias_classifier(tag, c, s):  # noqa: ARG001
        return ClassificationResult(
            orphan_tag=tag,
            bucket=OrphanClass.ALIAS_OF_EXISTING,
            target_character_id="c02",
            rationale="test classifier always says alias of MONTY",
        )

    outcome = repair_orphans(script, contract, classifier=alias_classifier)
    assert outcome.aliases_added == 1
    assert outcome.final_orphans == []
    monty = next(c for c in contract.characters if c.character_id == "c02")
    assert "ZORG" in monty.aliases


def test_repair_orphans_respects_max_iterations_cap():
    """Pass ``max_iterations=1`` and the loop must give up after exactly
    one classifier pass even if convergence hasn't been ruled out."""
    contract = _baseline_contract()
    script = "ZORG: stranger.\nQUARK: another stranger.\n"

    with pytest.raises(CastContractUnreparable) as exc:
        repair_orphans(script, contract, max_iterations=1)
    assert exc.value.iterations == 1


def test_repair_orphans_default_iteration_cap_is_3():
    """Sanity check on the constant matching the §5 design doc."""
    assert DEFAULT_MAX_ITERATIONS == 3


def test_repair_orphans_rejects_non_contract():
    with pytest.raises(TypeError, match="must be a CastContract"):
        repair_orphans("ZORG: hi.\n", "not a contract")  # type: ignore[arg-type]


def test_repair_outcome_is_dataclass():
    r = RepairOutcome(iterations=2, classifications_seen=3, aliases_added=1)
    assert r.iterations == 2
    assert r.classifications_seen == 3
    assert r.aliases_added == 1
    assert r.final_orphans == []
