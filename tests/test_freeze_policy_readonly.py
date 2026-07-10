"""FreezePolicy + proof-preserving freeze boundary (external QA ROUND 2
fold, 2026-07-10, fable2 S1b -> S2 runway).

The r2 P0 chain this file locks down:

  P0.1  the cascade invoked the text-mutating reviewer BEFORE its lane
        capability gate, so a sealed fable2 proof map could diverge from
        the live canonical text (Butterfly b2/b5, Einstein b3). Fix: ONE
        typed FreezePolicy resolved before any content pass; for
        content-owned lanes every legacy mutator is skipped and only
        read-only structural validation + Phase 10 run; a TAGGED bank
        that fails to resolve is TERMINAL, never fail-open.
  P0.3  Ledger.save() rebinds led.data; the cascade kept writing phase
        records and the terminal veto into the detached pre-save dict,
        so the persisted ledger had NO freeze_verdict and CastLock's
        missing-verdict legacy path let a needs_full_rerun episode
        through. Fix: refresh led.data immediately after review_ledger.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nodes import _otr_freeze_cascade as LFC  # noqa: E402
from nodes import _otr_story_routing as ROUTING  # noqa: E402
from nodes import production_ledger as _PL  # noqa: E402
from nodes._otr_ledger_reviewer import ReviewerDisposition  # noqa: E402

import test_fable2_assembly as _F2A  # noqa: E402  (golden-chain helpers)


@pytest.fixture(autouse=True)
def _fresh_registry_and_ledger():
    saved = _PL._CURRENT
    ROUTING._REGISTRY = None
    yield
    ROUTING._REGISTRY = None
    _PL._CURRENT = saved


def _fail_generate(*_a, **_k):
    raise AssertionError(
        "generate_fn must NEVER be called under a readonly/terminal policy")


def _forbid_legacy_passes(monkeypatch):
    """Spy every legacy content entry point: any call is a test failure."""
    def _boom(name):
        def _fn(*_a, **_k):
            raise AssertionError(f"{name} must not run under this policy")
        return _fn
    monkeypatch.setattr(LFC._OTRLR, "review_ledger",
                        _boom("review_ledger"))
    monkeypatch.setattr(LFC._OTRSC, "run_story_critic",
                        _boom("run_story_critic"))
    monkeypatch.setattr(LFC._OTRRR, "run_targeted_reroll",
                        _boom("run_targeted_reroll"))


def _golden_fable2_ledger(tmp_path, *, source_bank="scifi_fable2"):
    led, _parsed, meta = _F2A._assembled(tmp_path)
    if source_bank is not None:
        meta["source_bank"] = source_bank
    led.data.setdefault("meta", {}).update(meta)
    return led


# ---------------------------------------------------------------------------
# 1. Policy resolution (replaces the retired fail-open boolean)
# ---------------------------------------------------------------------------

def test_policy_untagged_ledger_is_legacy():
    pol = LFC.resolve_freeze_policy({})
    assert pol.name == "legacy_full"
    assert pol.run_legacy_content_passes is True
    assert pol.terminal_error == ""


def test_policy_seam_bank_is_legacy():
    pol = LFC.resolve_freeze_policy({"source_bank": "science_news"})
    assert pol.name == "legacy_full"
    assert pol.run_legacy_content_passes is True


def test_policy_fable2_is_content_owned_readonly():
    pol = LFC.resolve_freeze_policy({"source_bank": "scifi_fable2"})
    assert pol.name == "content_owned_readonly"
    assert pol.run_legacy_content_passes is False
    assert pol.terminal_error == ""


def test_policy_unresolvable_tagged_bank_is_terminal_not_fail_open():
    # r2 P1.2: "unknown means legacy applicable" is RETIRED for declared
    # sources -- a tagged bank that fails to resolve must halt, never
    # fall open into legacy content mutation.
    pol = LFC.resolve_freeze_policy({"source_bank": "no_such_bank"})
    assert pol.name == "policy_resolution_failed"
    assert pol.run_legacy_content_passes is False
    assert pol.terminal_error


# ---------------------------------------------------------------------------
# 2. Readonly cascade: proof invariance + zero legacy calls (P0.1)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("escalation_env", ["", "1"])
def test_readonly_cascade_freezes_without_any_content_mutation(
        tmp_path, monkeypatch, escalation_env):
    if escalation_env:
        monkeypatch.setenv("OTR_ENABLE_CRITIC_ESCALATION", escalation_env)
    else:
        monkeypatch.delenv("OTR_ENABLE_CRITIC_ESCALATION", raising=False)
    _forbid_legacy_passes(monkeypatch)
    led = _golden_fable2_ledger(tmp_path)
    before_lines = copy.deepcopy(led.data["lines"])
    before_proof = copy.deepcopy(led.data["meta"]["fable2"]["proof_map"])

    disp = LFC.run_freeze_cascade(_fail_generate, led)

    assert disp.verdict in ("frozen_clean", "frozen_with_warns"), (
        disp.verdict)
    meta = led.data["meta"]
    # canonical text + proof map byte-identical through the cascade
    assert [r["text"] for r in led.data["lines"]] == [
        r["text"] for r in before_lines]
    assert meta["fable2"]["proof_map"] == before_proof
    # truthful capability receipt (r2 P0.1)
    receipt = meta["freeze_capability_receipt"]
    assert receipt["policy"] == "content_owned_readonly"
    assert receipt["content_mutations"] == 0
    assert receipt["text_sha256_entry"] == receipt["text_sha256_exit"]
    assert receipt["proof_map_sha256_entry"] == (
        receipt["proof_map_sha256_exit"])
    for phase in LFC._LEGACY_CONTENT_PHASES:
        assert phase in receipt["skipped_phases"], phase
    # the reviewer result is a capability disposition, NOT a fake clean
    assert meta["reviewer_disposition_capability"] == (
        "not_applicable_content_owned")
    assert "story_critic_status" not in meta  # 5B genuinely did not run
    # Phase 7 canonical-text normalization skipped BY POLICY
    assert meta["audio_readiness"]["skipped_reason"] == (
        "content_owned_readonly_policy")


def test_readonly_cascade_proof_divergence_is_terminal(
        tmp_path, monkeypatch):
    _forbid_legacy_passes(monkeypatch)
    led = _golden_fable2_ledger(tmp_path)
    # tamper one proven line AFTER the seal (the Butterfly b2 shape)
    for row in led.data["lines"]:
        if row.get("speaker_role") == "character" and row.get("text"):
            row["text"] = row["text"] + " tampered"
            break
    disp = LFC.run_freeze_cascade(_fail_generate, led)
    assert disp.verdict == "needs_full_rerun"
    meta = led.data["meta"]
    assert meta["freeze_block_class"] == "structural"
    receipt = meta["freeze_capability_receipt"]
    assert receipt["structural_errors"], "proof divergence must be recorded"
    assert any("proof" in e for e in receipt["structural_errors"])


def test_unresolvable_tagged_bank_cascade_halts_before_any_pass(
        tmp_path, monkeypatch):
    _forbid_legacy_passes(monkeypatch)
    led = _golden_fable2_ledger(tmp_path, source_bank="no_such_bank")
    disp = LFC.run_freeze_cascade(_fail_generate, led)
    assert disp.verdict == "needs_full_rerun"
    meta = led.data["meta"]
    assert meta["freeze_block_class"] == "structural"
    receipt = meta["freeze_capability_receipt"]
    assert receipt["policy"] == "policy_resolution_failed"
    assert receipt["terminal_error"]


# ---------------------------------------------------------------------------
# 3. P0.3 -- reviewer save/rebind must not detach the terminal veto
# ---------------------------------------------------------------------------

def _rebinding_reviewer(verdict: str):
    """A reviewer double that performs a REAL rebinding save (the exact
    production sequence: mutate -> Ledger.save() -> self.data = merged)
    and then returns the requested verdict."""
    def _review(_generate_fn, led):
        led.data.setdefault("meta", {})["reviewer_probe"] = "ran"
        old_root = led.data
        led.save()  # rebinds led.data to the merged dict
        assert led.data is not old_root, (
            "precondition: save() must rebind led.data for this test "
            "to exercise the P0.3 detachment")
        return ReviewerDisposition(
            verdict=verdict,
            pre_audit_violations=0, pre_audit_repairs_applied=0,
            doctor_edits_proposed=0, doctor_edits_applied=0,
            post_audit_violations=0,
        )
    return _review


def test_terminal_reviewer_verdict_lands_on_live_root_and_disk(
        tmp_path, monkeypatch):
    # untagged ledger -> legacy_full policy -> reviewer runs
    led = _golden_fable2_ledger(tmp_path, source_bank=None)
    led.data["meta"].pop("source_bank", None)
    monkeypatch.setattr(LFC._OTRLR, "review_ledger",
                        _rebinding_reviewer("needs_full_rerun"))
    disp = LFC.run_freeze_cascade(_fail_generate, led)
    assert disp.verdict == "needs_full_rerun"
    # LIVE root carries the veto (pre-fix: only the detached dict did)
    meta = led.data["meta"]
    assert meta["freeze_verdict"] == "needs_full_rerun"
    assert meta["freeze_disposition"]["verdict"] == "needs_full_rerun"
    phase_names = [p.get("phase_name") for p in
                   (meta.get("cleanup_passes") or [])]
    assert "phase_1_2_9_reviewer_composite" in phase_names
    # persisted JSON carries it too (CastLock halts on the explicit veto)
    on_disk = json.loads(Path(led.path).read_text(encoding="utf-8"))
    assert on_disk["meta"]["freeze_verdict"] == "needs_full_rerun"


def test_commit_path_reviewer_rebind_keeps_phase_records_live(
        tmp_path, monkeypatch):
    led = _golden_fable2_ledger(tmp_path, source_bank=None)
    led.data["meta"].pop("source_bank", None)
    monkeypatch.setattr(LFC._OTRLR, "review_ledger",
                        _rebinding_reviewer("clean_no_edits"))
    # keep the rest of the legacy chain deterministic + offline
    monkeypatch.setattr(LFC._OTRSC, "run_story_critic",
                        lambda *_a, **_k: LFC._OTRSC.StoryCriticReport.clean())
    disp = LFC.run_freeze_cascade(_fail_generate, led)
    assert disp.verdict in (
        "frozen_clean", "frozen_with_warns", "frozen_with_doctor_edits")
    meta = led.data["meta"]
    assert meta.get("freeze_verdict") == disp.verdict
    phase_names = [p.get("phase_name") for p in
                   (meta.get("cleanup_passes") or [])]
    assert "phase_1_2_9_reviewer_composite" in phase_names
    receipt = meta["freeze_capability_receipt"]
    assert receipt["policy"] == "legacy_full"
