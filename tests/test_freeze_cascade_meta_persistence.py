"""tests/test_freeze_cascade_meta_persistence.py

Regression coverage for BUG-LOCAL-273 + BUG-LOCAL-274 (live run
signal_lost_bioluminescent_trench_descent_20260525_182002).

Root cause: production_ledger.Ledger.save() rebinds self.data to a new
merged dict on every save (production_ledger.py:1012, BUG-LOCAL-108
merge-with-disk pattern). The freeze cascade binds ledger_data + meta
from led.data at L570/L576; the Phase 1+2 reviewer at L622 calls
led.save() internally, detaching those locals from the live ledger.
Subsequent stamps on the stale meta (Sprint 5B critic at L683, Sprint 6
render_plan at L748) land on an orphaned dict that no consumer ever
sees -- the reroll, render_plan, and HuMo all re-read led.data and get
the live dict with no critic report and no render plan. The critic's
defaults (arc_verdict="strong", empty reroll_targets/flat_lines) then
fire silently because StoryCriticReport.model_validate({}) succeeds.

These tests pin the fix:
  * BUG-LOCAL-273 -- the critic stamp survives a save-rebind between
    cascade entry and the Sprint 5B block; reroll + render_plan see the
    real critic report, not clean()/defaults.
  * BUG-LOCAL-274 -- the render_plan stamp is persisted via led.save()
    so OTR_BatchHumoRender (which loads the on-disk ledger in a
    separate node invocation) sees meta.render_plan and honours it.

UTF-8 no BOM. No em-dashes (Windows cp1252 subprocess decode trap).
4-space indentation.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import _otr_freeze_cascade as _LFC_ORCH
from nodes import _otr_ledger_freeze as _LFC
from nodes import _otr_ledger_reviewer as _OTRLR
from nodes import _otr_story_critic as _OTRSC


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _clean_ledger_data() -> dict:
    """Minimal-valid L3 ledger -- same shape as the existing
    test_lfc_freeze_cascade_orchestrator fixture."""
    return {
        "schema_version": _LFC.EXPECTED_SCHEMA_VERSION,
        "episode_id": "ep_test_meta_persistence",
        "cast": [
            {"char_id": "c01", "name": "ALICE",
             "traits": "calm", "voice_preset": "v2/en_speaker_6"},
            {"char_id": "c02", "name": "BOB",
             "traits": "skeptical", "voice_preset": "v2/en_speaker_2"},
            {"char_id": "announcer", "name": "ANNOUNCER",
             "traits": "warm", "voice_preset": "v2/en_speaker_9"},
        ],
        "scenes": [], "shots": [],
        "beats": [{"beat_id": f"b00{i}"} for i in range(1, 5)],
        "lines": [
            {"line_id": "l001", "beat_id": "b001",
             "speaker_role": "announcer", "char_id": "announcer",
             "text": "Tonight, on Signal Lost.",
             "word_count": 4, "char_count": 24, "skip": False},
            {"line_id": "l002", "beat_id": "b002",
             "speaker_role": "character", "char_id": "c01",
             "text": "Bob, the readings are off the chart.",
             "word_count": 7, "char_count": 36, "skip": False},
            {"line_id": "l003", "beat_id": "b003",
             "speaker_role": "character", "char_id": "c02",
             "text": "Then someone tampered with the calibration.",
             "word_count": 6, "char_count": 43, "skip": False},
            {"line_id": "l004", "beat_id": "b004",
             "speaker_role": "announcer", "char_id": "announcer",
             "text": "Stay tuned.",
             "word_count": 2, "char_count": 11, "skip": False},
        ],
        "music": [], "clips": [],
        "meta": {
            "episode_title": "The Calibration Drift",
            "style": "mission_control_procedural",
        },
    }


class _RebindingLedger:
    """Fake Ledger that REBINDS .data on save() -- the exact behaviour
    of production_ledger.Ledger.save() at line 1012 (BUG-LOCAL-108
    merge-with-disk pattern) that causes BUG-LOCAL-273/274.

    Without this rebind in the test fixture, the staleness bug is
    invisible; the SimpleNamespace fixture used in the existing
    orchestrator tests has save = lambda: None which preserves the
    pre-bug reference behaviour.
    """

    def __init__(self, data: dict):
        self.data = data
        self.episode_id = data.get("episode_id", "ep")
        self.save_count = 0
        # Track every (snapshot of) data so a test can assert which
        # save captured which stamps.
        self.save_snapshots: list[dict] = []

    def save(self):
        """Mimic production save: build a NEW dict from current data
        and rebind self.data. The deep-copy via dict comprehension is
        intentional -- a shallow `dict(self.data)` would still share
        nested mutables and hide the bug surface."""
        self.save_count += 1
        # Deep enough to detach top-level + meta (the surfaces the
        # cascade mutates). Lines / cast lists shared by reference is
        # fine -- the bug is on meta keys.
        new_data = {k: v for k, v in self.data.items()}
        new_data["meta"] = dict(self.data.get("meta", {}))
        self.data = new_data
        # Snapshot AFTER rebind so the test sees the persisted shape.
        snap = {k: v for k, v in self.data.items()}
        snap["meta"] = dict(self.data.get("meta", {}))
        self.save_snapshots.append(snap)


def _stub_reviewer_disposition(verdict: str = "clean_no_edits"):
    return _OTRLR.ReviewerDisposition(
        verdict=verdict,
        pre_audit_violations=0,
        pre_audit_repairs_applied=0,
        doctor_edits_proposed=0,
        doctor_edits_applied=0,
        post_audit_violations=0,
    )


def _critic_with_targets() -> _OTRSC.StoryCriticReport:
    """A non-clean critic report: arc_verdict=uneven + 2 reroll_targets +
    1 flat_line. Mirrors the shape the live-run BUG-273 critic produced
    (4 targets / 4 flat / arc_verdict=uneven). Two targets / one flat is
    enough to prove the values survive the cascade-to-consumer handoff."""
    return _OTRSC.StoryCriticReport(
        continuity_issues=[],
        voice_drift=[],
        flat_lines=[
            _OTRSC.FlatLine(line_id="l002", reason="exposition dump"),
        ],
        arc_verdict="uneven",
        reroll_targets=[
            _OTRSC.RerollTarget(
                line_id="l002",
                hint="give Alice a sharper line; this one stalls",
            ),
            _OTRSC.RerollTarget(
                line_id="l003",
                hint="Bob's accusation needs more punch",
            ),
        ],
        render_priority=["l002", "l003"],
    )


def _critic_advisory_only() -> _OTRSC.StoryCriticReport:
    """Like `_critic_with_targets` but with ZERO reroll_targets so the
    Sprint 5C reroll early-exits at `_otr_reroll.py:507` without
    re-running the critic. That re-run is what otherwise overwrites
    meta.story_critic_report; for tests that pin the INITIAL critic
    stamp's reach, the empty-targets early-exit keeps the original
    stamp visible end-to-end. arc_verdict=uneven + 1 flat_line are
    preserved so render_plan still has non-default content to read."""
    return _OTRSC.StoryCriticReport(
        continuity_issues=[],
        voice_drift=[],
        flat_lines=[
            _OTRSC.FlatLine(line_id="l002", reason="exposition dump"),
        ],
        arc_verdict="uneven",
        reroll_targets=[],
        render_priority=["l002", "l003"],
    )


# ---------------------------------------------------------------------------
# BUG-LOCAL-273: critic stamp survives the reviewer's internal save-rebind
# ---------------------------------------------------------------------------


class TestBug273CriticStampSurvivesReviewerSave:
    """The Phase 1+2 reviewer at cascade L622 calls Ledger.save()
    internally. That save rebinds led.data. The Sprint 5B critic stamp
    must land on the LIVE (post-rebind) led.data["meta"], not on the
    cascade's stale L576 local."""

    def test_critic_report_lands_on_live_meta_after_reviewer_save(self):
        """End-to-end check: with an advisory-only critic (no
        reroll_targets so the reroll early-exits and does NOT re-run
        the critic / overwrite meta.story_critic_report), the original
        stamp must reach the live led.data['meta'] post-cascade. This
        is the BUG-273 surface in its purest form -- the stamp at L683
        must land on the post-reviewer-save dict, not on a stale local."""
        led = _RebindingLedger(_clean_ledger_data())
        critic_report = _critic_advisory_only()

        def fake_review(_fn, led_):
            # Reviewer-internal save: this rebinds led_.data. The bug
            # surface lives ONLY when this rebind actually happens.
            led_.save()
            return _stub_reviewer_disposition("clean_no_edits")

        def fake_critic(_fn, _candidate_ledger, _cast_rows):
            return critic_report

        with patch.object(_LFC_ORCH._OTRLR, "review_ledger",
                          side_effect=fake_review), \
             patch.object(_LFC_ORCH._OTRSC, "run_story_critic",
                          side_effect=fake_critic):
            _LFC_ORCH.run_freeze_cascade(lambda *a, **k: "", led)

        # Pre-fix this would be empty / clean() -- the cascade would
        # have stamped on the stale pre-reviewer-save meta reference.
        stamped = led.data["meta"].get("story_critic_report")
        assert stamped is not None, (
            "BUG-LOCAL-273: story_critic_report did not reach live meta "
            "-- the cascade stamped on a stale pre-save reference"
        )
        assert stamped.get("arc_verdict") == "uneven", (
            "arc_verdict lost in the cascade-to-meta handoff "
            "(read back as %r)" % stamped.get("arc_verdict")
        )
        assert len(stamped.get("flat_lines", [])) == 1, (
            "flat_lines lost -- read back %d items (expected 1)"
            % len(stamped.get("flat_lines", []))
        )

    def test_reroll_sees_real_critic_not_clean_fallback(self):
        """The downstream symptom: BUG-273's live run logged
        '[OTR_Reroll] critic named no reroll targets -- nothing to do'
        even though the critic produced 4 targets. With the fix, the
        reroll sees the real targets."""
        led = _RebindingLedger(_clean_ledger_data())

        def fake_review(_fn, led_):
            led_.save()
            return _stub_reviewer_disposition("clean_no_edits")

        def fake_critic(_fn, _candidate_ledger, _cast_rows):
            return _critic_with_targets()

        # Patch the reroll to record what report it actually receives.
        seen = {"reroll_targets": None, "arc_verdict": None}
        original_reroll = _LFC_ORCH._OTRRR.run_targeted_reroll

        def spy_reroll(generate_fn, led_):
            # Coerce the same way the production reroll does.
            from nodes._otr_reroll import _coerce_report
            meta = led_.data.setdefault("meta", {})
            report = _coerce_report(meta.get("story_critic_report"))
            seen["reroll_targets"] = list(report.reroll_targets)
            seen["arc_verdict"] = report.arc_verdict
            # Return a no-op disposition so the cascade continues.
            return original_reroll(generate_fn, led_)

        with patch.object(_LFC_ORCH._OTRLR, "review_ledger",
                          side_effect=fake_review), \
             patch.object(_LFC_ORCH._OTRSC, "run_story_critic",
                          side_effect=fake_critic), \
             patch.object(_LFC_ORCH._OTRRR, "run_targeted_reroll",
                          side_effect=spy_reroll):
            _LFC_ORCH.run_freeze_cascade(lambda *a, **k: "", led)

        assert seen["arc_verdict"] == "uneven", (
            "BUG-LOCAL-273: reroll saw arc_verdict=%r (expected 'uneven') "
            "-- critic stamp did not reach reroll's view of meta"
            % seen["arc_verdict"]
        )
        assert seen["reroll_targets"] is not None
        assert len(seen["reroll_targets"]) == 2, (
            "BUG-LOCAL-273: reroll saw %d targets (expected 2) -- the "
            "critic stamp was on a stale meta dict and the reroll "
            "re-read led.data after the rebind"
            % len(seen["reroll_targets"])
        )


# ---------------------------------------------------------------------------
# BUG-LOCAL-274: render_plan stamp is persisted to disk for HuMo
# ---------------------------------------------------------------------------


class TestBug274RenderPlanPersistedAfterStamp:
    """OTR_BatchHumoRender is a separate node invocation that loads the
    ledger via _load_ledger_with_path from *_ledger.json on disk. The
    cascade's Sprint 6 render_plan stamp must reach disk; pre-fix it
    only lived in-memory on a stale (often orphaned) meta dict and the
    next save call did not necessarily include it."""

    def test_render_plan_present_on_live_meta_after_cascade(self):
        led = _RebindingLedger(_clean_ledger_data())

        def fake_review(_fn, led_):
            led_.save()  # triggers the BUG-273 rebind surface too
            return _stub_reviewer_disposition("clean_no_edits")

        def fake_critic(_fn, _candidate_ledger, _cast_rows):
            return _critic_with_targets()

        with patch.object(_LFC_ORCH._OTRLR, "review_ledger",
                          side_effect=fake_review), \
             patch.object(_LFC_ORCH._OTRSC, "run_story_critic",
                          side_effect=fake_critic):
            _LFC_ORCH.run_freeze_cascade(lambda *a, **k: "", led)

        render_plan = led.data["meta"].get("render_plan")
        assert render_plan is not None, (
            "BUG-LOCAL-274: render_plan did not reach live meta "
            "post-cascade -- the Sprint 6 stamp was orphaned"
        )
        # Confirm the plan structure is intact.
        assert "line_ids" in render_plan
        assert "selection_mode" in render_plan
        assert "blocked" in render_plan

    def test_render_plan_persisted_via_explicit_save(self):
        """The fix adds led.save() after the render_plan stamp so HuMo's
        on-disk read sees the plan. This test asserts the save fired
        AFTER the stamp by checking the save_snapshots history."""
        led = _RebindingLedger(_clean_ledger_data())

        def fake_review(_fn, led_):
            led_.save()
            return _stub_reviewer_disposition("clean_no_edits")

        def fake_critic(_fn, _candidate_ledger, _cast_rows):
            return _critic_with_targets()

        with patch.object(_LFC_ORCH._OTRLR, "review_ledger",
                          side_effect=fake_review), \
             patch.object(_LFC_ORCH._OTRSC, "run_story_critic",
                          side_effect=fake_critic):
            _LFC_ORCH.run_freeze_cascade(lambda *a, **k: "", led)

        # At least one snapshot must contain render_plan -- proving a
        # save fired AFTER the stamp landed. Without the fix's
        # explicit led.save() the stamp would only have a chance to
        # land on disk via Phase 10's save, and even then only if the
        # stamp made it onto the live meta (BUG-273).
        snapshots_with_plan = [
            snap for snap in led.save_snapshots
            if snap.get("meta", {}).get("render_plan") is not None
        ]
        assert len(snapshots_with_plan) >= 1, (
            "BUG-LOCAL-274: no save snapshot captured meta.render_plan "
            "-- HuMo's on-disk read would see no plan and fall back to "
            "render-all (saved %d times total)" % led.save_count
        )

    def test_arc_verdict_lifts_render_plan_blocking_correctly(self):
        """Tied bug: arc_verdict=uneven must NOT block the render. The
        live run logged arc_verdict=strong (the default) instead of the
        real arc_verdict=uneven, which would have masked a real blocking
        condition. With the fix, build_render_plan sees the true verdict.

        Uses the advisory-only critic so the reroll early-exits and
        does NOT overwrite meta.story_critic_report with the re-run's
        result -- otherwise the render_plan would correctly read the
        re-run report (which is clean() under a no-op stub generate_fn)
        and the test would conflate two separate code paths."""
        led = _RebindingLedger(_clean_ledger_data())

        def fake_review(_fn, led_):
            led_.save()
            return _stub_reviewer_disposition("clean_no_edits")

        def fake_critic(_fn, _candidate_ledger, _cast_rows):
            return _critic_advisory_only()  # arc_verdict='uneven', 0 targets

        with patch.object(_LFC_ORCH._OTRLR, "review_ledger",
                          side_effect=fake_review), \
             patch.object(_LFC_ORCH._OTRSC, "run_story_critic",
                          side_effect=fake_critic):
            _LFC_ORCH.run_freeze_cascade(lambda *a, **k: "", led)

        render_plan = led.data["meta"].get("render_plan")
        assert render_plan is not None
        assert render_plan["source_arc_verdict"] == "uneven", (
            "render plan read arc_verdict=%r -- expected 'uneven' from "
            "the critic. Default 'strong' indicates the stamp did not "
            "reach build_render_plan's view of meta."
            % render_plan["source_arc_verdict"]
        )
        # 'uneven' does NOT block (only mid_collapse + flat at cap do).
        assert render_plan["blocked"] is False


# ---------------------------------------------------------------------------
# BUG-LOCAL-278: cascade meta is persisted at every cascade exit
# ---------------------------------------------------------------------------


class TestBug278CascadeMetaPersistedAtEveryExit:
    """BUG-LOCAL-278 surfaced from the Sprint 10A operator soak run
    signal_lost_..._154105: meta.freeze_verdict, meta.story_critic_report,
    meta.stage7_shadow_critic were all visible in the live cascade log
    but ABSENT from the persisted ledger.json after the run. KokoroAnnouncer
    reads from disk + writes back, silently dropping the cascade's
    in-memory modifications.

    Fix: _persist_cascade_meta(led) called at every cascade exit
    (terminal-skip from reviewer, reroll-exhaustion, Phase 10 reject,
    successful freeze). These tests pin that the save fires.

    The render_plan exit was already covered by BUG-LOCAL-274's fix
    (TestBug274RenderPlanPersistedAfterStamp above) and stays green.
    """

    def test_save_fires_on_reroll_exhaustion_exit(self):
        """The reroll-exhaustion path stamps freeze_verdict=needs_full_rerun
        + freeze_disposition + meta keys from Sprint 5B critic + Sprint 10A
        step 7 shadow critic. Pre-fix, the cascade returned without saving;
        Kokoro's subsequent disk read missed every one of them. The fix
        persists before the return.

        Test approach: patch _persist_cascade_meta itself and assert it
        was called from the reroll-exhaustion exit. This decouples the
        BUG-278 wiring check (does the call site exist?) from the
        BUG-273-family stale-meta concern (does the stamp reach live
        meta in the first place?). BUG-273's existing test class covers
        the stale-meta surface end-to-end."""
        led = _RebindingLedger(_clean_ledger_data())

        # Force reroll into exhaustion by returning a real reroll
        # disposition with verdict=needs_full_rerun. RerollDisposition
        # requires final_report; pass a minimal critic report.
        from nodes._otr_reroll import RerollDisposition
        final_report = _OTRSC.StoryCriticReport(
            continuity_issues=[], voice_drift=[], flat_lines=[],
            arc_verdict="uneven", reroll_targets=[],
            render_priority=[],
        )

        def fake_review(_fn, led_):
            led_.save()
            return _stub_reviewer_disposition("clean_no_edits")

        def fake_critic(_fn, _candidate_ledger, _cast_rows):
            return _critic_with_targets()

        def fake_reroll(_fn, _led):
            return RerollDisposition(
                verdict="needs_full_rerun",
                cycles_run=2,
                lines_rerolled=2,
                final_report=final_report,
            )

        persist_calls: list = []
        original_persist = _LFC_ORCH._persist_cascade_meta

        def spy_persist(led_arg):
            persist_calls.append(led_arg)
            return original_persist(led_arg)

        with patch.object(_LFC_ORCH._OTRLR, "review_ledger",
                          side_effect=fake_review), \
             patch.object(_LFC_ORCH._OTRSC, "run_story_critic",
                          side_effect=fake_critic), \
             patch.object(_LFC_ORCH._OTRRR, "run_targeted_reroll",
                          side_effect=fake_reroll), \
             patch.object(_LFC_ORCH, "_persist_cascade_meta",
                          side_effect=spy_persist):
            _LFC_ORCH.run_freeze_cascade(lambda *a, **k: "", led)

        # The reroll-exhaustion exit must have called the persist helper.
        # Other exits (Phase 10 reject, success freeze) also call it; the
        # reroll-exhaustion path returns before reaching them, so a
        # single call from THIS path is sufficient.
        assert len(persist_calls) >= 1, (
            "BUG-LOCAL-278: _persist_cascade_meta was not called from "
            "the reroll-exhaustion exit; the wiring is missing"
        )

    def test_save_fires_on_successful_freeze_exit(self):
        """The success path (frozen_clean / frozen_with_warns /
        frozen_with_doctor_edits) lands Phase 10 + stamps the final
        verdict. The fix's persist at the success exit ensures
        downstream consumers see the final state on disk."""
        led = _RebindingLedger(_clean_ledger_data())

        def fake_review(_fn, led_):
            led_.save()
            return _stub_reviewer_disposition("clean_no_edits")

        def fake_critic(_fn, _candidate_ledger, _cast_rows):
            # Advisory-only -- no reroll targets, no exhaustion.
            return _critic_advisory_only()

        pre_save_count = led.save_count
        with patch.object(_LFC_ORCH._OTRLR, "review_ledger",
                          side_effect=fake_review), \
             patch.object(_LFC_ORCH._OTRSC, "run_story_critic",
                          side_effect=fake_critic):
            _LFC_ORCH.run_freeze_cascade(lambda *a, **k: "", led)

        # The post-cascade save count must have grown -- at least one
        # save by the cascade itself (BUG-274 render_plan persist), one
        # by the new BUG-278 success-exit persist.
        assert led.save_count > pre_save_count + 1, (
            "BUG-LOCAL-278: cascade success exit did not save the ledger "
            "(only %d saves total since cascade entry, expected at least 2)"
            % (led.save_count - pre_save_count)
        )
        # freeze_verdict must be on the last snapshot.
        last_snap = led.save_snapshots[-1]
        verdict = last_snap.get("meta", {}).get("freeze_verdict")
        assert verdict is not None, (
            "BUG-LOCAL-278: meta.freeze_verdict missing from final save"
        )
        # Either frozen_clean, frozen_with_warns, or frozen_with_doctor_edits.
        assert verdict.startswith("frozen"), (
            "BUG-LOCAL-278: expected a frozen_* verdict on success exit; "
            "got %r" % verdict
        )

    def test_persist_helper_never_raises_on_save_failure(self):
        """PD1: the persist helper must never raise. If Ledger.save()
        throws (disk full, permission, etc.), the cascade still returns
        cleanly -- in-memory state preserved, on-disk state stale, the
        warning is logged."""

        class _BrokenSaveLedger(_RebindingLedger):
            def save(self):  # type: ignore[override]
                raise IOError("simulated disk failure")

        led = _BrokenSaveLedger(_clean_ledger_data())

        def fake_review(_fn, _led):
            # Don't call save() on the broken ledger; just return
            # disposition. The cascade's own _persist_cascade_meta is
            # what we're testing.
            return _stub_reviewer_disposition("clean_no_edits")

        def fake_critic(_fn, _candidate_ledger, _cast_rows):
            return _critic_advisory_only()

        with patch.object(_LFC_ORCH._OTRLR, "review_ledger",
                          side_effect=fake_review), \
             patch.object(_LFC_ORCH._OTRSC, "run_story_critic",
                          side_effect=fake_critic):
            # Must not raise.
            disp = _LFC_ORCH.run_freeze_cascade(lambda *a, **k: "", led)

        # In-memory state still valid.
        assert disp is not None
        assert led.data["meta"].get("freeze_verdict") is not None

    def test_persist_helper_handles_simplenamespace_fixture(self):
        """Test fixtures that stub `led` as a SimpleNamespace without a
        save method must not crash the cascade. The defensive
        getattr(led, 'save', None) guards this."""
        led = SimpleNamespace(
            data=_clean_ledger_data(),
            episode_id="ep_test_simplenamespace",
        )

        def fake_review(_fn, _led):
            return _stub_reviewer_disposition("clean_no_edits")

        def fake_critic(_fn, _candidate_ledger, _cast_rows):
            return _critic_advisory_only()

        with patch.object(_LFC_ORCH._OTRLR, "review_ledger",
                          side_effect=fake_review), \
             patch.object(_LFC_ORCH._OTRSC, "run_story_critic",
                          side_effect=fake_critic):
            # Must not raise.
            disp = _LFC_ORCH.run_freeze_cascade(lambda *a, **k: "", led)

        assert disp is not None
        assert led.data["meta"].get("freeze_verdict") is not None

    def test_persist_helper_is_called_at_terminal_reviewer_exit(self):
        """The reviewer-terminal exit (too_many_edits / needs_full_rerun
        from review_ledger itself) also needs to persist. Pin that
        _persist_cascade_meta is reached on this path.

        Test approach: same as test_save_fires_on_reroll_exhaustion_exit
        -- spy on _persist_cascade_meta itself rather than verifying
        post-save state. This isolates the BUG-278 wiring check from the
        BUG-273-family stale-meta-after-reviewer-rebind dynamic.
        """
        led = _RebindingLedger(_clean_ledger_data())

        def fake_review(_fn, led_):
            led_.save()
            return _stub_reviewer_disposition("needs_full_rerun")

        persist_calls: list = []
        original_persist = _LFC_ORCH._persist_cascade_meta

        def spy_persist(led_arg):
            persist_calls.append(led_arg)
            return original_persist(led_arg)

        with patch.object(_LFC_ORCH._OTRLR, "review_ledger",
                          side_effect=fake_review), \
             patch.object(_LFC_ORCH, "_persist_cascade_meta",
                          side_effect=spy_persist):
            _LFC_ORCH.run_freeze_cascade(lambda *a, **k: "", led)

        # Terminal reviewer exit must invoke the persist helper exactly
        # once -- the cascade returns before any other exit point fires.
        assert len(persist_calls) == 1, (
            "BUG-LOCAL-278: expected exactly 1 _persist_cascade_meta "
            "call from the terminal-reviewer exit, got %d"
            % len(persist_calls)
        )
