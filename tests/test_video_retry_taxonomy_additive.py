"""Additive CPU coverage for the render-time retry taxonomy (A-S7).

New, self-contained tests for nodes/_otr_shared/retry_taxonomy.py focused on the
DERIVED bits and fail-closed edges that complement
tests/test_video_retry_taxonomy.py: the max_attempts arithmetic, string<->enum
coercion in classify, the immutability of the pure dict transforms, and the
assert_decision_invariants guard rejecting hand-built bad decisions. Pure stdlib
+ the module under test. UTF-8, no BOM, ASCII-only, SFW.
"""
from __future__ import annotations

import pytest

from nodes._otr_shared import retry_taxonomy as rt


def test_classify_accepts_raw_string_value():
    assert rt.classify("oom") == rt.classify(rt.FailureKind.OOM)


def test_classify_unknown_kind_fails_closed():
    with pytest.raises(ValueError):
        rt.classify("not_a_real_failure_kind")


def test_block_class_of_unknown_string_raises():
    with pytest.raises(ValueError):
        rt.block_class_of("definitely_unknown")


def test_max_attempts_matches_retry_budget():
    # corrupt_output: 1 same-seed + 1 reseed -> 3 attempts on the engine.
    assert rt.classify(rt.FailureKind.CORRUPT_OUTPUT).max_attempts == 3
    # oom: 0 retries -> single attempt then escalate.
    assert rt.classify(rt.FailureKind.OOM).max_attempts == 1
    # transient_io: 1 + DEFAULT_TRANSIENT_IO_RETRIES.
    ti = rt.classify(rt.FailureKind.TRANSIENT_IO)
    assert ti.max_attempts == 1 + rt.DEFAULT_TRANSIENT_IO_RETRIES


def test_is_hard_is_warn_track_block_class_for_every_kind():
    for kind in rt.FailureKind:
        d = rt.classify(kind)
        assert d.kind is kind
        assert d.block_class is rt.block_class_of(kind)
        assert d.is_hard == (d.block_class is rt.BlockClass.HARD)
        assert d.is_warn == (d.block_class is rt.BlockClass.WARN)
        assert d.is_hard != d.is_warn


def test_every_classified_decision_honors_the_hard_invariants():
    # No emitted decision may ever drop a beat, touch audio, or abort.
    for kind in rt.FailureKind:
        d = rt.classify(kind)
        assert d.discards_output is False
        assert d.touches_audio is False
        assert d.aborts_episode is False


def test_warn_decisions_keep_output_hard_decisions_escalate():
    for kind in rt.WARN_KINDS:
        d = rt.classify(kind)
        assert d.keep_output is True
        assert d.escalate_to_fallback is False
    for kind in rt.HARD_KINDS:
        d = rt.classify(kind)
        assert d.escalate_to_fallback is True
        assert d.warn_only is False


def test_assert_decision_invariants_rejects_discard():
    bad = rt.RetryDecision(
        kind=rt.FailureKind.OOM, block_class=rt.BlockClass.HARD,
        discards_output=True)
    with pytest.raises(ValueError):
        rt.assert_decision_invariants(bad)


def test_assert_decision_invariants_rejects_audio_touch_and_abort():
    touches = rt.RetryDecision(
        kind=rt.FailureKind.OOM, block_class=rt.BlockClass.HARD,
        touches_audio=True)
    aborts = rt.RetryDecision(
        kind=rt.FailureKind.OOM, block_class=rt.BlockClass.HARD,
        aborts_episode=True)
    with pytest.raises(ValueError):
        rt.assert_decision_invariants(touches)
    with pytest.raises(ValueError):
        rt.assert_decision_invariants(aborts)


def test_assert_decision_invariants_rejects_warn_without_keep_output():
    bad = rt.RetryDecision(
        kind=rt.FailureKind.QUALITY, block_class=rt.BlockClass.WARN,
        keep_output=False)
    with pytest.raises(ValueError):
        rt.assert_decision_invariants(bad)


def test_build_fallback_decision_pins_revision_and_block_class():
    rec = rt.build_fallback_decision(
        shot_id="shot_0007", beat_id="b007", from_engine="humo",
        to_engine="latentsync", kind=rt.FailureKind.OOM, video_revision=1)
    assert rec["video_revision"] == 1
    assert rec["block_class"] == rt.BlockClass.HARD.value
    assert rec["failure_kind"] == "oom"
    assert rec["from_engine"] == "humo" and rec["to_engine"] == "latentsync"
    assert rec["attempt"] == 1


def test_append_runtime_fallback_decision_is_pure_and_revision_locked():
    section = {"video_revision": 2, "runtime_fallback_decisions": []}
    rec = rt.build_fallback_decision(
        shot_id="s", beat_id="b", from_engine="humo", to_engine="latentsync",
        kind=rt.FailureKind.OOM, video_revision=2)
    out = rt.append_runtime_fallback_decision(section, rec)
    assert out is not section                        # new dict
    assert section["runtime_fallback_decisions"] == []  # input unmutated
    assert out["video_revision"] == 2                # revision unchanged
    assert out["runtime_fallback_decisions"] == [rec]


def test_append_runtime_fallback_decision_rejects_revision_mismatch():
    section = {"video_revision": 2}
    rec = rt.build_fallback_decision(
        shot_id="s", beat_id="b", from_engine="humo", to_engine="latentsync",
        kind=rt.FailureKind.OOM, video_revision=3)
    with pytest.raises(ValueError):
        rt.append_runtime_fallback_decision(section, rec)


def test_restamp_shot_row_is_pure_and_appends_trail():
    row = {"engine_id": "humo", "family": "audio_driven_face"}
    out = rt.restamp_shot_row(
        row, to_engine="still_kenburns", to_family="static_motion",
        from_engine="humo", kind=rt.FailureKind.OOM)
    assert out is not row
    assert row == {"engine_id": "humo", "family": "audio_driven_face"}
    assert out["engine_id"] == "still_kenburns"
    assert out["family"] == "static_motion"
    assert out["degradation_trail"] == ["humo->still_kenburns (oom)"]


def test_format_swap_log_is_loud_and_reaffirms_audio_untouched():
    rec = rt.build_fallback_decision(
        shot_id="shot_0020", beat_id="b020", from_engine="hunyuan3d_talk",
        to_engine="humo", kind=rt.FailureKind.OOM, video_revision=1)
    line = rt.format_swap_log(rec)
    assert "LOUD FALLBACK" in line
    assert "frozen audio untouched" in line
    assert "hunyuan3d_talk" in line and "humo" in line
