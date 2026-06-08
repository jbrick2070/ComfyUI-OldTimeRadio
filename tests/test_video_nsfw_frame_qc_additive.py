"""Additive CPU coverage for the offline NSFW frame-QC sampler (A-S7).

New, self-contained tests for nodes/_otr_shared/nsfw_frame_qc.py: DEFAULT-OFF
(no flag -> ran=False), deterministic evenly-strided sampling, warn-only routing
(a flag maps to a WARN decision that keeps output), and robustness (a raising
detector is caught as an abstain, never breaking the render). Env is injected via
the env= param so os.environ is never touched. Complements
test_video_nsfw_frame_qc.py. Pure stdlib + the module under test. UTF-8, no BOM,
ASCII, SFW.
"""
from __future__ import annotations

from nodes._otr_shared import nsfw_frame_qc as qc
from nodes._otr_shared import retry_taxonomy as rt

ON = {qc.QC_ENABLE_FLAG: "1"}
OFF = {qc.QC_ENABLE_FLAG: "0"}


def test_qc_enabled_reads_the_flag():
    assert qc.qc_enabled(ON) is True
    assert qc.qc_enabled(OFF) is False
    assert qc.qc_enabled({}) is False                  # default-off


def test_default_off_returns_disabled_verdict():
    v = qc.run_frame_qc(frame_count=100, env={})
    assert v.ran is False
    assert v.flagged is False
    assert v.frames_sampled == 0


def test_sample_indices_empty_for_nonpositive():
    assert qc.sample_frame_indices(0) == ()
    assert qc.sample_frame_indices(-5) == ()


def test_sample_indices_all_when_shorter_than_max():
    assert qc.sample_frame_indices(3, max_samples=8) == (0, 1, 2)


def test_sample_indices_evenly_strided_first_to_last():
    idx = qc.sample_frame_indices(100, max_samples=5)
    assert len(idx) == 5
    assert idx[0] == 0
    assert idx[-1] == 99
    assert list(idx) == sorted(idx)
    # deterministic: same inputs -> same indices (render-twice stable)
    assert qc.sample_frame_indices(100, max_samples=5) == idx


def test_enabled_with_abstain_detector_runs_but_does_not_flag():
    v = qc.run_frame_qc(frame_count=30, env=ON)
    assert v.ran is True
    assert v.flagged is False
    assert v.max_score == 0.0
    assert v.frames_sampled == min(30, qc.DEFAULT_MAX_SAMPLES)


def test_enabled_flag_maps_to_a_warn_only_decision():
    def hot(frame):
        return 0.99

    v = qc.run_frame_qc(frame_count=30, detector=hot, env=ON)
    assert v.ran is True and v.flagged is True
    decision = qc.qc_decision(v)
    assert decision is not None
    assert decision.block_class is rt.BlockClass.WARN
    assert decision.keep_output is True
    assert decision.aborts_episode is False
    assert decision.discards_output is False
    assert decision.touches_audio is False


def test_clear_and_disabled_verdicts_have_no_decision():
    clear = qc.run_frame_qc(frame_count=10, env=ON)        # abstain -> clear
    assert qc.qc_decision(clear) is None
    disabled = qc.run_frame_qc(frame_count=10, env={})
    assert qc.qc_decision(disabled) is None


def test_raising_detector_is_caught_as_abstain():
    def boom(frame):
        raise RuntimeError("detector blew up")

    v = qc.run_frame_qc(frame_count=8, detector=boom, env=ON)
    assert v.ran is True
    assert v.flagged is False                  # never blocks on a broken detector
    assert v.detector_errors == v.frames_sampled
    assert "detector error" in v.reason


def test_build_qc_warning_is_warn_only():
    v = qc.run_frame_qc(frame_count=8, detector=lambda f: 0.99, env=ON)
    rec = qc.build_qc_warning(v, shot_id="shot_0001", beat_id="b001")
    assert rec["kind"] == "nsfw_frame_qc"
    assert rec["block_class"] == rt.BlockClass.WARN.value
    assert rec["action"] == "warn_only"
    assert rec["flagged"] is True


def test_format_qc_warn_log_is_loud_but_nonblocking():
    v = qc.run_frame_qc(frame_count=8, detector=lambda f: 0.99, env=ON)
    line = qc.format_qc_warn_log(v, "shot_0001")
    assert "WARN" in line
    assert "frozen audio untouched" in line
