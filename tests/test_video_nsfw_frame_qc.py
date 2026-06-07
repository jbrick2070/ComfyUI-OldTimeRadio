"""A-S7 CPU tests -- the offline NSFW frame-QC sampler (DEFAULT-OFF, warn-only).

SFW is an OTR invariant, so the sampler is a belt-and-suspenders safety check
that can only ever WARN: it is OFF unless OTR_NSFW_FRAME_QC=1, it routes a flag
through the shared retry taxonomy to a WARN decision (keeps output, never
aborts / discards / touches audio), and a detector exception never breaks the
render. These tests pin the env gate, the deterministic sampling stride, the
threshold compare, the warn-only routing, robustness to a raising detector, and
determinism. The live local detector + the real GPU frames are the operator
smoke, NOT covered here.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

from nodes._otr_shared import nsfw_frame_qc as qc
from nodes._otr_shared import retry_taxonomy as rt

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# DEFAULT-OFF env gate
# --------------------------------------------------------------------------- #
def test_qc_is_default_off(monkeypatch):
    monkeypatch.delenv(qc.QC_ENABLE_FLAG, raising=False)
    assert qc.qc_enabled() is False
    v = qc.run_frame_qc(frame_count=100,
                        detector=lambda f: 1.0)      # would flag if it ran
    assert v.ran is False and v.flagged is False
    assert v.frames_sampled == 0 and v.max_score == 0.0
    assert qc.qc_decision(v) is None                 # nothing to act on


def test_qc_enabled_reads_flag(monkeypatch):
    monkeypatch.setenv(qc.QC_ENABLE_FLAG, "1")
    assert qc.qc_enabled() is True
    monkeypatch.setenv(qc.QC_ENABLE_FLAG, "0")
    assert qc.qc_enabled() is False


# --------------------------------------------------------------------------- #
# deterministic sampling stride
# --------------------------------------------------------------------------- #
def test_sample_frame_indices_stride():
    assert qc.sample_frame_indices(0) == ()
    assert qc.sample_frame_indices(3, max_samples=8) == (0, 1, 2)   # short -> all
    idx = qc.sample_frame_indices(100, max_samples=5)
    assert idx == (0, 25, 50, 74, 99)                # evenly spaced, first..last
    assert len(idx) == 5 and idx[0] == 0 and idx[-1] == 99
    assert qc.sample_frame_indices(100, max_samples=5) == idx        # deterministic


# --------------------------------------------------------------------------- #
# threshold compare + warn-only routing
# --------------------------------------------------------------------------- #
def test_enabled_abstain_detector_never_flags(monkeypatch):
    monkeypatch.setenv(qc.QC_ENABLE_FLAG, "1")
    v = qc.run_frame_qc(frame_count=50)              # default = abstain_detector
    assert v.ran is True and v.flagged is False
    assert v.max_score == 0.0 and v.frames_sampled == qc.DEFAULT_MAX_SAMPLES
    assert qc.qc_decision(v) is None


def test_high_score_flags_and_routes_to_warn(monkeypatch):
    monkeypatch.setenv(qc.QC_ENABLE_FLAG, "1")
    scores = {i: 0.1 for i in range(60)}
    scores[30] = 0.97                                # one hot frame
    v = qc.run_frame_qc(frame_count=60, get_frame=None,
                        detector=lambda i: scores[i], max_samples=60)
    assert v.ran is True and v.flagged is True and v.max_score == pytest.approx(0.97)
    decision = qc.qc_decision(v)
    assert decision is not None
    assert decision.block_class is rt.BlockClass.WARN
    assert decision.warn_only is True and decision.keep_output is True
    # warn-only invariant: a flag can never abort / discard / touch audio.
    rt.assert_decision_invariants(decision)
    assert decision.aborts_episode is False and decision.touches_audio is False


def test_threshold_boundary(monkeypatch):
    monkeypatch.setenv(qc.QC_ENABLE_FLAG, "1")
    at = qc.run_frame_qc(frame_count=1, detector=lambda f: 0.85,
                         threshold=0.85)
    assert at.flagged is True                        # >= threshold flags
    below = qc.run_frame_qc(frame_count=1, detector=lambda f: 0.84,
                            threshold=0.85)
    assert below.flagged is False


# --------------------------------------------------------------------------- #
# robustness: a raising detector never breaks a render
# --------------------------------------------------------------------------- #
def test_detector_exception_abstains_never_raises(monkeypatch):
    monkeypatch.setenv(qc.QC_ENABLE_FLAG, "1")

    def boom(frame):
        raise RuntimeError("local detector crashed")

    v = qc.run_frame_qc(frame_count=20, detector=boom, max_samples=4)
    assert v.ran is True and v.flagged is False       # abstained, did not crash
    assert v.detector_errors == 4 and v.max_score == 0.0
    assert qc.qc_decision(v) is None


def test_one_good_detector_among_errors_still_scored(monkeypatch):
    monkeypatch.setenv(qc.QC_ENABLE_FLAG, "1")

    def flaky(i):
        if i == 50:
            return 0.95
        raise ValueError("nope")

    v = qc.run_frame_qc(frame_count=100, detector=flaky, max_samples=5)
    # indices for 100/5 = (0,25,50,74,99); only idx 50 scores, the rest abstain.
    assert v.flagged is True and v.max_score == pytest.approx(0.95)
    assert v.detector_errors == 4


# --------------------------------------------------------------------------- #
# warn-only audit record + LOUD-but-non-blocking log
# --------------------------------------------------------------------------- #
def test_build_qc_warning_is_warn_only():
    v = qc.QCVerdict(ran=True, flagged=True, max_score=0.9, frames_sampled=8,
                     threshold=0.85)
    rec = qc.build_qc_warning(v, shot_id="shot_07", beat_id="b07")
    assert rec["kind"] == "nsfw_frame_qc" and rec["flagged"] is True
    assert rec["block_class"] == "warn" and rec["action"] == "warn_only"
    assert rec["shot_id"] == "shot_07" and rec["beat_id"] == "b07"


def test_format_qc_warn_log_is_non_blocking():
    v = qc.QCVerdict(ran=True, flagged=True, max_score=0.91, frames_sampled=8,
                     threshold=0.85)
    line = qc.format_qc_warn_log(v, "shot_07")
    assert "NSFW frame-QC WARN" in line and "warn-only" in line
    assert "frozen audio untouched" in line and "shot_07" in line


# --------------------------------------------------------------------------- #
# determinism (render-twice)
# --------------------------------------------------------------------------- #
def test_run_frame_qc_is_deterministic(monkeypatch):
    monkeypatch.setenv(qc.QC_ENABLE_FLAG, "1")
    scores = {i: (i % 7) / 10.0 for i in range(40)}
    a = qc.run_frame_qc(frame_count=40, detector=lambda i: scores[i])
    b = qc.run_frame_qc(frame_count=40, detector=lambda i: scores[i])
    assert a == b


# --------------------------------------------------------------------------- #
# cold-import (V-12) + ASCII source
# --------------------------------------------------------------------------- #
def test_cold_import_nsfw_qc_no_heavy_libs():
    code = (
        "import sys;"
        "import nodes._otr_shared.nsfw_frame_qc as q;"
        "q.run_frame_qc(frame_count=10);"
        "heavy=[m for m in ('torch','transformers','diffusers','pydantic') "
        "if m in sys.modules];"
        "print('HEAVY', heavy);"
        "sys.exit(1 if heavy else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=str(REPO_ROOT),
                       capture_output=True, text=True)
    assert r.returncode == 0, f"heavy libs at import:\n{r.stdout}\n{r.stderr}"


def test_nsfw_qc_source_is_ascii_no_em_dash():
    src = (REPO_ROOT / "nodes" / "_otr_shared" / "nsfw_frame_qc.py").read_text(
        encoding="utf-8")
    assert chr(0x2014) not in src                 # em-dash (U+2014) forbidden
    src.encode("ascii")                           # ASCII-only source


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
