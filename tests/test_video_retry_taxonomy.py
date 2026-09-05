"""A-S7 CPU tests -- the render-time retry taxonomy (block_class split).

The taxonomy classifies every render-time failure into a HARD (renderability /
safety-integrity -> LOUD STOP; NO FALLBACKS, 2026-07-02) or WARN (subjective
quality / coherence / NSFW + A/V-sync -> warn only, keep output, never discard /
abort / touch audio) block class, with one deterministic action per kind. These
tests pin the per-kind policy and the cross-cutting invariants (no decision
ever discards a beat, touches the frozen audio, or aborts the episode). The
fallback-action API (build_fallback_decision / restamp_shot_row /
append_runtime_fallback_decision / format_swap_log) was DELETED in the Sprint A
rip -- these tests pin its absence. The live GPU re-render is the A-S7.5 soak
(operator), NOT covered here.
"""
from __future__ import annotations

import dataclasses
import pathlib
import subprocess
import sys

import pytest

from nodes._otr_shared import retry_taxonomy as rt
from nodes._otr_video_engines import registry as vreg
from nodes._otr_video_engines import schemas as sc
# Importing the adapters registers humo / humo_1.7B / still_motion for the
# no-fallback declaration checks below.
from nodes._otr_video_engines import eng_humo            # noqa: F401
from nodes._otr_video_engines import cheap_families      # noqa: F401

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# block-class mapping
# --------------------------------------------------------------------------- #
def test_hard_and_warn_kinds_partition_all_failure_kinds():
    assert rt.HARD_KINDS.isdisjoint(rt.WARN_KINDS)
    assert rt.HARD_KINDS | rt.WARN_KINDS == set(rt.FailureKind)
    for k in rt.HARD_KINDS:
        assert rt.block_class_of(k) is rt.BlockClass.HARD
    for k in rt.WARN_KINDS:
        assert rt.block_class_of(k) is rt.BlockClass.WARN


def test_block_class_of_is_fail_closed_on_unknown():
    with pytest.raises(ValueError):
        rt.block_class_of("not_a_kind")


# --------------------------------------------------------------------------- #
# per-kind deterministic policy
# --------------------------------------------------------------------------- #
def test_classify_is_deterministic_and_validated():
    for k in rt.FailureKind:
        d1, d2 = rt.classify(k), rt.classify(k)
        assert d1 == d2                          # deterministic
        assert d1.kind is k
        assert d1.block_class is rt.block_class_of(k)


def test_hard_renderability_kinds_fail_fast_zero_retries():
    for k in (rt.FailureKind.DEPENDENCY_MISSING, rt.FailureKind.ASSET_MISSING,
              rt.FailureKind.LICENSE_BLOCKED, rt.FailureKind.INVALID_DAG,
              rt.FailureKind.OOM, rt.FailureKind.TIMEOUT):
        d = rt.classify(k)
        assert d.is_hard
        assert d.same_seed_retries == 0 and d.reseed_retries == 0
        assert d.keep_output is False and d.warn_only is False
        assert d.max_attempts == 1               # fail fast, then STOP LOUD


def test_crash_before_load_retries_once_then_stops():
    d = rt.classify(rt.FailureKind.CRASH_BEFORE_LOAD)
    assert d.is_hard and d.same_seed_retries == 1 and d.reseed_retries == 0
    assert d.max_attempts == 2


def test_corrupt_output_same_seed_then_reseed_then_stops():
    d = rt.classify(rt.FailureKind.CORRUPT_OUTPUT)
    assert d.is_hard and d.same_seed_retries == 1 and d.reseed_retries == 1
    assert d.max_attempts == 3


def test_transient_io_bounded_retry():
    d = rt.classify(rt.FailureKind.TRANSIENT_IO)
    assert d.is_hard and d.same_seed_retries == rt.DEFAULT_TRANSIENT_IO_RETRIES


def test_warn_subjective_kinds_warn_only_keep_output():
    for k in (rt.FailureKind.QUALITY, rt.FailureKind.COHERENCE,
              rt.FailureKind.NSFW):
        d = rt.classify(k)
        assert d.is_warn and d.warn_only is True and d.keep_output is True


def test_av_sync_retimes_video_keeps_output_never_audio():
    d = rt.classify(rt.FailureKind.AV_SYNC)
    assert d.is_warn and d.retime is True and d.keep_output is True
    assert d.warn_only is False
    assert d.touches_audio is False              # retimes VIDEO frames only


# --------------------------------------------------------------------------- #
# cross-cutting invariants (never discard / abort / touch frozen audio)
# --------------------------------------------------------------------------- #
def test_no_decision_discards_aborts_or_touches_audio():
    for k in rt.FailureKind:
        d = rt.classify(k)
        assert d.discards_output is False
        assert d.touches_audio is False
        assert d.aborts_episode is False
    for k in rt.WARN_KINDS:                       # WARN never drops a beat
        assert rt.classify(k).keep_output is True


def test_classify_fail_closed_on_unknown_kind():
    with pytest.raises(ValueError):
        rt.classify("totally_unknown")


def test_assert_decision_invariants_rejects_a_bad_policy():
    good = rt.classify(rt.FailureKind.OOM)
    with pytest.raises(ValueError):              # would drop a beat
        rt.assert_decision_invariants(
            dataclasses.replace(good, discards_output=True))
    with pytest.raises(ValueError):              # would touch frozen audio
        rt.assert_decision_invariants(
            dataclasses.replace(good, touches_audio=True))
    with pytest.raises(ValueError):              # would abort the episode
        rt.assert_decision_invariants(
            dataclasses.replace(good, aborts_episode=True))
    warn = rt.classify(rt.FailureKind.QUALITY)
    with pytest.raises(ValueError):              # WARN must keep its output
        rt.assert_decision_invariants(
            dataclasses.replace(warn, keep_output=False))


# --------------------------------------------------------------------------- #
# NO FALLBACKS (Sprint A rip, 2026-07-02): the action API is GONE; the ledger
# schema slot survives (stamped never, A5 -- no schema churn)
# --------------------------------------------------------------------------- #
def test_fallback_action_api_stays_deleted():
    for name in ("build_fallback_decision", "restamp_shot_row",
                 "append_runtime_fallback_decision", "format_swap_log"):
        assert not hasattr(rt, name), (
            "%s must stay deleted (NO FALLBACKS 2026-07-02)" % name)
        assert name not in rt.__all__
    # the escalate flag died with the chain machinery.
    assert not hasattr(rt.classify(rt.FailureKind.OOM), "escalate_to_fallback")


def test_no_registered_engine_declares_a_fallback():
    for name in ("humo", "humo_1.7B", "still_motion"):
        assert getattr(vreg.get_engine(name), "fallback_engine", None) is None


# --------------------------------------------------------------------------- #
# cold-import (V-12) + ASCII source
# --------------------------------------------------------------------------- #
def test_cold_import_retry_taxonomy_no_heavy_libs():
    code = (
        "import sys;"
        "import nodes._otr_shared.retry_taxonomy as rt;"
        "rt.classify(rt.FailureKind.OOM);"
        "heavy=[m for m in ('torch','transformers','diffusers','pydantic') "
        "if m in sys.modules];"
        "print('HEAVY', heavy);"
        "sys.exit(1 if heavy else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=str(REPO_ROOT),
                       capture_output=True, text=True)
    assert r.returncode == 0, f"heavy libs at import:\n{r.stdout}\n{r.stderr}"


def test_retry_taxonomy_source_is_ascii_no_em_dash():
    src_path = REPO_ROOT / "nodes" / "_otr_shared" / "retry_taxonomy.py"
    src = src_path.read_text(encoding="utf-8")
    assert chr(0x2014) not in src             # em-dash (U+2014) forbidden
    src.encode("ascii")                           # ASCII-only source


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
