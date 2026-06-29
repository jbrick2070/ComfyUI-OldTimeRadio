"""A-S7 CPU tests -- the render-time retry taxonomy (block_class split).

The taxonomy classifies every render-time failure into a HARD (renderability /
safety-integrity -> walk the fallback chain, LOUD restamp) or WARN (subjective
quality / coherence / NSFW + A/V-sync -> warn only, keep output, never discard /
abort / touch audio) block class, with one deterministic action per kind. These
tests pin the per-kind policy, the cross-cutting invariants (no decision ever
discards a beat, touches the frozen audio, or aborts the episode), the durable
ledger restamp + the LOUD swap-log, and that the restamped shapes still validate
against the (extra='forbid') video schemas. The live GPU re-render is the
A-S7.5 soak (operator), NOT covered here.
"""
from __future__ import annotations

import dataclasses
import pathlib
import subprocess
import sys

import pytest

from nodes._otr_shared import retry_taxonomy as rt
from nodes._otr_shared.fallback import resolve_fallback_chain
from nodes._otr_video_engines import registry as vreg
from nodes._otr_video_engines import schemas as sc
# Importing the adapters registers humo / humo_1.7B / still_motion so the
# real declared fallback chain can be walked below.
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


def test_hard_renderability_kinds_walk_fallback_zero_retries():
    for k in (rt.FailureKind.DEPENDENCY_MISSING, rt.FailureKind.ASSET_MISSING,
              rt.FailureKind.LICENSE_BLOCKED, rt.FailureKind.INVALID_DAG,
              rt.FailureKind.OOM, rt.FailureKind.TIMEOUT):
        d = rt.classify(k)
        assert d.is_hard and d.escalate_to_fallback is True
        assert d.same_seed_retries == 0 and d.reseed_retries == 0
        assert d.keep_output is False and d.warn_only is False
        assert d.max_attempts == 1               # fail fast, then walk


def test_crash_before_load_retries_once_then_walks():
    d = rt.classify(rt.FailureKind.CRASH_BEFORE_LOAD)
    assert d.is_hard and d.same_seed_retries == 1 and d.reseed_retries == 0
    assert d.escalate_to_fallback is True and d.max_attempts == 2


def test_corrupt_output_same_seed_then_reseed_then_walk():
    d = rt.classify(rt.FailureKind.CORRUPT_OUTPUT)
    assert d.is_hard and d.same_seed_retries == 1 and d.reseed_retries == 1
    assert d.escalate_to_fallback is True and d.max_attempts == 3


def test_transient_io_bounded_retry():
    d = rt.classify(rt.FailureKind.TRANSIENT_IO)
    assert d.is_hard and d.same_seed_retries == rt.DEFAULT_TRANSIENT_IO_RETRIES
    assert d.escalate_to_fallback is True


def test_warn_subjective_kinds_warn_only_keep_output():
    for k in (rt.FailureKind.QUALITY, rt.FailureKind.COHERENCE,
              rt.FailureKind.NSFW):
        d = rt.classify(k)
        assert d.is_warn and d.warn_only is True and d.keep_output is True
        assert d.escalate_to_fallback is False


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
# durable ledger restamp + LOUD swap-log
# --------------------------------------------------------------------------- #
def test_build_fallback_decision_pins_same_revision_and_class():
    rec = rt.build_fallback_decision(
        shot_id="shot_05", beat_id="b05", from_engine="humo",
        to_engine="humo_1.7B", kind=rt.FailureKind.OOM,
        video_revision=3, attempt=1, detail="cuda oom at flux->humo boundary")
    assert rec["video_revision"] == 3            # unchanged (within-revision)
    assert rec["from_engine"] == "humo" and rec["to_engine"] == "humo_1.7B"
    assert rec["failure_kind"] == "oom" and rec["block_class"] == "hard"
    assert rec["shot_id"] == "shot_05" and rec["beat_id"] == "b05"


def test_restamp_shot_row_is_pure_and_chains_trail():
    row = {"shot_id": "shot_05", "engine_id": "humo",
           "family": "audio_driven_face", "degradation_trail": []}
    out = rt.restamp_shot_row(row, to_engine="humo_1.7B",
                              to_family="audio_driven_face", from_engine="humo",
                              kind=rt.FailureKind.OOM)
    assert out is not row and row["engine_id"] == "humo"      # input not mutated
    assert out["engine_id"] == "humo_1.7B"
    assert out["family"] == "audio_driven_face"
    assert out["degradation_trail"] == ["humo->humo_1.7B (oom)"]
    out2 = rt.restamp_shot_row(out, to_engine="still_motion",
                               to_family="static_motion",
                               from_engine="humo_1.7B",
                               kind=rt.FailureKind.OOM)
    assert out2["degradation_trail"] == [
        "humo->humo_1.7B (oom)", "humo_1.7B->still_motion (oom)"]
    assert out2["engine_id"] == "still_motion"


def test_append_runtime_fallback_decision_same_revision_fail_closed():
    section = {"video_revision": 2, "shots": []}
    rec = rt.build_fallback_decision(
        shot_id="s1", beat_id="b1", from_engine="humo", to_engine="humo_1.7B",
        kind=rt.FailureKind.OOM, video_revision=2)
    out = rt.append_runtime_fallback_decision(section, rec)
    assert out is not section
    assert out["video_revision"] == 2            # never bumped on a restamp
    assert out["runtime_fallback_decisions"] == [rec]
    bad = dict(rec, video_revision=3)            # no silent re-numbering
    with pytest.raises(ValueError):
        rt.append_runtime_fallback_decision(section, bad)


def test_format_swap_log_is_loud_and_reaffirms_audio_safety():
    rec = rt.build_fallback_decision(
        shot_id="s1", beat_id="b1", from_engine="humo", to_engine="humo_1.7B",
        kind=rt.FailureKind.OOM, video_revision=1)
    line = rt.format_swap_log(rec)
    assert "LOUD FALLBACK" in line
    assert "humo" in line and "humo_1.7B" in line
    assert "oom" in line and "frozen audio untouched" in line


# --------------------------------------------------------------------------- #
# restamped shapes still validate against the (extra='forbid') schemas
# --------------------------------------------------------------------------- #
def test_restamp_roundtrips_through_video_schemas():
    row = sc.ShotRow(shot_id="shot_01", engine_id="humo",
                     family="audio_driven_face").model_dump()
    restamped = rt.restamp_shot_row(row, to_engine="humo_1.7B",
                                    to_family="audio_driven_face",
                                    from_engine="humo", kind=rt.FailureKind.OOM)
    sc.ShotRow(**restamped)                       # still a valid ShotRow
    rec = rt.build_fallback_decision(
        shot_id="shot_01", beat_id="b01", from_engine="humo",
        to_engine="humo_1.7B", kind=rt.FailureKind.OOM, video_revision=1)
    section = sc.VideoLedgerSection(
        video_revision=1, shots=[sc.ShotRow(**restamped)]).model_dump()
    section = rt.append_runtime_fallback_decision(section, rec)
    validated = sc.VideoLedgerSection(**section)  # extra='forbid' accepts it
    assert len(validated.runtime_fallback_decisions) == 1
    assert validated.video_revision == 1


def test_real_humo_chain_resolves_and_restamps_to_floor():
    def fallback_of(name):
        return getattr(vreg.get_engine(name), "fallback_engine", None)
    chain = resolve_fallback_chain("humo", fallback_of)
    assert chain == ["humo", "humo_1.7B", "still_motion"]
    fam = {"humo": "audio_driven_face", "humo_1.7B": "audio_driven_face",
           "still_motion": "static_motion"}
    row = {"shot_id": "shot_09", "engine_id": "humo",
           "family": fam["humo"], "degradation_trail": []}
    for frm, to in zip(chain, chain[1:]):
        row = rt.restamp_shot_row(row, to_engine=to, to_family=fam[to],
                                  from_engine=frm, kind=rt.FailureKind.OOM)
    assert row["engine_id"] == "still_motion"
    assert fallback_of("still_motion") is None  # radio floor terminates
    assert row["degradation_trail"] == [
        "humo->humo_1.7B (oom)", "humo_1.7B->still_motion (oom)"]


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
