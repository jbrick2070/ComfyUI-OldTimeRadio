"""A-S7.5 CPU tests -- the full-episode soak fixture + harness (CPU portion).

NO FALLBACKS (Sprint A rip, 2026-07-02): the decision machinery is gone. The
soak drives a synthetic 40-beat, all-roles / all-families CLEAN episode through
an injected fake renderer TWICE back-to-back, then proves the LOUD-failure
contract on a separate forced-OOM leg. These tests pin: the fixture shape (all
roles + families; the char3d stub only on the contract leg), that every clean
beat produces a clip with NO degradation trail, that the frozen audio section
is byte-identical after the run, that two runs are deterministic with no
carryover, and that the forced OOM RAISES (never a swap, never a restamp).
The live 5080 render is the operator gate.
"""
from __future__ import annotations

import importlib.util
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SOAK_SRC = REPO_ROOT / "scripts" / "otr_video_soak.py"


def _load_soak():
    spec = importlib.util.spec_from_file_location("otr_video_soak", SOAK_SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SOAK = _load_soak()


# --------------------------------------------------------------------------- #
# fixture shape: all roles + families; char3d stub only when oom_index is set
# --------------------------------------------------------------------------- #
def test_clean_fixture_is_40_beats_all_roles_no_char3d():
    section, meta = SOAK.build_soak_fixture(n_beats=40, oom_index=None)
    shots = section["shots"]
    assert len(shots) == 40 and section["video_revision"] == 1
    roles = {s["role"] for s in shots}
    # rip-sfx-broll (2026-07-01): the rotation covers the 3 surviving roles.
    assert roles == {"announcer_visual", "music_visual", "character_video"}
    families = {s["family"] for s in shots}
    for fam in ("audio_driven_face", "image_to_video",
                "text_to_video", "static_image_gen", "static_motion"):
        assert fam in families
    # NO forced-OOM stub on the clean fixture; no trail stamped anywhere.
    # (Filter by ENGINE ID, not family: the stub was rebased onto the LIVE
    # audio_driven_face family on 2026-08-23 -- humo's -- so a family filter
    # would also match every legitimate humo profile row.)
    assert not any(s["engine_id"] == "soak_oom_heavy" for s in shots)
    assert all(s["degradation_trail"] == [] for s in shots)
    assert meta["oom_shot_id"] is None


def test_oom_fixture_injects_the_heavy_stub():
    section, meta = SOAK.build_soak_fixture(n_beats=40, oom_index=20)
    stub = [s for s in section["shots"] if s["engine_id"] == "soak_oom_heavy"]
    assert len(stub) == 1 and stub[0]["shot_id"] == meta["oom_shot_id"]
    # The stub rides a LIVE heavy family so the OOM contract outlives the
    # character_3d retirement (lean-mean order 4).
    assert stub[0]["family"] == "audio_driven_face"


def test_fixture_rejects_out_of_range_oom_index():
    with pytest.raises(ValueError):
        SOAK.build_soak_fixture(n_beats=10, oom_index=10)


# --------------------------------------------------------------------------- #
# the soak: clean completion + LOUD-failure contract + audio untouched
# --------------------------------------------------------------------------- #
def test_two_episode_soak_passes_all_invariants():
    result = SOAK.run_two_episode_soak(n_beats=40, oom_index=20)
    checks = SOAK.assert_soak_ok(result)             # raises SoakError on failure
    assert any("determinism" in c for c in checks)
    assert any("LOUD-failure contract" in c for c in checks)


def test_forced_oom_raises_and_never_restamps():
    result = SOAK.run_two_episode_soak(n_beats=40, oom_index=20)
    oc = result["oom_contract"]
    assert oc["raised"] is True and oc["error_type"] == "OomSignal"
    # the forced-OOM fixture is untouched: no swap, no trail, no restamp.
    oom_in = {s["shot_id"]: s for s in
              result["oom_input_ledger"]["video"]["shots"]}
    oom_shot = oom_in[result["oom_meta"]["oom_shot_id"]]
    assert oom_shot["engine_id"] == "soak_oom_heavy"
    assert oom_shot["degradation_trail"] == []


def test_every_clean_beat_renders_with_no_trail():
    result = SOAK.run_two_episode_soak(n_beats=40, oom_index=20)
    clips = result["e1"]["clips"]
    assert len(clips) == 40 and all(clips.values())
    assert clips["shot_0000"]["ok"] and clips["shot_0039"]["ok"]
    sec = result["e1"]["ledger"]["video"]
    assert all(s["degradation_trail"] == [] for s in sec["shots"])
    assert "runtime_fallback_decisions" not in sec


def test_frozen_audio_is_byte_identical_after_soak():
    result = SOAK.run_two_episode_soak(n_beats=12, oom_index=6)
    for ep in ("e1", "e2"):
        audio = result[ep]["ledger"]["audio"]
        assert audio["master_audio_sha256"] == SOAK.FROZEN_AUDIO_SHA
        assert audio["ledger_frozen"] is True


# --------------------------------------------------------------------------- #
# determinism + no carryover
# --------------------------------------------------------------------------- #
def test_two_runs_are_deterministic_and_leave_input_unmutated():
    result = SOAK.run_two_episode_soak(n_beats=24, oom_index=12)
    assert result["render_calls_1"] == result["render_calls_2"]
    # the shared input fixture is never mutated by a run (no carryover).
    for s in result["input_ledger"]["video"]["shots"]:
        assert s["degradation_trail"] == []


# --------------------------------------------------------------------------- #
# negative control: any mid-episode failure propagates (no silent pass)
# --------------------------------------------------------------------------- #
def test_any_engine_failure_propagates_loud():
    section, meta = SOAK.build_soak_fixture(n_beats=8, oom_index=None)
    ledger = SOAK.build_full_ledger(section)
    # Force a failure on an ordinary clean beat: NO FALLBACKS means it raises.
    bad = SOAK.SoakRenderer("shot_0000", {"humo"})
    with pytest.raises(SOAK.OomSignal):
        SOAK.run_episode_soak(ledger, renderer=bad)


# --------------------------------------------------------------------------- #
# main: CPU mode passes; GPU mode refuses to certify (operator gate)
# --------------------------------------------------------------------------- #
def test_main_cpu_passes_and_gpu_is_operator_gate():
    assert SOAK.main(["--mode", "cpu", "--beats", "16", "--oom-index", "8"]) == 0
    assert SOAK.main(["--mode", "gpu"]) == 2          # never a CPU-certifiable pass


def test_soak_source_is_ascii_no_em_dash():
    src = SOAK_SRC.read_text(encoding="utf-8")
    assert chr(0x2014) not in src                     # em-dash (U+2014) forbidden
    src.encode("ascii")                               # ASCII-only source


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
