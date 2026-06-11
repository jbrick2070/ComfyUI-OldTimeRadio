"""A-S7.5 CPU tests -- the full-episode soak fixture + harness (CPU portion).

The soak drives a synthetic 40-beat, all-roles / all-families episode through the
shipped A-S7 decision machinery TWICE back-to-back, forcing a mid-episode OOM on
the character_3d group. These tests pin: the fixture shape (all roles + families,
one character_3d group), that every beat produces a clip, that the character_3d
group converges triposg_talk -> humo -> humo_1.7B -> latentsync ->
still_kenburns (W7-pre: triposg_talk is the v1 3D lane) to the radio floor with
LOUD restamps at the SAME video_revision, that the frozen
audio section is byte-identical after the run, that two runs are deterministic
with no carryover, and (negative control) that a floor that cannot render fails
LOUDLY rather than passing silently. The live 5080 render is the operator gate.
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
# fixture shape: all roles + families, one character_3d group
# --------------------------------------------------------------------------- #
def test_fixture_is_40_beats_all_roles_all_families():
    section, meta = SOAK.build_soak_fixture(n_beats=40, oom_index=20)
    shots = section["shots"]
    assert len(shots) == 40 and section["video_revision"] == 1
    roles = {s["role"] for s in shots}
    assert roles == {"announcer_visual", "music_visual", "character_video",
                     "scene_broll", "background_abstract"}
    families = {s["family"] for s in shots}
    for fam in ("audio_driven_face", "lipsync_overlay", "image_to_video",
                "text_to_video", "static_image_gen", "static_motion",
                "abstract", "character_3d"):
        assert fam in families
    char3d = [s for s in shots if s["family"] == "character_3d"]
    assert len(char3d) == 1 and char3d[0]["shot_id"] == meta["oom_shot_id"]
    assert char3d[0]["engine_id"] == "triposg_talk"


def test_fixture_rejects_out_of_range_oom_index():
    with pytest.raises(ValueError):
        SOAK.build_soak_fixture(n_beats=10, oom_index=10)


# --------------------------------------------------------------------------- #
# the soak: convergence + LOUD restamp + audio untouched
# --------------------------------------------------------------------------- #
def test_two_episode_soak_passes_all_invariants():
    result = SOAK.run_two_episode_soak(n_beats=40, oom_index=20)
    checks = SOAK.assert_soak_ok(result)             # raises SoakError on failure
    assert any("converged" in c for c in checks)
    assert any("determinism" in c for c in checks)


def test_character_3d_oom_converges_to_floor_with_four_restamps():
    result = SOAK.run_two_episode_soak(n_beats=40, oom_index=20)
    sec = result["e1"]["ledger"]["video"]
    oom = {s["shot_id"]: s for s in sec["shots"]}[result["meta"]["oom_shot_id"]]
    assert oom["engine_id"] == "still_kenburns"      # converged to the radio floor
    assert oom["family"] == "static_motion"
    assert oom["degradation_trail"] == SOAK.EXPECTED_OOM_TRAIL
    decisions = sec["runtime_fallback_decisions"]
    # 4 hops now: hunyuan3d_talk -> humo -> humo_1.7B -> latentsync -> still_kenburns
    assert len(decisions) == 4
    assert all(d["failure_kind"] == "oom" and d["block_class"] == "hard"
               and d["video_revision"] == 1 for d in decisions)


def test_every_beat_renders_including_around_the_oom():
    result = SOAK.run_two_episode_soak(n_beats=40, oom_index=20)
    clips = result["e1"]["clips"]
    assert len(clips) == 40 and all(clips.values())
    assert clips["shot_0000"]["ok"] and clips["shot_0039"]["ok"]   # mid-episode OOM
    assert clips["shot_0020"]["engine_id"] == "still_kenburns"     # the OOM beat


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
    d1 = result["e1"]["ledger"]["video"]["runtime_fallback_decisions"]
    d2 = result["e2"]["ledger"]["video"]["runtime_fallback_decisions"]
    assert d1 == d2
    # the shared input fixture is never mutated by a run (no carryover).
    in_shots = {s["shot_id"]: s
                for s in result["input_ledger"]["video"]["shots"]}
    oom_in = in_shots[result["meta"]["oom_shot_id"]]
    assert oom_in["engine_id"] == "triposg_talk" and oom_in["degradation_trail"] == []


# --------------------------------------------------------------------------- #
# negative control: a floor that cannot render fails LOUDLY (no silent pass)
# --------------------------------------------------------------------------- #
def test_floor_failure_raises_soakerror():
    section, meta = SOAK.build_soak_fixture(n_beats=8, oom_index=4)
    ledger = SOAK.build_full_ledger(section)
    fb = SOAK.make_fallback_of()
    bad = SOAK.SoakRenderer(meta["oom_shot_id"],
                            SOAK.OOM_ENGINES | {"still_kenburns"})
    with pytest.raises(SOAK.SoakError):
        SOAK.run_episode_soak(ledger, fallback_of=fb, renderer=bad)


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
