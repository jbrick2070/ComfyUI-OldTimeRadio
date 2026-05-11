"""
test_build_test_ledger_from_director.py
=========================================

Unit tests for ``scripts/build_test_ledger_from_director.py``.

Validates:
  * extract_director_json_from_workflow finds OTR_VideoPlan node
  * director_json_to_test_ledger expands shots correctly for the
    LEMMY/Saturn TEST director (3 scenes, 2 shots/scene, 3 cast)
  * speaker rotation cycles across cast
  * round-trip JSON write/read produces a valid ledger consumable
    by render_humo_batch.filter_lines (scope='all')
  * orchestrator's filter_lines accepts the synthesized lines[]

Pure stdlib + pytest. No torch / diffusers / ComfyUI / GPU.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# Make sibling scripts/ importable as a top-level module.
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

LEMMY_SATURN_DIRECTOR = {
    "episode_title": "The Saturn Broadcast",
    "voice_assignments": {
        "LEMMY": {
            "voice_preset": "v2/en_speaker_8",
            "notes": "Male, 50s, gravelly",
        },
        "KENJI CROSS": {
            "voice_preset": "v2/en_speaker_3",
            "notes": "Male, 30s, urgent",
        },
        "ANNOUNCER": {
            "voice_preset": "v2/en_speaker_4",
            "notes": "Female, 40s, broadcast",
        },
    },
    "visual_plan": {
        "characters": {
            "LEMMY": {"portrait_prompt": "A weathered male spacer"},
            "KENJI CROSS": {"portrait_prompt": "A disciplined male officer"},
            "ANNOUNCER": {"portrait_prompt": "A poised broadcast announcer"},
        },
        "scenes": [
            {
                "scene_id": "scene_1",
                "shot_description": "LEMMY at the comms console",
                "visual_prompt": "Cinematic wide shot of dim cramped starship comms bay",
            },
            {
                "scene_id": "scene_2",
                "shot_description": "LEMMY and KENJI argue",
                "visual_prompt": "Cinematic medium shot of narrow starship corridor",
            },
            {
                "scene_id": "scene_3",
                "shot_description": "LEMMY alone in ready room",
                "visual_prompt": "Cinematic close shot of captain's ready room",
            },
        ],
    },
}


def _write_workflow(tmp_path: Path, director: dict) -> Path:
    """Write a minimal workflow JSON with OTR_VideoPlan node carrying director."""
    director_json_str = json.dumps(director)
    wf = {
        "last_node_id": 1,
        "last_link_id": 0,
        "nodes": [
            {
                "id": 1,
                "type": "OTR_VideoPlan",
                "widgets_values": [
                    director_json_str,
                    "(all)",
                    2,                  # shots_per_scene
                    "mission_control_procedural",
                    "cinematic, 35mm film look",
                    True,               # include_final_end_frame
                ],
            }
        ],
        "links": [],
        "groups": [],
        "config": {},
        "extra": {},
        "version": 0.4,
    }
    p = tmp_path / "test_workflow.json"
    p.write_text(json.dumps(wf, indent=2), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# extract_director_json_from_workflow
# ---------------------------------------------------------------------------


def test_extract_director_json_finds_video_plan_node(tmp_path: Path):
    from build_test_ledger_from_director import extract_director_json_from_workflow

    wf = _write_workflow(tmp_path, LEMMY_SATURN_DIRECTOR)
    s = extract_director_json_from_workflow(wf)
    parsed = json.loads(s)
    assert parsed["episode_title"] == "The Saturn Broadcast"
    assert "LEMMY" in parsed["visual_plan"]["characters"]


def test_extract_director_json_raises_when_missing(tmp_path: Path):
    from build_test_ledger_from_director import extract_director_json_from_workflow

    wf = tmp_path / "empty.json"
    wf.write_text(json.dumps({"nodes": []}), encoding="utf-8")
    with pytest.raises(RuntimeError):
        extract_director_json_from_workflow(wf)


def test_extract_widget_values_position_layout(tmp_path: Path):
    from build_test_ledger_from_director import extract_video_plan_widget_values

    wf = _write_workflow(tmp_path, LEMMY_SATURN_DIRECTOR)
    wv = extract_video_plan_widget_values(wf)
    assert isinstance(wv, list)
    # [0] director_json, [1] focus, [2] shots_per_scene, [3] genre, [4] style, [5] include_final
    assert wv[1] == "(all)"
    assert wv[2] == 2
    assert wv[3] == "mission_control_procedural"
    assert wv[5] is True


# ---------------------------------------------------------------------------
# director_json_to_test_ledger — shape + counts
# ---------------------------------------------------------------------------


def test_lemmy_saturn_ledger_has_three_cast_three_scenes_six_shots():
    """LEMMY/Saturn director has 3 cast, 3 scenes, 2 shots/scene = 6 shots."""
    from build_test_ledger_from_director import director_json_to_test_ledger

    led = director_json_to_test_ledger(
        json.dumps(LEMMY_SATURN_DIRECTOR),
        episode_id="test_humo_lemmy_saturn",
        shots_per_scene=2,
        include_final_end_frame=True,
    )
    assert len(led["cast"]) == 3
    assert {c["name"] for c in led["cast"]} == {"LEMMY", "KENJI CROSS", "ANNOUNCER"}
    assert len(led["scenes"]) == 3
    # 3 scenes × 2 shots = 6 shots. include_final_end_frame adds one FRAME
    # (in tokens) but does NOT add a shot — shots[] stays at 6.
    assert len(led["shots"]) == 6
    # One ledger line per shot.
    assert len(led["lines"]) == 6


def test_speaker_strategy_rotate_cycles_cast():
    from build_test_ledger_from_director import director_json_to_test_ledger

    led = director_json_to_test_ledger(
        json.dumps(LEMMY_SATURN_DIRECTOR),
        episode_id="test",
        shots_per_scene=2,
        speaker_strategy="rotate",
    )
    speakers = [ln["speaker"] for ln in led["lines"]]
    # 6 shots, 3 cast → each cast member gets exactly 2 lines
    assert len(speakers) == 6
    cast_names = {c["name"] for c in led["cast"]}
    for name in cast_names:
        assert speakers.count(name) == 2, (
            f"speaker {name!r} should appear 2x with rotate, got {speakers.count(name)}"
        )


def test_speaker_strategy_first_pins_to_first_cast():
    from build_test_ledger_from_director import director_json_to_test_ledger

    led = director_json_to_test_ledger(
        json.dumps(LEMMY_SATURN_DIRECTOR),
        episode_id="test",
        shots_per_scene=2,
        speaker_strategy="first",
    )
    speakers = [ln["speaker"] for ln in led["lines"]]
    first_name = led["cast"][0]["name"]
    assert all(s == first_name for s in speakers)


def test_clip_length_propagates_to_lines_and_total():
    from build_test_ledger_from_director import director_json_to_test_ledger

    led = director_json_to_test_ledger(
        json.dumps(LEMMY_SATURN_DIRECTOR),
        episode_id="test",
        shots_per_scene=2,
        clip_length_s=3.88,
    )
    for ln in led["lines"]:
        assert ln["start_s"] == 0.0
        assert ln["dur_s"] == pytest.approx(3.88)
    # 6 lines × 3.88 s
    assert led["total_episode_dur_s"] == pytest.approx(6 * 3.88)


def test_cast_carries_voice_preset_and_description():
    from build_test_ledger_from_director import director_json_to_test_ledger

    led = director_json_to_test_ledger(
        json.dumps(LEMMY_SATURN_DIRECTOR),
        episode_id="test",
    )
    by_name = {c["name"]: c for c in led["cast"]}
    assert by_name["LEMMY"]["voice_preset"] == "v2/en_speaker_8"
    # Field renamed 2026-05-10: description -> character_description.
    # Read either key for back-compat with builders that still emit the
    # old name; the assertion only cares the LEMMY description survived.
    cdesc = (
        by_name["LEMMY"].get("character_description")
        or by_name["LEMMY"].get("description")
        or ""
    )
    assert "weathered" in cdesc.lower()


def test_empty_director_yields_empty_ledger():
    from build_test_ledger_from_director import director_json_to_test_ledger

    led = director_json_to_test_ledger(
        json.dumps({}),
        episode_id="empty",
    )
    assert led["cast"] == []
    assert led["scenes"] == []
    assert led["shots"] == []
    assert led["lines"] == []


def test_each_line_has_required_orchestrator_keys():
    """Every key the HuMo batch orchestrator reads must be present per line."""
    from build_test_ledger_from_director import director_json_to_test_ledger

    led = director_json_to_test_ledger(
        json.dumps(LEMMY_SATURN_DIRECTOR),
        episode_id="test",
    )
    required = {"line_id", "speaker", "scene_id", "start_s", "dur_s"}
    for ln in led["lines"]:
        missing = required - set(ln.keys())
        assert not missing, f"line missing keys: {missing}"


def test_beats_present_and_match_shots():
    """Schema l2: beats[] must be populated, one per shot in this
    synthetic test (each shot is single-speaker by construction)."""
    from build_test_ledger_from_director import director_json_to_test_ledger

    led = director_json_to_test_ledger(
        json.dumps(LEMMY_SATURN_DIRECTOR),
        episode_id="test",
        shots_per_scene=2,
    )
    assert "beats" in led
    assert led["total_beats"] == len(led["shots"])
    # Each beat id follows shot_id_b1 convention (one beat per shot)
    for shot, beat in zip(led["shots"], led["beats"]):
        assert beat["shot_id"] == shot["shot_id"]
        assert beat["beat_id"] == f"{shot['shot_id']}_b1"
        assert beat["speaker"]  # non-empty


def test_lines_carry_beat_id_and_shot_start_boundary():
    """Schema l2: every line must reference its beat and carry a
    boundary value; for this synthetic test (one beat per shot) all
    lines are shot_start."""
    from build_test_ledger_from_director import director_json_to_test_ledger

    led = director_json_to_test_ledger(
        json.dumps(LEMMY_SATURN_DIRECTOR),
        episode_id="test",
        shots_per_scene=2,
    )
    for ln in led["lines"]:
        assert "beat_id" in ln and ln["beat_id"]
        assert ln["boundary"] == "shot_start"


def test_schema_version_marks_synthetic_test_ledger():
    from build_test_ledger_from_director import (
        director_json_to_test_ledger,
        TEST_LEDGER_SCHEMA,
    )

    led = director_json_to_test_ledger(
        json.dumps(LEMMY_SATURN_DIRECTOR),
        episode_id="test",
    )
    # Distinct from production "l1-2026-04-24" so downstream consumers
    # can detect a synthetic ledger.
    assert led["schema_version"] == TEST_LEDGER_SCHEMA
    assert led["schema_version"] != "l1-2026-04-24"


def test_audio_paths_left_none_for_user_supplied_master_wav():
    """final_audio_path / final_video_path must be None — orchestrator's
    --master-wav flag is the source of truth for TEST runs."""
    from build_test_ledger_from_director import director_json_to_test_ledger

    led = director_json_to_test_ledger(
        json.dumps(LEMMY_SATURN_DIRECTOR),
        episode_id="test",
    )
    assert led["final_audio_path"] is None
    assert led["final_video_path"] is None


# ---------------------------------------------------------------------------
# End-to-end with the orchestrator's filter_lines
# ---------------------------------------------------------------------------


def test_orchestrator_filter_all_scope_yields_every_line():
    """Synthesised ledger must work as input to render_humo_batch.filter_lines."""
    from build_test_ledger_from_director import director_json_to_test_ledger
    from render_humo_batch import filter_lines

    led = director_json_to_test_ledger(
        json.dumps(LEMMY_SATURN_DIRECTOR),
        episode_id="test",
        shots_per_scene=2,
    )
    filtered = filter_lines(led["lines"], "all", led["scenes"])
    assert len(filtered) == 6
    assert filtered == led["lines"]


def test_orchestrator_filter_first_per_scene_yields_three():
    from build_test_ledger_from_director import director_json_to_test_ledger
    from render_humo_batch import filter_lines

    led = director_json_to_test_ledger(
        json.dumps(LEMMY_SATURN_DIRECTOR),
        episode_id="test",
        shots_per_scene=2,
    )
    filtered = filter_lines(led["lines"], "first-per-scene", led["scenes"])
    assert len(filtered) == 3
    assert {ln["scene_id"] for ln in filtered} == {"scene_1", "scene_2", "scene_3"}


def test_orchestrator_filter_cold_open_yields_one():
    from build_test_ledger_from_director import director_json_to_test_ledger
    from render_humo_batch import filter_lines

    led = director_json_to_test_ledger(
        json.dumps(LEMMY_SATURN_DIRECTOR),
        episode_id="test",
        shots_per_scene=2,
    )
    filtered = filter_lines(led["lines"], "cold-open", led["scenes"])
    assert len(filtered) == 1


# ---------------------------------------------------------------------------
# CLI round-trip
# ---------------------------------------------------------------------------


def test_cli_round_trip_writes_valid_json(tmp_path: Path, monkeypatch, capsys):
    """End-to-end CLI: workflow -> ledger.json on disk, round-trip parse."""
    import build_test_ledger_from_director as adapter

    wf = _write_workflow(tmp_path, LEMMY_SATURN_DIRECTOR)
    out = tmp_path / "out_ledger.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_test_ledger_from_director.py",
            "--workflow", str(wf),
            "--out", str(out),
            "--episode-id", "test_cli",
        ],
    )
    rc = adapter.main()
    assert rc == 0
    assert out.exists()
    led = json.loads(out.read_text(encoding="utf-8"))
    assert led["episode_id"] == "test_cli"
    assert len(led["lines"]) == 6
    assert led["schema_version"] == adapter.TEST_LEDGER_SCHEMA
