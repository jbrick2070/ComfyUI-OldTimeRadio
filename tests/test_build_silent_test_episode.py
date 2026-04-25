"""
test_build_silent_test_episode.py
==================================

Unit tests for scripts/build_silent_test_episode.py and the
humo_length_for_dur helper in scripts/render_humo_batch.py.

Validates:
  * humo_length_for_dur snaps to nearest valid Wan-VAE 4n+1 frame count
  * estimate_dur_s computes word-count durations sanely
  * clip_durations_for_shot honours Jeffrey's fill-with-7s + average
    last two rule on every documented case
  * group_lines_into_shots respects scene boundaries + target dur cap
  * build_silent_test_episode end-to-end on a realistic mini ledger

Pure stdlib + pytest. No torch, diffusers, ComfyUI, or GPU.
ffmpeg is required for the end-to-end test (skipped if missing).
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


# ---------------------------------------------------------------------------
# humo_length_for_dur — Wan-VAE 4n+1 snapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dur_s,expected_length", [
    # Documented anchor points
    (3.88, 97),    # yesterday's measured floor
    (4.5, 113),    # 4n+1 = 113 -> 4.52s
    (5.0, 125),    # exact: 4n+1 = 125 -> 5.00s
    (6.0, 153),    # 4n+1 = 153 -> 6.12s
    (7.0, 177),    # 4n+1 = 177 -> 7.08s -- verified on hardware 2026-04-25
    (7.08, 177),   # already on grid
    # Edges
    (0.5, 33),     # below floor -> clamped to HUMO_MIN_FRAMES=33
    (0.0, 33),     # zero -> clamped
    (1.32, 33),    # 33-frame floor
    (1.4, 37),     # 4*9+1 = 37 -> 1.48s (smallest valid above 1.4s)
])
def test_humo_length_for_dur_snaps_to_4n_plus_1(dur_s, expected_length):
    from render_humo_batch import humo_length_for_dur
    assert humo_length_for_dur(dur_s) == expected_length


def test_humo_length_for_dur_caps_at_max():
    from render_humo_batch import humo_length_for_dur, HUMO_MAX_FRAMES
    # Anything beyond the empirical ceiling clamps to max
    assert humo_length_for_dur(10.0) == HUMO_MAX_FRAMES
    assert humo_length_for_dur(7.5) == HUMO_MAX_FRAMES  # 187 not yet verified


def test_humo_length_for_dur_returns_4n_plus_1():
    """Property test: every output must satisfy (length - 1) % 4 == 0."""
    from render_humo_batch import humo_length_for_dur
    for dur in [d / 100.0 for d in range(0, 800, 5)]:  # 0..8s @ 0.05 step
        length = humo_length_for_dur(dur)
        assert (length - 1) % 4 == 0, f"length={length} not 4n+1 for dur={dur}"


# ---------------------------------------------------------------------------
# estimate_dur_s — word-count duration
# ---------------------------------------------------------------------------


def test_estimate_dur_s_empty_text_returns_floor():
    from build_silent_test_episode import estimate_dur_s
    # Empty text returns max(1.0, pad_s)
    assert estimate_dur_s("") == 1.0
    assert estimate_dur_s("", pad_s=2.0) == 2.0


def test_estimate_dur_s_word_count():
    from build_silent_test_episode import estimate_dur_s
    # 5 words at 2.5 wps = 2.0s + 0.5s pad = 2.5s
    assert estimate_dur_s("one two three four five",
                          wps=2.5, pad_s=0.5) == pytest.approx(2.5)


def test_estimate_dur_s_punctuation_ignored():
    from build_silent_test_episode import estimate_dur_s
    # 11 words -> 11/2.5 + 0.5 = 4.9s
    text = "Welcome to Signal Lost. Tonight's broadcast takes us into the unknown."
    dur = estimate_dur_s(text, wps=2.5, pad_s=0.5)
    assert dur == pytest.approx(4.9)


def test_estimate_dur_s_floors_at_one_second():
    from build_silent_test_episode import estimate_dur_s
    # Single short word with zero pad -> floored at 1.0s
    assert estimate_dur_s("hi", wps=2.5, pad_s=0.0) == 1.0


# ---------------------------------------------------------------------------
# clip_durations_for_shot — Jeffrey's fill rule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shot_dur,expected", [
    # Clean multiples of 7
    (7.0, [7.0]),
    (14.0, [7.0, 7.0]),
    (21.0, [7.0, 7.0, 7.0]),
    (28.0, [7.0, 7.0, 7.0, 7.0]),
    # Less than base -> single clip
    (3.0, [3.0]),
    (5.0, [5.0]),
    (6.99, [6.99]),
    # Average-last-two cases
    (10.0, [5.0, 5.0]),       # last full 7 + 3 leftover -> 5+5
    (12.0, [6.0, 6.0]),       # 7 + 5 -> 6+6
    (16.0, [7.0, 4.5, 4.5]),  # keep first 7, then 7+2 -> 4.5+4.5
    (23.0, [7.0, 7.0, 4.5, 4.5]),
    (30.0, [7.0, 7.0, 7.0, 4.5, 4.5]),
    (14.5, [7.0, 3.75, 3.75]),  # tiny overshoot
    (8.0, [4.0, 4.0]),        # 7 + 1 -> 4+4
    (9.0, [4.5, 4.5]),
    # Zero or negative -> empty
    (0.0, []),
    (-1.0, []),
])
def test_clip_durations_for_shot_examples(shot_dur, expected):
    from build_silent_test_episode import clip_durations_for_shot
    result = clip_durations_for_shot(shot_dur)
    assert len(result) == len(expected), (
        f"shot_dur={shot_dur} got {result}, expected {expected}"
    )
    for got, want in zip(result, expected):
        assert got == pytest.approx(want), (
            f"shot_dur={shot_dur} got {result}, expected {expected}"
        )


def test_clip_durations_sum_equals_shot_dur():
    """Property: the clip durations must sum to the shot duration exactly."""
    from build_silent_test_episode import clip_durations_for_shot
    for shot_dur in [d / 10.0 for d in range(1, 350)]:  # 0.1..35s
        clips = clip_durations_for_shot(shot_dur)
        assert sum(clips) == pytest.approx(shot_dur, abs=1e-9), (
            f"shot_dur={shot_dur} clips={clips} sum={sum(clips)}"
        )


def test_clip_durations_no_clip_below_half_base():
    """Property: when the rule applies, no logical clip is < base/2.

    The whole point of the average-last-two rule is to avoid tiny
    trailing pieces. A clip should never end up shorter than half the
    base (3.5s for base=7) unless the entire shot is shorter than the
    base.
    """
    from build_silent_test_episode import clip_durations_for_shot
    base = 7.0
    for shot_dur in [d / 10.0 for d in range(70, 350)]:  # 7..35s
        clips = clip_durations_for_shot(shot_dur, base=base)
        for clip in clips:
            assert clip >= base / 2.0 - 1e-9, (
                f"shot_dur={shot_dur} produced clip={clip} below base/2={base/2}"
            )


def test_clip_durations_custom_base():
    from build_silent_test_episode import clip_durations_for_shot
    # base=8, shot=20 -> 8 + 12 leftover -> last full 8 + leftover 4 = 12, split 6+6
    # Wait: 20/8 = 2.5, n_full=2, remainder=4. 8+4=12, split into 6+6.
    # So [8, 6, 6]
    assert clip_durations_for_shot(20.0, base=8.0) == pytest.approx([8.0, 6.0, 6.0])


# ---------------------------------------------------------------------------
# group_lines_into_shots
# ---------------------------------------------------------------------------


def test_group_respects_target_dur():
    from build_silent_test_episode import group_lines_into_shots
    lines = [
        {"line_id": "l1", "scene_id": "s1", "dur_s": 3.0},
        {"line_id": "l2", "scene_id": "s1", "dur_s": 3.0},
        {"line_id": "l3", "scene_id": "s1", "dur_s": 3.0},  # closes shot at 9s
        {"line_id": "l4", "scene_id": "s1", "dur_s": 4.0},
        {"line_id": "l5", "scene_id": "s1", "dur_s": 6.0},  # closes shot at 10s
    ]
    shots = group_lines_into_shots(lines, target_shot_dur=9.0)
    assert len(shots) == 2
    assert shots[0]["dur_s"] == pytest.approx(9.0)
    assert shots[0]["line_ids"] == ["l1", "l2", "l3"]
    assert shots[1]["dur_s"] == pytest.approx(10.0)
    assert shots[1]["line_ids"] == ["l4", "l5"]


def test_group_never_crosses_scene_boundary():
    from build_silent_test_episode import group_lines_into_shots
    lines = [
        {"line_id": "l1", "scene_id": "s1", "dur_s": 4.0},
        {"line_id": "l2", "scene_id": "s2", "dur_s": 4.0},  # forces shot close
        {"line_id": "l3", "scene_id": "s2", "dur_s": 6.0},
    ]
    shots = group_lines_into_shots(lines, target_shot_dur=9.0)
    assert len(shots) == 2
    assert shots[0]["scene_id"] == "s1"
    assert shots[0]["line_ids"] == ["l1"]
    assert shots[1]["scene_id"] == "s2"
    assert shots[1]["line_ids"] == ["l2", "l3"]


def test_group_start_s_is_cumulative():
    from build_silent_test_episode import group_lines_into_shots
    lines = [
        {"line_id": "l1", "scene_id": "s1", "dur_s": 5.0},
        {"line_id": "l2", "scene_id": "s1", "dur_s": 5.0},  # closes at 10
        {"line_id": "l3", "scene_id": "s1", "dur_s": 5.0},
        {"line_id": "l4", "scene_id": "s1", "dur_s": 5.0},  # closes at 10
    ]
    shots = group_lines_into_shots(lines, target_shot_dur=9.0)
    assert shots[0]["start_s"] == 0.0
    assert shots[1]["start_s"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# build_silent_test_episode — end to end
# ---------------------------------------------------------------------------


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _mini_source_ledger() -> dict:
    """Realistic mini ledger structure matching production l1-2026-04-24."""
    return {
        "schema_version": "l1-2026-04-24",
        "episode_id": "mini_test",
        "cast": [
            {"char_id": "c01", "name": "ALICE"},
            {"char_id": "c02", "name": "BOB"},
        ],
        "scenes": [
            {"scene_id": "scene_01", "description": "lab"},
            {"scene_id": "scene_02", "description": "hall"},
        ],
        "lines": [
            {"line_id": "l001", "char_id": "c01", "scene_id": "scene_01",
             "text": "Welcome to Signal Lost. Tonight we explore the unknown.", },
            {"line_id": "l002", "char_id": "c02", "scene_id": "scene_01",
             "text": "Tell me what happened on Saturn last night.", },
            {"line_id": "l003", "char_id": "c01", "scene_id": "scene_01",
             "text": "We received a distress call at oh-three-hundred hours.", },
            {"line_id": "l004", "char_id": "c01", "scene_id": "scene_02",
             "text": "Down the hall, behind the bulkhead, a strange light flickers softly.", },
        ],
        "sfx": [],
        "music": [],
    }


def test_build_silent_episode_writes_ledger_and_meta(tmp_path: Path):
    from build_silent_test_episode import build_silent_test_episode
    src_path = tmp_path / "src_ledger.json"
    src_path.write_text(json.dumps(_mini_source_ledger()), encoding="utf-8")
    out_dir = tmp_path / "out"
    ledger = build_silent_test_episode(
        src_path, out_dir,
        target_shot_dur=9.0,
        clip_base_dur=7.0,
        write_audio=False,
    )
    assert (out_dir / "ledger.json").exists()
    assert (out_dir / "meta.json").exists()
    assert ledger["schema_version"] == "silent-test-2026-04-25"
    assert ledger["total_dialogue_lines"] == 4
    assert ledger["total_shots"] >= 1
    assert ledger["total_clips"] >= ledger["total_shots"]


def test_build_silent_episode_lines_have_humo_length(tmp_path: Path):
    from build_silent_test_episode import build_silent_test_episode
    src_path = tmp_path / "src_ledger.json"
    src_path.write_text(json.dumps(_mini_source_ledger()), encoding="utf-8")
    ledger = build_silent_test_episode(
        src_path, tmp_path / "out",
        write_audio=False,
    )
    # Every emitted line is one logical clip with a 4n+1 humo_length.
    assert ledger["lines"], "expected at least one logical-clip line"
    for ln in ledger["lines"]:
        assert "humo_length" in ln
        assert (ln["humo_length"] - 1) % 4 == 0
        assert ln["dur_s"] > 0
        assert ln["start_s"] >= 0


def test_build_silent_episode_clip_durations_sum_to_shot_dur(tmp_path: Path):
    from build_silent_test_episode import build_silent_test_episode
    src_path = tmp_path / "src_ledger.json"
    src_path.write_text(json.dumps(_mini_source_ledger()), encoding="utf-8")
    ledger = build_silent_test_episode(src_path, tmp_path / "out", write_audio=False)
    for shot in ledger["shots"]:
        assert sum(shot["clip_durations_s"]) == pytest.approx(shot["dur_s"])


def test_build_silent_episode_speaker_resolved_from_cast(tmp_path: Path):
    from build_silent_test_episode import build_silent_test_episode
    src_path = tmp_path / "src_ledger.json"
    src_path.write_text(json.dumps(_mini_source_ledger()), encoding="utf-8")
    ledger = build_silent_test_episode(src_path, tmp_path / "out", write_audio=False)
    speakers = {ln["speaker"] for ln in ledger["lines"]}
    # ALICE owns lines 1, 3, 4; BOB owns line 2. Both should appear.
    assert "ALICE" in speakers
    assert "BOB" in speakers


def test_build_silent_episode_lines_form_continuous_timeline(tmp_path: Path):
    """start_s of each clip line must equal sum of dur_s of preceding."""
    from build_silent_test_episode import build_silent_test_episode
    src_path = tmp_path / "src_ledger.json"
    src_path.write_text(json.dumps(_mini_source_ledger()), encoding="utf-8")
    ledger = build_silent_test_episode(src_path, tmp_path / "out", write_audio=False)
    cursor = 0.0
    for ln in ledger["lines"]:
        assert ln["start_s"] == pytest.approx(cursor, abs=1e-6), (
            f"timeline gap at {ln['line_id']}: start={ln['start_s']} expected={cursor}"
        )
        cursor += ln["dur_s"]


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg not on PATH")
def test_build_silent_episode_writes_silence_master_wav(tmp_path: Path):
    from build_silent_test_episode import build_silent_test_episode
    src_path = tmp_path / "src_ledger.json"
    src_path.write_text(json.dumps(_mini_source_ledger()), encoding="utf-8")
    out_dir = tmp_path / "out"
    ledger = build_silent_test_episode(src_path, out_dir, write_audio=True)
    master = out_dir / "audio" / "master.wav"
    assert master.exists()
    # File should be > 0 bytes (silence is still PCM samples, not zero file)
    assert master.stat().st_size > 1000
    assert ledger["final_audio_path"].endswith("master.wav")


# ---------------------------------------------------------------------------
# Orchestrator integration: filter_lines accepts the new schema
# ---------------------------------------------------------------------------


def test_orchestrator_filter_all_consumes_silent_ledger(tmp_path: Path):
    from build_silent_test_episode import build_silent_test_episode
    from render_humo_batch import filter_lines
    src_path = tmp_path / "src_ledger.json"
    src_path.write_text(json.dumps(_mini_source_ledger()), encoding="utf-8")
    ledger = build_silent_test_episode(src_path, tmp_path / "out", write_audio=False)
    selected = filter_lines(ledger["lines"], "all", ledger["scenes"])
    assert len(selected) == ledger["total_clips"]
