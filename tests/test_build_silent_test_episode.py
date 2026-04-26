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
# group_lines_into_shots — shot/beat hierarchy
# ---------------------------------------------------------------------------


def _flat_line_ids(shot: dict) -> list[str]:
    """Helper: collect all line_ids across all beats in a shot."""
    return [lid for b in shot["beats"] for lid in b["line_ids"]]


def test_group_single_speaker_monologue_stays_in_one_beat():
    """A long single-speaker monologue should be one shot with one beat,
    even when it exceeds target_shot_dur. Shot only closes at speaker
    boundaries (and scene boundaries)."""
    from build_silent_test_episode import group_lines_into_shots
    lines = [
        {"line_id": "l1", "scene_id": "s1", "speaker": "ALICE", "dur_s": 5.0},
        {"line_id": "l2", "scene_id": "s1", "speaker": "ALICE", "dur_s": 5.0},
        {"line_id": "l3", "scene_id": "s1", "speaker": "ALICE", "dur_s": 5.0},
    ]
    shots = group_lines_into_shots(lines, target_shot_dur=9.0)
    assert len(shots) == 1
    assert shots[0]["dur_s"] == pytest.approx(15.0)
    assert len(shots[0]["beats"]) == 1
    assert shots[0]["beats"][0]["speaker"] == "ALICE"
    assert _flat_line_ids(shots[0]) == ["l1", "l2", "l3"]
    assert shots[0]["speakers"] == ["ALICE"]


def test_group_speaker_change_creates_new_beat_within_shot():
    """When accumulated dur is below target, a speaker change opens a
    new beat inside the same shot rather than closing the shot."""
    from build_silent_test_episode import group_lines_into_shots
    lines = [
        {"line_id": "l1", "scene_id": "s1", "speaker": "ALICE", "dur_s": 3.0},
        {"line_id": "l2", "scene_id": "s1", "speaker": "BOB", "dur_s": 3.0},
        {"line_id": "l3", "scene_id": "s1", "speaker": "ALICE", "dur_s": 2.0},
    ]
    shots = group_lines_into_shots(lines, target_shot_dur=9.0)
    assert len(shots) == 1
    s = shots[0]
    assert s["dur_s"] == pytest.approx(8.0)
    assert len(s["beats"]) == 3
    assert [b["speaker"] for b in s["beats"]] == ["ALICE", "BOB", "ALICE"]
    assert s["speakers"] == ["ALICE", "BOB"]  # ordered, deduped


def test_group_speaker_change_after_target_closes_shot():
    """When accumulated shot dur reaches target, the next speaker change
    closes the shot at that boundary."""
    from build_silent_test_episode import group_lines_into_shots
    lines = [
        {"line_id": "l1", "scene_id": "s1", "speaker": "ALICE", "dur_s": 5.0},
        {"line_id": "l2", "scene_id": "s1", "speaker": "ALICE", "dur_s": 5.0},  # 10 >= 9
        {"line_id": "l3", "scene_id": "s1", "speaker": "BOB", "dur_s": 4.0},   # speaker change closes shot
    ]
    shots = group_lines_into_shots(lines, target_shot_dur=9.0)
    assert len(shots) == 2
    assert shots[0]["beats"][0]["speaker"] == "ALICE"
    assert shots[0]["dur_s"] == pytest.approx(10.0)
    assert shots[1]["beats"][0]["speaker"] == "BOB"
    assert shots[1]["dur_s"] == pytest.approx(4.0)


def test_group_never_crosses_scene_boundary():
    from build_silent_test_episode import group_lines_into_shots
    lines = [
        {"line_id": "l1", "scene_id": "s1", "speaker": "ALICE", "dur_s": 4.0},
        {"line_id": "l2", "scene_id": "s2", "speaker": "ALICE", "dur_s": 4.0},
        {"line_id": "l3", "scene_id": "s2", "speaker": "ALICE", "dur_s": 6.0},
    ]
    shots = group_lines_into_shots(lines, target_shot_dur=9.0)
    assert len(shots) == 2
    assert shots[0]["scene_id"] == "s1"
    assert _flat_line_ids(shots[0]) == ["l1"]
    assert shots[1]["scene_id"] == "s2"
    assert _flat_line_ids(shots[1]) == ["l2", "l3"]


def test_group_beat_start_s_is_cumulative():
    """beat.start_s tracks the cumulative master timeline, not in-shot offset."""
    from build_silent_test_episode import group_lines_into_shots
    lines = [
        {"line_id": "l1", "scene_id": "s1", "speaker": "ALICE", "dur_s": 3.0},
        {"line_id": "l2", "scene_id": "s1", "speaker": "BOB", "dur_s": 4.0},
        {"line_id": "l3", "scene_id": "s1", "speaker": "ALICE", "dur_s": 5.0},
    ]
    shots = group_lines_into_shots(lines, target_shot_dur=20.0)
    assert len(shots) == 1
    beats = shots[0]["beats"]
    assert beats[0]["start_s"] == 0.0
    assert beats[1]["start_s"] == pytest.approx(3.0)
    assert beats[2]["start_s"] == pytest.approx(7.0)


def test_group_shot_id_and_beat_id_format():
    from build_silent_test_episode import group_lines_into_shots
    lines = [
        {"line_id": "l1", "scene_id": "s1", "speaker": "ALICE", "dur_s": 5.0},
        {"line_id": "l2", "scene_id": "s1", "speaker": "BOB", "dur_s": 5.0},
        {"line_id": "l3", "scene_id": "s2", "speaker": "ALICE", "dur_s": 5.0},
    ]
    shots = group_lines_into_shots(lines, target_shot_dur=9.0)
    # 2 shots: scene change forces close. shot1 spans ALICE+BOB (under target)
    assert shots[0]["shot_id"] == "shot_001"
    assert shots[0]["beats"][0]["beat_id"] == "shot_001_b1"
    assert shots[0]["beats"][1]["beat_id"] == "shot_001_b2"
    assert shots[1]["shot_id"] == "shot_002"
    assert shots[1]["beats"][0]["beat_id"] == "shot_002_b1"


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
    assert ledger["total_beats"] >= ledger["total_shots"]
    assert ledger["total_clips"] >= ledger["total_beats"]


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


def test_build_silent_episode_clip_durations_sum_to_beat_dur(tmp_path: Path):
    """Clip-fill rule applied per BEAT (not per shot), so each beat's
    clip_durations sum to the beat's dur. Sum across all beats in a shot
    sums to the shot's dur."""
    from build_silent_test_episode import build_silent_test_episode
    src_path = tmp_path / "src_ledger.json"
    src_path.write_text(json.dumps(_mini_source_ledger()), encoding="utf-8")
    ledger = build_silent_test_episode(src_path, tmp_path / "out", write_audio=False)
    for shot in ledger["shots"]:
        for beat in shot["beats"]:
            assert sum(beat["clip_durations_s"]) == pytest.approx(beat["dur_s"]), (
                f"beat {beat['beat_id']} clips don't sum to dur: "
                f"{beat['clip_durations_s']} vs {beat['dur_s']}"
            )
        # Shot dur = sum of beat durs
        beat_total = sum(b["dur_s"] for b in shot["beats"])
        assert beat_total == pytest.approx(shot["dur_s"])


def test_build_silent_episode_lines_carry_beat_id_and_boundary(tmp_path: Path):
    """Every emitted clip line must carry beat_id + boundary so the
    orchestrator's anchor-mode lookup table works."""
    from build_silent_test_episode import build_silent_test_episode
    src_path = tmp_path / "src_ledger.json"
    src_path.write_text(json.dumps(_mini_source_ledger()), encoding="utf-8")
    ledger = build_silent_test_episode(src_path, tmp_path / "out", write_audio=False)
    valid_boundaries = {"shot_start", "beat_start", "beat_resume", "continue"}
    for ln in ledger["lines"]:
        assert "beat_id" in ln
        assert ln["beat_id"]  # non-empty
        assert "boundary" in ln
        assert ln["boundary"] in valid_boundaries


def _source_ledger_with_episode_dur() -> dict:
    """Source ledger with total_episode_dur_s set well above the
    cumulative dialogue duration, so the lip-synching RADIO atmospheric
    phantom shots get injected (cold open + closing)."""
    led = _mini_source_ledger()
    # Mini ledger's word-count-derived dialogue ≈ 13-14s. Set total to
    # 30s so gap = ~17s, comfortably above RADIO_GAP_FLOOR_S=2s.
    led["total_episode_dur_s"] = 30.0
    return led


def test_radio_phantom_shots_injected_when_gap_exists(tmp_path: Path):
    """Source ledger with total_episode_dur_s above cumulative dialogue
    triggers cold-open + closing RADIO phantom shots. Each is a single
    shot with one beat, kind=ambient, speaker=RADIO."""
    from build_silent_test_episode import build_silent_test_episode

    src_path = tmp_path / "src_ledger.json"
    src_path.write_text(
        json.dumps(_source_ledger_with_episode_dur()),
        encoding="utf-8",
    )
    ledger = build_silent_test_episode(
        src_path, tmp_path / "out", write_audio=False,
    )

    shot_ids = [s["shot_id"] for s in ledger["shots"]]
    assert "shot_atmos_open" in shot_ids
    assert "shot_atmos_close" in shot_ids
    # Cold open is first, closing is last
    assert shot_ids[0] == "shot_atmos_open"
    assert shot_ids[-1] == "shot_atmos_close"

    open_shot = next(s for s in ledger["shots"]
                     if s["shot_id"] == "shot_atmos_open")
    close_shot = next(s for s in ledger["shots"]
                      if s["shot_id"] == "shot_atmos_close")
    for atmos in (open_shot, close_shot):
        assert len(atmos["beats"]) == 1
        b = atmos["beats"][0]
        assert b["speaker"] == "RADIO"
        assert b["kind"] == "ambient"
        assert b["clip_count"] >= 1


def test_radio_injected_into_cast_when_phantom_shots_active(tmp_path: Path):
    """The synthetic RADIO cast member is appended to ledger.cast when
    atmospheric phantom shots get injected (so render_flux_batch can
    find it and emit the radio portrait)."""
    from build_silent_test_episode import build_silent_test_episode

    src_path = tmp_path / "src_ledger.json"
    src_path.write_text(
        json.dumps(_source_ledger_with_episode_dur()),
        encoding="utf-8",
    )
    ledger = build_silent_test_episode(
        src_path, tmp_path / "out", write_audio=False,
    )
    names = [c.get("name") for c in ledger["cast"]]
    assert "RADIO" in names
    radio = next(c for c in ledger["cast"] if c.get("name") == "RADIO")
    assert "vintage" in (radio.get("description") or "").lower()


def test_radio_clips_carry_kind_ambient(tmp_path: Path):
    """Every clip line emitted from a RADIO atmospheric beat must carry
    kind=ambient and speaker=RADIO so the orchestrator can route it
    correctly."""
    from build_silent_test_episode import build_silent_test_episode

    src_path = tmp_path / "src_ledger.json"
    src_path.write_text(
        json.dumps(_source_ledger_with_episode_dur()),
        encoding="utf-8",
    )
    ledger = build_silent_test_episode(
        src_path, tmp_path / "out", write_audio=False,
    )
    radio_clips = [ln for ln in ledger["lines"] if ln.get("speaker") == "RADIO"]
    assert radio_clips, "expected at least one RADIO clip"
    for ln in radio_clips:
        assert ln["kind"] == "ambient"
        assert ln["shot_id"] in ("shot_atmos_open", "shot_atmos_close")


def test_radio_phantom_skipped_when_no_episode_dur_gap(tmp_path: Path):
    """If total_episode_dur_s is missing or close to the cumulative
    dialogue dur, no phantom shots get injected (no-op behaviour for
    older ledgers)."""
    from build_silent_test_episode import build_silent_test_episode

    src_path = tmp_path / "src_ledger.json"
    # Mini ledger with no total_episode_dur_s -> no gap, no injection.
    src_path.write_text(json.dumps(_mini_source_ledger()), encoding="utf-8")
    ledger = build_silent_test_episode(
        src_path, tmp_path / "out", write_audio=False,
    )
    shot_ids = [s["shot_id"] for s in ledger["shots"]]
    assert "shot_atmos_open" not in shot_ids
    assert "shot_atmos_close" not in shot_ids
    # No RADIO injected into cast either
    names = [c.get("name") for c in ledger["cast"]]
    assert "RADIO" not in names


def test_dialogue_clips_keep_kind_dialogue(tmp_path: Path):
    """Dialogue clips emitted alongside RADIO atmospheric clips must
    still carry kind=dialogue, not ambient."""
    from build_silent_test_episode import build_silent_test_episode

    src_path = tmp_path / "src_ledger.json"
    src_path.write_text(
        json.dumps(_source_ledger_with_episode_dur()),
        encoding="utf-8",
    )
    ledger = build_silent_test_episode(
        src_path, tmp_path / "out", write_audio=False,
    )
    dialogue_clips = [ln for ln in ledger["lines"]
                      if ln.get("speaker") != "RADIO"]
    assert dialogue_clips, "expected dialogue clips alongside RADIO"
    for ln in dialogue_clips:
        assert ln["kind"] == "dialogue"


def _speaker_repeat_source_ledger() -> dict:
    """Source ledger that exercises beat_resume: a single shot where
    speaker A is interrupted by speaker B, then A returns. ALICE speaks
    short enough lines that the shot stays under the 9 s target until
    BOB joins, so all three lines stay in one shot."""
    return {
        "schema_version": "l1-2026-04-24",
        "episode_id": "speaker_repeat",
        "cast": [
            {"char_id": "c01", "name": "ALICE"},
            {"char_id": "c02", "name": "BOB"},
        ],
        "scenes": [{"scene_id": "scene_lab", "description": "lab"}],
        "lines": [
            {"line_id": "l001", "char_id": "c01", "scene_id": "scene_lab",
             "text": "First.", },                                # ALICE — short
            {"line_id": "l002", "char_id": "c02", "scene_id": "scene_lab",
             "text": "Hi.", },                                   # BOB interrupts
            {"line_id": "l003", "char_id": "c01", "scene_id": "scene_lab",
             "text": "Resume now.", },                            # ALICE returns
        ],
        "sfx": [], "music": [],
    }


def test_beat_resume_fires_when_speaker_returns_within_shot(tmp_path: Path):
    """ALICE → BOB → ALICE within one shot: ALICE's first clip after BOB
    must carry boundary=beat_resume (not beat_start), so the orchestrator
    chains from ALICE's last frame in this shot, not from a clean reset."""
    from build_silent_test_episode import build_silent_test_episode
    src_path = tmp_path / "src_ledger.json"
    src_path.write_text(json.dumps(_speaker_repeat_source_ledger()),
                        encoding="utf-8")
    ledger = build_silent_test_episode(
        src_path, tmp_path / "out",
        target_shot_dur=20.0,  # large so all 3 lines stay in one shot
        write_audio=False,
    )
    # Should be one shot with 3 beats: ALICE, BOB, ALICE
    assert ledger["total_shots"] == 1
    shot = ledger["shots"][0]
    assert len(shot["beats"]) == 3
    assert [b["speaker"] for b in shot["beats"]] == ["ALICE", "BOB", "ALICE"]

    # First clip of each beat carries the right boundary
    by_beat = {}
    for ln in ledger["lines"]:
        by_beat.setdefault(ln["beat_id"], []).append(ln)
    beat_ids = [b["beat_id"] for b in shot["beats"]]

    # b1 ALICE first clip = shot_start
    assert by_beat[beat_ids[0]][0]["boundary"] == "shot_start"
    # b2 BOB first clip = beat_start (BOB is new in this shot)
    assert by_beat[beat_ids[1]][0]["boundary"] == "beat_start"
    # b3 ALICE first clip = beat_resume (ALICE was already in this shot)
    assert by_beat[beat_ids[2]][0]["boundary"] == "beat_resume"


def test_beat_resume_does_not_leak_across_shots(tmp_path: Path):
    """If ALICE appears in shot 1 and again in shot 2 (after a scene
    change), shot 2's first ALICE beat is shot_start, not beat_resume.
    The speakers_seen tracker resets per shot."""
    from build_silent_test_episode import build_silent_test_episode
    src = {
        "schema_version": "l1-2026-04-24",
        "episode_id": "cross_shot",
        "cast": [{"char_id": "c01", "name": "ALICE"}],
        "scenes": [
            {"scene_id": "scene_a"},
            {"scene_id": "scene_b"},
        ],
        "lines": [
            {"line_id": "l001", "char_id": "c01",
             "scene_id": "scene_a", "text": "Scene one."},
            {"line_id": "l002", "char_id": "c01",
             "scene_id": "scene_b", "text": "Scene two."},
        ],
        "sfx": [], "music": [],
    }
    src_path = tmp_path / "src_ledger.json"
    src_path.write_text(json.dumps(src), encoding="utf-8")
    ledger = build_silent_test_episode(src_path, tmp_path / "out",
                                        write_audio=False)
    # Two shots (scene change forces close)
    assert ledger["total_shots"] == 2
    # Both first-beat clips are shot_start, neither is beat_resume
    for shot in ledger["shots"]:
        first_beat_id = shot["beats"][0]["beat_id"]
        first_clip = next(ln for ln in ledger["lines"]
                          if ln["beat_id"] == first_beat_id)
        assert first_clip["boundary"] == "shot_start", (
            f"shot {shot['shot_id']} first clip should be shot_start, "
            f"got {first_clip['boundary']}"
        )


def test_build_silent_episode_boundary_state_machine(tmp_path: Path):
    """The 4-state boundary machine must hold across the whole ledger:
       - first clip of each shot          -> shot_start
       - first clip of a beat whose speaker hasn't appeared in this shot
                                          -> beat_start
       - first clip of a beat whose speaker DID appear earlier in this shot
                                          -> beat_resume
       - any clip after the first in a beat
                                          -> continue
    """
    from build_silent_test_episode import build_silent_test_episode
    src_path = tmp_path / "src_ledger.json"
    src_path.write_text(json.dumps(_mini_source_ledger()), encoding="utf-8")
    ledger = build_silent_test_episode(src_path, tmp_path / "out", write_audio=False)

    seen_shots: set[str] = set()
    seen_beats_in_shot: dict[str, set[str]] = {}
    seen_speakers_in_shot: dict[str, set[str]] = {}
    for ln in ledger["lines"]:
        sid = ln["shot_id"]
        bid = ln["boundary"]
        bb = ln["beat_id"]
        sp = ln["speaker"]

        if sid not in seen_shots:
            assert bid == "shot_start", (
                f"first clip of new shot {sid} should be shot_start, got {bid}"
            )
            seen_shots.add(sid)
            seen_beats_in_shot[sid] = {bb}
            seen_speakers_in_shot[sid] = {sp}
        elif bb not in seen_beats_in_shot[sid]:
            # New beat in this shot -- depends on whether speaker is new
            if sp in seen_speakers_in_shot[sid]:
                assert bid == "beat_resume", (
                    f"speaker {sp} returning in shot {sid} (beat {bb}) "
                    f"should be beat_resume, got {bid}"
                )
            else:
                assert bid == "beat_start", (
                    f"first clip of new speaker {sp} in shot {sid} "
                    f"(beat {bb}) should be beat_start, got {bid}"
                )
            seen_beats_in_shot[sid].add(bb)
            seen_speakers_in_shot[sid].add(sp)
        else:
            assert bid == "continue", (
                f"continuing clip in beat {bb} (shot {sid}) should be "
                f"continue, got {bid}"
            )


def test_build_silent_episode_shots_carry_beats_inline(tmp_path: Path):
    """ledger.shots[i].beats[] must be present, populated, and consistent
    with the lines[] emit."""
    from build_silent_test_episode import build_silent_test_episode
    src_path = tmp_path / "src_ledger.json"
    src_path.write_text(json.dumps(_mini_source_ledger()), encoding="utf-8")
    ledger = build_silent_test_episode(src_path, tmp_path / "out", write_audio=False)
    for shot in ledger["shots"]:
        assert "beats" in shot
        assert shot["beats"], f"shot {shot['shot_id']} has no beats"
        assert "speakers" in shot
        # Total clip count from beats matches shot's clip_count
        assert sum(b["clip_count"] for b in shot["beats"]) == shot["clip_count"]
    # Sum across shots matches total_clips
    total = sum(s["clip_count"] for s in ledger["shots"])
    assert total == ledger["total_clips"]


def test_build_silent_episode_no_clip_spans_two_speakers(tmp_path: Path):
    """The whole point of beats: a clip's audio window must come from
    one speaker only. Verify by checking each clip's [start_s, start_s+dur_s]
    falls entirely within one source line's window (or contiguous lines
    of the SAME speaker)."""
    from build_silent_test_episode import build_silent_test_episode
    src_path = tmp_path / "src_ledger.json"
    src_path.write_text(json.dumps(_mini_source_ledger()), encoding="utf-8")
    ledger = build_silent_test_episode(src_path, tmp_path / "out", write_audio=False)
    # Build (start_s, end_s, speaker) triples from source_lines
    src_windows = []
    cursor = 0.0
    for ln in ledger["source_lines"]:
        src_windows.append((cursor, cursor + ln["dur_s"], ln["speaker"]))
        cursor += ln["dur_s"]
    for clip in ledger["lines"]:
        c_start = clip["start_s"]
        c_end = c_start + clip["dur_s"]
        # Find which source-line window(s) overlap this clip
        overlap_speakers = set()
        for s_start, s_end, s_speaker in src_windows:
            # Strictly-inside overlap (more than rounding tolerance)
            if s_end > c_start + 1e-6 and s_start < c_end - 1e-6:
                overlap_speakers.add(s_speaker)
        assert len(overlap_speakers) <= 1, (
            f"clip {clip['line_id']} ({c_start}-{c_end}) spans speakers {overlap_speakers}"
        )


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
