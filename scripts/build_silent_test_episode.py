#!/usr/bin/env python3
"""Build a silent-audio test episode from a prior FULL ledger.

Goal 1 helper for HuMo full-episode coverage testing without paying the
Bark / SceneSequencer cost. Reads an existing FULL-pipeline ledger that
has real `cast[]` + `lines[].text` but null per-line timing, backfills
durations from word counts, walks lines into shots at a configurable
target duration, applies Jeffrey's clip-fill rule, snaps each logical
clip's frame count to a Wan-VAE-valid (4n+1) value, and emits:

  <out_dir>/ledger.json          augmented ledger with per-line dur_s,
                                 per-shot grouping, per-clip frame plan
  <out_dir>/audio/master.wav     mono 16 kHz silence at total episode
                                 duration (HuMo-compatible, lip-sync
                                 will be static -- intentional for
                                 orchestration testing)
  <out_dir>/meta.json            run metadata (source ledger, params,
                                 totals, est wall clock)

The orchestrator (scripts/render_humo_batch.py) consumes the ledger
with --scope all and renders one HuMo MP4 per logical clip.

Clip-fill rule (locked 2026-04-25 with Jeffrey):
  - Walk lines, close a shot when accumulated dur_s >= target_shot_dur
  - For each shot of duration X, fill with 7s clips while X >= 7
  - If a leftover < 7s remains, take the LAST FULL 7s clip + the
    leftover and split the combined duration into TWO equal pieces
  - This avoids a tiny trailing shorty; trailing two clips are always
    "good size"
  - Examples:
      14s shot -> [7, 7]
      16s shot -> [7, 4.5, 4.5]
      23s shot -> [7, 7, 4.5, 4.5]
      10s shot -> [5, 5]
      5s shot  -> [5]                  (shorter than 7, single clip)

HuMo length (per-clip, 4n+1 snap): the smallest length such that
`length / 25 >= clip_dur_s`. Empirical ceiling on RTX 5080 16GB at
640x640 fp8 is length=177 / 7.08s (verified 2026-04-25). length=33
is the practical floor (smaller hangs the system).

Usage:
  python scripts/build_silent_test_episode.py \\
      --source-ledger output/old_time_radio/signal_lost_<title>_ledger.json \\
      --out-dir output/old_time_radio/silent_test_<title> \\
      --target-shot-dur 9.0 \\
      --clip-base-dur 7.0
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

# Repo root one level up from scripts/. Reuse helpers from render_humo_batch.py.
_THIS = Path(__file__).resolve()
_REPO = _THIS.parent.parent
_SCRIPTS = _THIS.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from render_humo_batch import (  # noqa: E402  -- after sys.path setup
    HUMO_FPS,
    HUMO_MAX_FRAMES,
    HUMO_MIN_FRAMES,
    humo_length_for_dur,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_WPS = 2.5            # English broadcast cadence (~150 wpm)
DEFAULT_PAD_PER_LINE = 0.5   # silence padding per line (beat pause)
DEFAULT_TARGET_SHOT_DUR = 9.0
DEFAULT_CLIP_BASE_DUR = 7.0
SCHEMA_VERSION = "silent-test-2026-04-25"


# ---------------------------------------------------------------------------
# Word-count duration estimator
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'\-]*")


def estimate_dur_s(text: str, *, wps: float = DEFAULT_WPS,
                   pad_s: float = DEFAULT_PAD_PER_LINE) -> float:
    """Estimate spoken duration of `text` from word count.

    dur_s = max(1.0, word_count / wps + pad_s).
    Floor of 1.0s avoids degenerate zero-length lines.
    """
    if not text:
        return max(1.0, pad_s)
    words = _WORD_RE.findall(text)
    return max(1.0, len(words) / float(wps) + float(pad_s))


# ---------------------------------------------------------------------------
# Line -> shot/beat grouping
# ---------------------------------------------------------------------------

# Hierarchy:
#   Scene > Shot > Beat > Clip
#
#   Scene: high-level narrative location (e.g., "AstroTech Research Facility").
#          Sourced from line.scene_id when present.
#   Shot:  continuous visual unit. Same framing, same lighting. Multiple
#          speakers OK *within* a shot (modeled as beats below).
#   Beat:  single-speaker continuous turn within a shot. Closes on every
#          speaker change. The unit at which the 7s clip-fill rule applies.
#   Clip:  one HuMo render call. dur_s comes from the clip-fill rule
#          (clip_durations_for_shot, applied per beat).
#
# Shot-close rules:
#   - Scene change: always closes the current shot.
#   - Speaker change AND accumulated shot dur >= target_shot_dur: closes
#     the shot at the speaker boundary (so we don't grow shots past
#     target on long single-speaker monologues).
#   - Otherwise: speaker change opens a new beat *inside* the same shot.
#
# Beat-close rules:
#   - Closes whenever the parent shot closes.
#   - Closes on every speaker change (a beat is single-speaker by definition).


def group_lines_into_shots(
    lines: list[dict[str, Any]],
    *,
    target_shot_dur: float,
) -> list[dict[str, Any]]:
    """Walk lines and group into shots, with beats inside each shot.

    Returns a list of shot dicts:
      {
        shot_id, scene_id, dur_s, start_s,
        speakers: [str, ...]  # ordered, deduped
        beats: [
          {beat_id, speaker, line_ids[], dur_s, start_s},
          ...
        ]
      }

    `dur_s` on the shot is the sum of its beats' durations.
    `start_s` is the shot's start timestamp on the master timeline; the
    same field on a beat is the beat's start within the master timeline.
    """
    shots: list[dict[str, Any]] = []
    current_shot: dict[str, Any] | None = None
    current_beat: dict[str, Any] | None = None
    timeline_pos = 0.0

    def _open_shot(scene_id: Any, start_s: float) -> dict[str, Any]:
        return {
            "shot_id": f"shot_{len(shots) + 1:03d}",
            "scene_id": scene_id,
            "start_s": start_s,
            "dur_s": 0.0,
            "beats": [],
        }

    def _open_beat(shot: dict[str, Any], speaker: str,
                   start_s: float) -> dict[str, Any]:
        return {
            "beat_id": f"{shot['shot_id']}_b{len(shot['beats']) + 1}",
            "speaker": speaker,
            "line_ids": [],
            "start_s": start_s,
            "dur_s": 0.0,
        }

    def _finalize_shot(shot: dict[str, Any]) -> None:
        # Speakers in order of first appearance, deduplicated.
        seen: set[str] = set()
        ordered: list[str] = []
        for b in shot["beats"]:
            sp = b.get("speaker") or ""
            if sp and sp not in seen:
                seen.add(sp)
                ordered.append(sp)
        shot["speakers"] = ordered

    for ln in lines:
        ln_dur = float(ln.get("dur_s") or 0.0)
        ln_scene = ln.get("scene_id")
        ln_speaker = (ln.get("speaker") or "").strip()
        ln_id = ln.get("line_id")

        # ----- decide: new shot? -----
        need_new_shot = False
        if current_shot is None:
            need_new_shot = True
        elif (
            current_shot["scene_id"] is not None
            and ln_scene is not None
            and ln_scene != current_shot["scene_id"]
        ):
            need_new_shot = True
        elif (
            current_beat is not None
            and ln_speaker != current_beat["speaker"]
            and current_shot["dur_s"] >= target_shot_dur
        ):
            need_new_shot = True

        if need_new_shot:
            if current_shot is not None:
                _finalize_shot(current_shot)
                shots.append(current_shot)
            current_shot = _open_shot(ln_scene, timeline_pos)
            current_beat = _open_beat(current_shot, ln_speaker, timeline_pos)
            current_shot["beats"].append(current_beat)
        else:
            # ----- decide: new beat inside same shot? -----
            assert current_shot is not None
            if (
                current_beat is None
                or ln_speaker != current_beat["speaker"]
            ):
                current_beat = _open_beat(current_shot, ln_speaker, timeline_pos)
                current_shot["beats"].append(current_beat)

        # ----- append the line -----
        assert current_beat is not None and current_shot is not None
        current_beat["line_ids"].append(ln_id)
        current_beat["dur_s"] += ln_dur
        current_shot["dur_s"] += ln_dur
        timeline_pos += ln_dur

    if current_shot is not None:
        _finalize_shot(current_shot)
        shots.append(current_shot)

    return shots


# ---------------------------------------------------------------------------
# Jeffrey's clip-fill rule
# ---------------------------------------------------------------------------

def clip_durations_for_shot(shot_dur: float,
                            base: float = DEFAULT_CLIP_BASE_DUR,
                            *,
                            tol: float = 1e-6) -> list[float]:
    """Apply the fill-with-7s + average-last-two rule.

    Examples (base=7):
        7  -> [7]
        14 -> [7, 7]
        21 -> [7, 7, 7]
        16 -> [7, 4.5, 4.5]
        10 -> [5, 5]
        12 -> [6, 6]
        23 -> [7, 7, 4.5, 4.5]
        5  -> [5]              (shorter than base, single clip)
    """
    base = float(base)
    shot_dur = float(shot_dur)
    if shot_dur <= 0:
        return []
    if shot_dur < base - tol:
        # Too short for a full base clip -- single clip at shot_dur.
        return [shot_dur]
    n_full = int((shot_dur + tol) // base)
    remainder = shot_dur - n_full * base
    if remainder <= tol:
        # Clean multiple of base; no averaging needed.
        return [base] * n_full
    # Take the last full clip + the leftover, split equally.
    avg = (base + remainder) / 2.0
    return [base] * (n_full - 1) + [avg, avg]


# ---------------------------------------------------------------------------
# Silence WAV generation
# ---------------------------------------------------------------------------

def generate_silence_wav(out_path: Path, dur_s: float) -> None:
    """Write a mono 16 kHz PCM silence WAV at the given duration.

    Matches HuMo's required input audio format (Whisper expects 16 kHz mono).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-f", "lavfi",
        "-i", f"anullsrc=channel_layout=mono:sample_rate=16000",
        "-t", f"{dur_s:.3f}",
        "-c:a", "pcm_s16le",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_silent_test_episode(
    source_ledger_path: Path,
    out_dir: Path,
    *,
    target_shot_dur: float = DEFAULT_TARGET_SHOT_DUR,
    clip_base_dur: float = DEFAULT_CLIP_BASE_DUR,
    wps: float = DEFAULT_WPS,
    pad_per_line: float = DEFAULT_PAD_PER_LINE,
    write_audio: bool = True,
) -> dict[str, Any]:
    """Build the augmented ledger + silence audio + meta.

    Returns the ledger dict (also written to disk under out_dir).
    """
    with open(source_ledger_path, "r", encoding="utf-8") as f:
        src = json.load(f)

    cast = list(src.get("cast") or [])
    scenes_in = list(src.get("scenes") or [])
    lines_in = list(src.get("lines") or [])

    # Map char_id -> name for quick speaker lookup.
    char_id_to_name = {c.get("char_id"): c.get("name") for c in cast if c.get("char_id")}

    # ----- Phase 1: backfill per-line dur_s + start_s -----
    enriched_lines: list[dict[str, Any]] = []
    cursor = 0.0
    for idx, ln in enumerate(lines_in):
        text = (ln.get("text") or "").strip()
        dur_s = estimate_dur_s(text, wps=wps, pad_s=pad_per_line)
        speaker = char_id_to_name.get(ln.get("char_id")) or ln.get("speaker") or ""
        out_ln = dict(ln)
        out_ln.setdefault("line_id", f"l{idx + 1:03d}")
        out_ln["speaker"] = speaker
        out_ln["dur_s"] = dur_s
        out_ln["start_s"] = cursor
        enriched_lines.append(out_ln)
        cursor += dur_s

    total_episode_dur = cursor

    # ----- Phase 2: group lines into shots -----
    shots = group_lines_into_shots(
        enriched_lines, target_shot_dur=target_shot_dur,
    )

    # ----- Phase 3: clip-fill each beat, build orchestrator-ready lines -----
    # The orchestrator iterates ledger.lines[]. We REPLACE the original
    # lines list with one entry per LOGICAL CLIP. The clip-fill rule
    # (clip_durations_for_shot) is applied PER BEAT, not per shot --
    # speaker boundaries are honoured so no clip ever spans two speakers.
    #
    # Each clip carries `boundary` ∈ {shot_start, beat_start, continue}
    # so the orchestrator's Goal 3 hybrid-anchor logic knows whether to
    # do a full reset (new shot), a clean portrait reset (new speaker
    # inside the same shot), or a daisy-chain α-blend (same speaker,
    # same shot, continuing).
    cast_name_to_char_id = {
        (c.get("name") or "").strip(): c.get("char_id")
        for c in cast if c.get("name")
    }

    clip_lines: list[dict[str, Any]] = []
    total_clips = 0
    for shot in shots:
        shot_id = shot["shot_id"]
        scene_id = shot["scene_id"]
        # Track which speakers have appeared in this shot already so we
        # can distinguish a fresh speaker (`beat_start`) from a returning
        # speaker who was interrupted earlier (`beat_resume`). Cleared
        # per shot since the chain-from-this-speaker's-last-frame
        # mechanic only makes sense within one shot.
        speakers_seen_in_shot: set[str] = set()

        for beat_idx, beat in enumerate(shot["beats"]):
            beat_id = beat["beat_id"]
            speaker = beat["speaker"]
            char_id = cast_name_to_char_id.get(speaker)
            beat_start = beat["start_s"]
            beat_dur = beat["dur_s"]
            speaker_already_in_shot = speaker in speakers_seen_in_shot

            clip_durs = clip_durations_for_shot(beat_dur, base=clip_base_dur)
            beat["clip_count"] = len(clip_durs)
            beat["clip_durations_s"] = clip_durs

            offset = 0.0
            for clip_idx, clip_dur in enumerate(clip_durs):
                length = humo_length_for_dur(clip_dur)
                clip_id = f"{beat_id}_c{clip_idx + 1}"

                # Boundary type drives the orchestrator's anchor mode:
                #   shot_start  : first clip of new shot, full visual reset
                #   beat_start  : same shot, NEW speaker (never appeared in
                #                 this shot before) -- clean portrait, no
                #                 chain
                #   beat_resume : same shot, RETURNING speaker (interrupted
                #                 earlier in this shot by another beat) --
                #                 chain from this speaker's last frame in
                #                 this shot
                #   continue    : same beat, same speaker -- chain from
                #                 immediately preceding clip's last frame
                if beat_idx == 0 and clip_idx == 0:
                    boundary = "shot_start"
                elif clip_idx == 0:
                    boundary = ("beat_resume" if speaker_already_in_shot
                                else "beat_start")
                else:
                    boundary = "continue"

                clip_lines.append({
                    "line_id": clip_id,
                    "shot_id": shot_id,
                    "beat_id": beat_id,
                    "scene_id": scene_id,
                    "speaker": speaker,
                    "char_id": char_id,
                    "boundary": boundary,
                    "text": "",
                    "start_s": beat_start + offset,
                    "dur_s": clip_dur,
                    "humo_length": length,
                    "humo_duration_s": length / HUMO_FPS,
                })
                offset += clip_dur
                total_clips += 1

            # End of beat: this speaker has now appeared in this shot.
            # Subsequent beats with the same speaker in this shot get
            # boundary=beat_resume.
            if speaker:
                speakers_seen_in_shot.add(speaker)

    # ----- Phase 4: per-shot summary (beats inline, with clip plan) -----
    shots_out: list[dict[str, Any]] = []
    for shot in shots:
        beats_out: list[dict[str, Any]] = []
        shot_clip_count = 0
        for beat in shot["beats"]:
            beats_out.append({
                "beat_id": beat["beat_id"],
                "speaker": beat["speaker"],
                "line_ids": list(beat["line_ids"]),
                "start_s": beat["start_s"],
                "dur_s": beat["dur_s"],
                "clip_count": beat.get("clip_count", 0),
                "clip_durations_s": list(beat.get("clip_durations_s", [])),
            })
            shot_clip_count += beat.get("clip_count", 0)
        shots_out.append({
            "shot_id": shot["shot_id"],
            "scene_id": shot["scene_id"],
            "start_s": shot["start_s"],
            "dur_s": shot["dur_s"],
            "speakers": list(shot.get("speakers", [])),
            "beat_count": len(shot["beats"]),
            "beats": beats_out,
            "clip_count": shot_clip_count,
        })

    total_beats = sum(len(s["beats"]) for s in shots)

    # ----- Phase 5: silence master WAV -----
    audio_dir = out_dir / "audio"
    master_wav_path = audio_dir / "master.wav"
    if write_audio:
        generate_silence_wav(master_wav_path, total_episode_dur)

    # ----- Phase 6: assemble augmented ledger -----
    ledger = {
        "schema_version": SCHEMA_VERSION,
        "episode_id": src.get("episode_id"),
        "source_ledger": str(source_ledger_path),
        "source_schema": src.get("schema_version"),
        "build_params": {
            "target_shot_dur": target_shot_dur,
            "clip_base_dur": clip_base_dur,
            "wps": wps,
            "pad_per_line": pad_per_line,
            "humo_fps": HUMO_FPS,
            "humo_min_frames": HUMO_MIN_FRAMES,
            "humo_max_frames": HUMO_MAX_FRAMES,
        },
        "total_episode_dur_s": total_episode_dur,
        "total_dialogue_lines": len(enriched_lines),
        "total_shots": len(shots),
        "total_beats": total_beats,
        "total_clips": total_clips,
        "cast": cast,
        "scenes": scenes_in,
        "shots": shots_out,
        "lines": clip_lines,
        "source_lines": enriched_lines,
        "sfx": src.get("sfx") or [],
        "music": src.get("music") or [],
        "clips": [],
        "final_audio_path": str(master_wav_path) if write_audio else None,
        "final_video_path": None,
    }

    # ----- Phase 7: write outputs -----
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = out_dir / "ledger.json"
    with open(ledger_path, "w", encoding="utf-8") as f:
        json.dump(ledger, f, indent=2, ensure_ascii=False)
        f.write("\n")
    meta = {
        "schema_version": SCHEMA_VERSION,
        "source_ledger": str(source_ledger_path),
        "out_dir": str(out_dir),
        "ledger_path": str(ledger_path),
        "master_wav_path": str(master_wav_path) if write_audio else None,
        "totals": {
            "episode_dur_s": total_episode_dur,
            "lines": len(enriched_lines),
            "shots": len(shots),
            "beats": total_beats,
            "clips": total_clips,
        },
        "params": ledger["build_params"],
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return ledger


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a silent-audio test episode from a FULL ledger.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--source-ledger",
        type=Path,
        required=True,
        help="Path to a prior FULL pipeline ledger with cast + lines (text).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Where to write the augmented ledger + silence audio.",
    )
    p.add_argument(
        "--target-shot-dur",
        type=float,
        default=DEFAULT_TARGET_SHOT_DUR,
        help="Close a shot when accumulated line dur reaches this (s).",
    )
    p.add_argument(
        "--clip-base-dur",
        type=float,
        default=DEFAULT_CLIP_BASE_DUR,
        help="Base logical-clip duration for the fill rule (s).",
    )
    p.add_argument(
        "--wps",
        type=float,
        default=DEFAULT_WPS,
        help="Words-per-second rate for duration estimation.",
    )
    p.add_argument(
        "--pad-per-line",
        type=float,
        default=DEFAULT_PAD_PER_LINE,
        help="Per-line padding (s) added to word-count estimate.",
    )
    p.add_argument(
        "--no-audio",
        action="store_true",
        help="Skip silence WAV generation; ledger only.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.source_ledger.exists():
        print(f"FATAL: source ledger not found: {args.source_ledger}", file=sys.stderr)
        return 2
    ledger = build_silent_test_episode(
        args.source_ledger,
        args.out_dir,
        target_shot_dur=args.target_shot_dur,
        clip_base_dur=args.clip_base_dur,
        wps=args.wps,
        pad_per_line=args.pad_per_line,
        write_audio=not args.no_audio,
    )
    print(
        f"[ok] {args.out_dir}\n"
        f"     episode_id          = {ledger.get('episode_id')}\n"
        f"     total_episode_dur_s = {ledger['total_episode_dur_s']:.2f}\n"
        f"     source lines        = {ledger['total_dialogue_lines']}\n"
        f"     shots               = {ledger['total_shots']}\n"
        f"     beats (single-spkr) = {ledger['total_beats']}\n"
        f"     logical clips       = {ledger['total_clips']}\n"
        f"     wall-clock @ 6 min/clip cold, ~5 min warm = "
        f"~{(6 + (ledger['total_clips'] - 1) * 5) / 60:.1f} h"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
