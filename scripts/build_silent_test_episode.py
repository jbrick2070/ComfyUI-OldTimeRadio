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
# Line -> shot grouping
# ---------------------------------------------------------------------------

def group_lines_into_shots(
    lines: list[dict[str, Any]],
    *,
    target_shot_dur: float,
) -> list[dict[str, Any]]:
    """Walk lines and close a shot the moment accumulated dur >= target.

    Returns a list of shot dicts:
      { shot_id, scene_id, line_ids[], dur_s, start_s }

    `dur_s` is the sum of the shot's lines (may exceed `target_shot_dur`
    by up to one full line's duration). `start_s` is the shot's start
    timestamp on the master timeline.
    """
    shots: list[dict[str, Any]] = []
    current: dict[str, Any] = {
        "scene_id": None,
        "line_ids": [],
        "dur_s": 0.0,
    }
    timeline_pos = 0.0

    def _close(start_s: float) -> None:
        if not current["line_ids"]:
            return
        sid = f"shot_{len(shots) + 1:03d}"
        shots.append({
            "shot_id": sid,
            "scene_id": current["scene_id"],
            "line_ids": list(current["line_ids"]),
            "dur_s": current["dur_s"],
            "start_s": start_s,
        })

    shot_start = 0.0
    for ln in lines:
        ln_dur = float(ln.get("dur_s") or 0.0)
        ln_scene = ln.get("scene_id")
        # Close on scene boundary (shots never cross scenes)
        if current["scene_id"] is not None and ln_scene != current["scene_id"]:
            _close(shot_start)
            shot_start = timeline_pos
            current = {"scene_id": None, "line_ids": [], "dur_s": 0.0}
        if current["scene_id"] is None:
            current["scene_id"] = ln_scene
        current["line_ids"].append(ln.get("line_id"))
        current["dur_s"] += ln_dur
        timeline_pos += ln_dur
        if current["dur_s"] >= target_shot_dur:
            _close(shot_start)
            shot_start = timeline_pos
            current = {"scene_id": None, "line_ids": [], "dur_s": 0.0}
    _close(shot_start)
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

    # ----- Phase 3: clip-fill each shot, build orchestrator-ready lines -----
    # The orchestrator iterates ledger.lines[]. We REPLACE the original
    # lines list with one entry per LOGICAL CLIP. Speaker per clip is
    # resolved by finding the source line that's "currently being
    # spoken" at the clip's start offset within the shot -- so a shot
    # that spans multiple speakers correctly hands each clip the right
    # speaker (and therefore the right portrait when rendered).
    line_id_to_ln = {ln["line_id"]: ln for ln in enriched_lines}

    def _speaker_at_offset(shot_line_ids: list[str], offset_in_shot: float
                           ) -> tuple[str, str | None]:
        """Find the source line covering `offset_in_shot` within the shot.

        Returns (speaker, char_id). Falls back to the last line if the
        offset overshoots (rounding edge cases).
        """
        cursor = 0.0
        last_ln: dict[str, Any] = {}
        for ln_id in shot_line_ids:
            src = line_id_to_ln.get(ln_id) or {}
            ln_dur = float(src.get("dur_s") or 0.0)
            if cursor + ln_dur > offset_in_shot + 1e-9:
                return (src.get("speaker") or "", src.get("char_id"))
            cursor += ln_dur
            last_ln = src
        return (last_ln.get("speaker") or "", last_ln.get("char_id"))

    clip_lines: list[dict[str, Any]] = []
    total_clips = 0
    for shot in shots:
        shot_dur = shot["dur_s"]
        shot_start = shot["start_s"]
        clip_durs = clip_durations_for_shot(shot_dur, base=clip_base_dur)
        offset = 0.0
        for clip_idx, clip_dur in enumerate(clip_durs):
            speaker, char_id = _speaker_at_offset(shot["line_ids"], offset)
            length = humo_length_for_dur(clip_dur)
            clip_id = f"{shot['shot_id']}_c{clip_idx + 1}"
            clip_lines.append({
                "line_id": clip_id,
                "shot_id": shot["shot_id"],
                "scene_id": shot["scene_id"],
                "speaker": speaker,
                "char_id": char_id,
                "text": "",
                "start_s": shot_start + offset,
                "dur_s": clip_dur,
                "humo_length": length,
                "humo_duration_s": length / HUMO_FPS,
            })
            offset += clip_dur
            total_clips += 1

    # ----- Phase 4: per-shot summary block -----
    shots_out: list[dict[str, Any]] = []
    for shot in shots:
        clip_durs = clip_durations_for_shot(shot["dur_s"], base=clip_base_dur)
        shots_out.append({
            "shot_id": shot["shot_id"],
            "scene_id": shot["scene_id"],
            "start_s": shot["start_s"],
            "dur_s": shot["dur_s"],
            "line_count": len(shot["line_ids"]),
            "line_ids": shot["line_ids"],
            "clip_count": len(clip_durs),
            "clip_durations_s": clip_durs,
        })

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
        f"     logical clips       = {ledger['total_clips']}\n"
        f"     wall-clock @ 6 min/clip cold, ~5 min warm = "
        f"~{(6 + (ledger['total_clips'] - 1) * 5) / 60:.1f} h"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
