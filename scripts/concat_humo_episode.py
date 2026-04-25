#!/usr/bin/env python3
"""HuMo episode concat / overlay (the second half of Pattern B).

Takes a directory of HuMo clip MP4s + a master episode WAV and produces a
final episode MP4. Two modes:

  --mode concat
      Sequentially concatenate clips back-to-back, then replace the
      audio track with the master WAV. Best when clips already cover
      the timeline (one HuMo window per timeline slice).

  --mode overlay
      Composite clips ONTO a base video track at their start_s/dur_s
      positions (per ledger timing). Gaps are filled by the base
      track. Best when HuMo clips are sparse cutaways and the
      remainder is atmospheric base footage.

Usage (concat mode, full-coverage episode):
  python scripts/concat_humo_episode.py \\
      --mode concat \\
      --clips-dir output/old_time_radio/<title>_humo \\
      --master-wav output/old_time_radio/<title>.wav \\
      --out output/old_time_radio/<title>_humo_full.mp4

Usage (overlay mode, sparse cutaways):
  python scripts/concat_humo_episode.py \\
      --mode overlay \\
      --clips-dir output/old_time_radio/<title>_humo \\
      --ledger output/old_time_radio/<title>_ledger.json \\
      --base-video output/old_time_radio/<title>.mp4 \\
      --out output/old_time_radio/<title>_humo_overlay.mp4

Concat ordering: clips are sorted by filename (e.g. l001.mp4, l002.mp4 …).
For overlay, ledger.lines[].line_id maps to the matching <line_id>.mp4.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Concat HuMo clips into an episode MP4.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--mode",
        choices=["concat", "overlay"],
        default="concat",
    )
    p.add_argument(
        "--clips-dir",
        type=Path,
        required=True,
        help="Directory containing per-line HuMo MP4s (named lXXX.mp4).",
    )
    p.add_argument(
        "--master-wav",
        type=Path,
        default=None,
        help="Master episode WAV. Required for concat mode.",
    )
    p.add_argument(
        "--ledger",
        type=Path,
        default=None,
        help="Ledger JSON with per-line timing. Required for overlay mode.",
    )
    p.add_argument(
        "--base-video",
        type=Path,
        default=None,
        help="Base video track for overlay mode (e.g. existing waveform "
        "+ treatment HUD video).",
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output episode MP4 path.",
    )
    p.add_argument(
        "--ffmpeg",
        type=str,
        default="ffmpeg",
        help="ffmpeg binary path (defaults to ffmpeg on PATH).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
    )
    return p.parse_args()


def list_clips(clips_dir: Path) -> list[Path]:
    """Return clips sorted by filename (lexical = line_id order)."""
    clips = sorted(clips_dir.glob("*.mp4"))
    if not clips:
        raise SystemExit(f"FATAL: no MP4s in {clips_dir}")
    return clips


def run_ffmpeg(cmd: list[str], dry_run: bool = False) -> None:
    print("[ffmpeg]", " ".join(f'"{c}"' if " " in c else c for c in cmd))
    if dry_run:
        return
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("---ffmpeg stderr---", file=sys.stderr)
        print(res.stderr[-4000:], file=sys.stderr)
        raise SystemExit(f"ffmpeg failed (exit {res.returncode})")


def concat_clips(args: argparse.Namespace) -> None:
    """Sequentially concat clips back-to-back, replace audio with master."""
    if not args.master_wav:
        raise SystemExit("--master-wav required for concat mode")

    clips = list_clips(args.clips_dir)
    print(f"[concat] {len(clips)} clip(s) found")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: build a concat demuxer file list.
    list_file = args.out.with_suffix(".concat.txt")
    with open(list_file, "w", encoding="utf-8") as f:
        for c in clips:
            # ffmpeg concat demuxer needs forward slashes / quotes.
            posix = str(c.resolve()).replace("\\", "/")
            f.write(f"file '{posix}'\n")

    # Step 2: concat video (video-only, drop the per-clip audio), then
    # mux against the master WAV. Single ffmpeg pass.
    cmd = [
        args.ffmpeg,
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_file),
        "-i", str(args.master_wav),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        "-movflags", "+faststart",
        str(args.out),
    ]
    run_ffmpeg(cmd, args.dry_run)
    print(f"[done] {args.out}")
    if not args.dry_run:
        list_file.unlink(missing_ok=True)


def load_ledger_lines(ledger_path: Path) -> list[dict]:
    with open(ledger_path, "r", encoding="utf-8") as f:
        led = json.load(f)
    return led.get("lines") or []


def overlay_clips(args: argparse.Namespace) -> None:
    """Composite each clip onto base video at line.start_s / line.dur_s.
    Gaps are filled by the base. Audio comes from the base."""
    if not args.base_video:
        raise SystemExit("--base-video required for overlay mode")
    if not args.ledger:
        raise SystemExit("--ledger required for overlay mode")

    lines = load_ledger_lines(args.ledger)
    clips_dir = args.clips_dir

    # Build (clip_path, start_s, dur_s) tuples in line order.
    placements: list[tuple[Path, float, float]] = []
    for ln in lines:
        line_id = str(ln.get("line_id") or "")
        clip_path = clips_dir / f"{line_id}.mp4"
        if not clip_path.exists():
            continue
        start_s = ln.get("start_s")
        dur_s = ln.get("dur_s")
        if start_s in (None, "") or dur_s in (None, ""):
            continue
        placements.append((clip_path, float(start_s), float(dur_s)))

    if not placements:
        raise SystemExit("FATAL: no clips matched ledger lines with timing")

    print(f"[overlay] {len(placements)} clip(s) placed onto {args.base_video}")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Build ffmpeg filter graph:
    #   [0:v]  base video        (input 0)
    #   [1:v], [2:v], ... [Nv:v] HuMo clips (inputs 1..N)
    #   chained overlays with enable='between(t, start, start+dur)'
    inputs = ["-i", str(args.base_video)]
    for clip_path, _start, _dur in placements:
        inputs += ["-i", str(clip_path)]

    chain: list[str] = []
    prev_label = "0:v"
    for idx, (_clip, start_s, dur_s) in enumerate(placements):
        clip_label = f"{idx + 1}:v"
        next_label = f"v{idx + 1}"
        end_s = start_s + dur_s
        # Center the clip on the base. Adjust if base resolution differs.
        chain.append(
            f"[{prev_label}][{clip_label}]"
            f"overlay=x=(W-w)/2:y=(H-h)/2:"
            f"enable='between(t,{start_s:.3f},{end_s:.3f})'"
            f"[{next_label}]"
        )
        prev_label = next_label
    final_label = prev_label

    filter_graph = ";".join(chain)

    cmd = (
        [args.ffmpeg, "-y"]
        + inputs
        + [
            "-filter_complex", filter_graph,
            "-map", f"[{final_label}]",
            "-map", "0:a?",  # base track audio if present
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "20",
            "-c:a", "aac",
            "-b:a", "192k",
            "-movflags", "+faststart",
            str(args.out),
        ]
    )
    run_ffmpeg(cmd, args.dry_run)
    print(f"[done] {args.out}")


def main() -> int:
    args = parse_args()
    if args.mode == "concat":
        concat_clips(args)
    else:
        overlay_clips(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
