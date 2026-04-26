"""
render_episode_concat.py -- stitch HuMo per-clip mp4s into one episode mp4.

Reads a silent_test_episode ledger, finds each clip's HuMo mp4 (rendered
by render_humo_batch.py), concatenates them in beat order via ffmpeg,
muxes the master audio episode WAV (or mp4) over the top, and writes
the final episode mp4 plus a .vtt subtitle sidecar.

Pattern matches render_flux_batch.py / render_humo_batch.py: pure stdlib
+ ffmpeg subprocess. No ComfyUI required for this step.

Inputs (from the silent_test ledger):
  - cast[]          -> cast roster for the closing card
  - lines[]         -> beat order, line_id, speaker, text, start_s, dur_s
  - episode_id      -> output naming
  - total_episode_dur_s

Discovery rule for HuMo clip mp4s:
  output/otr_videos/<episode_id>/humo_<line_id>_*.mp4
  (matches render_humo_batch.py's save_prefix scheme)

Audio source:
  --audio-mp4  the SignalLostVideo / EpisodeAssembler output mp4
                (audio extracted, muxed over the concatenated HuMo video)
  OR
  --audio-wav  the SceneSequencer master WAV directly

Outputs:
  output/otr_videos/<episode_id>/<episode_id>.mp4
  output/otr_videos/<episode_id>/<episode_id>.vtt   (subtitle sidecar)

Usage:
  python scripts/render_episode_concat.py \\
      --ledger   output/old_time_radio/silent_test_<episode>/ledger.json \\
      --audio-mp4 output/old_time_radio/<episode>.mp4 \\
      --comfy-output-dir C:/Users/jeffr/Documents/ComfyUI/output

Subtitles: every dialogue line becomes one .vtt cue with start_s,
start_s + dur_s, and the line text. Atmospheric (kind=ambient) beats
are skipped from subtitles by default (no spoken content).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Concat HuMo per-clip mp4s + mux audio + write .vtt",
    )
    p.add_argument("--ledger", required=True, type=Path,
                   help="Path to silent_test_<episode>/ledger.json")
    p.add_argument("--comfy-output-dir", type=Path,
                   default=Path(r"C:/Users/jeffr/Documents/ComfyUI/output"),
                   help="ComfyUI output directory (HuMo clips live in "
                        "<this>/otr_videos/<episode_id>/)")
    audio = p.add_mutually_exclusive_group(required=True)
    audio.add_argument("--audio-mp4", type=Path,
                       help="Path to the audio episode mp4 (audio extracted)")
    audio.add_argument("--audio-wav", type=Path,
                       help="Path to the master WAV directly")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Where to write final mp4 + .vtt. Defaults to "
                        "<comfy-output-dir>/otr_videos/<episode_id>/")
    p.add_argument("--ffmpeg", default="ffmpeg",
                   help="ffmpeg binary path (default: 'ffmpeg' on PATH)")
    p.add_argument("--include-ambient-subs", action="store_true",
                   help="Include atmospheric (kind=ambient) beats as cues "
                        "marked [ambient] in the .vtt. Off by default.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print plan, do not run ffmpeg.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Ledger inspection
# ---------------------------------------------------------------------------

def load_ledger(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def find_humo_clip(comfy_out: Path, episode_id: str, line_id: str) -> Path | None:
    """Locate the HuMo mp4 for a given line. Picks the newest match if
    multiple runs left siblings (ComfyUI appends _NNNNN before the ext)."""
    folder = comfy_out / "otr_videos" / episode_id
    candidates = sorted(
        folder.glob(f"humo_{line_id}_*.mp4"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


# ---------------------------------------------------------------------------
# Subtitles (.vtt)
# ---------------------------------------------------------------------------

def _ts(seconds: float) -> str:
    """Format seconds as WebVTT timestamp HH:MM:SS.mmm."""
    if seconds < 0:
        seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds - h * 3600 - m * 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def write_vtt(ledger: dict, out_path: Path, *,
              include_ambient: bool = False) -> int:
    """Write a WebVTT sidecar from ledger.lines[]. Returns cue count."""
    cues = []
    cast_by_id = {c.get("char_id"): c.get("name", "")
                  for c in (ledger.get("cast") or [])}
    n = 0
    for ln in ledger.get("lines", []) or []:
        kind = ln.get("kind") or "dialogue"
        if kind == "ambient" and not include_ambient:
            continue
        start_s = ln.get("start_s")
        dur_s = ln.get("dur_s")
        if start_s is None or dur_s is None:
            continue
        speaker = cast_by_id.get(ln.get("char_id"), "")
        text = (ln.get("text") or "").strip()
        if not text:
            continue
        cue_label = f"<v {speaker}>" if speaker else ""
        end_s = float(start_s) + float(dur_s)
        cues.append(
            f"{n + 1}\n{_ts(float(start_s))} --> {_ts(end_s)}\n"
            f"{cue_label}{text}\n"
        )
        n += 1
    body = "WEBVTT\n\n" + "\n".join(cues)
    out_path.write_text(body, encoding="utf-8")
    return n


# ---------------------------------------------------------------------------
# ffmpeg orchestration
# ---------------------------------------------------------------------------

def write_concat_list(clips: list[Path], list_path: Path) -> None:
    """Write the ffmpeg concat-demuxer file list."""
    lines = []
    for c in clips:
        # ffmpeg concat demuxer needs forward-slash paths and quoted spec
        path_str = str(c).replace("\\", "/").replace("'", r"\'")
        lines.append(f"file '{path_str}'")
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_concat_with_audio(*, clips: list[Path], audio_src: Path,
                          out_mp4: Path, ffmpeg_bin: str,
                          dry_run: bool) -> int:
    """Concat HuMo clips, mux audio_src as the audio track. Audio src may
    be a WAV or any file ffmpeg can decode an audio stream from."""
    list_path = out_mp4.with_suffix(".concat.txt")
    write_concat_list(clips, list_path)

    # First pass: concat HuMo clips into a video-only intermediate. We
    # discard the per-clip audio (HuMo's TTS dub) because the master
    # audio episode is the canonical mix.
    intermediate = out_mp4.with_suffix(".video_only.mp4")
    concat_cmd = [
        ffmpeg_bin, "-y", "-loglevel", "warning",
        "-f", "concat", "-safe", "0", "-i", str(list_path),
        "-c:v", "copy", "-an",
        str(intermediate),
    ]

    # Second pass: mux master audio over the concatenated video.
    mux_cmd = [
        ffmpeg_bin, "-y", "-loglevel", "warning",
        "-i", str(intermediate),
        "-i", str(audio_src),
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-map", "0:v:0", "-map", "1:a:0",
        "-shortest",
        str(out_mp4),
    ]

    print("[ffmpeg] concat:")
    print("  " + " ".join(concat_cmd))
    print("[ffmpeg] mux audio:")
    print("  " + " ".join(mux_cmd))

    if dry_run:
        return 0

    rc = subprocess.call(concat_cmd)
    if rc != 0:
        print(f"FATAL: ffmpeg concat failed rc={rc}", file=sys.stderr)
        return rc
    rc = subprocess.call(mux_cmd)
    if rc != 0:
        print(f"FATAL: ffmpeg mux failed rc={rc}", file=sys.stderr)
        return rc
    # Cleanup intermediates
    for p in (list_path, intermediate):
        try:
            p.unlink()
        except OSError:
            pass
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    if not args.ledger.exists():
        print(f"FATAL: ledger not found: {args.ledger}", file=sys.stderr)
        return 2
    audio_src = args.audio_mp4 or args.audio_wav
    if not audio_src.exists():
        print(f"FATAL: audio source not found: {audio_src}", file=sys.stderr)
        return 2
    if not shutil.which(args.ffmpeg) and not Path(args.ffmpeg).exists():
        print(f"FATAL: ffmpeg not found at {args.ffmpeg!r}", file=sys.stderr)
        return 3

    led = load_ledger(args.ledger)
    episode_id = led.get("episode_id") or args.ledger.parent.name
    out_dir = args.out_dir or (
        args.comfy_output_dir / "otr_videos" / episode_id
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    out_mp4 = out_dir / f"{episode_id}.mp4"
    out_vtt = out_dir / f"{episode_id}.vtt"

    # ---- collect HuMo clips in beat order ----
    lines = led.get("lines", []) or []
    clips: list[Path] = []
    missing: list[str] = []
    for ln in lines:
        line_id = ln.get("line_id")
        if not line_id:
            continue
        clip = find_humo_clip(args.comfy_output_dir, episode_id, line_id)
        if clip:
            clips.append(clip)
        else:
            missing.append(line_id)

    print(f"[ledger] episode_id={episode_id}")
    print(f"[ledger] lines={len(lines)} (cast={len(led.get('cast') or [])}, "
          f"shots={len(led.get('shots') or [])}, "
          f"beats={len(led.get('beats') or [])})")
    print(f"[clips]  found {len(clips)} HuMo mp4(s)  missing={len(missing)}")
    if missing:
        print(f"  missing line_ids: {missing[:10]}"
              f"{' ...' if len(missing) > 10 else ''}")

    # ---- write subtitles regardless (cheap) ----
    n_cues = write_vtt(led, out_vtt,
                       include_ambient=args.include_ambient_subs)
    print(f"[vtt]    wrote {n_cues} cues -> {out_vtt}")

    if not clips:
        print("FATAL: no HuMo clips found; cannot concat.", file=sys.stderr)
        print("       Check that render_humo_batch.py finished and that "
              "save_prefix routes to otr_videos/<episode_id>/humo_<line_id>",
              file=sys.stderr)
        return 4

    # ---- concat + mux ----
    rc = run_concat_with_audio(
        clips=clips, audio_src=audio_src,
        out_mp4=out_mp4, ffmpeg_bin=args.ffmpeg,
        dry_run=args.dry_run,
    )
    if rc != 0:
        return rc
    if args.dry_run:
        print(f"[dry-run] would write {out_mp4}")
        return 0
    print(f"[done]   {out_mp4}")
    print(f"[done]   {out_vtt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
