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

Discovery rule for HuMo clip mp4s (BUG-LOCAL-075):
  output/otr/videos/<episode_id>/<line_id>.mp4              (canonical post-move)
  output/otr/videos/<episode_id>/humo_<line_id>_*.mp4       (raw SaveVideo, partial-run recovery)
  output/otr_videos/<episode_id>/...                        (mid-day legacy, back-compat only)

Audio source:
  --audio-mp4  the SignalLostVideo / EpisodeAssembler output mp4
                (audio extracted, muxed over the concatenated HuMo video)
  OR
  --audio-wav  the SceneSequencer master WAV directly

Outputs (BUG-LOCAL-084 -- broadcast tree, sibling of output/otr/):
  output/otr/episodes/<episode_id>.mp4
  output/otr/episodes/<episode_id>.vtt   (subtitle sidecar)

Usage:
  python scripts/render_episode_concat.py \\
      --ledger   output/old_time_radio/silent_test_<episode>/ledger.json \\
      --audio-mp4 output/old_time_radio/<episode>.mp4 \\
      --comfy-output-dir output  # or omit to use OTR_OUTPUT_DIR / auto-resolve

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

# Make ``nodes._otr_paths`` importable when running this script directly
# from the repo root (no ComfyUI process required).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from nodes._otr_paths import comfy_output_dir  # noqa: E402


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
                   default=comfy_output_dir(),
                   help="ComfyUI output directory (HuMo clips live in "
                        "<this>/otr/videos/<episode_id>/ as <line_id>.mp4 "
                        "or humo_<line_id>_*.mp4; mid-day legacy fallback "
                        "to <this>/otr_videos/<episode_id>/). Override "
                        "globally via the OTR_OUTPUT_DIR env var.")
    audio = p.add_mutually_exclusive_group(required=True)
    audio.add_argument("--audio-mp4", type=Path,
                       help="Path to the audio episode mp4 (audio extracted)")
    audio.add_argument("--audio-wav", type=Path,
                       help="Path to the master WAV directly")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Where to write final mp4 + .vtt. Defaults to "
                        "<comfy-output-dir>/otr/episodes/ "
                        "(canonical post BUG-LOCAL-084 -- sibling of otr/, "
                        "OBS-streaming-safe).")
    p.add_argument("--ffmpeg", default="ffmpeg",
                   help="ffmpeg binary path (default: 'ffmpeg' on PATH)")
    p.add_argument("--include-ambient-subs", action="store_true",
                   help="Include atmospheric (kind=ambient) beats as cues "
                        "marked [ambient] in the .vtt. Off by default.")
    p.add_argument("--trim", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Pre-trim each HuMo clip to its ledger line's "
                        "dur_s before concat so the video timeline matches "
                        "proc-gen audio mix frame-accurately. Off via "
                        "--no-trim if you want raw 4n+1 HuMo durations. "
                        "(default: on)")
    p.add_argument("--embed-subs", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Embed the .vtt as a mov_text subtitle track "
                        "inside the mp4 container so players (VLC, mpv, "
                        "modern browsers) get a toggleable CC track "
                        "without needing the sidecar file. Sidecar .vtt "
                        "is still written either way. "
                        "Off via --no-embed-subs. (default: on)")
    p.add_argument("--subs-lang", default="eng",
                   help="Language tag for the embedded subtitle stream "
                        "(ISO 639-2/3-letter, default 'eng').")
    p.add_argument("--dry-run", action="store_true",
                   help="Print plan, do not run ffmpeg.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Ledger inspection
# ---------------------------------------------------------------------------

def load_ledger(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def find_humo_clip(comfy_out: Path, episode_id: str, line_id: str) -> Path | None:
    """Locate the HuMo mp4 for a given line.

    Two filename forms exist on disk:
      1. ``<line_id>.mp4`` -- the canonical post-move form written by
         render_humo_batch.py's move block (line 1053). This is what
         every successful run produces and the form concat must
         resolve against.
      2. ``humo_<line_id>_NNNNN_.mp4`` -- ComfyUI's SaveVideo raw
         output before render_humo_batch's move-rename. Only seen on
         disk when render_humo_batch crashed mid-run before the move
         could happen (see BUG-LOCAL-074 / BUG-LOCAL-075). Kept as a
         fallback so partial-run recovery still works.

    Two folder roots are searched:
      - ``output/otr/videos/<id>/`` -- canonical OTR location.
      - ``output/otr_videos/<id>/`` -- mid-day legacy, retained for
        back-compat with episodes rendered before the path rename.

    Picks the newest match across all roots and forms.
    """
    folders = [
        comfy_out / "otr" / "videos" / episode_id,
        comfy_out / "otr_videos" / episode_id,   # mid-day legacy
    ]
    candidates: list[Path] = []
    for folder in folders:
        if not folder.exists():
            continue
        # Canonical post-move form first; raw SaveVideo form second.
        canonical = folder / f"{line_id}.mp4"
        if canonical.exists():
            candidates.append(canonical)
        candidates.extend(folder.glob(f"humo_{line_id}_*.mp4"))
    candidates = sorted(candidates,
                        key=lambda p: p.stat().st_mtime, reverse=True)
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


def is_clip_readable(src: Path, ffmpeg_bin: str = "ffmpeg") -> bool:
    """Return True if ffprobe can read the mp4's format duration.

    Guards against partially-written / corrupt HuMo clips that would
    silently break the concat. Edge case: if render_humo_batch.py is
    racing with concat (rare but possible), one of its in-flight mp4s
    may not yet have a valid moov atom. ffprobe rejects those cleanly.
    """
    # Derive ffprobe path from the ffmpeg path (they ship together).
    ffprobe = ffmpeg_bin.replace("ffmpeg", "ffprobe")
    try:
        rc = subprocess.call(
            [ffprobe, "-v", "error",
             "-show_entries", "format=duration",
             "-of", "csv=p=0",
             str(src)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return rc == 0
    except FileNotFoundError:
        # ffprobe missing -- skip the check rather than blocking concat.
        return True


def trim_clip_to_dur(*, src: Path, dst: Path, dur_s: float,
                     ffmpeg_bin: str, dry_run: bool) -> int:
    """Trim a HuMo clip to exactly dur_s seconds via ffmpeg stream copy.

    HuMo renders at 4n+1 frame counts (Wan VAE temporal compression), so
    a line whose actual dialogue is 3.4 s gets a 97-frame / 3.88 s clip.
    Without this trim, concat'd video runs slightly long vs proc-gen
    audio mix and ffmpeg --shortest clips the tail. This pass keeps the
    video exactly aligned with the ledger timeline.

    Stream copy avoids re-encoding (fast, lossless). The trim point may
    snap to the nearest keyframe (~1-frame tolerance at 25 fps), which
    is visually invisible.
    """
    cmd = [
        ffmpeg_bin, "-y", "-loglevel", "error",
        "-i", str(src),
        "-t", f"{dur_s:.3f}",
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        str(dst),
    ]
    if dry_run:
        return 0
    return subprocess.call(cmd)


def run_concat_with_audio(*, clips: list[Path], audio_src: Path,
                          out_mp4: Path, ffmpeg_bin: str,
                          dry_run: bool,
                          embed_subs_vtt: Path | None = None,
                          subs_lang: str = "eng") -> int:
    """Concat HuMo clips, mux audio_src as the audio track. Audio src may
    be a WAV or any file ffmpeg can decode an audio stream from.

    If embed_subs_vtt is provided, also embeds the .vtt as a mov_text
    subtitle stream inside the output mp4 (language tag = subs_lang).
    Players that honor mp4 subtitle tracks (VLC, mpv, QuickTime, modern
    web browsers via <video><track>) will show a toggleable CC option
    without needing the sidecar file alongside.
    """
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

    # Second pass: mux master audio over the concatenated video. If the
    # caller passed a .vtt, also fold it in as an mp4 mov_text subtitle
    # stream.
    mux_cmd = [
        ffmpeg_bin, "-y", "-loglevel", "warning",
        "-i", str(intermediate),
        "-i", str(audio_src),
    ]
    if embed_subs_vtt is not None:
        mux_cmd.extend(["-i", str(embed_subs_vtt)])
    mux_cmd.extend([
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        # ITU BS.1770 broadcast loudness target (per peer review
        # 2026-04-26). Bark + Kokoro + MusicGen + AudioGen levels
        # currently drift across the mix; -loudnorm two-pass would be
        # cleaner but the single-pass form is fine for delivery and
        # adds zero round-trips to the existing mux call.
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
    ])
    if embed_subs_vtt is not None:
        mux_cmd.extend(["-c:s", "mov_text"])
    mux_cmd.extend(["-map", "0:v:0", "-map", "1:a:0"])
    if embed_subs_vtt is not None:
        mux_cmd.extend([
            "-map", "2:s:0",
            f"-metadata:s:s:0", f"language={subs_lang}",
            f"-metadata:s:s:0", "title=English",
        ])
    mux_cmd.extend([
        "-shortest",
        str(out_mp4),
    ])

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
    # Final episode mp4 lives at output/otr/episodes/<episode_id>.mp4
    # -- FLAT layout (no per-episode subfolder). OBS's directory_sorter
    # reads this single dir sorted by mtime / filename and queues each
    # finished episode in turn. Per-line clip pieces stay separated at
    # output/otr/videos/<episode_id>/ so OBS never sees them.
    # History: output/episodes_for_obs/<id>/  (BUG-LOCAL-084 era,
    # sibling of otr/) -> output/otr/episodes/<id>/  (nested under otr/,
    # earlier on 2026-05-02) -> output/otr/episodes/  (flat, current).
    out_dir = args.out_dir or (
        args.comfy_output_dir / "otr" / "episodes"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    out_mp4 = out_dir / f"{episode_id}.mp4"
    out_vtt = out_dir / f"{episode_id}.vtt"

    # ---- collect HuMo clips in beat order, paired with their ledger
    #      line's dur_s so the trim pass knows the target length.
    #      ffprobe-validate each clip to drop any that are partially
    #      written or corrupt (per peer review 2026-04-26 — guards
    #      against silent broken-episode failures). ----
    lines = led.get("lines", []) or []
    paired: list[tuple[Path, float | None, str]] = []   # (clip, dur_s, line_id)
    clips: list[Path] = []
    missing: list[str] = []
    corrupt: list[str] = []
    for ln in lines:
        line_id = ln.get("line_id")
        if not line_id:
            continue
        clip = find_humo_clip(args.comfy_output_dir, episode_id, line_id)
        if not clip:
            missing.append(line_id)
            continue
        if not is_clip_readable(clip, args.ffmpeg):
            corrupt.append(f"{line_id} ({clip.name})")
            continue
        dur_s = ln.get("dur_s")
        paired.append((clip, dur_s, line_id))
        clips.append(clip)

    print(f"[ledger] episode_id={episode_id}")
    print(f"[ledger] lines={len(lines)} (cast={len(led.get('cast') or [])}, "
          f"shots={len(led.get('shots') or [])}, "
          f"beats={len(led.get('beats') or [])})")
    print(f"[clips]  found {len(clips)} HuMo mp4(s)  "
          f"missing={len(missing)}  corrupt={len(corrupt)}")
    if missing:
        print(f"  missing line_ids: {missing[:10]}"
              f"{' ...' if len(missing) > 10 else ''}")
    if corrupt:
        print(f"  corrupt clips dropped (ffprobe failed): "
              f"{corrupt[:10]}"
              f"{' ...' if len(corrupt) > 10 else ''}")

    # ---- write subtitles regardless (cheap) ----
    n_cues = write_vtt(led, out_vtt,
                       include_ambient=args.include_ambient_subs)
    print(f"[vtt]    wrote {n_cues} cues -> {out_vtt}")

    if not clips:
        print("FATAL: no HuMo clips found; cannot concat.", file=sys.stderr)
        print("       Check that render_humo_batch.py finished and that "
              "clips landed at otr/videos/<episode_id>/<line_id>.mp4 "
              "(canonical) or otr/videos/<episode_id>/humo_<line_id>_*.mp4 "
              "(raw SaveVideo).",
              file=sys.stderr)
        return 4

    # ---- pre-trim pass (default ON): trim each HuMo clip to the
    #      ledger line's dur_s so concat'd video matches proc-gen audio
    #      timeline frame-accurately. Lines without dur_s in the ledger
    #      pass through untrimmed. ----
    trimmed_clips: list[Path] = []
    trim_dir: Path | None = None
    n_trimmed = 0
    n_passthrough = 0
    if args.trim:
        trim_dir = out_dir / ".trim_temp"
        trim_dir.mkdir(parents=True, exist_ok=True)
        for src, dur_s, line_id in paired:
            if dur_s is None or float(dur_s) <= 0:
                trimmed_clips.append(src)
                n_passthrough += 1
                continue
            dst = trim_dir / f"trim_{line_id}_{src.name}"
            rc = trim_clip_to_dur(
                src=src, dst=dst, dur_s=float(dur_s),
                ffmpeg_bin=args.ffmpeg, dry_run=args.dry_run,
            )
            if rc != 0 and not args.dry_run:
                print(f"WARN: trim failed for {src.name} (rc={rc}); "
                      f"using untrimmed source", file=sys.stderr)
                trimmed_clips.append(src)
                n_passthrough += 1
                continue
            trimmed_clips.append(dst if not args.dry_run else src)
            n_trimmed += 1
        print(f"[trim]   {n_trimmed} clip(s) trimmed to dur_s, "
              f"{n_passthrough} passthrough (no dur_s)")
    else:
        trimmed_clips = clips
        print("[trim]   skipped (--no-trim); using raw 4n+1 HuMo durations")

    # ---- concat + mux (with optional embedded soft subs) ----
    embed_vtt = out_vtt if (args.embed_subs and n_cues > 0) else None
    rc = run_concat_with_audio(
        clips=trimmed_clips, audio_src=audio_src,
        out_mp4=out_mp4, ffmpeg_bin=args.ffmpeg,
        dry_run=args.dry_run,
        embed_subs_vtt=embed_vtt,
        subs_lang=args.subs_lang,
    )
    if embed_vtt is not None and not args.dry_run:
        print(f"[subs]   embedded {n_cues} cue(s) as mov_text "
              f"(language={args.subs_lang}) inside {out_mp4.name}")
    if rc != 0:
        return rc
    if args.dry_run:
        print(f"[dry-run] would write {out_mp4}")
        return 0
    # Clean up trim temp dir if we created one (ignore errors -- the
    # individual mp4s are no longer needed after concat).
    if trim_dir is not None and trim_dir.exists():
        for f in trim_dir.glob("trim_*"):
            try:
                f.unlink()
            except OSError:
                pass
        try:
            trim_dir.rmdir()
        except OSError:
            pass
    print(f"[done]   {out_mp4}")
    print(f"[done]   {out_vtt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
