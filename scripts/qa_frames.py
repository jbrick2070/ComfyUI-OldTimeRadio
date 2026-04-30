r"""qa_frames.py -- intelligent frame extractor for OTR QA passes.

Extracts JPEG frames from the final episode .mp4 at smart intervals:
  - Always grabs frames at every clip boundary in ledger.clips[]
    (where HuMo concatenation tends to drift / glitch)
  - Always grabs a mid-clip "stable" frame so I can compare baseline
  - Light coverage of the radio-bookend FLUX stills
  - Hard caps at 60 frames total (manageable to Read in one pass)

Output:
    <ComfyUI/output>/otr/qa_frames/<episode_id>/
        manifest.json                          metadata for every frame
        frame_001_t0001.50_clip0_pre.jpg       1.5s, clip 0, pre-boundary
        frame_002_t0002.00_clip0_start.jpg     2.0s, clip 0 start
        ...

Each frame's filename embeds the timestamp + which clip + role
(start / mid / pre / post). The manifest.json is the source of truth
for any analysis pass that needs to map frame -> clip -> ledger line.

Usage:
    C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe ^
        scripts\qa_frames.py --ledger PATH\TO\<episode_id>_ledger.json

    # or pull the most recent ledger automatically:
    python scripts\qa_frames.py --latest

Returns exit 0 on success. Failures during individual frame extracts
log a warning but never abort the run -- partial coverage is better
than nothing.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [qa_frames] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("qa_frames")

REPO_ROOT = Path(__file__).resolve().parent.parent
COMFY_OUT = REPO_ROOT.parent.parent / "output"
LEDGER_DIR = COMFY_OUT / "otr" / "audio"

# Frame budget. Tuned so I can Read every frame in a QA pass without
# blowing context. Frames per clip = max(2, ceil(MAX / clip_count)).
_DEFAULT_MAX_FRAMES = 60


def _find_latest_ledger() -> Optional[Path]:
    """Return the most recent *_ledger.json in the OTR audio dir."""
    if not LEDGER_DIR.exists():
        return None
    cands = list(LEDGER_DIR.glob("*_ledger.json"))
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def _ffmpeg_path() -> str:
    """Locate ffmpeg. Same fallback chain the rest of OTR uses."""
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    # Common Windows install spots
    for p in (
        Path(r"C:\ffmpeg\bin\ffmpeg.exe"),
        Path(r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"),
    ):
        if p.exists():
            return str(p)
    raise RuntimeError("ffmpeg not found on PATH or in known install dirs")


def _extract_frame(
    ffmpeg: str,
    video: Path,
    t_seconds: float,
    out_path: Path,
    width: int = 768,
) -> bool:
    """Extract one frame at exactly t_seconds. Returns True on success.

    Width default 768 is the QA-readable sweet spot -- big enough that
    I can resolve facial features and on-screen text, small enough that
    a 60-frame batch doesn't blow context.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",                # overwrite
        "-ss", f"{t_seconds:.3f}",
        "-i", str(video),
        "-frames:v", "1",
        "-vf", f"scale={width}:-2",
        "-q:v", "3",         # JPEG quality 1-31; 3 = high
        str(out_path),
    ]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )
        if result.returncode != 0:
            log.warning(
                "ffmpeg returncode=%d at t=%.2fs: %s",
                result.returncode,
                t_seconds,
                (result.stderr or b"").decode(errors="replace")[:200],
            )
            return False
        return out_path.exists()
    except subprocess.TimeoutExpired:
        log.warning("ffmpeg timeout at t=%.2fs", t_seconds)
        return False
    except Exception as exc:  # noqa: BLE001
        log.warning("ffmpeg failed at t=%.2fs: %s", t_seconds, exc)
        return False


def _build_extraction_plan(
    clips: list[dict],
    duration_s: float,
    max_frames: int = 60,
) -> list[dict]:
    """Decide which timestamps to extract and what each frame represents.

    Strategy:
      - Each clip gets at minimum: 1 frame at start+0.2s, 1 frame at
        mid-point, 1 frame at end-0.2s. The +/-0.2s offsets dodge
        ffmpeg keyframe-snap weirdness right at the boundary.
      - If the budget allows, additional frames at clip boundary - 1s
        and boundary + 1s to catch concatenation glitches.
      - Always one intro frame at 0.5s (FLUX bookend) and one outro
        frame at duration_s - 0.5s.

    Returns a list of {t: float, role: str, clip_idx: int|None} dicts
    in time order. Capped at max_frames.
    """
    plan: list[dict] = []
    plan.append({"t": 0.5, "role": "intro_bookend", "clip_idx": None})

    if clips:
        # Frame budget for clip frames after subtracting the 2 bookends.
        per_clip = max(2, (max_frames - 2) // max(1, len(clips)))
        for idx, c in enumerate(clips):
            start = float(c.get("start_s", 0.0))
            dur = float(c.get("dur_s", 0.0))
            if dur <= 0:
                continue
            end = start + dur
            mid = start + dur / 2.0
            # Always: clip start and mid
            plan.append({"t": start + 0.2, "role": f"clip{idx}_start", "clip_idx": idx})
            if per_clip >= 2:
                plan.append({"t": mid, "role": f"clip{idx}_mid", "clip_idx": idx})
            if per_clip >= 3:
                plan.append({"t": end - 0.2, "role": f"clip{idx}_end", "clip_idx": idx})
            # Boundary stress frames if budget left
            if per_clip >= 5 and idx > 0:
                plan.append({"t": max(0.0, start - 1.0), "role": f"clip{idx}_pre", "clip_idx": idx})
                plan.append({"t": min(duration_s, start + 1.0), "role": f"clip{idx}_post", "clip_idx": idx})

    plan.append({"t": max(0.0, duration_s - 0.5), "role": "outro_bookend", "clip_idx": None})

    # Sort, dedupe by rounded timestamp, cap.
    plan.sort(key=lambda x: x["t"])
    seen_buckets: set[int] = set()
    unique: list[dict] = []
    for item in plan:
        bucket = int(round(item["t"] * 4))   # 0.25s rounding bucket
        if bucket in seen_buckets:
            continue
        seen_buckets.add(bucket)
        unique.append(item)
    if len(unique) > max_frames:
        log.info(
            "plan had %d frames, trimming to max_frames=%d "
            "(keeping bookends + every Nth clip frame)",
            len(unique), max_frames,
        )
        # Always keep first + last (bookends), then evenly stride
        if len(unique) >= 2:
            head, tail = unique[0], unique[-1]
            middle = unique[1:-1]
            stride = max(1, len(middle) // (max_frames - 2))
            unique = [head] + middle[::stride][: max_frames - 2] + [tail]
    return unique


def _probe_duration(ffmpeg: str, video: Path) -> float:
    """Use ffprobe (sibling of ffmpeg) to get the video's duration in seconds."""
    ffprobe = "ffprobe"
    if ffmpeg != "ffmpeg":
        candidate = Path(ffmpeg).with_name("ffprobe.exe")
        if candidate.exists():
            ffprobe = str(candidate)
    cmd = [
        ffprobe,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video),
    ]
    try:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=True, timeout=15,
        )
        return float(result.stdout.decode().strip() or 0.0)
    except Exception as exc:  # noqa: BLE001
        log.warning("ffprobe failed (%s) - falling back to ledger sum", exc)
        return 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--ledger", type=Path, help="Path to <episode_id>_ledger.json")
    g.add_argument("--latest", action="store_true", help="Use most recent ledger")
    ap.add_argument(
        "--max-frames", type=int, default=_DEFAULT_MAX_FRAMES,
        help=f"Hard cap on extracted frames (default {_DEFAULT_MAX_FRAMES})",
    )
    ap.add_argument(
        "--width", type=int, default=768,
        help="Output JPEG width in pixels (default 768)",
    )
    args = ap.parse_args()

    ledger_path = args.ledger if args.ledger else _find_latest_ledger()
    if not ledger_path or not ledger_path.exists():
        log.error("ledger not found: %s", ledger_path)
        return 2
    log.info("ledger: %s", ledger_path)

    try:
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.error("failed to parse ledger: %s", exc)
        return 2

    episode_id = ledger.get("episode_id") or ledger_path.stem.replace("_ledger", "")
    final_video = ledger.get("final_video_path")
    if not final_video:
        log.error("ledger has no final_video_path -- has the run finished?")
        return 2
    video = Path(final_video)
    if not video.exists():
        log.error("final_video_path does not exist on disk: %s", video)
        return 2

    ffmpeg = _ffmpeg_path()
    duration_s = _probe_duration(ffmpeg, video)
    if duration_s <= 0:
        # Fallback: take the max end_s across clips
        clips = ledger.get("clips") or []
        duration_s = max(
            (float(c.get("start_s", 0)) + float(c.get("dur_s", 0)) for c in clips),
            default=0.0,
        )
    if duration_s <= 0:
        log.error("could not determine video duration; aborting")
        return 2
    log.info("video: %s  duration=%.2fs", video.name, duration_s)

    clips = ledger.get("clips") or []
    log.info("clips in ledger: %d", len(clips))

    plan = _build_extraction_plan(clips, duration_s, max_frames=args.max_frames)
    log.info("extraction plan: %d frames", len(plan))

    out_root = COMFY_OUT / "otr" / "qa_frames" / episode_id
    out_root.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    success_count = 0
    for i, item in enumerate(plan, start=1):
        t = float(item["t"])
        fname = f"frame_{i:03d}_t{t:07.2f}_{item['role']}.jpg"
        fpath = out_root / fname
        if _extract_frame(ffmpeg, video, t, fpath, width=args.width):
            success_count += 1
            manifest.append({
                "index":     i,
                "filename":  fname,
                "t_s":       round(t, 3),
                "role":      item["role"],
                "clip_idx":  item["clip_idx"],
                "exists":    True,
            })
        else:
            manifest.append({
                "index":     i,
                "filename":  fname,
                "t_s":       round(t, 3),
                "role":      item["role"],
                "clip_idx":  item["clip_idx"],
                "exists":    False,
            })

    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "episode_id":  episode_id,
                "video_path":  str(video),
                "ledger_path": str(ledger_path),
                "duration_s":  round(duration_s, 3),
                "clip_count":  len(clips),
                "frame_count": success_count,
                "frames":      manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    log.info(
        "wrote %d/%d frames to %s",
        success_count, len(plan), out_root,
    )
    log.info("manifest: %s", manifest_path)
    return 0 if success_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
