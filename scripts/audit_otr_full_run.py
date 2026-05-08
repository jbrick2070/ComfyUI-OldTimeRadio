"""scripts/audit_otr_full_run.py

Post-run acceptance audit for an OTR full run. Walks the episode folder
and reports every S3.x acceptance bullet plus boomerang/timebase health.

Usage (PowerShell, with the OTR venv active):
    & C:\\Users\\jeffr\\Documents\\ComfyUI\\.venv\\Scripts\\python.exe `
        scripts\\audit_otr_full_run.py `
        --episode "C:\\Users\\jeffr\\Documents\\ComfyUI\\output\\otr\\episodes\\<ep_id>" `
        --log "C:\\Users\\jeffr\\Documents\\ComfyUI\\user\\comfyui.log"

Exit codes:
    0 - all acceptance bullets pass
    1 - one or more failures (full report still printed)
    2 - script error (cannot read episode folder, etc.)
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

DUR_TOLERANCE_S = 0.25
FAIL_PATTERNS = (
    "Non-monotonous DTS",
    "boomerang FAILED",
    "duration contract VIOLATED",
    "strict_c7=True",
    "derived ledger from .mp4 not found",
    "audio may be truncated",
    # Per-clip exception path inside BatchLTXRender; catches everything
    # the boomerang grep misses.
    re.compile(r"\[BatchLTXRender\] \w+ failed:"),
)


def ffprobe_duration_s(mp4: Path) -> float | None:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error",
             "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1",
             str(mp4)],
            capture_output=True, text=True, check=True,
        )
        return float(r.stdout.strip())
    except Exception:
        return None


def ffprobe_dimensions(mp4: Path) -> tuple[int, int] | None:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error",
             "-select_streams", "v:0",
             "-show_entries", "stream=width,height",
             "-of", "csv=p=0:s=x",
             str(mp4)],
            capture_output=True, text=True, check=True,
        )
        w, h = r.stdout.strip().split("x")
        return int(w), int(h)
    except Exception:
        return None


def ffprobe_timebase(mp4: Path) -> str | None:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error",
             "-select_streams", "v:0",
             "-show_entries", "stream=time_base",
             "-of", "default=noprint_wrappers=1:nokey=1",
             str(mp4)],
            capture_output=True, text=True, check=True,
        )
        return r.stdout.strip()
    except Exception:
        return None


def load_ledger(episode_dir: Path) -> dict | None:
    candidates = [
        episode_dir / "ledger.json",
        episode_dir / "metadata" / "ledger.json",
    ]
    for c in candidates:
        if c.is_file():
            try:
                return json.loads(c.read_text(encoding="utf-8"))
            except Exception:
                continue
    # Fall back to the canonical location used by the OTR pipeline:
    # output/otr/episodes/<ep>/audio/<ep>_ledger.json
    audio_dir = episode_dir / "audio"
    if audio_dir.is_dir():
        for c in audio_dir.glob("*_ledger.json"):
            try:
                return json.loads(c.read_text(encoding="utf-8"))
            except Exception:
                continue
    return None


def audit_clips(episode_dir: Path, ledger: dict) -> tuple[list[str], int, int]:
    """Returns (failure_lines, ltx_count, short_count)."""
    fails: list[str] = []
    videos_dir = episode_dir / "videos"
    if not videos_dir.is_dir():
        fails.append(f"FAIL: missing {videos_dir}")
        return fails, 0, 0

    lines = ledger.get("lines") or []
    clips = ledger.get("clips") or []
    clip_by_id = {c.get("line_id"): c for c in clips if isinstance(c, dict)}

    ltx_count = 0
    static_count = 0
    short_count = 0

    for ln in lines:
        if not isinstance(ln, dict):
            continue
        line_id = ln.get("line_id")
        if not line_id:
            continue
        speaker_role = ln.get("speaker_role", "")
        expected_dur = float(ln.get("dur_s") or 0.0)
        mp4 = videos_dir / f"{line_id}.mp4"

        clip_meta = clip_by_id.get(line_id) or {}
        source_kind = clip_meta.get("source_kind")

        if speaker_role in ("announcer", "music_open", "music_close",
                            "music_inter", "sfx"):
            if source_kind == "ltx":
                ltx_count += 1
            elif source_kind in (None, "static"):
                static_count += 1
            else:
                fails.append(
                    f"FAIL: {line_id} ({speaker_role}) source_kind="
                    f"{source_kind!r}, expected 'ltx' or 'static'"
                )

        if not mp4.is_file():
            # Static-fill lines may not have a per-line mp4 -- skip.
            continue

        actual_dur = ffprobe_duration_s(mp4)
        if actual_dur is None:
            fails.append(f"FAIL: ffprobe could not read {mp4.name}")
            continue
        if expected_dur > 0 and actual_dur < expected_dur - DUR_TOLERANCE_S:
            short_count += 1
            fails.append(
                f"FAIL: {line_id} duration {actual_dur:.2f}s < "
                f"expected {expected_dur:.2f}s (gap "
                f"{expected_dur - actual_dur:.2f}s)"
            )

    print(f"OK: ledger source_kind summary -- "
          f"{ltx_count} ltx, {static_count} static")
    return fails, ltx_count, short_count


def audit_finals(episode_dir: Path) -> list[str]:
    fails: list[str] = []
    pre = next(episode_dir.glob("*.mp4"), None)
    upscaled = next(episode_dir.glob("*_1080p.mp4"), None)

    if pre is None:
        fails.append(f"FAIL: no episode mp4 found in {episode_dir}")
        return fails

    dims = ffprobe_dimensions(pre)
    if dims is None:
        fails.append(f"FAIL: ffprobe dims failed for {pre.name}")
    elif dims not in ((832, 480), (1920, 1080)):
        fails.append(
            f"WARN: {pre.name} dims={dims}, expected (832,480) "
            f"pre-upscale or (1920,1080) post"
        )
    else:
        print(f"OK: {pre.name} dims={dims}")

    if upscaled is not None:
        u_dims = ffprobe_dimensions(upscaled)
        if u_dims != (1920, 1080):
            fails.append(
                f"FAIL: {upscaled.name} dims={u_dims}, expected (1920,1080)"
            )
        else:
            print(f"OK: {upscaled.name} dims=(1920, 1080)")

    return fails


def audit_log(log_path: Path) -> list[str]:
    fails: list[str] = []
    if not log_path.is_file():
        return [f"WARN: log file not found at {log_path}"]
    text = log_path.read_text(encoding="utf-8", errors="replace")
    for pattern in FAIL_PATTERNS:
        if isinstance(pattern, str):
            if pattern in text:
                fails.append(f"FAIL: log contains '{pattern}'")
        else:
            m = pattern.search(text)
            if m:
                fails.append(f"FAIL: log contains '{m.group(0)}'")
    if not fails:
        print(f"OK: log clean of failure strings ({log_path.name})")
    return fails


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True, type=Path)
    ap.add_argument("--log", required=False, type=Path, default=None)
    args = ap.parse_args()

    episode_dir = args.episode
    if not episode_dir.is_dir():
        print(f"ERROR: episode dir not found: {episode_dir}", file=sys.stderr)
        return 2

    ledger = load_ledger(episode_dir)
    if ledger is None:
        print(f"ERROR: cannot load ledger.json from {episode_dir}",
              file=sys.stderr)
        return 2

    print("=" * 64)
    print(f"OTR FULL run audit: {episode_dir}")
    print("=" * 64)

    all_fails: list[str] = []
    clip_fails, ltx_n, short_n = audit_clips(episode_dir, ledger)
    all_fails += clip_fails
    all_fails += audit_finals(episode_dir)
    if args.log:
        all_fails += audit_log(args.log)

    print()
    if all_fails:
        print(f"AUDIT FAILED ({len(all_fails)} issue(s), "
              f"{short_n} short clips, {ltx_n} LTX clips):")
        for f in all_fails:
            print(f"  {f}")
        return 1
    print(f"AUDIT PASSED ({ltx_n} LTX clips, 0 short, 0 log failures)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
