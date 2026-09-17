"""List recent episode dirs and video artifacts under output/otr/episodes."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(r"C:\Users\jeffr\Documents\ComfyUI\output\otr\episodes")
OBS = Path(r"C:\Users\jeffr\Documents\ComfyUI\output\otr\obs")
VIDEO_EXT = {".mp4", ".webm", ".mov", ".mkv"}


def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def _fmt(ts: float) -> str:
    if not ts:
        return "-"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _dir_latest(path: Path) -> float:
    latest = _mtime(path)
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            dirnames[:] = [d for d in dirnames if d not in {".git", "__pycache__"}]
            for name in filenames:
                latest = max(latest, _mtime(Path(dirpath) / name))
    except OSError:
        pass
    return latest


def _summarize(path: Path) -> None:
    n_files = 0
    n_vid = 0
    bytes_vid = 0
    newest_vid = None
    newest_vid_ts = 0.0
    newest_any = None
    newest_any_ts = 0.0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            dirnames[:] = [d for d in dirnames if d not in {".git", "__pycache__"}]
            for name in filenames:
                p = Path(dirpath) / name
                n_files += 1
                ts = _mtime(p)
                if ts >= newest_any_ts:
                    newest_any_ts = ts
                    newest_any = p
                if p.suffix.lower() in VIDEO_EXT:
                    n_vid += 1
                    try:
                        bytes_vid += p.stat().st_size
                    except OSError:
                        pass
                    if ts >= newest_vid_ts:
                        newest_vid_ts = ts
                        newest_vid = p
    except OSError as exc:
        print("WALK_FAIL", path.name, exc)
        return
    rel_vid = str(newest_vid.relative_to(path)) if newest_vid else "-"
    rel_any = str(newest_any.relative_to(path)) if newest_any else "-"
    print(
        "DIR %-48s files=%4d vids=%3d vid_mb=%7.1f latest=%s newest=%s newest_vid=%s"
        % (
            path.name[:48],
            n_files,
            n_vid,
            bytes_vid / (1024 * 1024),
            _fmt(_dir_latest(path)),
            rel_any,
            rel_vid,
        )
    )


def main() -> int:
    print("ROOT", ROOT, "exists", ROOT.is_dir())
    if not ROOT.is_dir():
        return 2
    dirs = [p for p in ROOT.iterdir() if p.is_dir()]
    dirs.sort(key=_dir_latest, reverse=True)
    print("episode_dirs", len(dirs))
    print("--- newest 20 episode dirs ---")
    for p in dirs[:20]:
        _summarize(p)

    print("--- obs top-level files ---")
    if OBS.is_dir():
        files = [p for p in OBS.iterdir() if p.is_file()]
        files.sort(key=_mtime, reverse=True)
        print("obs_files", len(files))
        for p in files[:15]:
            print("OBS", _fmt(_mtime(p)), "%8d" % p.stat().st_size, p.name)
        # also recent mp4 anywhere under obs
        vids = []
        for dirpath, dirnames, filenames in os.walk(OBS):
            dirnames[:] = [d for d in dirnames if d not in {"bark_calibration", "music_recipe_bench", "music_model_bench", "earthsearch"}]
            for name in filenames:
                p = Path(dirpath) / name
                if p.suffix.lower() in VIDEO_EXT:
                    vids.append(p)
        vids.sort(key=_mtime, reverse=True)
        print("obs_videos_n", len(vids))
        for p in vids[:12]:
            print("OBSV", _fmt(_mtime(p)), "%8d" % p.stat().st_size, p.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
