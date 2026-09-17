"""Inspect episode dirs immediately before the_fourth_bowl."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

ROOT = Path(r"C:\Users\jeffr\Documents\ComfyUI\output\otr\episodes")
NAMES = [
    "pending_20260915_223522",
    "pending_20260915_223031",
    "pending_20260915_222733",
    "pending_20260915_222438",
    "pending_20260915_230027",
    "pending_20260915_230227",
    "pending_20260915_230312",
    "pending_20260915_230413",
    "pending_20260915_230541",
    "pending_20260915_230617",
    "pending_20260915_231029",
    "pending_20260915_231335",
    "pending_20260915_231452",
    "signal_lost_the_knife_before_the_portrait_20260915_211250",
    "signal_lost_the_bird_of_dawning_20260915_200747",
]


def _files(p: Path):
    out = []
    if not p.is_dir():
        return out
    for child in p.rglob("*"):
        if child.is_file():
            st = child.stat()
            out.append((st.st_mtime, st.st_size, str(child.relative_to(p))))
    out.sort(reverse=True)
    return out


def _ledger_bits(p: Path) -> None:
    for led in p.rglob("*ledger.json"):
        try:
            data = json.loads(led.read_text(encoding="utf-8"))
        except Exception as exc:
            print("  ledger_unreadable", led.name, exc)
            continue
        meta = data.get("meta") if isinstance(data, dict) else {}
        if not isinstance(meta, dict):
            meta = {}
        title = meta.get("episode_title") or data.get("episode_title")
        lines = data.get("lines") or []
        video = data.get("video") if isinstance(data, dict) else {}
        shots = (video or {}).get("shots") if isinstance(video, dict) else []
        err = meta.get("error") or meta.get("last_error") or data.get("error")
        print("  ledger", led.name)
        print("    title", title)
        print("    n_lines", len(lines), "n_video_shots", len(shots or []))
        sb = meta.get("source_bank")
        if isinstance(meta.get("story_input"), dict):
            sb = meta["story_input"].get("source_bank") or sb
            print("    bank", meta["story_input"].get("source_bank"),
                  "style", meta["story_input"].get("visual_style"))
        else:
            print("    source_bank", sb)
        if err:
            print("    error", str(err)[:400])
        # hunt exception-ish keys
        for k in ("abort_reason", "writer_error", "status"):
            if meta.get(k):
                print("   ", k, str(meta.get(k))[:300])


def main() -> int:
    for name in NAMES:
        p = ROOT / name
        print("====", name, "exists", p.is_dir())
        if not p.is_dir():
            continue
        files = _files(p)
        print("  n_files", len(files))
        for ts, size, rel in files[:12]:
            print("   ", datetime.fromtimestamp(ts).strftime("%H:%M:%S"),
                  "%8d" % size, rel)
        _ledger_bits(p)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
