"""Confirm the 3-act obs publish. No keys."""
from __future__ import annotations

from pathlib import Path

OBS = Path(r"C:\Users\jeffr\Documents\ComfyUI\output\otr\obs")
LOG = Path(
    r"C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio"
    r"\tmp\comfy_cpu_8000_night.log"
)


def main() -> int:
    if LOG.exists():
        text = LOG.read_text(encoding="utf-8", errors="replace")
        print(f"log_has_obs_publish_ok={'obs_publish OK' in text}")
        print(f"log_has_fanout={'cloud fan-out' in text}")
    mp4s = sorted(OBS.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    print(f"obs_mp4={len(mp4s)}")
    for p in mp4s[:4]:
        print(f"{p.name} bytes={p.stat().st_size} mtime={p.stat().st_mtime}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
