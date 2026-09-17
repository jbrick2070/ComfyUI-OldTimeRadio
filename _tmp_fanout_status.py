"""Queue / port / obs snapshot. No keys printed."""
from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

OBS = Path(r"C:\Users\jeffr\Documents\ComfyUI\output\otr\obs")
LOG = Path(
    r"C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio"
    r"\tmp\comfy_cpu_8000_night.log"
)


def _get(url: str):
    try:
        with urllib.request.urlopen(url, timeout=4) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return resp.status, raw
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def _queue_brief(raw: str) -> str:
    try:
        body = json.loads(raw)
    except json.JSONDecodeError:
        return raw[:200]
    running = body.get("queue_running") or []
    pending = body.get("queue_pending") or []
    run_ids = []
    for item in running:
        if isinstance(item, (list, tuple)) and len(item) > 1:
            run_ids.append(str(item[1]))
    return f"running={len(running)} pending={len(pending)} ids={run_ids}"


def main() -> int:
    for port in (8000, 8188):
        status, raw = _get(f"http://127.0.0.1:{port}/queue")
        if status == 200:
            print(f":{port} UP {_queue_brief(raw)}")
        else:
            print(f":{port} DOWN {raw}")
    if LOG.exists():
        text = LOG.read_text(encoding="utf-8", errors="replace")
        print(f"log_bytes={LOG.stat().st_size} fanout_in_log={'cloud fan-out' in text}")
        for needle in (
            "cloud fan-out",
            "Prompt executed",
            "obs_publish",
            "ERROR",
            "SUCCESS",
        ):
            print(f"  log_has {needle!r}={needle in text}")
    else:
        print("log missing")
    if OBS.exists():
        mp4s = sorted(OBS.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        print(f"obs_mp4={len(mp4s)}")
        for p in mp4s[:6]:
            print(f"  {p.name} {p.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
