"""smoke_watcher.py -- single-pass status check for a running smoke episode.

Reads:
  * runtime log tail (last ~40 lines from otr_runtime.log)
  * /history/<prompt_id> status
  * /queue snapshot (running, pending counts)
  * LibreHardwareMonitor http://localhost:8085/data.json -> GPU memory used

Usage:
    python scripts/smoke_watcher.py <prompt_id>

Built on scripts/otr_api.py. Caller invokes on a cadence to track progress
without holding a long-lived shell.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import requests

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from otr_api import COMFYUI_URL, queue_snapshot  # noqa: E402

OTR_RUNTIME_LOG = os.path.join(
    os.path.dirname(_HERE),
    "otr_runtime.log",
)
LHM_URL = "http://localhost:8085/data.json"


def tail_log(path: str, n: int = 40) -> list[str]:
    if not os.path.isfile(path):
        return [f"[runtime log not found: {path}]"]
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            f.seek(0, 2)
            end = f.tell()
            f.seek(max(0, end - 24000))
            lines = f.readlines()
            return [ln.rstrip() for ln in lines[-n:]]
    except OSError as e:
        return [f"[log read error: {e}]"]


def history_status(prompt_id: str) -> dict[str, Any]:
    try:
        h = requests.get(f"{COMFYUI_URL}/history/{prompt_id}", timeout=10).json()
        if prompt_id in h:
            return h[prompt_id].get("status", {}) or {}
        return {"status_str": "not_yet_in_history"}
    except Exception as e:
        return {"error": str(e)[:200]}


def vram_gb_from_lhm() -> float | None:
    """Walk LHM data.json for GPU memory used (NOT system RAM).

    LHM trees are nested. We look specifically under hardware nodes whose
    name or hardwareType contains "GpuNvidia" / "GPU" before matching the
    "Memory Used" / "GPU Memory Used" leaf, so we don't pick up the box's
    system RAM "Memory Used" reading by mistake.
    """
    try:
        data = requests.get(LHM_URL, timeout=5).json()
    except Exception:
        return None

    def parse_value(v: str) -> float | None:
        try:
            num, unit = v.split(" ")
            num_f = float(num.replace(",", "."))
            if unit.upper() == "MB":
                num_f /= 1024.0
            return num_f
        except Exception:
            return None

    def walk(n: dict, in_gpu: bool) -> float | None:
        text = (n.get("Text") or "").lower()
        # Detect entering a GPU subtree.
        gpu_now = in_gpu or any(
            k in text for k in ("gpunvidia", "gpu", "geforce", "rtx ")
        )
        # Match GPU memory leaves only inside a GPU subtree.
        if gpu_now and any(
            t in text for t in (
                "gpu memory used",
                "memory used",
                "dedicated memory used",
                "vram used",
            )
        ):
            v = parse_value(n.get("Value", "") or "")
            if v is not None:
                return v
        for c in n.get("Children", []) or []:
            r = walk(c, gpu_now)
            if r is not None:
                return r
        return None

    return walk(data, in_gpu=False)


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: smoke_watcher.py <prompt_id>")
        return 2
    prompt_id = sys.argv[1]

    running, pending = queue_snapshot()
    status = history_status(prompt_id)
    vram = vram_gb_from_lhm()
    tail = tail_log(OTR_RUNTIME_LOG, n=40)

    print(f"prompt_id     = {prompt_id}")
    print(f"queue         = running={running} pending={pending}")
    print(f"history       = {json.dumps(status, default=str)[:300]}")
    print(f"vram_used_gb  = {vram if vram is not None else 'n/a'}")
    print("-- runtime tail --")
    for ln in tail:
        print(ln)
    return 0


if __name__ == "__main__":
    sys.exit(main())
