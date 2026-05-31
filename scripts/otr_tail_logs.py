#!/usr/bin/env python
"""scripts/otr_tail_logs.py -- tail the ComfyUI + OTR runtime logs and query the
ComfyUI history/queue API for authoritative job status during an episode run.

Why a python tailer (NOT type / tail / Get-Content):
  - The logs live under OneDrive-virtualized paths that cmd often cannot read;
    the venv python reads them fine.
  - `tail` does not exist on Windows.
  - The logs contain box-drawing / emoji characters that crash the default
    cp1252 decode -- we force UTF-8 with errors="replace".

Gotcha: ComfyUI prints its in-place progress bar to STDERR, so advancing
"NN%|...| s/it" lines that get tagged [error] are NORMAL progress, not failures.
This tool filters pure progress-bar lines out of the highlights section.

Run with the venv python (system `python` is not on PATH):
  C:\\Users\\jeffr\\Documents\\ComfyUI\\.venv\\Scripts\\python.exe scripts\\otr_tail_logs.py [--lines 40]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request

LOGS = [
    (r"C:\Users\jeffr\AppData\Roaming\ComfyUI\logs\comfyui.log",
     "ComfyUI console (s/it, VRAM, FFmpeg)"),
    (r"C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\otr_runtime.log",
     "OTR phase log"),
]
API = "http://127.0.0.1:8000"

# Signals worth surfacing from a wider window than the raw tail.
_KEY = re.compile(
    r"(s/it|it/s|VRAM|allocated|reserved|cuda free|PARSE_FATAL|FFmpeg|ffmpeg|"
    r"guardrail|GUARDRAIL|OOM|out of memory|Traceback|CastingFailedError|"
    r"needs_full_rerun|writer_llm_unload|llm_naming|freeze_verdict)",
    re.I,
)
# Pure in-place progress-bar lines are noise, not signal.
_PROGRESS = re.compile(r"%\|")


def _read_lines(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read().splitlines()
    except FileNotFoundError:
        return None
    except Exception as e:  # noqa: BLE001
        return ["<error reading log: %r>" % e]


def _api(path):
    try:
        with urllib.request.urlopen(API + path, timeout=5) as r:
            return json.load(r)
    except Exception as e:  # noqa: BLE001
        return {"__error__": repr(e)}


def main():
    # Force UTF-8 OUTPUT too: the console/redirect is cp1252 by default and the
    # logs carry box/emoji chars that crash a default-encoded print().
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 -- older / non-reconfigurable stdout
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", type=int, default=40)
    args = ap.parse_args()
    n = args.lines

    for path, label in LOGS:
        print("===== %s =====" % label)
        print(path)
        lines = _read_lines(path)
        if lines is None:
            print("  <missing -- not created yet?>")
            print()
            continue
        for ln in lines[-n:]:
            print(ln)
        hi = [ln for ln in lines[-400:]
              if _KEY.search(ln) and not _PROGRESS.search(ln)]
        if hi:
            print("--- highlights (key signals; progress bars excluded) ---")
            for ln in hi[-20:]:
                print(ln)
        print()

    # Authoritative status: /queue (running + pending) and /history (completed).
    q = _api("/queue")
    if "__error__" in q:
        print("QUEUE API unreachable (ComfyUI down?): %s" % q["__error__"])
    else:
        run = q.get("queue_running", [])
        pend = q.get("queue_pending", [])
        print("QUEUE: running=%d pending=%d" % (len(run), len(pend)))
        for item in run[:3]:
            pid = item[1] if isinstance(item, list) and len(item) > 1 else "?"
            print("  RUNNING prompt_id=%s" % pid)

    h = _api("/history")
    if "__error__" in h:
        print("HISTORY API unreachable: %s" % h["__error__"])
    elif not h:
        print("HISTORY: empty (no prompt finished yet)")
    else:
        print("HISTORY (last 5):")
        for pid, info in list(h.items())[-5:]:
            st = (info or {}).get("status", {})
            print("  %s  status_str=%s  completed=%s" % (
                pid[:12], st.get("status_str", "?"), st.get("completed")))


if __name__ == "__main__":
    main()
