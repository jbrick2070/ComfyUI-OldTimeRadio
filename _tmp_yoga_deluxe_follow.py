"""Follow yoga deluxe 1-act on :8000. Write status. Do not submit."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

PID = "1aeb2b01-7718-4500-81f7-4c80135b31c2"
BASE = "http://127.0.0.1:8000"
REPO = Path(__file__).resolve().parent
LOG = REPO / "tmp" / "comfy_cpu_8000_night.log"
STATUS = REPO / "tmp" / "yoga_deluxe_follow.json"
OBS = Path(r"C:\Users\jeffr\Documents\ComfyUI\output\otr\obs")
EP = Path(r"C:\Users\jeffr\Documents\ComfyUI\output\otr\episodes")

MARKERS = (
    "obs_publish OK",
    "Prompt executed",
    "FoleyStemError",
    "content_filtered",
    "CONTENT_REFUSED",
    "budget_floor",
    "CloudMediaBudgetExceeded",
    "EXCEPTION",
    "Traceback",
    "RESULT SUCCESS",
    "RESULT FAIL",
)


def get(path: str):
    with urllib.request.urlopen(BASE + path, timeout=8) as resp:
        return json.loads(resp.read().decode("utf-8"))


def queue_ids(q):
    ids = []
    for item in (q.get("queue_running") or []) + (q.get("queue_pending") or []):
        if not isinstance(item, list) or len(item) < 2:
            continue
        part = item[1]
        if isinstance(part, str):
            ids.append(part)
        elif isinstance(part, dict):
            ids.append(part.get("prompt_id") or part.get("prompt"))
    return ids


def newest_obs():
    if not OBS.is_dir():
        return []
    rows = []
    for p in OBS.iterdir():
        if p.is_file() and p.suffix.lower() in {".mp4", ".mkv", ".webm"}:
            if "ltx25_foley_mux_smoke" in p.name:
                continue
            rows.append((p.stat().st_mtime, p.name, p.stat().st_size))
    rows.sort(reverse=True)
    return rows[:8]


def newest_ep():
    if not EP.is_dir():
        return []
    rows = []
    for p in EP.iterdir():
        if not p.is_dir():
            continue
        m = p.stat().st_mtime
        if m >= time.time() - 8 * 3600:
            rows.append((m, p.name))
    rows.sort(reverse=True)
    return rows[:6]


def last_log_hits(n=80):
    if not LOG.is_file():
        return {"size": 0, "hits": []}
    text = LOG.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    hits = [ln for ln in lines if any(m in ln for m in MARKERS)]
    return {
        "size": LOG.stat().st_size,
        "tail": lines[-12:],
        "hits": hits[-n:],
    }


def once():
    row = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pid": PID,
    }
    try:
        q = get("/queue")
        ids = queue_ids(q)
        row["queue_running"] = len(q.get("queue_running") or [])
        row["queue_pending"] = len(q.get("queue_pending") or [])
        row["queue_has_pid"] = PID in ids
        row["queue_ids"] = ids
    except Exception as exc:
        row["queue_err"] = f"{type(exc).__name__}: {exc}"
    try:
        hist = get("/history/" + PID)
        entry = hist.get(PID) or {}
        status = entry.get("status") or {}
        row["hist_status"] = status.get("status_str")
        row["hist_completed"] = status.get("completed")
        msgs = status.get("messages") or []
        if msgs:
            last = msgs[-1]
            row["hist_last"] = last[0] if isinstance(last, list) else last
    except urllib.error.HTTPError as exc:
        row["hist"] = f"HTTP {exc.code}"
    except Exception as exc:
        row["hist"] = f"{type(exc).__name__}: {exc}"
    row["log"] = last_log_hits()
    row["obs"] = newest_obs()
    row["episodes"] = newest_ep()
    STATUS.write_text(json.dumps(row, indent=2), encoding="utf-8")
    print(
        f"[{row['ts']}] q={row.get('queue_running')}/{row.get('queue_pending')} "
        f"has={row.get('queue_has_pid')} hist={row.get('hist_status')} "
        f"obs={len(row.get('obs') or [])} log={row['log']['size']}",
        flush=True,
    )
    if row.get("hist_completed") or (
        row.get("hist_status") in {"success", "error"} and not row.get("queue_has_pid")
    ):
        print("DONE", row.get("hist_status"), flush=True)
        return True
    return False


def main():
    loops = int(os.environ.get("YOGA_FOLLOW_LOOPS", "180"))
    sleep_s = int(os.environ.get("YOGA_FOLLOW_SLEEP", "20"))
    for i in range(loops):
        try:
            if once():
                return 0
        except Exception as exc:
            print("follow_err", type(exc).__name__, exc, flush=True)
        time.sleep(sleep_s)
    print("TIMEOUT still running", flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
