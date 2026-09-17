"""Poll the two mystory cloud prompt_ids until both terminal."""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STATUS = ROOT / "tmp" / "mystory_cloud_queue.json"
URL = "http://127.0.0.1:8000"


def _get(path: str):
    with urllib.request.urlopen(URL + path, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _one(prompt_id: str) -> str:
    try:
        hist = _get("/history/" + prompt_id)
    except Exception:
        return "unreachable"
    row = hist.get(prompt_id) or {}
    status = (row.get("status") or {})
    if status.get("completed"):
        if status.get("status_str") == "success":
            return "SUCCESS"
        return "FAIL"
    return str(status.get("status_str") or "pending")


def main() -> int:
    payload = json.loads(STATUS.read_text(encoding="utf-8"))
    deluxe = payload["deluxe_prompt_id"]
    low = payload["low_prompt_id"]
    start = time.time()
    while True:
        q = _get("/queue")
        running = len(q.get("queue_running") or [])
        pending = len(q.get("queue_pending") or [])
        ds = _one(deluxe)
        ls = _one(low)
        print(
            "[follow] t=%ds deluxe=%s low=%s queue r=%d p=%d"
            % (int(time.time() - start), ds, ls, running, pending),
            flush=True,
        )
        if ds in ("SUCCESS", "FAIL") and ls in ("SUCCESS", "FAIL"):
            print("[follow] DONE deluxe=%s low=%s" % (ds, ls), flush=True)
            return 0 if ds == "SUCCESS" and ls == "SUCCESS" else 1
        time.sleep(20)


if __name__ == "__main__":
    sys.exit(main())
