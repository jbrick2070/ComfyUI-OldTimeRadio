"""Follow deluxe 1-act prompt on :8000. Print status, do not submit."""
from __future__ import annotations

import json
import sys
import urllib.request

PID = "b65b0811-f04a-4af6-8636-23c6af3515d9"
BASE = "http://127.0.0.1:8000"

def get(path: str):
    with urllib.request.urlopen(BASE + path, timeout=8) as resp:
        return json.loads(resp.read().decode("utf-8"))

q = get("/queue")
running = q.get("queue_running") or []
pending = q.get("queue_pending") or []
print(f"queue running={len(running)} pending={len(pending)}")
ids = []
for item in running + pending:
    if isinstance(item, list) and len(item) > 1 and isinstance(item[1], dict):
        ids.append(item[1].get("prompt_id") or item[1].get("prompt"))
print("queue ids", ids)

try:
    hist = get("/history/" + PID)
except Exception as exc:
    print("history", type(exc).__name__, exc)
    sys.exit(0)
entry = hist.get(PID) or {}
status = (entry.get("status") or {})
print("history status_str", status.get("status_str"), "completed", status.get("completed"))
msgs = status.get("messages") or []
if msgs:
    last = msgs[-1]
    print("last message", last[0] if isinstance(last, list) else last)
