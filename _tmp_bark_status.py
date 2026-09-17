"""What is :8188 doing right now, and does the process have Lumina env?"""
from __future__ import annotations

import json
import urllib.request

ids = (
    "219394d3-96fe-43b4-9bb8-a07c990b3965",
    "c1fd411b-1677-447b-b140-cdf7cf3fbc60",
)

with urllib.request.urlopen("http://127.0.0.1:8188/queue", timeout=5) as resp:
    q = json.loads(resp.read().decode("utf-8"))
print("running", len(q.get("queue_running") or []))
print("pending", len(q.get("queue_pending") or []))

with urllib.request.urlopen("http://127.0.0.1:8188/prompt", timeout=5) as resp:
    print("prompt_endpoint", resp.status)

for pid in ids:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:8188/history/{pid}", timeout=8) as resp:
            hist = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        print(pid, "hist_fail", type(exc).__name__, exc)
        continue
    row = hist.get(pid) or {}
    status = (row.get("status") or {})
    print("===", pid, "===")
    print(" status_str", status.get("status_str"))
    print(" completed", status.get("completed"))
    msgs = status.get("messages") or []
    # last few event names
    names = []
    for item in msgs[-12:]:
        if isinstance(item, (list, tuple)) and item:
            names.append(str(item[0]))
    print(" last_events", names)
