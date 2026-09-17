"""Peek CPU :8000 queue without touching it."""
from __future__ import annotations

import json
import urllib.request

for url in (
    "http://127.0.0.1:8000/queue",
    "http://127.0.0.1:8000/system_stats",
):
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        print("===", url, "===")
        data = json.loads(body)
        if "queue_running" in data or "queue_pending" in data:
            print("running", len(data.get("queue_running") or []))
            print("pending", len(data.get("queue_pending") or []))
        else:
            print(json.dumps(data, indent=2)[:1500])
    except Exception as exc:
        print("FAIL", url, type(exc).__name__, exc)
