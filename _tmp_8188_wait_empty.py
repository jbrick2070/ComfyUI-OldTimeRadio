"""Wait until :8188 queue is empty after interrupt."""
from __future__ import annotations

import json
import time
import urllib.request

deadline = time.time() + 60
while time.time() < deadline:
    with urllib.request.urlopen("http://127.0.0.1:8188/queue", timeout=5) as resp:
        q = json.loads(resp.read().decode("utf-8"))
    running = len(q.get("queue_running") or [])
    pending = len(q.get("queue_pending") or [])
    print(f"running={running} pending={pending}")
    if running == 0 and pending == 0:
        raise SystemExit(0)
    time.sleep(2)
print("TIMEOUT still busy")
raise SystemExit(2)
