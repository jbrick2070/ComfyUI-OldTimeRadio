"""Print :8188 queue snapshot for the Bark A/B."""
from __future__ import annotations

import json
import urllib.request

with urllib.request.urlopen("http://127.0.0.1:8188/queue", timeout=5) as resp:
    data = json.loads(resp.read().decode("utf-8"))
running = data.get("queue_running") or []
pending = data.get("queue_pending") or []
print("running", len(running))
for item in running:
    # [number, prompt_id, prompt, extra]
    pid = item[1] if len(item) > 1 else item
    print("  run", pid)
print("pending", len(pending))
for item in pending:
    pid = item[1] if len(item) > 1 else item
    print("  wait", pid)
