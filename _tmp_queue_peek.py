import json
import urllib.request

url = "http://127.0.0.1:8000/queue"
with urllib.request.urlopen(url, timeout=5) as r:
    q = json.loads(r.read().decode("utf-8", "replace"))
running = q.get("queue_running") or []
pending = q.get("queue_pending") or []
print("running", len(running), "pending", len(pending))
for item in running:
    # [number, prompt_id, prompt, extra]
    pid = None
    if isinstance(item, list) and len(item) >= 2:
        pid = item[1]
    print("run_id", pid)
for item in pending:
    pid = None
    if isinstance(item, list) and len(item) >= 2:
        pid = item[1]
    print("pend_id", pid)
