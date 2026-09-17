import json
import urllib.request

print("HEAD check is shell-side")
try:
    with urllib.request.urlopen("http://127.0.0.1:8000/queue", timeout=3) as r:
        q = json.loads(r.read().decode("utf-8", "replace"))
    running = q.get("queue_running") or []
    pending = q.get("queue_pending") or []
    print("8000_up", True)
    print("running", len(running))
    print("pending", len(pending))
except Exception as e:
    print("8000_up", False)
    print("err", type(e).__name__, e)
try:
    with urllib.request.urlopen("http://127.0.0.1:8188/queue", timeout=2) as r:
        print("8188_up", True)
except Exception:
    print("8188_up", False)
