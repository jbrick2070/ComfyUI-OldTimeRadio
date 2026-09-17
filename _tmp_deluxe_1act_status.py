import json
import urllib.request

pid = "445ce606-ab44-45a5-b879-165499a82f07"
q = json.loads(urllib.request.urlopen("http://127.0.0.1:8000/queue", timeout=5).read().decode("utf-8", "replace"))
running = q.get("queue_running") or []
pending = q.get("queue_pending") or []
print("running", len(running), "pending", len(pending))
ids = []
for item in running + pending:
    if isinstance(item, list) and len(item) >= 2:
        ids.append(str(item[1]))
print("queue_ids", ids)
try:
    hist = json.loads(urllib.request.urlopen(
        "http://127.0.0.1:8000/history/" + pid, timeout=8).read().decode("utf-8", "replace"))
except Exception as e:
    print("history_err", type(e).__name__, e)
    hist = {}
row = hist.get(pid) or {}
status = ((row.get("status") or {}).get("status_str")
          or (row.get("status") or {}).get("completed"))
print("history_status", status)
print("history_keys", list(row.keys())[:12])
meta = row.get("meta") or {}
print("meta_keys", list(meta.keys())[:12] if isinstance(meta, dict) else type(meta))
