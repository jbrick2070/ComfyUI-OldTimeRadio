import json
import urllib.request

BASE = "http://127.0.0.1:8000"


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=8) as resp:
        return json.loads(resp.read().decode("utf-8"))


q = get("/queue")
print("running", len(q.get("queue_running") or []))
print("pending", len(q.get("queue_pending") or []))
for label, items in (
    ("running", q.get("queue_running") or []),
    ("pending", q.get("queue_pending") or []),
):
    for item in items:
        print(label, "type", type(item).__name__, "len", len(item) if isinstance(item, list) else None)
        if isinstance(item, list):
            for i, part in enumerate(item[:4]):
                print(f"  [{i}] {type(part).__name__}", end=" ")
                if isinstance(part, dict):
                    print("keys", list(part)[:12], "prompt_id", part.get("prompt_id"))
                elif isinstance(part, str):
                    print(part[:80])
                else:
                    print(repr(part)[:120])
