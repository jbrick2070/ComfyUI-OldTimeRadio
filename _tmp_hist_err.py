"""Print Comfy history error for a prompt_id."""
from __future__ import annotations

import json
import sys
import urllib.request

URL = "http://127.0.0.1:8000"
pid = sys.argv[1]
with urllib.request.urlopen(URL + "/history/" + pid, timeout=20) as resp:
    hist = json.loads(resp.read().decode("utf-8"))
row = hist.get(pid) or {}
status = row.get("status") or {}
print("status_str=", status.get("status_str"))
print("completed=", status.get("completed"))
msgs = status.get("messages") or []
for m in msgs[-8:]:
    kind = m[0] if isinstance(m, list) and m else m
    body = m[1] if isinstance(m, list) and len(m) > 1 else {}
    if kind in ("execution_error", "execution_interrupted"):
        print("KIND", kind)
        for k in (
            "node_type", "node_id", "exception_type", "exception_message",
        ):
            if k in body:
                print(" ", k, "=", body.get(k))
        tb = body.get("traceback") or []
        if isinstance(tb, list):
            print(" TB", "\n".join(tb[-20:]))
        elif tb:
            print(" TB", str(tb)[-2000:])
outputs = row.get("outputs") or {}
print("output_nodes", list(outputs)[:12])
