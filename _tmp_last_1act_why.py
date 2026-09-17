"""Dump :8000 queue + recent history statuses."""
from __future__ import annotations

import json
import urllib.request

URL = "http://127.0.0.1:8000"
KNOWN = [
    "e4d69d68-eb7d-4408-a37d-4212ed20f0af",
    "79269151-4239-4676-a280-e0ebf9d5b47a",
    "d0c48d3d-0e09-4f74-a64c-e2494660dbc8",
    "b88be265-dbb8-4b7e-94ef-670ad9727cf1",
    "eb2feb3f",
    "4489d0cd",
    "6822d511",
    "7b6d6633",
]


def get(path: str):
    with urllib.request.urlopen(URL + path, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def summarize(pid: str, row: dict) -> None:
    st = row.get("status") or {}
    print("PID", pid)
    print("  status_str", st.get("status_str"), "completed", st.get("completed"))
    for m in (st.get("messages") or [])[-10:]:
        kind = m[0] if isinstance(m, list) and m else type(m).__name__
        body = m[1] if isinstance(m, list) and len(m) > 1 else {}
        if kind in ("execution_error", "execution_interrupted"):
            print("  KIND", kind)
            for k in ("node_type", "node_id", "exception_type", "exception_message"):
                if isinstance(body, dict) and k in body:
                    print("   ", k, "=", body.get(k))
            tb = body.get("traceback") if isinstance(body, dict) else None
            if isinstance(tb, list) and tb:
                print("    TB_TAIL", "".join(tb[-12:]))
            elif tb:
                print("    TB_TAIL", str(tb)[-1500:])
        elif kind in ("execution_start", "execution_success"):
            print("  KIND", kind)


def main() -> int:
    q = get("/queue")
    running = q.get("queue_running") or []
    pending = q.get("queue_pending") or []
    print("running", len(running), "pending", len(pending))
    for row in running:
        print(" RUN", row[1] if isinstance(row, list) and len(row) > 1 else row)
    for row in pending[:8]:
        print(" PEND", row[1] if isinstance(row, list) and len(row) > 1 else row)
    hist = get("/history?max_items=20")
    print("history_n", len(hist))
    for pid, row in hist.items():
        summarize(pid, row if isinstance(row, dict) else {})
        print("---")
    for pid in KNOWN:
        try:
            body = get("/history/" + pid)
        except Exception as exc:
            print("lookup", pid, exc)
            continue
        row = body.get(pid)
        if not row:
            print("lookup", pid, "missing")
            continue
        summarize(pid, row)
        print("---")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
