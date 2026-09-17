"""Dump /queue raw and try delete+interrupt until deluxe is gone."""
from __future__ import annotations

import json
import time
import urllib.request

URL = "http://127.0.0.1:8000"
DELUXE = "eb2feb3f-ae06-4baf-a05a-9ec7ead264fe"
LOW = "07a4d59b-6912-4603-8e3a-3df977692c3d"


def _get(path: str):
    with urllib.request.urlopen(URL + path, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _post(path: str, payload: dict | None) -> tuple[int, str]:
    data = b"" if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        URL + path, data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return int(resp.status), resp.read().decode("utf-8")[:300]


def main() -> int:
    q = _get("/queue")
    print("RAW running keys", type(q.get("queue_running")), flush=True)
    if q.get("queue_running"):
        print("RAW row0 types", [type(x).__name__ for x in q["queue_running"][0][:4]], flush=True)
        print("RAW row0[1]", q["queue_running"][0][1] if len(q["queue_running"][0]) > 1 else None, flush=True)
    st, body = _post("/interrupt", {})
    print("interrupt", st, body, flush=True)
    st, body = _post("/queue", {"delete": [DELUXE]})
    print("delete deluxe", st, body, flush=True)
    time.sleep(2)
    q2 = _get("/queue")
    print("after running", [r[1] for r in (q2.get("queue_running") or []) if isinstance(r, list) and len(r) > 1], flush=True)
    print("after pending", [r[1] for r in (q2.get("queue_pending") or []) if isinstance(r, list) and len(r) > 1], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
