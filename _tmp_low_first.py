"""Leave only the cheap-cloud 1-act on :8000. Interrupt deluxe, drop audio-in."""
from __future__ import annotations

import json
import urllib.request

URL = "http://127.0.0.1:8000"
DELUXE = "eb2feb3f-ae06-4baf-a05a-9ec7ead264fe"
LOW = "07a4d59b-6912-4603-8e3a-3df977692c3d"
AUDIO = "9b1bc295-de04-4228-8af0-fab70e14a7b5"


def _get(path: str):
    with urllib.request.urlopen(URL + path, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _post(path: str, payload: dict) -> int:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        URL + path, data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return int(resp.status)


def _ids(rows) -> list[str]:
    out = []
    for row in rows or []:
        if isinstance(row, list) and len(row) > 1:
            out.append(str(row[1]))
        elif isinstance(row, dict) and row.get("prompt_id"):
            out.append(str(row["prompt_id"]))
    return out


def main() -> int:
    q = _get("/queue")
    running = _ids(q.get("queue_running"))
    pending = _ids(q.get("queue_pending"))
    print("[low-first] before running=%s pending=%s" % (running, pending), flush=True)
    drop = [pid for pid in pending if pid in {DELUXE, AUDIO}]
    if drop:
        _post("/queue", {"delete": drop})
        print("[low-first] deleted pending %s" % drop, flush=True)
    if running and running[0] != LOW:
        _post("/interrupt", {})
        print("[low-first] interrupted %s" % running[0], flush=True)
    q2 = _get("/queue")
    print(
        "[low-first] after running=%s pending=%s"
        % (_ids(q2.get("queue_running")), _ids(q2.get("queue_pending"))),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
