"""Cancel paid cloud runs on :8000. Do not touch :8188 unless it is also queued."""
from __future__ import annotations

import json
import urllib.error
import urllib.request

URLS = {
    "8000": "http://127.0.0.1:8000",
    "8188": "http://127.0.0.1:8188",
}


def _get(url: str, path: str, timeout: float = 8):
    with urllib.request.urlopen(url + path, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def _post(url: str, path: str, payload: dict, timeout: float = 8) -> str:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url + path, data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return "HTTP %s" % resp.status
    except urllib.error.HTTPError as exc:
        return "HTTP %s %s" % (exc.code, exc.read()[:200])
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return "FAIL %s" % exc


def _peek(label: str, url: str) -> tuple[int, int]:
    try:
        q = _get(url, "/queue", timeout=5)
    except Exception as exc:
        print("[%s] DOWN %s" % (label, exc), flush=True)
        return -1, -1
    running = q.get("queue_running") or []
    pending = q.get("queue_pending") or []
    print("[%s] running=%d pending=%d" % (label, len(running), len(pending)), flush=True)
    for row in running[:4]:
        print("[%s] RUN %s" % (label, row[1] if isinstance(row, list) and len(row) > 1 else row), flush=True)
    for row in pending[:6]:
        print("[%s] PEND %s" % (label, row[1] if isinstance(row, list) and len(row) > 1 else row), flush=True)
    return len(running), len(pending)


def main() -> int:
    r8000, p8000 = _peek("8000", URLS["8000"])
    r8188, p8188 = _peek("8188", URLS["8188"])
    if r8000 > 0 or p8000 > 0:
        print("[8000] clear", _post(URLS["8000"], "/queue", {"clear": True}), flush=True)
        print("[8000] interrupt", _post(URLS["8000"], "/interrupt", {}), flush=True)
        print("[8000] interrupt2", _post(URLS["8000"], "/interrupt", {}), flush=True)
    else:
        print("[8000] already idle", flush=True)
    if r8188 > 0 or p8188 > 0:
        print("[8188] HAS JOBS -- not clearing unless paid; listing only", flush=True)
    print("--- after ---", flush=True)
    _peek("8000", URLS["8000"])
    _peek("8188", URLS["8188"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
