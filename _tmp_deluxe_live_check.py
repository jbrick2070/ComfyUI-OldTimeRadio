"""Confirm the recycled :8000 loaded Sol/Luna and is running 3e6e56fb."""
from __future__ import annotations

import json
import urllib.request

URL = "http://127.0.0.1:8000"


def _get(path: str):
    with urllib.request.urlopen(URL + path, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def main() -> int:
    q = _get("/queue")
    running = q.get("queue_running") or []
    pending = q.get("queue_pending") or []
    print("running=%d pending=%d" % (len(running), len(pending)))
    info = _get("/object_info")
    writer = info.get("OTR_LedgerScriptWriter") or {}
    req = ((writer.get("input") or {}).get("required") or {})
    opt = ((writer.get("input") or {}).get("optional") or {})
    slot_a = req.get("comfy_slot_a_model") or opt.get("comfy_slot_a_model") or []
    choices = slot_a[0] if slot_a and isinstance(slot_a[0], list) else slot_a
    needed = (
        "openai/gpt-5.6-sol",
        "openai/gpt-5.6-luna",
        "anthropic/claude-sonnet-5",
        "openai/gpt-5.5",
    )
    for name in needed:
        print("catalog %s = %s" % (name, "yes" if name in choices else "NO"))
    hist = _get("/history/3e6e56fb-98d0-469a-8f09-75556175ddbc")
    row = hist.get("3e6e56fb-98d0-469a-8f09-75556175ddbc") or hist
    status = ((row.get("status") or {}) if isinstance(row, dict) else {})
    print("history status=%s completed=%s" % (
        status.get("status_str"), status.get("completed")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
