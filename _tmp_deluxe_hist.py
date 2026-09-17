"""How many :8000 jobs ran today and how they ended."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE = "http://127.0.0.1:8000"
OBS = Path(r"C:\Users\jeffr\Documents\ComfyUI\output\otr\obs")
EP = Path(r"C:\Users\jeffr\Documents\ComfyUI\output\otr\episodes")
KNOWN = "b65b0811-f04a-4af6-8636-23c6af3515d9"


def get(path: str):
    with urllib.request.urlopen(BASE + path, timeout=8) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    try:
        q = get("/queue")
        print("QUEUE running=%d pending=%d" % (
            len(q.get("queue_running") or []),
            len(q.get("queue_pending") or [])))
    except Exception as exc:
        print("QUEUE ERR", type(exc).__name__, exc)
        q = None

    try:
        hist = get("/history")
    except Exception as exc:
        print("HISTORY ERR", type(exc).__name__, exc)
        hist = {}

    print("HISTORY n=%d" % len(hist))
    rows = []
    for pid, entry in hist.items():
        if not isinstance(entry, dict):
            continue
        st = entry.get("status") or {}
        meta = entry.get("meta") or {}
        prompt = entry.get("prompt")
        act = None
        profile = None
        if isinstance(prompt, list) and len(prompt) > 2 and isinstance(prompt[2], dict):
            nodes = prompt[2]
        elif isinstance(prompt, dict):
            nodes = prompt
        else:
            nodes = {}
        for node in (nodes or {}).values():
            if not isinstance(node, dict):
                continue
            if node.get("class_type") == "OTR_LedgerScriptWriter":
                act = (node.get("inputs") or {}).get("act_count")
            if node.get("class_type") == "OTR_WorkflowValidator":
                profile = (node.get("inputs") or {}).get("profile_id") or (
                    node.get("inputs") or {}).get("workflow_json_path")
        msgs = st.get("messages") or []
        last = msgs[-1][0] if msgs and isinstance(msgs[-1], list) else None
        rows.append((
            str(pid),
            st.get("status_str"),
            st.get("completed"),
            act,
            profile,
            last,
        ))
    for pid, status, completed, act, profile, last in rows:
        mark = " <-- THIS 1-act" if pid == KNOWN else ""
        print("  %s status=%s completed=%s act=%s profile=%s last=%s%s" % (
            pid[:8], status, completed, act, profile, last, mark))

    print("\nOBS mtimes today (local):")
    if OBS.is_dir():
        items = sorted(OBS.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        for p in items[:12]:
            ts = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            print("  %s  %s" % (ts, p.name[:110]))
    else:
        print("  missing", OBS)

    print("\nEpisode dirs touched today:")
    if EP.is_dir():
        today = []
        for p in EP.iterdir():
            if not p.is_dir():
                continue
            name = p.name
            if "20260916" in name or name.startswith("pending_20260916"):
                today.append(p)
        for p in sorted(today, key=lambda x: x.stat().st_mtime, reverse=True)[:20]:
            ts = datetime.fromtimestamp(p.stat().st_mtime).strftime("%H:%M:%S")
            clips = list((p / "video").glob("*.mp4")) if (p / "video").is_dir() else []
            print("  %s  %s  video_mp4=%d" % (ts, p.name, len(clips)))


if __name__ == "__main__":
    main()
