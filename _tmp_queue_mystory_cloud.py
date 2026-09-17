"""Queue 1-act deluxe Foley + 1-act cheap-cloud, my_story + recur_frac.

Recycles idle :8000 first when the live object_info still lacks
cloud_ltx25_foley_plus. Does not kill a busy queue. Does not touch :8188.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = r"C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe"
URL = "http://127.0.0.1:8000"
DELUXE = ROOT / "workflows" / "variants" / "otr_cloud_deluxe_7act.json"
LOW = ROOT / "workflows" / "variants" / "otr_cloud_low_1act.json"
STATUS = ROOT / "tmp" / "mystory_cloud_queue.json"


def _hydrate_user_env(name: str) -> None:
    if os.environ.get(name):
        return
    try:
        import winreg

        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment")
        try:
            value, _ = winreg.QueryValueEx(key, name)
        finally:
            key.Close()
        if value:
            os.environ[name] = str(value)
    except OSError:
        return


def _get(path: str, timeout: float = 15):
    with urllib.request.urlopen(URL + path, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _queue():
    try:
        return _get("/queue", timeout=5)
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def _engine_blob() -> str:
    info = _get("/object_info/OTR_VideoDirector")
    node = info.get("OTR_VideoDirector") or {}
    inputs = ((node.get("input") or {}).get("required") or {})
    combo = inputs.get("announcer_video_model") or []
    options = combo[0] if combo and isinstance(combo[0], list) else []
    return " ".join(str(x) for x in options)


def _has_foley() -> bool:
    try:
        return "cloud_ltx25_foley_plus" in _engine_blob()
    except Exception:
        return False


def _recycle_if_stale() -> None:
    q = _queue()
    if q is None:
        print("[mystory] :8000 down -- boot", flush=True)
    else:
        running = q.get("queue_running") or []
        pending = q.get("queue_pending") or []
        if running or pending:
            raise SystemExit(
                "BUSY running=%d pending=%d -- not recycling"
                % (len(running), len(pending))
            )
        if _has_foley():
            print("[mystory] :8000 already has cloud_ltx25_foley_plus", flush=True)
            return
        print("[mystory] idle stale object_info -- recycling", flush=True)
    rc = subprocess.call(
        [PY, str(ROOT / "_tmp_recycle_8000_fanout.py")],
        cwd=str(ROOT),
    )
    if rc != 0:
        raise SystemExit("recycle rc=%s" % rc)
    deadline = time.time() + 120
    while time.time() < deadline:
        if _has_foley():
            print("[mystory] Foley engine live on :8000", flush=True)
            return
        time.sleep(2)
    raise SystemExit("recycle finished but cloud_ltx25_foley_plus still missing")


def _submit(label: str, workflow: Path, act_count: str) -> str:
    cmd = [
        PY, str(ROOT / "_tmp_submit_cloud.py"),
        "--workflow", str(workflow),
        "--act-count", act_count,
        "--comfyui-url", URL,
        "--source-bank", "my_story",
        "--visual-style", "recur_frac",
        "--source-ref", "",
        "--no-wait",
    ]
    print("[mystory] submit %s %s" % (label, workflow.name), flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    sys.stdout.write(proc.stdout or "")
    sys.stderr.write(proc.stderr or "")
    if proc.returncode != 0:
        raise SystemExit("submit %s rc=%s" % (label, proc.returncode))
    prompt_id = ""
    for line in (proc.stdout or "").splitlines():
        if "QUEUED prompt_id=" in line:
            prompt_id = line.split("QUEUED prompt_id=", 1)[1].strip()
    if not prompt_id:
        raise SystemExit("submit %s produced no prompt_id" % label)
    return prompt_id


def main() -> int:
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("OTR_ENABLE_COMFY_CREDITS", "1")
    os.environ.setdefault("OTR_CLOUD_VIDEO_FANOUT", "8")
    _hydrate_user_env("OTR_COMFY_API_KEY")
    if not (os.environ.get("OTR_COMFY_API_KEY") or "").strip():
        raise SystemExit("OTR_COMFY_API_KEY missing")
    _recycle_if_stale()
    deluxe_id = _submit("deluxe", DELUXE, "1")
    low_id = _submit("low", LOW, "1")
    payload = {
        "deluxe_prompt_id": deluxe_id,
        "low_prompt_id": low_id,
        "source_bank": "my_story",
        "visual_style": "recur_frac",
        "act_count": "1",
    }
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("[mystory] wrote %s" % STATUS, flush=True)
    print("[mystory] deluxe=%s" % deluxe_id, flush=True)
    print("[mystory] low=%s" % low_id, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
