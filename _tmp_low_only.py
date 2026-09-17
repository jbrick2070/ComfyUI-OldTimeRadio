"""Wipe :8000 queue, then queue ONLY cheap-cloud 1-act my_story + recur_frac."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = r"C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe"
URL = "http://127.0.0.1:8000"
WF = ROOT / "workflows" / "variants" / "otr_cloud_low_1act.json"
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


def _post(path: str, payload: dict) -> None:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        URL + path, data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        resp.read()


def main() -> int:
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("OTR_ENABLE_COMFY_CREDITS", "1")
    _hydrate_user_env("OTR_COMFY_API_KEY")
    _post("/queue", {"clear": True})
    try:
        _post("/interrupt", {})
    except Exception as exc:
        print("[low-only] interrupt %s" % exc, flush=True)
    print("[low-only] queue cleared", flush=True)
    cmd = [
        PY, str(ROOT / "_tmp_submit_cloud.py"),
        "--workflow", str(WF),
        "--act-count", "1",
        "--comfyui-url", URL,
        "--source-bank", "my_story",
        "--visual-style", "recur_frac",
        "--source-ref", "",
        "--no-wait",
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    sys.stdout.write(proc.stdout or "")
    sys.stderr.write(proc.stderr or "")
    if proc.returncode != 0:
        return int(proc.returncode)
    prompt_id = ""
    for line in (proc.stdout or "").splitlines():
        if "QUEUED prompt_id=" in line:
            prompt_id = line.split("QUEUED prompt_id=", 1)[1].strip()
    STATUS.write_text(json.dumps({
        "low_prompt_id": prompt_id,
        "source_bank": "my_story",
        "visual_style": "recur_frac",
        "act_count": "1",
    }, indent=2), encoding="utf-8")
    print("[low-only] low=%s" % prompt_id, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
