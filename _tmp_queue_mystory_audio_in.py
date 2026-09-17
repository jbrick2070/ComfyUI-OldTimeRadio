"""Queue 1-act deluxe AUDIO-IN behind the live Foley+low jobs. Do not recycle."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = r"C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe"
WF = ROOT / "workflows" / "variants" / "otr_cloud_deluxe_audio_in_7act.json"
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


def main() -> int:
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("OTR_ENABLE_COMFY_CREDITS", "1")
    _hydrate_user_env("OTR_COMFY_API_KEY")
    if not WF.is_file():
        raise SystemExit("missing %s" % WF)
    cmd = [
        PY, str(ROOT / "_tmp_submit_cloud.py"),
        "--workflow", str(WF),
        "--act-count", "1",
        "--comfyui-url", "http://127.0.0.1:8000",
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
    if not prompt_id:
        raise SystemExit("no prompt_id")
    payload = {}
    if STATUS.is_file():
        payload = json.loads(STATUS.read_text(encoding="utf-8"))
    payload["audio_in_prompt_id"] = prompt_id
    payload["source_bank"] = "my_story"
    payload["visual_style"] = "recur_frac"
    STATUS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("[mystory] audio_in=%s" % prompt_id, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
