"""Boot one --cpu ComfyUI on :8000 for cloud graphs. Do not touch Desktop :8188."""
from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

PY = r"C:\Users\jeffr\ComfyUI-Installs\ComfyUI (1)\ComfyUI\.venv\Scripts\python.exe"
MAIN = r"C:\Users\jeffr\ComfyUI-Installs\ComfyUI (1)\ComfyUI\main.py"
CWD = r"C:\Users\jeffr\ComfyUI-Installs\ComfyUI (1)\ComfyUI"
OUT = r"C:\Users\jeffr\Documents\ComfyUI\output"
OBS = r"C:\Users\jeffr\Documents\ComfyUI\output\otr\obs"
LOG = r"C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\tmp\comfy_cpu_8000_night.log"
URL = "http://127.0.0.1:8000/queue"


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


def _queue_ok() -> bool:
    try:
        with urllib.request.urlopen(URL, timeout=3) as resp:
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def main() -> int:
    if _queue_ok():
        print("READY already listening on :8000", flush=True)
        return 0
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    os.makedirs(OBS, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["OTR_ENABLE_COMFY_CREDITS"] = "1"
    env["OTR_COMFY_MAX_TOKENS_PER_RUN"] = os.environ.get(
        "OTR_COMFY_MAX_TOKENS_PER_RUN", "1000000")
    # Vidu Q2 I2V on Comfy Cloud routinely GET-retries; 900s killed a live
    # 1-act on shot_b001 after writer/Luma had already paid (2026-09-15).
    env["OTR_CLOUD_VIDEO_TIMEOUT_S"] = os.environ.get(
        "OTR_CLOUD_VIDEO_TIMEOUT_S", "1800")
    # Overnight boot does not inject a USD ceiling. Unset = no local cap;
    # a partner 402 is the stop. Do not inherit a leftover 90/300 from
    # this shell (operator 2026-09-16: do not hole a published episode).
    env.pop("OTR_CLOUD_MEDIA_BUDGET_USD", None)
    env["OTR_CLOUD_FANOUT"] = "4"
    env["OTR_CLOUD_VIDEO_FANOUT"] = "4"
    env["OTR_OBS_DIR"] = OBS
    env["OTR_OUTPUT_DIR"] = OUT
    _hydrate_user_env("OTR_COMFY_API_KEY")
    env["OTR_COMFY_API_KEY"] = os.environ.get("OTR_COMFY_API_KEY", "")
    key_set = "yes" if env.get("OTR_COMFY_API_KEY") else "NO"
    print(f"[boot] OTR_COMFY_API_KEY set={key_set}", flush=True)
    print(f"[boot] OTR_CLOUD_VIDEO_TIMEOUT_S={env.get('OTR_CLOUD_VIDEO_TIMEOUT_S')}", flush=True)
    print(f"[boot] OTR_CLOUD_MEDIA_BUDGET_USD={env.get('OTR_CLOUD_MEDIA_BUDGET_USD') or 'unset'}", flush=True)
    print(f"[boot] OTR_CLOUD_FANOUT={env.get('OTR_CLOUD_FANOUT')}", flush=True)
    print(f"[boot] OTR_OBS_DIR={OBS}", flush=True)
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    if os.path.isfile(LOG) and os.path.getsize(LOG) > 0:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        rotated = LOG.replace(".log", "_%s.log" % stamp)
        try:
            os.replace(LOG, rotated)
            print(f"[boot] rotated previous log to {rotated}", flush=True)
        except OSError as exc:
            print(f"[boot] log rotate skipped: {exc}", flush=True)
    log = open(LOG, "w", encoding="utf-8")
    args = [
        PY,
        "-s",
        MAIN,
        "--listen",
        "127.0.0.1",
        "--port",
        "8000",
        "--cpu",
        "--disable-metadata",
        "--database-url",
        "sqlite:///:memory:",
        "--output-directory",
        OUT,
    ]
    proc = subprocess.Popen(
        args,
        cwd=CWD,
        stdout=log,
        stderr=subprocess.STDOUT,
        env=env,
    )
    print(f"[boot] pid={proc.pid} log={LOG}", flush=True)
    deadline = time.time() + 90
    while time.time() < deadline:
        if proc.poll() is not None:
            print(f"[boot] DEAD rc={proc.returncode}", flush=True)
            return 1
        if _queue_ok():
            print("[boot] READY http://127.0.0.1:8000/queue", flush=True)
            return 0
        time.sleep(2)
    print("[boot] TIMEOUT waiting for /queue", flush=True)
    return 2


if __name__ == "__main__":
    sys.exit(main())
