"""Boot one GPU ComfyUI on :8188 for 16 GB graphs. Do not touch CPU :8000."""
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
LOG = r"C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\tmp\comfy_gpu_8188_night.log"
URL = "http://127.0.0.1:8188/queue"
CPU_URL = "http://127.0.0.1:8000/queue"


def _queue_ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def main() -> int:
    if not _queue_ok(CPU_URL):
        print("REFUSE: CPU :8000 is down -- not booting GPU until cloud 1-act is safe", flush=True)
        return 3
    if _queue_ok(URL):
        print("READY already listening on :8188", flush=True)
        return 0
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    os.makedirs(OBS, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["HF_HOME"] = r"C:\ComfyUI-Models\huggingface"
    env["OTR_OBS_DIR"] = OBS
    env["OTR_OUTPUT_DIR"] = OUT
    print(f"[boot] OTR_OBS_DIR={OBS}", flush=True)
    log = open(LOG, "w", encoding="utf-8")
    args = [
        PY, "-s", MAIN,
        "--listen", "127.0.0.1",
        "--port", "8188",
        "--disable-metadata",
        "--output-directory", OUT,
    ]
    proc = subprocess.Popen(
        args, cwd=CWD, stdout=log, stderr=subprocess.STDOUT, env=env,
    )
    print(f"[boot] pid={proc.pid} log={LOG}", flush=True)
    deadline = time.time() + 120
    while time.time() < deadline:
        if proc.poll() is not None:
            print(f"[boot] DEAD rc={proc.returncode}", flush=True)
            return 1
        if _queue_ok(URL):
            if not _queue_ok(CPU_URL):
                print("[boot] READY :8188 but :8000 died -- investigate", flush=True)
                return 4
            print("[boot] READY http://127.0.0.1:8188/queue (cpu :8000 still up)", flush=True)
            return 0
        time.sleep(2)
    print("[boot] TIMEOUT waiting for /queue", flush=True)
    return 2


if __name__ == "__main__":
    sys.exit(main())
