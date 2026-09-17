"""After cheap-cloud 1-act SUCCESS on :8000, queue the 5-act cheap Vidu graph.

Does not touch Desktop :8188. Exits without queueing if 1-act fails.
"""
from __future__ import annotations

import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ONE_ACT_ID = (
    sys.argv[1] if len(sys.argv) > 1
    else "b3fe1522-7328-4e28-b7b9-809cc2a9580a")
QUEUE = "http://127.0.0.1:8000/queue"
HISTORY = "http://127.0.0.1:8000/history/"
OBS = Path(r"C:\Users\jeffr\Documents\ComfyUI\output\otr\obs")


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


def _get(url: str, timeout: float = 8.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _one_act_status() -> str:
    try:
        raw = _get(HISTORY + ONE_ACT_ID, timeout=12)
    except (urllib.error.URLError, TimeoutError, OSError):
        return "unknown"
    if not raw or raw.strip() in ("{}", ""):
        return "pending"
    if '"status"' in raw and '"completed"' in raw.lower():
        # Comfy history: status.completed true + status.status_str success
        low = raw.lower()
        if "error" in low and "success" not in low:
            return "fail"
        if '"status_str": "success"' in low or '"status_str":"success"' in low:
            return "success"
        if "success" in low:
            return "success"
        if "error" in low:
            return "fail"
    return "pending"


def _submit_5act() -> int:
    _hydrate_user_env("OTR_COMFY_API_KEY")
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("OTR_ENABLE_COMFY_CREDITS", "1")
    os.environ.setdefault("OTR_OBS_DIR", str(OBS))
    os.environ.setdefault(
        "OTR_OUTPUT_DIR", r"C:\Users\jeffr\Documents\ComfyUI\output")
    sys.path.insert(0, str(ROOT))
    import _tmp_submit_cloud as submit

    sys.argv = [
        "_tmp_submit_cloud.py",
        "--workflow",
        str(ROOT / "workflows" / "variants" / "otr_cloud_low.json"),
        "--act-count",
        "5",
        "--comfyui-url",
        "http://127.0.0.1:8000",
        "--timeout",
        "18000",
        "--poll-s",
        "20",
    ]
    print("[night-5act] submitting otr_cloud_low act_count=5", flush=True)
    return int(submit.main())


def main() -> int:
    print(f"[night-5act] waiting on 1-act {ONE_ACT_ID}", flush=True)
    deadline = time.time() + 9000
    while time.time() < deadline:
        st = _one_act_status()
        print(f"[night-5act] 1-act status={st}", flush=True)
        if st == "success":
            rainbow = list(OBS.glob("*rainbow*"))
            print(f"[night-5act] obs rainbow matches={len(rainbow)}", flush=True)
            return _submit_5act()
        if st == "fail":
            print("[night-5act] 1-act FAILED -- not queueing 5-act", flush=True)
            return 2
        time.sleep(30)
    print("[night-5act] TIMEOUT waiting for 1-act", flush=True)
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
