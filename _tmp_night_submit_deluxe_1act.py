"""Queue deluxe-cloud 1-act on :8000 (Wan audio-in + Sol/Luna).

Uses the shipping deluxe graph with act_count patched to 1 so we prove
the deluxe engines and the cloud video fan-out without a 5-act bill.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


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
    _hydrate_user_env("OTR_COMFY_API_KEY")
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("OTR_ENABLE_COMFY_CREDITS", "1")
    os.environ.setdefault("OTR_CLOUD_VIDEO_FANOUT", "8")
    os.environ.setdefault(
        "OTR_OBS_DIR", r"C:\Users\jeffr\Documents\ComfyUI\output\otr\obs")
    os.environ.setdefault(
        "OTR_OUTPUT_DIR", r"C:\Users\jeffr\Documents\ComfyUI\output")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import _tmp_submit_cloud as submit

    print("[deluxe-1act] workflow=otr_cloud_deluxe_7act act_count=1 "
          "fanout=%s" % os.environ.get("OTR_CLOUD_VIDEO_FANOUT"), flush=True)
    sys.argv = [
        "_tmp_submit_cloud.py",
        "--workflow",
        r"C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\workflows\variants\otr_cloud_deluxe_7act.json",
        "--act-count",
        "1",
        "--comfyui-url",
        "http://127.0.0.1:8000",
        "--timeout",
        "9000",
        "--poll-s",
        "15",
    ]
    return int(submit.main())


if __name__ == "__main__":
    sys.exit(main())
