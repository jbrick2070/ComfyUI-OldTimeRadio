"""Queue cheap-cloud 5-act on :8000. Hydrate OTR_COMFY_API_KEY from User env."""
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
    os.environ.setdefault(
        "OTR_OBS_DIR", r"C:\Users\jeffr\Documents\ComfyUI\output\otr\obs")
    os.environ.setdefault(
        "OTR_OUTPUT_DIR", r"C:\Users\jeffr\Documents\ComfyUI\output")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import _tmp_submit_cloud as submit

    sys.argv = [
        "_tmp_submit_cloud.py",
        "--workflow",
        str(Path(__file__).resolve().parent / "workflows" / "variants" / "otr_cloud_low.json"),
        "--act-count",
        "5",
        "--comfyui-url",
        "http://127.0.0.1:8000",
        "--timeout",
        "18000",
        "--poll-s",
        "20",
    ]
    return int(submit.main())


if __name__ == "__main__":
    sys.exit(main())
