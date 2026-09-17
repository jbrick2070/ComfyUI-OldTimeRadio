"""Queue cheap-cloud 5-act sci-fi news + recur_frac on :8000. No wait."""
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
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import _tmp_submit_cloud as submit

    root = Path(__file__).resolve().parent
    sys.argv = [
        "_tmp_submit_cloud.py",
        "--workflow",
        str(root / "workflows" / "variants" / "otr_cloud_low.json"),
        "--act-count",
        "5",
        "--source-bank",
        "scifi_news_pro",
        "--visual-style",
        "recur_frac",
        "--comfyui-url",
        "http://127.0.0.1:8000",
        "--no-wait",
    ]
    return int(submit.main())


if __name__ == "__main__":
    raise SystemExit(main())
