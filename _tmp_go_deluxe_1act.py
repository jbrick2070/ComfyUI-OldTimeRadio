"""Recycle idle :8000 onto current pack, queue deluxe Foley 1-act, do not wait.

$100 Credits is enough for one 1-act deluxe Foley fan-out proof, not a 5-act.
Local media cap is $90 so LTX floors before the wallet is empty.
Does not touch :8188.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

# Leave ~$10 of the $100 wallet outside the media cap so a last LTX 402
# is less likely to also 402 the next writer probe. Video floors on cap.
os.environ["PYTHONUTF8"] = "1"
os.environ["OTR_ENABLE_COMFY_CREDITS"] = "1"
os.environ["OTR_CLOUD_FANOUT"] = "8"
os.environ["OTR_CLOUD_VIDEO_FANOUT"] = "8"
os.environ["OTR_CLOUD_MEDIA_BUDGET_USD"] = "90"
os.environ["OTR_CLOUD_VIDEO_TIMEOUT_S"] = "1800"

sys.path.insert(0, str(ROOT))
import _tmp_recycle_8000_fanout as recycle  # noqa: E402


def _object_info_ok() -> bool:
    url = "http://127.0.0.1:8000/object_info/OTR_LedgerScriptWriter"
    try:
        with urllib.request.urlopen(url, timeout=8) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return bool(body)
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return False


def main() -> int:
    print("[go] deluxe Foley 1-act fan-out; media cap $90; no 5-act", flush=True)
    rc = int(recycle.main())
    if rc != 0:
        print(f"[go] recycle failed rc={rc}", flush=True)
        return rc
    deadline = time.time() + 60
    while time.time() < deadline:
        if _object_info_ok():
            print("[go] object_info OTR_LedgerScriptWriter OK", flush=True)
            break
        time.sleep(2)
    else:
        print("[go] object_info never came up", flush=True)
        return 5

    import _tmp_night_submit_deluxe_1act as submit

    sys.argv = [
        "_tmp_night_submit_deluxe_1act.py",
    ]
    # Patch submit.main by calling the cloud submit with --no-wait.
    os.environ.setdefault("OTR_ENABLE_COMFY_CREDITS", "1")
    submit._hydrate_user_env("OTR_COMFY_API_KEY")
    import _tmp_submit_cloud as cloud

    sys.argv = [
        "_tmp_submit_cloud.py",
        "--workflow",
        str(ROOT / "workflows" / "variants" / "otr_cloud_deluxe_7act.json"),
        "--act-count",
        "1",
        "--comfyui-url",
        "http://127.0.0.1:8000",
        "--no-wait",
    ]
    print("[go] submitting otr_cloud_deluxe_7act act_count=1 --no-wait", flush=True)
    return int(cloud.main())


if __name__ == "__main__":
    raise SystemExit(main())
