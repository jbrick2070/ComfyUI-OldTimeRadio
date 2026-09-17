"""Boot :8000 if down, then queue ONLY cheap-cloud 1-act.

my_story + recur_frac + fan-out 8. Does not touch :8188.
Does not queue deluxe or 5-act.
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


def _queue():
    try:
        with urllib.request.urlopen(URL + "/queue", timeout=5) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def _has_vidu() -> bool:
    try:
        with urllib.request.urlopen(
            URL + "/object_info/OTR_VideoDirector", timeout=15
        ) as resp:
            info = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return False
    node = info.get("OTR_VideoDirector") or {}
    inputs = ((node.get("input") or {}).get("required") or {})
    combo = inputs.get("announcer_video_model") or []
    options = combo[0] if combo and isinstance(combo[0], list) else []
    return "cloud_vidu_q2_pro_fast_720p" in " ".join(str(x) for x in options)


def _boot() -> None:
    q = _queue()
    if q is not None:
        running = q.get("queue_running") or []
        pending = q.get("queue_pending") or []
        if running or pending:
            raise SystemExit(
                "BUSY running=%d pending=%d -- not queuing over a live job"
                % (len(running), len(pending))
            )
        if _has_vidu():
            print("[low1] :8000 already up with Vidu", flush=True)
            return
        print("[low1] idle without Vidu -- will not recycle a live box", flush=True)
        return
    os.environ["OTR_CLOUD_FANOUT"] = "8"
    os.environ["OTR_CLOUD_VIDEO_FANOUT"] = "8"
    os.environ["OTR_COMFY_MAX_TOKENS_PER_RUN"] = "1000000"
    import _tmp_boot_cpu_8000 as boot
    rc = int(boot.main())
    if rc != 0:
        raise SystemExit("boot rc=%s" % rc)
    deadline = time.time() + 120
    while time.time() < deadline:
        if _has_vidu():
            print("[low1] Vidu live on :8000", flush=True)
            return
        time.sleep(2)
    raise SystemExit("boot finished but Vidu missing from object_info")


def main() -> int:
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("OTR_ENABLE_COMFY_CREDITS", "1")
    os.environ.setdefault("OTR_CLOUD_FANOUT", "8")
    os.environ.setdefault("OTR_CLOUD_VIDEO_FANOUT", "8")
    _hydrate_user_env("OTR_COMFY_API_KEY")
    if not (os.environ.get("OTR_COMFY_API_KEY") or "").strip():
        raise SystemExit("OTR_COMFY_API_KEY missing")
    if not WF.is_file():
        raise SystemExit("missing %s" % WF)
    _boot()
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
    print("[low1] submit %s my_story recur_frac fanout=8" % WF.name, flush=True)
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
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps({
        "low_prompt_id": prompt_id,
        "source_bank": "my_story",
        "visual_style": "recur_frac",
        "act_count": "1",
        "fanout": 8,
    }, indent=2), encoding="utf-8")
    print("[low1] low=%s" % prompt_id, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
