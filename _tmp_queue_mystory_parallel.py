"""Force-recycle :8000, then queue three my_story + recur_frac 1-acts.

Cheap Vidu, deluxe Foley, deluxe audio-in. Parallel AV needs a fresh
process so ThreadPool stills/TTS/music/video load. Does not touch :8188.
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
LOW = ROOT / "workflows" / "variants" / "otr_cloud_low_1act.json"
DELUXE = ROOT / "workflows" / "variants" / "otr_cloud_deluxe_7act.json"
AUDIO = ROOT / "workflows" / "variants" / "otr_cloud_deluxe_audio_in_7act.json"
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


def _get(path: str, timeout: float = 15):
    with urllib.request.urlopen(URL + path, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _post(path: str, payload: dict, timeout: float = 15) -> None:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        URL + path, data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print("[parallel] POST %s %s" % (path, exc), flush=True)


def _queue():
    try:
        return _get("/queue", timeout=5)
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def _engine_blob() -> str:
    info = _get("/object_info/OTR_VideoDirector")
    node = info.get("OTR_VideoDirector") or {}
    inputs = ((node.get("input") or {}).get("required") or {})
    combo = inputs.get("announcer_video_model") or []
    options = combo[0] if combo and isinstance(combo[0], list) else []
    return " ".join(str(x) for x in options)


def _has_new_engines() -> bool:
    try:
        blob = _engine_blob()
    except Exception:
        return False
    return (
        "cloud_ltx25_foley_plus" in blob
        and "cloud_ltx25_audio_in" in blob
        and "cloud_vidu_q2_pro_fast_720p" in blob
    )


def _kill_8000() -> None:
    ps = (
        "$procs = Get-CimInstance Win32_Process -Filter \"Name='python.exe'\"; "
        "foreach ($p in $procs) { "
        "  $c = [string]$p.CommandLine; "
        "  if ($c -match 'ComfyUI' -and $c -match '--port\\s+8000') { "
        "    Write-Output ('KILL ' + $p.ProcessId); "
        "    Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue "
        "  } "
        "}"
    )
    subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps],
        check=False,
        timeout=30,
    )


def _force_recycle() -> None:
    q = _queue()
    if q is not None:
        running = q.get("queue_running") or []
        pending = q.get("queue_pending") or []
        print(
            "[parallel] retarget running=%d pending=%d -- interrupt+recycle"
            % (len(running), len(pending)),
            flush=True,
        )
        _post("/queue", {"clear": True})
        _post("/interrupt", {})
        time.sleep(2)
    _kill_8000()
    deadline = time.time() + 25
    while time.time() < deadline:
        if _queue() is None:
            break
        time.sleep(1)
    else:
        raise SystemExit("port :8000 still up after kill")
    os.environ["OTR_CLOUD_FANOUT"] = "8"
    os.environ["OTR_CLOUD_VIDEO_FANOUT"] = "8"
    os.environ["OTR_COMFY_MAX_TOKENS_PER_RUN"] = "1000000"
    import _tmp_boot_cpu_8000 as boot
    rc = int(boot.main())
    if rc != 0:
        raise SystemExit("boot rc=%s" % rc)
    deadline = time.time() + 120
    while time.time() < deadline:
        if _has_new_engines():
            print("[parallel] Foley + audio-in + Vidu live on :8000", flush=True)
            return
        time.sleep(2)
    raise SystemExit("boot finished but new engines missing from object_info")


def _submit(label: str, workflow: Path, act_count: str = "1") -> str:
    cmd = [
        PY, str(ROOT / "_tmp_submit_cloud.py"),
        "--workflow", str(workflow),
        "--act-count", act_count,
        "--comfyui-url", URL,
        "--source-bank", "my_story",
        "--visual-style", "recur_frac",
        "--source-ref", "",
        "--no-wait",
    ]
    print("[parallel] submit %s %s" % (label, workflow.name), flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    sys.stdout.write(proc.stdout or "")
    sys.stderr.write(proc.stderr or "")
    if proc.returncode != 0:
        raise SystemExit("submit %s rc=%s" % (label, proc.returncode))
    prompt_id = ""
    for line in (proc.stdout or "").splitlines():
        if "QUEUED prompt_id=" in line:
            prompt_id = line.split("QUEUED prompt_id=", 1)[1].strip()
    if not prompt_id:
        raise SystemExit("submit %s produced no prompt_id" % label)
    return prompt_id


def main() -> int:
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("OTR_ENABLE_COMFY_CREDITS", "1")
    os.environ.setdefault("OTR_CLOUD_FANOUT", "8")
    os.environ.setdefault("OTR_CLOUD_VIDEO_FANOUT", "8")
    _hydrate_user_env("OTR_COMFY_API_KEY")
    if not (os.environ.get("OTR_COMFY_API_KEY") or "").strip():
        raise SystemExit("OTR_COMFY_API_KEY missing")
    for path in (LOW, DELUXE):
        if not path.is_file():
            raise SystemExit("missing %s" % path)
    _force_recycle()
    low_id = _submit("low", LOW, "1")
    deluxe_id = _submit("deluxe_foley", DELUXE, "1")
    deluxe5_id = _submit("deluxe_foley_5act", DELUXE, "5")
    payload = {
        "low_prompt_id": low_id,
        "deluxe_prompt_id": deluxe_id,
        "deluxe_5act_prompt_id": deluxe5_id,
        "source_bank": "my_story",
        "visual_style": "recur_frac",
        "fanout": 8,
    }
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("[parallel] wrote %s" % STATUS, flush=True)
    print("[parallel] low=%s" % low_id, flush=True)
    print("[parallel] deluxe_foley=%s" % deluxe_id, flush=True)
    print("[parallel] deluxe_foley_5act=%s" % deluxe5_id, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
