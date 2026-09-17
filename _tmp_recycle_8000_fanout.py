"""Idle recycle of :8000 so fan-out code loads. Never kills a busy queue."""
from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.request

QUEUE = "http://127.0.0.1:8000/queue"


def _queue() -> dict | None:
    try:
        with urllib.request.urlopen(QUEUE, timeout=4) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception:
        return None


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


def main() -> int:
    q = _queue()
    if q is not None:
        running = q.get("queue_running") or []
        pending = q.get("queue_pending") or []
        if running or pending:
            print(f"BUSY running={len(running)} pending={len(pending)} -- not recycling")
            return 3
        print("idle on :8000 -- recycling for fan-out")
    else:
        print(":8000 not listening -- boot only")
    _kill_8000()
    deadline = time.time() + 20
    while time.time() < deadline:
        if _queue() is None:
            break
        time.sleep(1)
    else:
        print("port still up after kill")
        return 4
    os.environ["PYTHONUTF8"] = "1"
    os.environ["OTR_ENABLE_COMFY_CREDITS"] = "1"
    os.environ.pop("OTR_CLOUD_MEDIA_BUDGET_USD", None)
    os.environ["OTR_CLOUD_FANOUT"] = "4"
    os.environ["OTR_CLOUD_VIDEO_FANOUT"] = "4"
    import _tmp_boot_cpu_8000 as boot
    return int(boot.main())


if __name__ == "__main__":
    raise SystemExit(main())
