"""Kill the :8000 Comfy process. Do not boot it again. Do not touch other pythons."""
from __future__ import annotations

import json
import subprocess
import time
import urllib.request

QUEUE = "http://127.0.0.1:8000/queue"


def _queue():
    try:
        with urllib.request.urlopen(QUEUE, timeout=4) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception:
        return None


def main() -> int:
    q = _queue()
    if q is None:
        print("already down", flush=True)
        return 0
    running = q.get("queue_running") or []
    pending = q.get("queue_pending") or []
    print("before running=%d pending=%d" % (len(running), len(pending)), flush=True)
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
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=False, timeout=30)
    deadline = time.time() + 20
    while time.time() < deadline:
        if _queue() is None:
            print("DOWN :8000", flush=True)
            return 0
        time.sleep(1)
    print("STILL UP after kill", flush=True)
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
