"""Identify which ComfyUI is on 8000/8188 and whether GPU is free."""
from __future__ import annotations

import json
import subprocess
import urllib.request

print("=== nvidia-smi ===")
try:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
         "--format=csv,noheader"],
        text=True, timeout=10,
    )
    print(out.strip())
except Exception as exc:
    print("smi fail", exc)

print("=== listeners ===")
ps = subprocess.check_output(
    ["powershell", "-NoProfile", "-Command",
     "Get-NetTCPConnection -LocalPort 8000,8188 -State Listen -ErrorAction SilentlyContinue | "
     "Select-Object LocalPort,OwningProcess | ConvertTo-Json"],
    text=True, timeout=15,
)
print(ps.strip())
try:
    rows = json.loads(ps)
except json.JSONDecodeError:
    rows = []
if isinstance(rows, dict):
    rows = [rows]
pids = sorted({int(r["OwningProcess"]) for r in rows if r})
for pid in pids:
    cmd = subprocess.check_output(
        ["powershell", "-NoProfile", "-Command",
         f"(Get-CimInstance Win32_Process -Filter \"ProcessId={pid}\").CommandLine"],
        text=True, timeout=15,
    )
    print(f"pid {pid}: {cmd.strip()[:500]}")

print("=== /history tails ===")
for port in (8000, 8188):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/history", timeout=8) as resp:
            hist = json.loads(resp.read().decode("utf-8", errors="replace"))
        keys = list(hist.keys())
        print(f"port {port} history_n={len(keys)} last5={keys[-5:]}")
        for k in reversed(keys[-3:]):
            st = (hist[k].get("status") or {})
            print(f"  {k[:8]} status={st.get('status_str')} completed={st.get('completed')}")
    except Exception as exc:
        print(f"port {port} history fail {type(exc).__name__}: {exc}")
