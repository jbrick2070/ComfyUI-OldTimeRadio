from __future__ import annotations
import json, urllib.request, urllib.error
from pathlib import Path

KEY = (Path.home() / ".gemini" / "antigravity" / "scratch" / "fah-campaign" / "runpod_key.txt").read_text(encoding="utf-8").strip()
REST = "https://rest.runpod.io/v1/pods"
UA = "Mozilla/5.0"
H = {"Authorization": f"Bearer {KEY}", "User-Agent": UA}

def get(url):
    req = urllib.request.Request(url, headers=H, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:200]

code, data = get(REST)
print("LIST", code)
pods = data if isinstance(data, list) else (data.get("pods") or [])
for p in pods:
    print(p.get("id"), p.get("name"), p.get("desiredStatus"), "gpu", (p.get("machine") or {}).get("gpuDisplayName"), "ip", p.get("publicIp"))
    rt = p.get("runtime") or {}
    ports = rt.get("ports") if isinstance(rt, dict) else None
    if ports:
        print("  runtime_ports", [
            {k: x.get(k) for k in ("ip","publicIp","privatePort","publicPort","type") if isinstance(x, dict)}
            for x in (ports if isinstance(ports, list) else [])
        ])
