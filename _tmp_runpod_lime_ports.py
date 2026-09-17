from __future__ import annotations
import json, urllib.request
from pathlib import Path

KEY = (Path.home() / ".gemini" / "antigravity" / "scratch" / "fah-campaign" / "runpod_key.txt").read_text(encoding="utf-8").strip()
PID = "zuihlk2y9dpl82"
req = urllib.request.Request(
    f"https://rest.runpod.io/v1/pods/{PID}",
    headers={"Authorization": f"Bearer {KEY}", "User-Agent": "Mozilla/5.0"},
    method="GET",
)
with urllib.request.urlopen(req, timeout=30) as r:
    raw = json.loads(r.read().decode())
pod = raw.get("pod") or raw
pod.pop("env", None)
print("top_keys", sorted(pod.keys()))
rt = pod.get("runtime")
print("runtime_type", type(rt).__name__, sorted(rt.keys()) if isinstance(rt, dict) else rt)
if isinstance(rt, dict):
    rt.pop("env", None)
    ports = rt.get("ports")
    print("ports", json.dumps(ports, indent=2)[:2000])
print("publicIp", pod.get("publicIp"))
print("desired", pod.get("desiredStatus"))
# some REST shapes put port mappings at top
for k in ("portMappings", "ports", "runtime"):
    if k in pod and k != "runtime":
        print(k, json.dumps(pod.get(k), default=str)[:800])
