"""Dump safe RunPod fields and try starting every EXITED pod. No secrets."""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

KEY_PATH = Path.home() / ".gemini" / "antigravity" / "scratch" / "fah-campaign" / "runpod_key.txt"
REST = "https://rest.runpod.io/v1/pods"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
SAFE = (
    "id", "name", "desiredStatus", "costPerHr", "gpuCount", "gpuTypeId",
    "machineId", "machineType", "memoryInGb", "vcpuCount", "volumeInGb",
    "containerDiskInGb", "volumeMountPath", "imageName", "lastStatusChange",
    "networkVolumeId", "templateId", "interruptible", "locked", "publicIp",
)


def _read(req):
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            return resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")


def headers(key, extra=None):
    h = {"Authorization": f"Bearer {key}", "User-Agent": UA}
    if extra:
        h.update(extra)
    return h


def main() -> int:
    key = KEY_PATH.read_text(encoding="utf-8").strip()
    req = urllib.request.Request(REST, headers=headers(key), method="GET")
    code, body = _read(req)
    print(f"LIST {code}")
    pods = json.loads(body)
    if isinstance(pods, dict):
        pods = pods.get("pods") or pods.get("data") or []
    for p in pods:
        row = {k: p.get(k) for k in SAFE}
        machine = p.get("machine") or {}
        row["gpuDisplayName"] = machine.get("gpuDisplayName") or p.get("gpu")
        row["machine_keys"] = sorted(machine.keys()) if isinstance(machine, dict) else []
        extra = sorted(set(p.keys()) - {"env", "environment", "ports"})
        print("---")
        print(json.dumps(row, indent=2, default=str))
        print("all_keys", extra)
        # try start if EXITED
        if str(p.get("desiredStatus") or "").upper() == "EXITED":
            pid = p["id"]
            sreq = urllib.request.Request(
                f"{REST}/{pid}/start",
                data=b"{}",
                headers=headers(key, {"Content-Type": "application/json"}),
                method="POST",
            )
            sc, sb = _read(sreq)
            snippet = sb[:240]
            if "JUPYTER" in sb or "PUBLIC_KEY" in sb or "OPENROUTER" in sb:
                snippet = "(body redacted)"
            print(f"START {pid} {p.get('name')!r} -> {sc} {snippet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
