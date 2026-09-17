"""Stop the wrong pod; wait for lime (shared OTR volume) SSH. No secrets."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

KEY_PATH = Path.home() / ".gemini" / "antigravity" / "scratch" / "fah-campaign" / "runpod_key.txt"
REST = "https://rest.runpod.io/v1/pods"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
LIME = "zuihlk2y9dpl82"
SECURE = "hwpbc1hevngne2"


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


def get_pod(key, pid):
    req = urllib.request.Request(f"{REST}/{pid}", headers=headers(key), method="GET")
    code, body = _read(req)
    if code >= 400:
        raise SystemExit(f"GET {pid} {code} {body[:200]}")
    data = json.loads(body)
    return data.get("pod") or data


def stop_pod(key, pid):
    req = urllib.request.Request(
        f"{REST}/{pid}/stop",
        data=b"{}",
        headers=headers(key, {"Content-Type": "application/json"}),
        method="POST",
    )
    code, body = _read(req)
    snippet = "(redacted)" if any(s in body for s in ("JUPYTER", "PUBLIC_KEY", "OPENROUTER")) else body[:180]
    print(f"STOP {pid} -> {code} {snippet}")


def ssh_of(pod):
    runtime = pod.get("runtime") or {}
    ports = runtime.get("ports") or []
    if isinstance(ports, list):
        for p in ports:
            if not isinstance(p, dict):
                continue
            private = str(p.get("privatePort") or p.get("port") or "")
            itype = str(p.get("type") or p.get("name") or "").lower()
            if private == "22" or "ssh" in itype:
                ip = p.get("ip") or p.get("publicIp")
                pub = p.get("publicPort")
                if ip and pub:
                    return str(ip), int(pub)
    return None, None


def main():
    key = KEY_PATH.read_text(encoding="utf-8").strip()
    stop_pod(key, SECURE)
    deadline = time.time() + 240
    last = ""
    while time.time() < deadline:
        pod = get_pod(key, LIME)
        status = str(pod.get("desiredStatus") or "").upper()
        ip, port = ssh_of(pod)
        line = f"lime desired={status} ssh={ip}:{port}"
        if line != last:
            print(line)
            last = line
        if status == "RUNNING" and ip and port:
            Path(__file__).with_name("_tmp_runpod_ssh.json").write_text(
                json.dumps({"id": LIME, "name": pod.get("name"), "host": ip, "port": port}, indent=2)
                + "\n",
                encoding="utf-8",
            )
            print(f"READY host={ip} port={port}")
            return 0
        time.sleep(5)
    print("LIME_NOT_READY")
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
