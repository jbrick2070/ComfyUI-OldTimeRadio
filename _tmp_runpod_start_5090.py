"""Start the 5090 Foley pod and print SSH host:port only. Never print env/secrets."""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

KEY_PATH = Path.home() / ".gemini" / "antigravity" / "scratch" / "fah-campaign" / "runpod_key.txt"
POD_ID = "3o8mxio6jm3t3n"
REST = "https://rest.runpod.io/v1/pods"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"


def _read(req: urllib.request.Request) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            return resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")


def _headers(key: str, extra=None) -> dict:
    h = {"Authorization": f"Bearer {key}", "User-Agent": UA}
    if extra:
        h.update(extra)
    return h


def get_pod(key: str) -> dict:
    req = urllib.request.Request(f"{REST}/{POD_ID}", headers=_headers(key), method="GET")
    code, body = _read(req)
    if code >= 400:
        raise SystemExit(f"GET pod HTTP {code}: {body[:300]}")
    data = json.loads(body)
    return data.get("pod") or data


def start_pod(key: str) -> dict:
    req = urllib.request.Request(
        f"{REST}/{POD_ID}/start",
        data=b"{}",
        headers=_headers(key, {"Content-Type": "application/json"}),
        method="POST",
    )
    code, body = _read(req)
    print(f"START HTTP {code}")
    if code >= 400:
        raise SystemExit(f"start failed: {body[:400]}")
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return {}


def _ssh(pod: dict) -> tuple[str | None, int | None]:
    runtime = pod.get("runtime") or {}
    ports = runtime.get("ports") or pod.get("portMappings") or []
    # REST variants: list of dicts with ip/publicIp + publicPort, or mapping.
    if isinstance(ports, dict):
        ssh = ports.get("22") or ports.get(22)
        if isinstance(ssh, dict):
            return ssh.get("ip") or ssh.get("publicIp"), int(
                ssh.get("publicPort") or ssh.get("port") or 0
            ) or None
    if isinstance(ports, list):
        for p in ports:
            if not isinstance(p, dict):
                continue
            private = str(p.get("privatePort") or p.get("private") or p.get("port") or "")
            itype = str(p.get("type") or p.get("name") or "").lower()
            if private == "22" or "ssh" in itype:
                ip = p.get("ip") or p.get("publicIp") or p.get("ipAddress")
                pub = p.get("publicPort") or p.get("public") or p.get("port")
                if ip and pub:
                    return str(ip), int(pub)
        public = runtime.get("ports") if False else None
    ip = (
        runtime.get("publicIp")
        or runtime.get("ip")
        or pod.get("publicIp")
        or pod.get("machine", {}).get("podHostId")
    )
    return (str(ip) if ip else None), None


def _safe_print(pod: dict) -> None:
    runtime = pod.get("runtime") or {}
    machine = pod.get("machine") or {}
    status = str(pod.get("desiredStatus") or pod.get("status") or "")
    print(
        f"id={pod.get('id')} name={pod.get('name')!r} desired={status} "
        f"gpu={machine.get('gpuDisplayName') or pod.get('gpu')} "
        f"uptime={runtime.get('uptimeInSeconds')}"
    )
    ip, port = _ssh(pod)
    print(f"ssh={ip}:{port}")
    # Dump port list without env
    ports = runtime.get("ports") or []
    if isinstance(ports, list):
        for p in ports:
            if isinstance(p, dict):
                print(
                    "port",
                    {k: p.get(k) for k in (
                        "ip", "publicIp", "privatePort", "publicPort",
                        "type", "name", "isIpPublic",
                    ) if k in p}
                )
    elif isinstance(ports, dict):
        print("port_keys", sorted(str(k) for k in ports.keys()))


def main() -> int:
    key = KEY_PATH.read_text(encoding="utf-8").strip()
    pod = get_pod(key)
    _safe_print(pod)
    status = str(pod.get("desiredStatus") or "").upper()
    if status != "RUNNING":
        start_pod(key)
    deadline = time.time() + 240
    last = ""
    while time.time() < deadline:
        pod = get_pod(key)
        status = str(pod.get("desiredStatus") or "").upper()
        ip, port = _ssh(pod)
        line = f"poll desired={status} ssh={ip}:{port}"
        if line != last:
            print(line)
            last = line
        if status == "RUNNING" and ip and port:
            out = Path(__file__).with_name("_tmp_runpod_ssh.json")
            out.write_text(
                json.dumps({"id": POD_ID, "host": ip, "port": port}, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"READY host={ip} port={port}")
            return 0
        time.sleep(5)
    _safe_print(get_pod(key))
    print("NOT_READY")
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
