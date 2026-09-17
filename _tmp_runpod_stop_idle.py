"""List RunPod pods; stop any that are running. Never print the API key."""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

KEY_PATH = Path.home() / ".gemini" / "antigravity" / "scratch" / "fah-campaign" / "runpod_key.txt"
GQL = "https://api.runpod.io/graphql"
REST = "https://rest.runpod.io/v1/pods"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"

LIST_Q = """
query {
  myself {
    pods {
      id
      name
      desiredStatus
      lastStatusChange
      costPerHr
      machine { gpuDisplayName }
      runtime { uptimeInSeconds }
    }
  }
}
"""
STOP_Q = """
mutation Stop($podId: String!) {
  podStop(input: {podId: $podId}) { id desiredStatus }
}
"""


def _read(req: urllib.request.Request) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")


def rest_list(key: str):
    req = urllib.request.Request(
        REST,
        headers={"Authorization": f"Bearer {key}", "User-Agent": UA},
        method="GET",
    )
    code, body = _read(req)
    print(f"REST GET /v1/pods -> {code}")
    if code >= 400:
        print(f"  body={body[:500]}")
        return []
    data = json.loads(body)
    if isinstance(data, dict):
        return data.get("pods") or data.get("data") or data.get("items") or []
    if isinstance(data, list):
        return data
    return []


def rest_stop(key: str, pod_id: str) -> None:
    req = urllib.request.Request(
        f"{REST}/{pod_id}/stop",
        data=b"{}",
        headers={
            "Authorization": f"Bearer {key}",
            "User-Agent": UA,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    code, body = _read(req)
    print(f"REST STOP {pod_id} -> {code} {body[:400]}")


def gql(key: str, query: str, variables=None) -> dict:
    payload = {"query": query}
    if variables:
        payload["variables"] = variables
    req = urllib.request.Request(
        f"{GQL}?api_key={key}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": UA},
        method="POST",
    )
    code, raw_text = _read(req)
    print(f"GQL -> {code}")
    if code >= 400:
        print(f"  body={raw_text[:400]}")
        return {}
    raw = json.loads(raw_text)
    if raw.get("errors"):
        print(f"graphql errors: {raw['errors']}")
    return raw.get("data") or {}


def _status(pod: dict) -> str:
    return str(
        pod.get("desiredStatus")
        or pod.get("desired_status")
        or pod.get("status")
        or ""
    ).upper()


def _print_pod(prefix: str, p: dict) -> None:
    machine = p.get("machine") or {}
    runtime = p.get("runtime") or {}
    print(
        f"{prefix} id={p.get('id')} name={p.get('name')!r} "
        f"desired={_status(p)} gpu={machine.get('gpuDisplayName') or p.get('gpu')} "
        f"costPerHr={p.get('costPerHr') or p.get('costPerGpu')} "
        f"uptime_s={runtime.get('uptimeInSeconds') or p.get('uptimeSeconds')} "
        f"last={p.get('lastStatusChange')}"
    )


def main() -> int:
    if not KEY_PATH.is_file():
        print(f"NO_KEY at {KEY_PATH}", file=sys.stderr)
        return 2
    key = KEY_PATH.read_text(encoding="utf-8").strip()
    if not key:
        print("EMPTY_KEY", file=sys.stderr)
        return 2

    pods = rest_list(key)
    if not pods:
        data = gql(key, LIST_Q)
        pods = (data.get("myself") or {}).get("pods") or []
        print(f"GQL pods={len(pods)}")

    print(f"account pods={len(pods)}")
    to_stop = []
    seen = set()
    for p in pods:
        _print_pod("  ", p)
        pid = p.get("id")
        if pid and pid not in seen and _status(p) == "RUNNING":
            seen.add(pid)
            to_stop.append(p)

    if not to_stop:
        print("NO_RUNNING_PODS")
        return 0

    for p in to_stop:
        pid = p["id"]
        print(f"STOPPING {pid} {p.get('name')!r}")
        rest_stop(key, pid)
        gql(key, STOP_Q, {"podId": pid})

    pods2 = rest_list(key)
    if not pods2:
        pods2 = ((gql(key, LIST_Q).get("myself") or {}).get("pods") or [])
    for p in pods2:
        _print_pod("AFTER ", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
