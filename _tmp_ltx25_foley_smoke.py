"""Contained LTX 2.5 Foley smoke: LoadImage -> LtxApi25ImageToVideo -> SaveVideo.

No writer, no canonical episode. Proves the partner node our
cloud_ltx25_foley_plus pin uses (generate_audio=True) on :8000.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

import requests

URL = "http://127.0.0.1:8000"
OBS = Path(r"C:\Users\jeffr\Documents\ComfyUI\output\otr\obs")
# Same Luma Photon Flash start frame as the Vidu episodes in obs
# (bird_of_dawning -- cvdu + clum). Foley I2V holds this look.
STILL = Path(
    r"C:\Users\jeffr\Documents\ComfyUI\output\otr\episodes"
    r"\signal_lost_the_bird_of_dawning_20260915_200747"
    r"\stills\still_music_opening_001_c181a09bae3a.png")
PREFIX = "otr/obs/ltx25_foley_luma_still"


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


def _upload_image() -> str:
    with STILL.open("rb") as fh:
        resp = requests.post(
            f"{URL}/upload/image",
            files={"image": (STILL.name, fh, "image/png")},
            data={"overwrite": "true"},
            timeout=30,
        )
    resp.raise_for_status()
    body = resp.json()
    name = str(body.get("name") or STILL.name)
    print(f"[ltx-smoke] uploaded still={name}", flush=True)
    return name


def _prompt(image_name: str) -> dict:
    return {
        "4": {
            "class_type": "LoadImage",
            "inputs": {"image": image_name},
        },
        "5": {
            "class_type": "LtxApi25ImageToVideo",
            "inputs": {
                "image": ["4", 0],
                "model": "LTX-2.5 (Fast)",
                "model.duration": "8",
                "model.resolution": "1920x1080",
                "model.fps": "25",
                "model.generate_audio": True,
                "prompt": (
                    "Use the start image as the first frame and keep this "
                    "radio, this wet stone, and this dusk. The camera dollies "
                    "in from a three-quarter angle as the three vacuum tubes "
                    "flare from amber to white-hot. The speaker cloth "
                    "vibrates. Orange dial light pulses. Wind pushes mist "
                    "across the battlement. Continuous uncut take, hard "
                    "sidelight, no text, no black frames, no voices."
                ),
                "seed": 42,
            },
        },
        "2": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["5", 0],
                "filename_prefix": PREFIX,
                "format": "auto",
                "codec": "auto",
            },
        },
    }


def main() -> int:
    _hydrate_user_env("OTR_COMFY_API_KEY")
    key = (os.environ.get("OTR_COMFY_API_KEY") or "").strip()
    if not key:
        print("FAIL missing OTR_COMFY_API_KEY")
        return 3
    print(f"[ltx-smoke] key_len={len(key)} prefix={key[:8]}", flush=True)
    q = requests.get(f"{URL}/queue", timeout=10).json()
    running = q.get("queue_running") or []
    pending = q.get("queue_pending") or []
    if running or pending:
        print(f"FAIL queue not idle running={len(running)} pending={len(pending)}")
        return 3
    if not STILL.is_file():
        print("FAIL missing luma still %s" % STILL)
        return 3
    image_name = _upload_image()
    client_id = str(uuid.uuid4())
    resp = requests.post(
        f"{URL}/prompt",
        json={
            "prompt": _prompt(image_name),
            "client_id": client_id,
            "extra_data": {"api_key_comfy_org": key},
        },
        timeout=30,
    )
    if resp.status_code != 200:
        print(f"FAIL POST /prompt HTTP {resp.status_code}: {resp.text[:800]}")
        return 2
    body = resp.json()
    if body.get("error") or body.get("node_errors"):
        print(f"FAIL submit {json.dumps(body)[:800]}")
        return 2
    prompt_id = body.get("prompt_id")
    print(f"[ltx-smoke] QUEUED prompt_id={prompt_id}", flush=True)
    deadline = time.time() + 900
    while time.time() < deadline:
        hist = requests.get(f"{URL}/history/{prompt_id}", timeout=15).json()
        row = hist.get(prompt_id) if isinstance(hist, dict) else None
        if row:
            status = ((row.get("status") or {}).get("status_str") or "")
            if status in ("success", "error"):
                print(f"[ltx-smoke] RESULT {status} prompt_id={prompt_id}", flush=True)
                if status != "success":
                    print(json.dumps(row.get("status") or {}, indent=2)[:1500])
                    return 2
                outputs = row.get("outputs") or {}
                print(json.dumps(outputs, indent=2)[:2000], flush=True)
                return 0
        print(f"[ltx-smoke] waiting prompt_id={prompt_id}", flush=True)
        time.sleep(10)
    print("FAIL timeout 900s")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
