"""
Extract audio from a completed ComfyUI run by prompt_id.

Usage:
  python tests/v2/_extract_from_history.py <prompt_id>

Fetches the history entry for the given prompt, downloads the PreviewAudio
output (FLAC), converts to WAV, and saves baseline fixtures.
"""

import hashlib
import json
import os
import sys

COMFYUI_BASE = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8000")
_HERE = os.path.dirname(os.path.abspath(__file__))
_FIXTURES = os.path.join(_HERE, "fixtures")

PREVIEW_NID = "9999"


def main():
    import requests

    if len(sys.argv) < 2:
        print("Usage: python _extract_from_history.py <prompt_id>")
        sys.exit(1)

    prompt_id = sys.argv[1]
    print(f"Fetching history for prompt {prompt_id}...")

    resp = requests.get(f"{COMFYUI_BASE}/history/{prompt_id}", timeout=15)
    resp.raise_for_status()
    history = resp.json()

    if prompt_id not in history:
        print(f"ERROR: prompt {prompt_id} not found in history")
        print(f"Available keys: {list(history.keys())[:5]}")
        sys.exit(1)

    entry = history[prompt_id]
    outputs = entry.get("outputs", {})
    status = entry.get("status", {})
    print(f"Status: {json.dumps(status, indent=2)}")
    print(f"Output nodes: {list(outputs.keys())}")

    # Find audio
    audio_list = outputs.get(PREVIEW_NID, {}).get("audio", [])
    if not audio_list:
        for nid, node_out in outputs.items():
            if "audio" in node_out:
                audio_list = node_out["audio"]
                print(f"Found audio on node #{nid}")
                break

    if not audio_list:
        print("ERROR: No audio output found")
        sys.exit(1)

    audio_info = audio_list[0]
    filename = audio_info["filename"]
    subfolder = audio_info.get("subfolder", "")
    audio_type = audio_info.get("type", "temp")

    print(f"Downloading: {filename} (type={audio_type})")
    params = {"filename": filename, "type": audio_type}
    if subfolder:
        params["subfolder"] = subfolder
    resp = requests.get(f"{COMFYUI_BASE}/view", params=params, timeout=60)
    resp.raise_for_status()
    audio_bytes = resp.content
    print(f"  Downloaded {len(audio_bytes):,} bytes")

    # Convert to WAV
    if filename.endswith(".flac"):
        import subprocess, tempfile
        with tempfile.NamedTemporaryFile(suffix=".flac", delete=False) as f:
            f.write(audio_bytes)
            tmp_in = f.name
        tmp_out = tmp_in.replace(".flac", ".wav")
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", tmp_in,
                "-acodec", "pcm_s16le", "-ar", "48000", "-ac", "1", tmp_out
            ], capture_output=True, check=True, timeout=120)
            with open(tmp_out, "rb") as f:
                wav_bytes = f.read()
        finally:
            for p in (tmp_in, tmp_out):
                try: os.unlink(p)
                except: pass
    else:
        wav_bytes = audio_bytes

    print(f"  WAV: {len(wav_bytes):,} bytes")

    # Save fixtures
    os.makedirs(_FIXTURES, exist_ok=True)
    wav_path = os.path.join(_FIXTURES, "baseline_v1.5.wav")
    sha_path = os.path.join(_FIXTURES, "baseline_v1.5.sha256")

    with open(wav_path, "wb") as f:
        f.write(wav_bytes)

    sha = hashlib.sha256(wav_bytes).hexdigest()
    with open(sha_path, "w", encoding="utf-8") as f:
        f.write(sha)

    print(f"\nBaseline saved:")
    print(f"  WAV: {wav_path}")
    print(f"  SHA: {sha_path}")
    print(f"  SHA-256: {sha}")
    print("\nPhase 0 baseline captured successfully.")


if __name__ == "__main__":
    main()
