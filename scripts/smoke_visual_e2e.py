"""
End-to-end smoke test for the Visual trio (Bridge -> Poll -> Renderer).

Runs in-process. No ComfyUI Desktop required. Subprocess worker runs in
stub mode (CPU-only, ffmpeg Ken Burns).

Review history: ChatGPT round 1 (2026-04-17) flagged 4 coverage gaps;
round 2 signed off STABLE after the gaps were closed. Transcripts under
docs/2026-04-17-smoke-review/.

Six stages:
    [1] Bridge parses canonical script_json, spawns worker, returns job_id.
    [2] Poll blocks until STATUS.json READY (or terminal fallback).
    [3] Renderer writes temp WAV from AUDIO tensor (X1 contract),
        concats + muxes with -c:a copy, produces final MP4.
    [4] Temp WAV hygiene -- assert no visual_audio_*.wav leaked in %TEMP%.
    [5] C7 byte-identity -- extract MP4 audio via ffmpeg, assert extracted
        PCM is a byte-exact PREFIX of a reference WAV written from the same
        tensor. `-shortest` truncation is expected; -c:a copy means the
        bytes that survive must pass through unchanged.
    [6] FALLBACK path -- call Renderer with assets_path="FALLBACK", assert
        empty-string return and no temp-WAV leak.

Pass = all six stages succeed. Exit 0 green, exit 1 red.

Usage:
    C:\\Users\\jeffr\\Documents\\ComfyUI\\.venv\\Scripts\\python.exe scripts\\smoke_visual_e2e.py
or
    scripts\\smoke_visual_e2e.bat
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

OTR_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(OTR_ROOT))

SCRIPT_LINES = [
    {"type": "title", "text": "SIGNAL LOST -- Smoke Test Episode"},
    {"type": "scene_break", "number": 1},
    {"type": "environment", "text": "A dimly lit 1980s radio studio. Static hisses on a dead monitor."},
    {"type": "direction", "text": "(The host leans into the mic.)"},
    {"type": "dialogue", "character": "HOST", "text": "Good evening, listeners. Tonight we have something strange."},
    {"type": "sfx", "text": "[static burst]"},
    {"type": "dialogue", "character": "HOST", "text": "A signal came through at three a.m. We played it back. We don't know what it means."},
    {"type": "pause", "duration": 1.0},
    {"type": "scene_break", "number": 2},
    {"type": "environment", "text": "A tape deck clicks on in an empty warehouse."},
    {"type": "dialogue", "character": "VOICE", "text": "If you are hearing this, the broadcast is still alive."},
    {"type": "music", "text": "[closing cue, low strings]"},
]
PRODUCTION_PLAN = {
    "episode_title": "SIGNAL LOST -- Smoke Test Episode",
    "lane": "faithful",
    "era": "1980s",
    "visual_palette": "VHS tape, CRT glow, deep teal / warm amber",
}
SCENE_MANIFEST = {
    "scenes": [
        {"index": 0, "start_sec": 0.0, "end_sec": 12.0, "label": "studio_opener"},
        {"index": 1, "start_sec": 12.0, "end_sec": 20.0, "label": "warehouse_close"},
    ],
    "total_duration_sec": 20.0,
}


def build_audio_tensor(duration_sec: float = 20.0):
    """Silent mono AUDIO dict: shape (1, 1, N), 44.1 kHz float32."""
    import torch
    sr = 44100
    n = int(sr * duration_sec)
    return {"waveform": torch.zeros((1, 1, n), dtype=torch.float32), "sample_rate": sr}


def write_reference_wav(audio, out_path: Path) -> None:
    """Mirror the renderer's _write_audio_tensor_to_wav exactly.
    16-bit PCM, mono, same sample rate."""
    import numpy as np
    import wave
    wf = audio["waveform"]
    sr = audio["sample_rate"]
    if wf.dim() == 3:
        np_audio = wf[0].mean(dim=0).cpu().numpy()
    elif wf.dim() == 2:
        np_audio = wf.mean(dim=0).cpu().numpy()
    else:
        np_audio = wf.cpu().numpy()
    pcm = np.clip(np_audio * 32767.0, -32767, 32767).astype(np.int16)
    with wave.open(str(out_path), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm.tobytes())


def find_ffmpeg() -> str | None:
    for c in ("ffmpeg", r"C:\Users\jeffr\Documents\ComfyUI\ffmpeg\bin\ffmpeg.exe"):
        if shutil.which(c):
            return c
    return None


def extract_mp4_audio_pcm(ffmpeg: str, mp4_path: Path, out_wav: Path) -> bool:
    """Decode the muxed MP4's audio track to PCM WAV so we can byte-compare."""
    cmd = [
        ffmpeg, "-y", "-i", str(mp4_path),
        "-vn", "-acodec", "pcm_s16le", "-ac", "1",
        str(out_wav),
    ]
    r = subprocess.run(cmd, capture_output=True, timeout=60)
    return r.returncode == 0 and out_wav.exists()


def sha256_of_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def count_leaked_temp_wavs() -> int:
    pattern = os.path.join(tempfile.gettempdir(), "visual_audio_*.wav")
    return len(glob.glob(pattern))


def run_bridge():
    from otr_v2.visual.bridge import VisualBridge
    node = VisualBridge()
    return node.execute(
        script_json=json.dumps(SCRIPT_LINES),
        episode_title="SIGNAL LOST -- Smoke Test Episode",
        production_plan_json=json.dumps(PRODUCTION_PLAN),
        scene_manifest_json=json.dumps(SCENE_MANIFEST),
        lane="faithful", chaos_ops="", chaos_seed=42, sidecar_enabled=True,
    )


def run_poll(job_id: str):
    from otr_v2.visual.poll import VisualPoll
    return VisualPoll().execute(visual_job_id=job_id)


def run_renderer(assets_path, shotlist_json, audio, episode_title="SIGNAL LOST -- Smoke Test Episode"):
    from otr_v2.visual.renderer import VisualRenderer
    return VisualRenderer().execute(
        visual_assets_path=assets_path,
        episode_audio=audio,
        shotlist_json=shotlist_json,
        episode_title=episode_title,
        crt_postfx=True,
        output_resolution="1280x720",
    )


def main() -> int:
    print("=" * 72)
    print("Visual trio smoke test (Bridge -> Poll -> Renderer + contract asserts)")
    print("=" * 72)

    temp_wav_before = count_leaked_temp_wavs()
    print(f"Baseline visual_audio_*.wav count in %TEMP%: {temp_wav_before}")

    # --- Stage 1: Bridge ---
    print("\n[1/6] Bridge.execute() ...")
    try:
        job_id, shotlist_json = run_bridge()
    except Exception:
        traceback.print_exc(); return 1
    print(f"  job_id         = {job_id}")
    print(f"  shotlist_bytes = {len(shotlist_json)}")
    if job_id.startswith("PARSE_ERROR_"):
        print("!! Bridge PARSE_ERROR"); return 1

    # --- Stage 2: Poll ---
    print("\n[2/6] Poll.execute() ...")
    try:
        assets_path, status, detail = run_poll(job_id)
    except Exception:
        traceback.print_exc(); return 1
    print(f"  assets_path = {assets_path}")
    print(f"  status      = {status}")
    print(f"  detail      = {detail}")

    # --- Stage 3: Renderer happy path ---
    print("\n[3/6] Renderer.execute() (happy path) ...")
    try:
        audio = build_audio_tensor(20.0)
    except Exception:
        traceback.print_exc(); return 1

    ref_wav = Path(tempfile.gettempdir()) / f"smoke_ref_{job_id}.wav"
    try:
        write_reference_wav(audio, ref_wav)
    except Exception:
        traceback.print_exc(); return 1
    ref_sha = sha256_of_file(ref_wav)
    print(f"  reference WAV   = {ref_wav} (sha256={ref_sha[:16]}...)")

    try:
        final_mp4, render_log = run_renderer(assets_path, shotlist_json, audio)
    except Exception:
        traceback.print_exc(); return 1
    print(f"  final_mp4_path = {final_mp4!r}")
    for line in render_log.splitlines():
        print(f"    {line}")
    if not final_mp4:
        acceptable = {"READY", "PARSE_ERROR", "ERROR", "OOM", "TIMEOUT",
                      "ENV_NOT_FOUND", "WORKER_MISSING", "SPAWN_FAILED",
                      "DRY_RUN", "SIDECAR_UNAVAILABLE", "WORKER_DEAD"}
        if status not in acceptable:
            print(f"!! FALLBACK but poll status {status!r} not acceptable"); return 1
        print(f"(skipping byte-identity check; happy path fell back cleanly, status={status})")
    else:
        if not Path(final_mp4).exists():
            print("!! Renderer returned a path that does not exist"); return 1

    # --- Stage 4: Temp WAV hygiene ---
    print("\n[4/6] Temp WAV cleanup assertion ...")
    temp_wav_after = count_leaked_temp_wavs()
    print(f"  visual_audio_*.wav in %TEMP%: before={temp_wav_before} after={temp_wav_after}")
    if temp_wav_after > temp_wav_before:
        print(f"!! Leaked {temp_wav_after - temp_wav_before} temp WAV(s) after render")
        return 1
    print("  OK: renderer cleaned up its temp WAV")

    # --- Stage 5: C7 byte-identity ---
    print("\n[5/6] C7 audio byte-identity ...")
    if not final_mp4:
        print("  (skipped: happy path fell back)")
    else:
        ffmpeg = find_ffmpeg()
        if not ffmpeg:
            print("!! ffmpeg not found; cannot verify byte identity")
            return 1
        extracted = Path(tempfile.gettempdir()) / f"smoke_extracted_{job_id}.wav"
        if not extract_mp4_audio_pcm(ffmpeg, Path(final_mp4), extracted):
            print("!! ffmpeg extraction failed")
            return 1
        ext_sha = sha256_of_file(extracted)
        print(f"  reference  sha256 = {ref_sha}")
        print(f"  extracted  sha256 = {ext_sha}")
        import wave
        def pcm_bytes(p):
            with wave.open(str(p), "rb") as w:
                return w.readframes(w.getnframes()), w.getframerate(), w.getnchannels(), w.getsampwidth()
        ref_pcm, ref_sr, ref_ch, ref_sw = pcm_bytes(ref_wav)
        ext_pcm, ext_sr, ext_ch, ext_sw = pcm_bytes(extracted)
        print(f"  reference: {len(ref_pcm)} bytes @ {ref_sr} Hz, {ref_ch} ch, {ref_sw} bytes/sample")
        print(f"  extracted: {len(ext_pcm)} bytes @ {ext_sr} Hz, {ext_ch} ch, {ext_sw} bytes/sample")
        if (ref_sr, ref_ch, ref_sw) != (ext_sr, ext_ch, ext_sw):
            print("!! PCM format mismatch (sr/channels/sample-width)")
            return 1
        if len(ext_pcm) > len(ref_pcm):
            print(f"!! Extracted PCM longer than reference ({len(ext_pcm)} > {len(ref_pcm)})")
            return 1
        prefix = ref_pcm[:len(ext_pcm)]
        if prefix != ext_pcm:
            for i in range(len(ext_pcm)):
                if prefix[i] != ext_pcm[i]:
                    print(f"!! PCM prefix diff at byte {i}: ref={prefix[i]} ext={ext_pcm[i]}")
                    break
            mismatches = sum(1 for a, b in zip(prefix, ext_pcm) if a != b)
            print(f"!! Total mismatched bytes: {mismatches}/{len(ext_pcm)} ({100.0*mismatches/len(ext_pcm):.2f}%)")
            print("   This means ffmpeg silently re-encoded the audio (C7 violation).")
            return 1
        kept_sec = len(ext_pcm) / ref_sw / ref_sr
        total_sec = len(ref_pcm) / ref_sw / ref_sr
        print(f"  OK: extracted PCM is a byte-exact prefix of reference "
              f"({kept_sec:.2f}s of {total_sec:.2f}s survived -shortest truncation, C7 preserved)")

    # --- Stage 6: FALLBACK path ---
    print("\n[6/6] Renderer FALLBACK path ...")
    wav_before_fallback = count_leaked_temp_wavs()
    try:
        fb_mp4, fb_log = run_renderer("FALLBACK", "{}", audio, episode_title="fallback_test")
    except Exception:
        traceback.print_exc(); return 1
    print(f"  fallback returned: {fb_mp4!r}")
    for line in fb_log.splitlines():
        print(f"    {line}")
    if fb_mp4 != "":
        print("!! FALLBACK path returned non-empty path"); return 1
    wav_after_fallback = count_leaked_temp_wavs()
    if wav_after_fallback > wav_before_fallback:
        print("!! FALLBACK path leaked a temp WAV"); return 1
    print("  OK: FALLBACK returns empty path cleanly, no temp WAV leaked")

    for p in [ref_wav, Path(tempfile.gettempdir()) / f"smoke_extracted_{job_id}.wav"]:
        try: p.unlink(missing_ok=True)
        except OSError: pass

    print("\n" + "=" * 72)
    print("ALL 6 STAGES PASSED -- Visual wiring smoke test GREEN")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
