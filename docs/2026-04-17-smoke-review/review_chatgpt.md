# ChatGPT smoke-test review -- model=gpt-4.1 elapsed=11.4s

## User prompt

Below is the smoke test and its first-run output. Review for:

1. **Does PASS actually mean what it claims?** Specifically, does a green run here prove the wiring is sound, or are there cases the test would still miss that would blow up in a real ComfyUI execution?

2. **Coverage gaps worth adding before we call the HyWorld wiring stable.** For each gap, say: what to add, why, and whether it's a blocker or a nice-to-have. Be honest — if coverage is adequate, say so.

3. **The FALLBACK branch is not exercised here.** Is that a problem for a smoke test, or is that a separate test's job?

4. **Contract invariants the smoke test should assert that it currently doesn't.** Especially anything touching C7 audio byte-identity, PID/process cleanup, or the worker's sidecar_pid.txt lifecycle.

5. **Honest verdict.** Is the test green result meaningful enough to call the HyWorld wiring "stable", or do we need another iteration? One-line answer: STABLE / NEEDS-ITERATION.

Format: numbered sections matching 1-5. Keep it tight. Bullet points only when useful.

----- SMOKE TEST SOURCE -----
```python
"""
End-to-end smoke test for the HyWorld trio (Bridge -> Poll -> Renderer).

Runs in-process (no ComfyUI Desktop required). Exercises:
    1. Bridge: parses canonical script_json, generates shotlist, writes
       contract files, spawns the sidecar worker subprocess.
    2. Poll: blocks until STATUS.json READY or terminal-fallback status.
    3. Renderer: writes temp WAV from AUDIO tensor (X1 contract), calls
       ffmpeg concat + mux, produces final MP4 or clean FALLBACK.

Pass criteria: no exceptions thrown; final_mp4_path is either a real file
on disk OR empty string ("FALLBACK"); render_log parses; audio temp WAV
was cleaned up.

Exit 0 = green, exit 1 = red.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

OTR_ROOT = Path(r"C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio")
sys.path.insert(0, str(OTR_ROOT))

# Minimal synthetic script with canonical Audio Tokens (Bridge requires at
# least one canonical-typed line; we give it a small but complete scene).
SCRIPT_LINES = [
    {"type": "title", "text": "SIGNAL LOST — Smoke Test Episode"},
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
    "episode_title": "SIGNAL LOST — Smoke Test Episode",
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


def build_audio_tensor():
    """Return a ComfyUI-shaped AUDIO dict: silent mono, 20 s, 44.1 kHz."""
    import torch  # torch is in the ComfyUI venv only
    sr = 44100
    duration_sec = 20.0
    n_samples = int(sr * duration_sec)
    # shape (batch=1, channels=1, samples) — matches what Node 7 emits
    waveform = torch.zeros((1, 1, n_samples), dtype=torch.float32)
    return {"waveform": waveform, "sample_rate": sr}


def run_bridge():
    from otr_v2.hyworld.bridge import HyworldBridge
    node = HyworldBridge()
    job_id, shotlist_json = node.execute(
        script_json=json.dumps(SCRIPT_LINES),
        episode_title="SIGNAL LOST — Smoke Test Episode",
        production_plan_json=json.dumps(PRODUCTION_PLAN),
        scene_manifest_json=json.dumps(SCENE_MANIFEST),
        lane="faithful",
        chaos_ops="",
        chaos_seed=42,
        sidecar_enabled=True,
    )
    return job_id, shotlist_json


def run_poll(job_id: str):
    from otr_v2.hyworld.poll import HyworldPoll
    node = HyworldPoll()
    return node.execute(hyworld_job_id=job_id)


def run_renderer(assets_path: str, shotlist_json: str, audio):
    from otr_v2.hyworld.renderer import HyworldRenderer
    node = HyworldRenderer()
    return node.execute(
        hyworld_assets_path=assets_path,
        episode_audio=audio,
        shotlist_json=shotlist_json,
        episode_title="SIGNAL LOST — Smoke Test Episode",
        crt_postfx=True,
        output_resolution="1280x720",
    )


def main() -> int:
    print("=" * 70)
    print("HyWorld trio smoke test (Bridge -> Poll -> Renderer)")
    print("=" * 70)

    # --- Stage 1: Bridge ---
    print("\n[Stage 1/3] Bridge.execute() ...")
    try:
        job_id, shotlist_json = run_bridge()
    except Exception:
        print("!! Bridge threw:")
        traceback.print_exc()
        return 1
    print(f"  job_id        = {job_id}")
    print(f"  shotlist bytes = {len(shotlist_json)}")
    if job_id.startswith("PARSE_ERROR_"):
        print("!! Bridge reported PARSE_ERROR — script_json validation failed.")
        return 1

    # --- Stage 2: Poll ---
    print("\n[Stage 2/3] Poll.execute() (may block up to 10 min) ...")
    try:
        assets_path, status, detail = run_poll(job_id)
    except Exception:
        print("!! Poll threw:")
        traceback.print_exc()
        return 1
    print(f"  assets_path = {assets_path}")
    print(f"  status      = {status}")
    print(f"  detail      = {detail}")

    # --- Stage 3: Renderer ---
    print("\n[Stage 3/3] Renderer.execute() ...")
    try:
        audio = build_audio_tensor()
    except Exception:
        print("!! build_audio_tensor threw (torch import problem?):")
        traceback.print_exc()
        return 1
    try:
        final_mp4_path, render_log = run_renderer(assets_path, shotlist_json, audio)
    except Exception:
        print("!! Renderer threw:")
        traceback.print_exc()
        return 1
    print(f"  final_mp4_path = {final_mp4_path!r}")
    print("  render_log:")
    for line in render_log.splitlines():
        print(f"    {line}")

    # --- Acceptance ---
    print("\n" + "=" * 70)
    if final_mp4_path:
        fp = Path(final_mp4_path)
        if not fp.exists():
            print(f"!! Renderer returned path {fp} but file does not exist on disk.")
            return 1
        size_mb = fp.stat().st_size / (1024 * 1024)
        print(f"PASS: final MP4 exists ({size_mb:.2f} MB) at {fp}")
    else:
        # Clean FALLBACK — still a pass for the smoke test as long as the chain
        # did not throw. Status must be in the known fallback-set so we know
        # the Poll node made an honest decision (not a silent hang).
        acceptable = {
            "READY",          # poll saw READY but renderer decided to fallback
            "PARSE_ERROR",
            "ERROR", "OOM", "TIMEOUT",
            "ENV_NOT_FOUND", "WORKER_MISSING", "SPAWN_FAILED",
            "DRY_RUN", "SIDECAR_UNAVAILABLE", "WORKER_DEAD",
        }
        if status not in acceptable:
            print(f"!! Renderer returned FALLBACK but poll status {status!r} is not in the acceptable set.")
            return 1
        print(f"PASS: clean FALLBACK with status {status!r}")

    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())

```

----- FIRST-RUN OUTPUT (PASS) -----
```

======================================================================
HyWorld trio smoke test (Bridge -> Poll -> Renderer)
======================================================================

[Stage 1/3] Bridge.execute() ...
  job_id        = hw_53723e8e69ea
  shotlist bytes = 972

[Stage 2/3] Poll.execute() (may block up to 10 min) ...
  assets_path = C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\io\hyworld_out\hw_53723e8e69ea
  status      = READY
  detail      = Motion stub: 2 Ken Burns MP4 clips generated

[Stage 3/3] Renderer.execute() ...
  final_mp4_path = 'C:\\Users\\jeffr\\Documents\\ComfyUI\\custom_nodes\\ComfyUI-OldTimeRadio\\output\\hyworld_renders\\SIGNAL LOST _ Smoke Test Episode_20260417_115932.mp4'
  render_log:
    [11:59:32] Assets: C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\io\hyworld_out\hw_53723e8e69ea
    [11:59:32] Shotlist: 2 shots
    [11:59:32]   s01_01: video clip (6.2s)
    [11:59:32]   s02_01: video clip (3s)
    [11:59:32] Audio: wrote 882000 samples @ 44100 Hz to hyworld_audio_negqtmza.wav
    [11:59:32] ffmpeg: ffmpeg
    [11:59:32] Audio: C:\Users\jeffr\AppData\Local\Temp\hyworld_audio_negqtmza.wav (read-only, byte-identical guarantee)
    [11:59:32] Concatenating 2 clips...
    [11:59:32] Muxing video + audio (audio passthrough, C7 guaranteed)...
    [11:59:32] Output: C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\output\hyworld_renders\SIGNAL LOST _ Smoke Test Episode_20260417_115932.mp4 (0.8 MB)

======================================================================
PASS: final MP4 exists (0.79 MB) at C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\output\hyworld_renders\SIGNAL LOST _ Smoke Test Episode_20260417_115932.mp4
======================================================================

```


---

## Assistant reply

### 1. Does PASS actually mean what it claims?

**Mostly, but not fully.**  
A green run here proves the basic Bridge→Poll→Renderer chain is wired and working for the happy path (READY status, stub motion, audio tensor contract, MP4 output).  
**However, it does *not* guarantee:**
- That the nodes work when called as ComfyUI nodes (with ComfyUI's async, graph, and tensor lifecycles).
- That all contract invariants (e.g., C7 audio byte-identity, temp file cleanup, sidecar process cleanup) are met.
- That error/fallback paths are robust (e.g., worker crash, OOM, parse errors).
- That the system is robust to concurrency or multiple jobs in flight.
- That the temp WAV is actually deleted (only implied, not asserted).
- That the output MP4’s audio is truly byte-identical to the input tensor (not checked).

### 2. Coverage gaps worth adding

**Blockers (should add before calling "stable"):**
- **C7 Audio Byte-Identity:**  
  - *What to add:* After render, extract the audio track from the MP4 and compare its PCM bytes to the original tensor (after round-tripping through WAV).  
  - *Why:* Contract says audio must be byte-identical (C7 guarantee). Not currently checked.
- **Temp WAV Cleanup:**  
  - *What to add:* Assert that the temp WAV file (e.g., `hyworld_audio_*.wav`) is deleted after render.  
  - *Why:* Contract requires temp cleanup; not asserted.
- **Sidecar Worker Cleanup:**  
  - *What to add:* Assert that the worker process is dead and `sidecar_pid.txt` is deleted after render.  
  - *Why:* Prevents orphaned processes; not checked.
- **Renderer FALLBACK Path:**  
  - *What to add:* At least one test that forces a FALLBACK (e.g., by corrupting assets or simulating a worker crash).  
  - *Why:* Ensures fallback logic and cleanup are exercised.

**Nice-to-haves:**
- **Multiple Job IDs:**  
  - *What to add:* Run two jobs in sequence to check for job ID collisions or state bleed.
  - *Why:* Ensures isolation.
- **Non-canonical Script Edge Cases:**  
  - *What to add:* Try a script with missing/extra fields to check Bridge validation.
  - *Why:* Robustness.
- **Output File Integrity:**  
  - *What to add:* Check that the MP4 can be opened and played (e.g., via ffprobe).
  - *Why:* Surface-level smoke, but not a core contract.

### 3. FALLBACK branch not exercised — is that a problem?

**For a smoke test, not a blocker, but...**  
- The main smoke test should pass the happy path, but a *minimal* separate test should force FALLBACK to ensure that path is exercised and cleanup is correct.
- If FALLBACK is never tested, regressions in fallback handling (e.g., temp files left, wrong status, orphaned processes) could go unnoticed.

### 4. Contract invariants not currently asserted

- **C7 Audio Byte-Identity:**  
  - *Not asserted.* Should compare original tensor→WAV→MP4 audio bytes.
- **Temp WAV Cleanup:**  
  - *Not asserted.* Should check temp WAV is deleted after render.
- **Worker Process/sidecar_pid.txt:**  
  - *Not asserted.* Should check worker process is dead and `sidecar_pid.txt` is gone after completion (both success and fallback).
- **No orphaned temp files or processes:**  
  - *Not asserted.* Should check for lingering temp files or processes after test.
- **MP4 output is valid and playable:**  
  - *Not asserted.* Should at least check file opens with ffprobe.

### 5. Honest verdict

**NEEDS-ITERATION**  
- The test is a solid happy-path smoke, but does *not* fully guarantee contract invariants or robust cleanup.  
- Add at least: C7 audio byte check, temp WAV/sidecar cleanup assertions, and a forced FALLBACK test before calling the HyWorld wiring "stable".
