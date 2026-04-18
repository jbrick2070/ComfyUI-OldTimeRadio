"""
Visual Full Pipeline Test — runs the complete OTR workflow with Visual nodes
wired in, via ComfyUI /prompt API. Supersoaker pattern.

This exercises the REAL pipeline end-to-end:
  ScriptWriter -> Director -> VisualBridge (fires early, parallel with audio)
                           -> BatchBark -> Kokoro -> AudioGen -> MusicGen
                              -> SceneSequencer -> AudioEnhance -> EpisodeAssembler
  VisualPoll (waits for sidecar) + EpisodeAssembler audio -> VisualRenderer -> final MP4

Visual Bridge runs RIGHT AFTER Director (no SceneSequencer dependency),
so the sidecar can render video while the audio pipeline is still running.
SignalLostVideo is kept as fallback only — Visual Renderer is the primary
video output node.

Visual sidecar is ENABLED. The worker runs in stub mode (placeholder
stills per shot) since WorldMirror 2.0 is not yet installed. The renderer
composites these stills into video and muxes with the real episode audio.
This proves the full end-to-end path: Bridge -> Worker -> Poll -> Renderer -> MP4.

Usage:
    python scripts/_visual_full_pipeline_test.py
"""
import json
import sys
import time
import uuid
import urllib.request
import urllib.error

COMFYUI = "http://localhost:8000"
TIMEOUT_S = 2400   # 40 min max for full episode
POLL_S = 5

# The full OTR pipeline + Visual nodes appended
WORKFLOW = {
    "1": {
        "class_type": "OTR_Gemma4ScriptWriter",
        "inputs": {
            "episode_title": "Visual Integration Smoke",
            "genre_flavor": "cyberpunk",
            "target_words": 350,
            "num_characters": 2,
            "model_id": "mistralai/Mistral-Nemo-Instruct-2407",
            "custom_premise": "",
            "include_act_breaks": True,
            "self_critique": False,
            "open_close": True,
            "target_length": "short (3 acts)",
            "style_variant": "tense claustrophobic",
            "creativity": "safe & tight",
            "arc_enhancer": False,
            "project_state": "Standard"
        }
    },
    "2": {
        "class_type": "OTR_Gemma4Director",
        "inputs": {
            "script_text": ["1", 0],
            "temperature": 0.4,
            "tts_engine": "bark (standard 8GB)",
            "vintage_intensity": "subtle",
            "project_state": "Standard"
        }
    },
    "11": {
        "class_type": "OTR_BatchBarkGenerator",
        "inputs": {
            "script_json": ["1", 1],
            "production_plan_json": ["2", 0]
        }
    },
    "13": {
        "class_type": "OTR_KokoroAnnouncer",
        "inputs": {
            "script_json": ["1", 1],
            "episode_seed": "",
            "voice_override": "random"
        }
    },
    "15": {
        "class_type": "OTR_BatchAudioGenGenerator",
        "inputs": {
            "script_json": ["1", 1],
            "production_plan_json": ["2", 0],
            "episode_seed": "",
            "model_id": "facebook/audiogen-medium"
        }
    },
    "14": {
        "class_type": "OTR_MusicGenTheme",
        "inputs": {
            "production_plan_json": ["2", 0],
            "episode_seed": "",
            "model_id": "facebook/musicgen-medium",
            "guidance_scale": 3.0
        }
    },
    "3": {
        "class_type": "OTR_SceneSequencer",
        "inputs": {
            "script_json": ["1", 1],
            "production_plan_json": ["2", 0],
            "tts_audio_clips": ["11", 0],
            "announcer_audio_clips": ["13", 0],
            "sfx_audio_clips": ["15", 0],
            "start_line": 0,
            "end_line": 999
        }
    },
    "4": {
        "class_type": "OTR_AudioEnhance",
        "inputs": {
            "audio": ["3", 0],
            "target_sample_rate": 48000,
            "spatial_width": 0.3,
            "haas_delay_ms": 0.8,
            "bass_warmth": 0.15,
            "lpf_cutoff_hz": 16000.0,
            "tape_emulation": "subtle",
            "normalize_dbfs": -1.0
        }
    },
    "7": {
        "class_type": "OTR_EpisodeAssembler",
        "inputs": {
            "scene_audio": ["4", 0],
            "opening_theme_audio": ["14", 0],
            "closing_theme_audio": ["14", 1],
            "episode_title": "Visual Integration Smoke",
            "opening_duration_sec": 10.0,
            "closing_duration_sec": 8.0,
            "crossfade_ms": 500
        }
    },
    "12": {
        "class_type": "OTR_SignalLostVideo",
        "inputs": {
            "audio": ["7", 0],
            "script_json": ["1", 1],
            "production_plan_json": ["2", 0],
            "news_used": ["1", 2]
        }
    },
    # ── Visual v2.0 nodes (dry run, no sidecar) ──
    # Bridge fires RIGHT AFTER Director — no SceneSequencer dependency.
    # This lets the sidecar start rendering while audio is still generating.
    "20": {
        "class_type": "OTR_VisualBridge",
        "inputs": {
            "script_json": ["1", 1],
            "episode_title": "Visual Integration Smoke",
            "production_plan_json": ["2", 0],
            "lane": "faithful",
            "chaos_ops": "",
            "chaos_seed": 42,
            "sidecar_enabled": True
        }
    },
    "21": {
        "class_type": "OTR_VisualPoll",
        "inputs": {
            "visual_job_id": ["20", 0],
            "timeout_sec": 120,
            "poll_interval_sec": 2.0
        }
    },
    "22": {
        "class_type": "OTR_VisualRenderer",
        "inputs": {
            "visual_assets_path": ["21", 0],
            "final_audio_path": ["7", 1],
            "shotlist_json": ["20", 1],
            "episode_title": "Visual Integration Smoke",
            "crt_postfx": True,
            "output_resolution": "1280x720"
        }
    }
}


def api_request(endpoint, data=None):
    url = f"{COMFYUI}{endpoint}"
    if data is not None:
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
    else:
        req = urllib.request.Request(url)
    resp = urllib.request.urlopen(req, timeout=10)
    return json.loads(resp.read())


def main():
    print("=" * 65)
    print("Visual FULL PIPELINE Test (supersoaker pattern)")
    print(f"ComfyUI: {COMFYUI}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config: short (3 acts), 350 words, 2 chars, dry-run Visual")
    print("=" * 65)

    # 1. Pre-flight
    print("\n[1] Pre-flight...")
    try:
        info = api_request("/object_info")
        needed = [
            "OTR_Gemma4ScriptWriter", "OTR_Gemma4Director",
            "OTR_BatchBarkGenerator", "OTR_KokoroAnnouncer",
            "OTR_BatchAudioGenGenerator", "OTR_MusicGenTheme",
            "OTR_SceneSequencer", "OTR_AudioEnhance",
            "OTR_EpisodeAssembler", "OTR_SignalLostVideo",
            "OTR_VisualBridge", "OTR_VisualPoll", "OTR_VisualRenderer",
        ]
        missing = [n for n in needed if n not in info]
        if missing:
            print(f"    FAIL: missing nodes: {missing}")
            return 1
        print(f"    All {len(needed)} required nodes registered")
        print("    [PASS] Pre-flight OK")
    except Exception as e:
        print(f"    FAIL: {e}")
        return 1

    # 2. Queue
    print("\n[2] Posting full pipeline workflow...")
    client_id = uuid.uuid4().hex[:16]
    try:
        result = api_request("/prompt", {"prompt": WORKFLOW, "client_id": client_id})
        prompt_id = result.get("prompt_id", "")
        if not prompt_id:
            err = result.get("error", result.get("node_errors", "unknown"))
            print(f"    FAIL: no prompt_id. Error: {err}")
            if "node_errors" in result:
                for nid, nerr in result["node_errors"].items():
                    print(f"    Node {nid}: {json.dumps(nerr, indent=2)[:300]}")
            return 1
        print(f"    Queued: {prompt_id}")
        print("    [PASS] Workflow accepted")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:800]
        print(f"    FAIL: HTTP {e.code}")
        print(f"    {body}")
        return 1
    except Exception as e:
        print(f"    FAIL: {e}")
        return 1

    # 3. Poll
    print(f"\n[3] Polling (timeout={TIMEOUT_S // 60}min)...")
    print("    This runs the full LLM + TTS + SFX + Music + Assembly + Video pipeline.")
    print("    Expect 10-25 minutes depending on VRAM and model load times.")
    start = time.monotonic()
    status = "PENDING"
    outputs = {}
    last_dot = start

    while (time.monotonic() - start) < TIMEOUT_S:
        try:
            history = api_request(f"/history/{prompt_id}")
            if prompt_id in history:
                entry = history[prompt_id]
                status_info = entry.get("status", {})
                status = status_info.get("status_str", "unknown")
                outputs = entry.get("outputs", {})
                if status in ("success", "error"):
                    break
        except Exception:
            pass

        # Progress dots every 30s
        now = time.monotonic()
        if now - last_dot > 30:
            elapsed_min = (now - start) / 60
            print(f"    ... {elapsed_min:.1f}min elapsed, status={status}")
            last_dot = now

        time.sleep(POLL_S)

    elapsed = time.monotonic() - start
    elapsed_min = elapsed / 60
    print(f"\n    Final status: {status} ({elapsed_min:.1f} min)")

    # 4. Analyze
    print("\n[4] Results...")
    if status == "success":
        print("    [PASS] Full pipeline completed successfully")
    elif status == "error":
        print("    [WARN] Pipeline completed with error")
        # Check which node errored
        status_data = history.get(prompt_id, {}).get("status", {})
        msgs = status_data.get("messages", [])
        for msg in msgs:
            if isinstance(msg, list) and len(msg) > 1:
                print(f"    {msg[0]}: {json.dumps(msg[1])[:200]}")
    else:
        print(f"    FAIL: status={status} after {TIMEOUT_S}s")
        return 1

    # Check key outputs
    output_nodes = {
        "12": "SignalLostVideo",
        "22": "VisualRenderer",
    }
    for nid, name in output_nodes.items():
        out = outputs.get(nid, {})
        if out:
            print(f"    {name} (node {nid}): output present")
            for k, v in out.items():
                val_str = str(v)[:100]
                print(f"      {k}: {val_str}")
        else:
            print(f"    {name} (node {nid}): no output captured")

    # 5. Check io/ artifacts from Bridge
    print("\n[5] Checking Visual contract artifacts...")
    import os
    io_dir = os.path.join(
        r"C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio",
        "io", "visual_in"
    )
    if os.path.isdir(io_dir):
        jobs = os.listdir(io_dir)
        print(f"    Jobs in io/visual_in/: {len(jobs)}")
        for j in jobs:
            jpath = os.path.join(io_dir, j)
            files = os.listdir(jpath) if os.path.isdir(jpath) else []
            total_bytes = sum(
                os.path.getsize(os.path.join(jpath, f))
                for f in files if os.path.isfile(os.path.join(jpath, f))
            )
            print(f"    {j}: {len(files)} files, {total_bytes:,} bytes")
            # Check shotlist has real OTR-parsed content
            sl_path = os.path.join(jpath, "shotlist.json")
            if os.path.isfile(sl_path):
                with open(sl_path, "r", encoding="utf-8") as f:
                    sl = json.load(f)
                shots = sl.get("shots", [])
                scenes = sl.get("scene_count", 0)
                print(f"      shotlist: {scenes} scenes, {len(shots)} shots")
                if shots:
                    print(f"      first shot: {shots[0].get('env_prompt', '?')[:60]}")
                    print(f"      camera: {shots[0].get('camera', '?')}")
                    print("      [PASS] Shotlist generated from live LLM script")
                else:
                    print("      [WARN] Shotlist has 0 shots")
    else:
        print(f"    io/visual_in/ not found (Bridge may not have written)")

    # Summary
    print("\n" + "=" * 65)
    print(f"FULL PIPELINE TEST: status={status}, elapsed={elapsed_min:.1f}min")
    if status == "success":
        print("VERDICT: PIPELINE COMPLETE — audio + video + Visual integration OK")
    else:
        print(f"VERDICT: {status}")
    print("=" * 65)

    return 0 if status in ("success",) else 1


if __name__ == "__main__":
    sys.exit(main())