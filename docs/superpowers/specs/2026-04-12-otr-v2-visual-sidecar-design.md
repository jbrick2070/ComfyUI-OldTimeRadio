# OTR v2.0 Visual Sidecar

Add a visual generation pipeline to ComfyUI-OldTimeRadio that runs alongside the existing audio pipeline without modifying it. The visual sidecar generates B-roll diffusion clips and (later) 3D character lipsync shots, compositing them under the CRT post-process to produce a single audio-visual MP4 episode.

## Motivation

v1.5 produces audio-only episodes with a CRT static video overlay. v2 adds real visual content — environment B-roll via LTX-2.3 diffusion and character shots via Blender+Rhubarb NG — while treating the audio pipeline as frozen and immutable. The audio DAG must remain byte-identical to v1.5 output at every step.

## Hardware Constraints

- RTX 5080, 16 GB VRAM, Blackwell architecture, single GPU, no cloud
- LTX-2.3 22B DiT at `torch.float8_e4m3fn` uses ~11 GB, leaving 5 GB for audio residuals + OS + ffmpeg
- All visual generation must run in subprocesses spawned with `multiprocessing.get_context("spawn")` for OS-level VRAM reclaim

## Architecture

Six-phase build, each gated by an audio regression test (SHA-256 match against v1.5 baseline).

### Phase 0: Audio Regression Baseline

Capture a known-good audio output from `otr_scifi_16gb_full.json` with a fixed seed and prompt. Save the WAV + SHA-256 hash. Write a regression test that re-runs the workflow and asserts byte-identical output. This test must pass at every subsequent gate.

### Phase 1: Visual Plan Schema Extension

Teach `OTR_Gemma4Director` to emit a `visual_plan` block inside its existing `production_plan_json` output. No IO changes to the node — only the system prompt and post-processing change. The schema defines scenes with `scene_id`, `shot_type` (establishing/character/insert), `character_ids`, `environment`, `camera`, `duration_s`, `audio_offset_s`, and `seed`. On schema validation failure, emit an empty `visual_plan` (graceful degradation — audio still ships).

### Phase 2: VisualGateway Sink Node

A new node downstream of the audio pipeline that consumes only strings + final audio: `production_plan_json` (STRING from Director), `scene_manifest_json` (STRING from SceneSequencer), `episode_audio` (AUDIO from EpisodeAssembler). Outputs a bundled `gateway_payload_json`. This node does nothing computational — it establishes the wiring contract before heavy work goes behind it. Because it only takes strings and the final audio output, upstream widget drift is impossible.

### Phase 3: VisualSidecar (Diffusion B-roll)

First real visual output. For each scene in `visual_plan` where `character_ids` is empty, spawn `ltx_runner.py` as a subprocess. The runner loads LTX-2.3 with `torch.float8_e4m3fn`, generates the clip, writes MP4 to temp, exits. Clips over 12 seconds auto-chunk into segments and crossfade-concat with ffmpeg. Character scenes get black frame placeholders. Every 3-4 clips, kill and respawn the subprocess to defeat VRAM fragmentation.

### Phase 4: Composite into SignalLostVideo

Add one new optional input to `OTR_SignalLostVideo` (node #12): `visual_overlay` (STRING path to MP4). Place it as the last input slot to minimize widget drift. When present, ffmpeg-composite it under the CRT scanline/static layer. When absent, behave exactly as v1.5.

### Phase 5: Blender + Rhubarb NG for Character Shots

Replace black-frame placeholders with rigged 3D characters lipsynced to the audio. Spawn Blender headless as subprocess, call Rhubarb NG on audio slices for visemes, apply as blendshape keyframes, render PNG sequences. Requires a minimal asset library (2 rigged characters + 2 environments). Highest-risk phase — do not start until Gate 4 is solid.

### Phase 6: Ship

Run full v1.5 regression suite (89 tests) + full v2 test suite. Generate one full episode end-to-end as smoke test. Tag v2.0, merge to main.

## Key Technical Decisions

**Subprocess isolation (C3):** All visual generation runs in subprocesses spawned with `multiprocessing.get_context("spawn")`. This is the only reliable way to reclaim VRAM across PyTorch + Blender on a single GPU.

**LTX-2.3 clip cap (C4):** 10-12 seconds max (257 frames @ 24fps). 20-second clips OOM intermittently on 5080 with IP-Adapter + dual encoders.

**Blackwell-native FP8 (C5):** `torch.float8_e4m3fn` drops the 22B DiT to ~11 GB.

**IP-Adapter environments only (C6):** Never for characters with lipsync — aspect-ratio tensor misalignment causes the "Silent Lip Bug."

**No CheckpointLoaderSimple (C2):** Stock diffusion nodes load checkpoints into the ComfyUI process while audio models hold residual VRAM, causing OOM on 16 GB.

## Subprocess Hardening Checklist

Every subprocess runner must:
- Set `os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"` before `import torch`
- Call explicit `torch.cuda.init()` in child
- Wrap generation in `with torch.cuda.device(0):`
- Run `del model; gc.collect(); torch.cuda.empty_cache()` before exit
- Propagate exit codes for error handling in parent

## Directory Structure

```
custom_nodes/ComfyUI-OldTimeRadio/
  otr_v2/
    __init__.py
    visual_gateway.py          # Phase 2
    visual_sidecar.py          # Phase 3+
    subprocess_runners/
      __init__.py
      ltx_runner.py            # Phase 3
      blender_runner.py        # Phase 5
    schema/
      visual_plan.schema.json  # Phase 1
  tests/v2/
    test_audio_byte_identical.py
    test_visual_plan_schema.py
    test_subprocess_isolation.py
    fixtures/
      baseline_v1.5.wav
      baseline_v1.5.sha256
  workflows/
    otr_v2_sidecar.json        # Phase 6
```

## Visual Plan Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "OTR v2 Visual Plan",
  "type": "object",
  "properties": {
    "visual_plan": {
      "type": "object",
      "properties": {
        "scenes": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["scene_id", "shot_type", "duration_s", "audio_offset_s"],
            "properties": {
              "scene_id":        { "type": "string" },
              "shot_type":       { "enum": ["establishing", "character", "insert"] },
              "character_ids":   { "type": "array", "items": { "type": "string" } },
              "environment":     { "type": "string" },
              "camera":          { "type": "string" },
              "duration_s":      { "type": "number", "minimum": 0.5, "maximum": 60.0 },
              "audio_offset_s":  { "type": "number", "minimum": 0.0 },
              "seed":            { "type": "integer" }
            }
          }
        }
      },
      "required": ["scenes"]
    }
  }
}
```

## Troubleshooting

- **OOM during LTX run:** Reduce clip cap from 12s to 8s in `ltx_runner.py`. Do NOT increase VRAM by killing audio models.
- **Audio regression test fails:** Revert the change immediately. The audio path is non-negotiable.
- **Subprocess won't release VRAM:** Check `CUDA_VISIBLE_DEVICES` is inherited and `start_method="spawn"` is set explicitly. Never use `fork`.
- **Widget drift suspected:** Diff `widgets_values` arrays before/after. If any v1.5 node's array changed length, the change violates C1.
- **`visual_plan` missing from Director output:** Graceful degrade to empty plan. Episode ships audio-only with black visual track. Never crash the audio pipeline.

## Out of Scope

- Real-time audio-reactive video generation
- Multi-GPU support
- Cloud diffusion fallback
- NVIDIA Audio2Face (requires incompatible RTX A-series)
- Rewriting any v1.5 audio node internals
- Adding a second LLM for visual planning
