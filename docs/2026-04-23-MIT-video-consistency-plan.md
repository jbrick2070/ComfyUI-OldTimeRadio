# OTR MIT Video-Consistency Implementation Plan

**Date:** 2026-04-23
**Branch:** `v2.0-alpha-video-stack` (off `v2.0-alpha`; do NOT touch `main`)
**Author:** Claude + Jeffrey Brick
**Status:** Plan; phase 1 shipped (OTR_VideoConcat), phase 2-3 pending

---

## Why this doc exists

Jeffrey wants "character consistency + audio-length video" for OTR episodes, and explicitly rejected three paths during the 2026-04-23 session:

1. **Face-identity anchors (PuLID):** not the need — "I don't want faces really."
2. **Vendoring community packs as OTR_* wrappers:** OTR's past custom video nodes have bottlenecked; better to use existing community code.
3. **Accepting GPL-3 code into the OTR tree:** `VideoHelperSuite` and `SVI-Pro-FLF` are GPL-3.0; swallowing them would relicense OTR from MIT to GPL-3 one-way. Rejected.

The resulting direction: **write our own MIT-original nodes for the gaps, using native ComfyUI + Apache-licensed upstreams as reference only (never code copy).**

This doc is the spec the next session executes against.

---

## Scope — what we need, what we have, what's missing

### Pipeline target

```
Mistral-Nemo polish -> OTR_CheckpointLoaderGated -> [IP-Adapter style lock]
                                                          |
                                                          v
                                            OTR_BatchFluxRender (batched stills)
                                                          |
                                                          v
                                                 OTR_UnloadAll
                                                          |
                                                          v
                                          [FLF video engine: Wan 2.2 FLF2V]
                                                          |
                                                          v
                                                 OTR_VideoConcat
                                                          |
                                                          v
                                              final mux with audio (C7)
```

### Node inventory

| Node | Purpose | Status | License |
|---|---|---|---|
| `OTR_BatchFluxRender` | Batched FLUX stills with Mistral->FLUX handoff | **Shipped** 2026-04-23 | MIT |
| `OTR_CheckpointLoaderGated` | Gate FLUX load on Mistral unload | **Shipped** 2026-04-22 | MIT |
| `OTR_UnloadAll` | Release VRAM between stages | **Shipped** 2026-04-22 | MIT |
| `OTR_VideoConcat` | ffmpeg-concat N clips, C7 audio passthrough | **Shipped** 2026-04-23 | MIT |
| `OTR_FluxIpAdapter` | Environment/style anchor for per-shot FLUX batch | **Pending** (phase 2) | MIT |
| `OTR_WanFlfVideo` | First-last-frame chained I2V for audio-length coverage | **Pending** (phase 3) | MIT |

---

## Phase 2 — `OTR_FluxIpAdapter`

### Goal

Feed one hero reference image (`episode_anchor.png`) to every shot in an `OTR_BatchFluxRender` batch so all N shots share the same world, lighting, palette, and art direction. Environment-mode only (per C6: IP-Adapter for environments, never characters — avoids the Silent Lip Bug).

### Reference (read, don't copy)

- `XLabs-AI/x-flux-comfyui` (Apache 2.0, staged at `/tmp/otr_vendor_stage/x-flux-comfyui/` during 2026-04-23 session) — canonical FLUX IP-Adapter implementation. Reading the API + architecture is MIT-compatible; verbatim code copy is not.
- `Shakker-Labs/FLUX.1-dev-IP-Adapter` HF model card — weight format, expected preprocessing.
- `XLabs-AI/flux-ip-adapter-v2` HF model card — v2 weights with better fp8 compatibility.

### Architecture (to implement fresh under MIT)

```python
class OTR_FluxIpAdapter:
    """
    Applies IP-Adapter style conditioning to a FLUX MODEL.  Accepts
    a reference image + weight, returns a patched MODEL.  Runs in
    the ComfyUI main graph (no sidecar — MODEL passthrough pattern
    matches OTR_CheckpointLoaderGated).
    """

    INPUT_TYPES = {
        "required": {
            "model": ("MODEL",),
            "clip_vision": ("CLIP_VISION",),
            "ip_adapter_weights": (list_files("ip_adapter/"), ),
            "reference_image": ("IMAGE",),
            "weight": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0}),
        },
        "optional": {
            "start_percent": ("FLOAT", {"default": 0.0}),
            "end_percent":   ("FLOAT", {"default": 1.0}),
        },
    }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "OldTimeRadio/flux"
```

### Implementation steps (estimate: 2-3 focused sessions)

1. **Read XLabs code.** Understand how they hook the cross-attention in FLUX's double/single stream blocks. Document the attention-injection pattern in a scratch note; do NOT copy source lines.
2. **Choose weight pack.** Default to `XLabs-AI/flux-ip-adapter-v2` (better fp8 compat on Blackwell). Add `ip_adapter/` to ComfyUI's model path so weights drop in cleanly.
3. **Build CLIP-Vision encoder wrapper.** Standard ComfyUI `CLIP_VISION` flow — probably no new code needed, just consume the upstream output.
4. **Write the attention patch.** The IP-Adapter pattern adds a second cross-attention that conditions on the CLIP-Vision reference embedding instead of (or alongside) the prompt embedding. FLUX has double-stream (img+txt) and single-stream (unified) blocks — patch both. Implement via ComfyUI's `ModelPatcher.set_model_attn2_patch()` so we don't modify the underlying model class.
5. **Test with stub mode first.** `OTR_FLUX_IP_ADAPTER_STUB=1` returns the model unpatched — lets unit tests run without weights.
6. **Real-mode smoke test.** Render 4 shots from `OTR_BatchFluxRender` with IP-Adapter on vs off. Visual diff should show obvious style carry-over on the "on" run.
7. **Blackwell fp8 kernel check.** If the attention patch breaks on sm_120 fp8, fall back to bf16 cast on the IP-Adapter weights only (FLUX body stays fp8).

### Kill criteria

- IP-Adapter weights fail to load on Blackwell fp8 → switch to SDXL IP-Adapter + SDXL anchors as Stage 1 fallback (existing ROADMAP fallback).
- Model patch breaks the base FLUX render (regression in `OTR_BatchFluxRender` output without IP-Adapter even bypassed) → revert patch; IP-Adapter becomes a separate MODEL-producing node in a parallel branch.
- VRAM peak > 14.5 GB on 1024x1024 FLUX + IP-Adapter → quantize IP-Adapter weights to fp8 or drop weight parameter ceiling.

### Unit tests to ship

- Registry test (OTR_FluxIpAdapter in NODE_CLASS_MAPPINGS)
- INPUT_TYPES schema
- Stub mode: model passes through unpatched
- Weight-missing path returns stub + logs reason
- RETURN_TYPES == ("MODEL",)
- No torch import at module load (torch only imported inside `apply()`)

---

## Phase 3 — `OTR_WanFlfVideo`

### Goal

Given N+1 FLUX stills (from IP-Adapter-locked `OTR_BatchFluxRender`), generate N video clips where clip_i runs from still_i to still_i+1. Each clip's last frame IS the next clip's first frame by construction, so `OTR_VideoConcat` produces a seamless video with zero crossfade artefacts.

### Two candidate paths (prototype both, pick winner after Blackwell test)

#### Path A — Native ComfyUI Wan 2.2 FLF2V template (preferred if it works)

ComfyUI core ships a Wan 2.2 FLF2V (First-Last-Frame to Video) template as of late 2025. Built on diffusers' `WanImageToVideoPipeline` with `image` and `last_image` inputs.

**Pros:**
- Zero new code — we just wire existing nodes into the OTR workflow JSON
- Apache 2.0 (ComfyUI core license) — fully MIT-compatible
- Maintained by the Comfy-Org team; updates arrive automatically

**Cons:**
- Blackwell sm_120 fp8 kernel compatibility unknown until tested
- Not in OTR's code — harder to coordinate with OTR's VRAM/unload discipline

**Implementation:** mostly a workflow-JSON wiring exercise. One small OTR helper node (`OTR_WanFlfShotList`) to convert `OTR_BatchFluxRender`'s IMAGE batch into the (start_image, end_image) pair sequence expected by the native Wan FLF2V node. That helper is ~60 lines of pure Python.

#### Path B — OTR-native Wan 2.2 FLF2V node (fallback if native template fails)

Thin wrapper around `diffusers.WanImageToVideoPipeline` loaded via Comfy-Org safetensors (same pattern as `OTR_CheckpointLoaderGated` for FLUX). Accepts (start_image, end_image, prompt, duration_s) per shot, emits .mp4 path.

```python
class OTR_WanFlfVideo:
    INPUT_TYPES = {
        "required": {
            "model_name": (list_files("diffusers/wan22-i2v/"), ),
            "image_batch": ("IMAGE",),   # N+1 stills from OTR_BatchFluxRender
            "prompts": ("STRING", {"multiline": True}),  # N newline-separated prompts
            "duration_s": ("FLOAT", {"default": 8.0, "min": 2.0, "max": 10.0}),
            "output_dir": ("STRING",),
        },
    }
    RETURN_TYPES = ("STRING",)   # newline-separated .mp4 paths for OTR_VideoConcat
    RETURN_NAMES = ("clip_paths",)
    FUNCTION = "render"
    CATEGORY = "OldTimeRadio/video"
```

**Implementation steps (estimate: 3-4 focused sessions):**

1. **Read `visual/backends/wan21_loop.py`** — existing OTR Wan 2.1 I2V sidecar backend (Day 6 of video stack sprint). Adapt the loading + FP8 + VRAMCoordinator pattern; change the target to Wan 2.2 I2V + add `last_image` conditioning.
2. **Download Wan 2.2 I2V weights** to the centralized models folder (Jeffrey's model-tidy policy). The diffusers format is what we want.
3. **Build the shot-pair iterator.** `images[0..N-1]` as starts, `images[1..N]` as ends. Each pair becomes one pipeline call.
4. **Iterate pipeline calls with explicit teardown.** Per `reference_chained_backend_teardown.md`: after each clip, `remove_all_hooks + CPU-move + gc + empty_cache` before the next call. Single MODEL load at the start, torn down at the end.
5. **Duration clamp** at 10.0s per C4.
6. **Stub mode** (`OTR_WAN_FLF_STUB=1`) emits minimal-valid MP4s with per-clip unique payloads (so tests can verify ordering). Same pattern as `visual/backends/wan21_loop.py`.

**Pros:**
- Full OTR-native discipline (VRAM gate, C7 audio-safe, per-clip meta.json)
- Can evolve with OTR's pipeline invariants
- MIT-clean (our code, with Apache-license diffusers as a library dependency)

**Cons:**
- More code to maintain
- Risk of the "custom video nodes bottleneck" historical pattern — mitigation: keep this as a thin pipeline caller, not a re-encoding or preprocessing loop

### Test-first Blackwell check

Before committing to A or B, verify Wan 2.2 I2V FP8 loads on sm_120 at all:

```powershell
# 30-minute smoke test — decides A vs B
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe -c "
import torch
from diffusers import WanImageToVideoPipeline
pipe = WanImageToVideoPipeline.from_pretrained('<WAN22_MODEL_ID>', torch_dtype=torch.float8_e4m3fn)
pipe.to('cuda')
print('Wan2.2 I2V fp8 load: OK')
print('Free VRAM after load:', torch.cuda.mem_get_info()[0] / 1e9, 'GB')
"
```

If this lands without OOM or kernel errors, Path A is reachable. If it crashes, Path B's custom sidecar isolation is the safer bet (because we control exactly when the model touches VRAM).

### Kill criteria

- Wan 2.2 FP8 peaks > 14.5 GB during inference on 576x1024 24fps clips → downgrade to fp16 with `enable_model_cpu_offload` (accept 2x slowdown) or fall back to Wan 2.1 I2V (existing backend) + accept non-FLF chain (visible seams at clip boundaries).
- Temporal drift across clips despite shared endpoint frames → VAE re-encode each endpoint frame from the previous clip's last frame before passing in, so both clips actually see the same VAE-encoded latent.
- > 8-hour render time for a 20-min episode on Blackwell → cap at 10-min episodes until Wan 2.3 or newer arrives.

### Unit tests to ship

- Registry test
- INPUT_TYPES schema
- Shot-pair iterator: N stills → N-1 pairs, correctly ordered
- Stub mode: emits N valid MP4s, byte-distinct from each other
- C4 duration clamp enforced
- Stub-mode clip path strings returned in concat-ready order
- No torch import at module load

---

## Shipping checklist (when all three phases land)

- [ ] OTR_VideoConcat (shipped 2026-04-23)
- [ ] OTR_FluxIpAdapter
- [ ] OTR_WanFlfVideo (Path A or B)
- [ ] Workflow JSON: `workflows/otr_scifi_16gb_TEST.json` updated with new nodes wired in, preserving the "one big intact JSON" goal
- [ ] README.md updated: capability list, what the new nodes do
- [ ] LICENSE unchanged (still MIT)
- [ ] All new code UTF-8 no BOM
- [ ] No "dummy" in code/comments — "placeholder" / "stub" only
- [ ] No profanity in code/comments/output
- [ ] Regression suites green: `test_core.py`, `test_dropdown_guardrails.py`, `test_workflow_json_guardrails.py`, Bug Bible if sister repo mounted
- [ ] C7 audio byte-identical gate green (`tests/v2/test_audio_byte_identical.py`)

---

## Deferred to v2.1 or later

- **Scene-Geometry-Vault** (P2 in ROADMAP): cross-episode geometry lock. Phase 2 of this doc gives shot-to-shot style carry within one episode; full world-persistence across episodes is a separate problem.
- **Crossfade mode in OTR_VideoConcat**: ffmpeg `xfade` filter between clips. Only needed if we ever use non-FLF video engines; FLF chains don't need it by construction.
- **OTR_StyleAnchorCache**: save the IP-Adapter reference image + its CLIP-Vision embedding to disk so Scene 1 of Episode 5 can load the same anchor as Scene 1 of Episode 4.

---

## Status snapshot

- Phase 1 (OTR_VideoConcat): **done** in this session, 2026-04-23. Tests in `tests/test_otr_video_concat.py`. Wire into workflow JSON once OTR_WanFlfVideo ships.
- Phase 2 (OTR_FluxIpAdapter): **planned**. Next session's primary focus.
- Phase 3 (OTR_WanFlfVideo): **planned**. Blackwell fp8 smoke test gates path A vs B.

Jeffrey owns: pull request review, tag cuts, the Blackwell smoke tests.
Next Claude session owns: implementing phase 2, running regression on phase 1.
