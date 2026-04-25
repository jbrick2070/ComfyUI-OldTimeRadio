# HuMo POC recipe — RTX 5080 16 GB Blackwell

**Status:** ready to download + run. Everything verified against the live
ComfyUI registry on Jeffrey's machine 2026-04-24.

## Why we already have native support

The `WanHuMoImageToVideo` node is **already registered** in your running
ComfyUI (verified via `GET /object_info`). Despite the Desktop wrapper
showing "0.19.5", the underlying ComfyUI core is at 0.3.59+ which ships
HuMo as a first-class node alongside Wan 2.1/2.2.

The official template is also already on disk at:
`C:\Users\jeffr\Documents\ComfyUI\.venv\Lib\site-packages\comfyui_workflow_templates_media_video\templates\video_humo.json`

A copy is now in this repo at:
`workflows/external_examples/video_humo_official_native.json`

## Models you need (5 files, ~18-19 GB total disk)

Download via `huggingface-cli` or browser, place in the listed
ComfyUI subdirectories:

| File | Goes to | URL | Size | Likely have? |
|---|---|---|---|---|
| `humo_17B_fp8_e4m3fn.safetensors` | `models/diffusion_models/` | [Comfy-Org/HuMo_ComfyUI](https://huggingface.co/Comfy-Org/HuMo_ComfyUI/resolve/main/split_files/diffusion_models/humo_17B_fp8_e4m3fn.safetensors) | **15.89 GB** | NO — main download (over 16 GB ceiling, see VRAM note below) |
| `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | `models/text_encoders/` | [Comfy-Org/Wan_2.1_ComfyUI_repackaged](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors) | ~5 GB | maybe — Wan 2.1 share |
| `wan_2.1_vae.safetensors` | `models/vae/` | [Comfy-Org/Wan_2.2_ComfyUI_Repackaged](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors) | ~500 MB | maybe |
| `whisper_large_v3_fp16.safetensors` | `models/audio_encoders/` | [Comfy-Org/HuMo_ComfyUI](https://huggingface.co/Comfy-Org/HuMo_ComfyUI/resolve/main/split_files/audio_encoders/whisper_large_v3_fp16.safetensors) | ~3 GB | NO |
| `lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors` | `models/loras/` | [Kijai/WanVideo_comfy](https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors) | ~150 MB | NO |

## VRAM math

**CORRECTED 2026-04-25:** the 17B fp8 model is **15.89 GB on disk**, not
~10 GB as the original recipe stated. ComfyUI stages it at **16266 MB**
on GPU — **over** the 5080 Laptop's 16 GB ceiling. This triggers
NVIDIA's Sysmem Fallback Policy: the driver silently spills overflow
weights into pinned system RAM, then pages them over PCIe per layer.
Result: each KSampler step takes ~42 seconds instead of the ~12s it
would take if the weights actually fit on the GPU.

Measured wall clock with stock fp8: **6 steps × ~42s = 4:14 sampling**,
total run **~4:30**. Functional but slow.

The `lightx2v_I2V_14B_480p_cfg_step_distill_rank64` LoRA reduces sampling
from typical 50 steps down to 6 — keep it loaded, don't strip it. The
"lora key not loaded" warnings during runs are noise (they're the I2V
cross-attention layers that HuMo's audio-conditioning architecture
doesn't have); the LoRA still applies to all the layers that DO match.

```
Phase load chart (with stock fp8):
  Phase 1: umt5_xxl  ─────────  6.4 GB  (encoded twice — pos + neg)
  Phase 2: Whisper   ─────────  1.2 GB  (audio encoder)
  Phase 3: WanVAE    ─────────  0.2 GB
  Phase 4: HuMo 17B fp8 ──────  16.3 GB ← OVER CEILING, fallback engaged
                                          ALL prior models stay resident
                                          ("0 models unloaded")
  Phase 5: VAEDecode + Save ──  CPU mostly
```

**Two paths to fix the spillover:**

1. **Diagnostic — disable Sysmem Fallback for ComfyUI's python.exe.**
   NVIDIA Control Panel → Manage 3D Settings → Program Settings → add
   `C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe` →
   "CUDA - Sysmem Fallback Policy" → **Prefer No Sysmem Fallback**.
   Re-queue. Should OOM cleanly at the WAN21_HuMo load phase, which
   confirms the spillover diagnosis.

2. **Fix — switch to a GGUF quant that fits.** Install city96's
   `ComfyUI-GGUF` custom node, download a smaller weight set, replace
   `UNETLoader` with `UnetLoaderGGUF` in the workflow. Best 17B HuMo
   quants (from VeryAladeen/Wan2_1-HuMo_17B-GGUF on Hugging Face):

   | Variant | Size on disk | Notes |
   |---|---|---|
   | Q5_K_M | **11.86 GB** | recommended — near-lossless, fits with ~4 GB headroom |
   | Q5_K_S | 11.22 GB | similar to Q5_K_M, slightly more compressed |
   | Q4_K_S | 9.60 GB | fallback if Q5_K_M still tight |
   | Q3_K_M | 7.83 GB | aggressive — risks lip-sync detail loss |
   | Q2_K   | 5.81 GB | too aggressive for face-centric model |

   **Q4_K_M does NOT exist for HuMo 17B** in either VeryAladeen or
   calcuis repos — don't go looking for it. Q4_K_S is the equivalent
   next-tier-down.

   **Don't drop below Q4_K_S for keeper takes.** HuMo's whole job is
   lip sync; quantization damage hits mouth-shape precision and
   phoneme timing first.

If still tight after Q5_K_M: drop length 97 → 65 (2.6s instead of
3.88s) to roughly halve activation memory. Frame count cuts only as
last resort — at 25 fps, 33 frames is 1.32 seconds, fine for batch
prompt validation but too short for keeper takes.

## The graph topology (simplified)

```
UNETLoader(humo_17B_fp8) ─→ LoraLoaderModelOnly(lightx2v) ─→ ModelSamplingSD3(shift=8) ─┐
                                                                                         ├─→ KSampler ─→ VAEDecode ─→ CreateVideo ─→ SaveVideo
CLIPLoader(umt5_xxl) ─→ CLIPTextEncode(positive + negative) ─→ WanHuMoImageToVideo ─────┤
VAELoader(wan_2.1_vae) ─────────────────────────────────────→ WanHuMoImageToVideo ─────┤
LoadImage(reference) ───────────────────────────────────────→ WanHuMoImageToVideo ─────┤
LoadAudio(speech.wav) ─→ AudioEncoderEncode(Whisper) ───────→ WanHuMoImageToVideo ─────┘
                       └────────────────────────────────────→ CreateVideo (audio mux)
```

KSampler config: **steps=6, cfg=1.0, sampler=uni_pc, scheduler=simple, ModelSamplingSD3 shift=8.**

These are NOT default KSampler values — they're the lightx2v-distilled
fast-inference values. Don't change without understanding the trade.

## Resolution + length

- **Default in template:** 640×640 (square), length=97
- **HuMo native preference:** 480×832 (portrait) or 720×1280 — both supported
- **For OTR POC:** start with 480×832 to match native training, see [HuMo I/O artifact](otr-humo-inputs-outputs)
- **Length=97 is fixed** at HuMo's native (97 frames @ 25fps = 3.88s)

## Step-by-step POC procedure

1. **Confirm disk space** — 18-19 GB free for models
2. **Download all 5 files** via browser or `huggingface-cli download <repo>`
3. **Place each file in the listed subdirectory** — wrong dir = won't load
4. **Restart ComfyUI Desktop** so it picks up the new files
5. **Open `workflows/external_examples/video_humo_official_native.json`**
6. **Replace LoadImage with one of OTR's character portraits** — any single-character PASS1 PNG from a previous run
7. **Replace LoadAudio with a 3-4 sec clip** — quick way: `ffmpeg -i existing.mp4 -ss 30 -t 3.88 -ac 1 -ar 16000 humo_test.wav` (Whisper wants mono 16 kHz)
8. **Update CLIPTextEncode positive prompt** — describe the OTR character + setting (the existing prompt is "A young boy in sci-fi style clothing")
9. **Set width=480, height=832, length=97** on the WanHuMoImageToVideo node
10. **Queue Prompt**

## What success looks like

- One MP4 written to `output/old_time_radio/humo_test/clip_<NNNN>.mp4` (97 frames, 25 fps, 3.88s)
- Character's mouth moves with the audio (lip-sync readable)
- Background looks reasonable (text prompt influences scene)
- Wall clock per clip:
  - **stock fp8**: 4-7 min (sysmem fallback PCIe-thrash, validated 2026-04-25)
  - **Q5_K_M GGUF**: ~1:30-2:00 (target after VRAM fits cleanly)
- VRAM peak (LHM live monitor):
  - stock fp8: 16+ GB with sysmem fallback engaged
  - Q5_K_M GGUF: ~13 GB clean, no spillover

## What failure looks like

- OOM at WAN21_HuMo load (with Sysmem Fallback disabled): expected for stock fp8 on a 16 GB card; switch to Q5_K_M GGUF
- OOM at WAN21_HuMo load (with Sysmem Fallback enabled): something else is wrong — encoders might not be releasing, or another model is resident from a prior run
- Black/garbled output: probably Whisper audio format mismatch — re-encode to mono 16 kHz WAV
- Character looks wrong (different person every time): expected — HuMo TIA matches the reference image; if quality is poor, the FLUX portrait may need higher detail
- Mushy mouths / bad phoneme timing: quant too aggressive (Q3 or below); step back up to Q4_K_S or Q5_K_M
- Wall-clock > 10 min/clip on a fitting quant: lightx2v LoRA isn't loading; verify the LoraLoaderModelOnly weight is at 1.0. (The "lora key not loaded: ..._img.lora_*" lines are normal — those are I2V layers HuMo doesn't have. The LoRA still applies to layers it can match.)

## Better variant for OTR — `video_humo_native_unlimited_workflow.json`

**This is probably the workflow we actually want for OTR.** Same model
files as the official template, but adds three things:

1. **832×480 LANDSCAPE** instead of 480×832 portrait — matches OTR's
   cinematic radio drama aesthetic without needing Option C composite tricks
2. **Variable length** via `ContextWindowsManual` (sliding-window inference)
   — beyond HuMo's native 97-frame ceiling. Default in template is 249
   frames = 9.96s.
3. **Auto-derives frame count from audio duration** via a `MathExpression`
   node: `ceil(a*25/4)*4+1`. Feed it any WAV, it computes the frame count.

**Source:** [amao2001/ganloss-latent-space](https://github.com/amao2001/ganloss-latent-space/blob/main/workflow/2025-09-27%20video_humo_native_unlimited_workflow.json)

**Local copy:** `workflows/external_examples/video_humo_native_unlimited_workflow.json`

**Custom nodes required:**

| Node | Status on your machine | If missing |
|---|---|---|
| `ContextWindowsManual` | INSTALLED | core sliding-window machinery |
| `VHS_LoadAudioUpload` | INSTALLED | from VideoHelperSuite |
| `VHS_VideoCombine` | INSTALLED | from VideoHelperSuite |
| `MathExpression\|pysssss` | MISSING | install `pysssss/ComfyUI-Custom-Scripts` OR hard-code frame count |
| `easy showAnything` | MISSING | install `easyuse/ComfyUI-Easy-Use` OR delete the node — debug-only |

The two missing nodes are quality-of-life only. Path of least resistance:
delete both nodes from the workflow and hard-code length on
`WanHuMoImageToVideo` widgets (e.g. set length=249 manually instead of
having MathExpression compute it).

**Honest concerns about this variant:**

- **VRAM at long lengths.** ContextWindowsManual keeps multiple windows
  in memory during blending. Going from 97 frames to 249 frames may
  push above 14.5 GB peak. Measure first; drop length if needed.
- **Boundary artifacts.** Window-blending sometimes shows subtle motion
  glitches at 81-frame boundaries (every ~65 frames after first window).
  Usually fine for casual viewing, visible if you look for it.
- **Off-axis from native training.** HuMo trained portrait; running
  landscape works but quality may differ slightly. Fall back to portrait
  if results look wrong.
- **Not author-endorsed.** ContextWindowsManual is community technique
  for HuMo. Treat as "experimental but tested" not "guaranteed."

**Recommended POC sequence:**

1. Try the **official native template first** — proves the 5 models load
   and HuMo runs at all on the hardware. 97 frames, portrait, no surprises.
2. If green, switch to **video_humo_native_unlimited** with length=97 first
   (so it's apples-to-apples with the official template, just landscape).
3. Then bump length to 161, then 249, watching VRAM each step.
4. Stop at the highest length that fits 14 GB peak with no boundary
   artifacts you can't tolerate.

## After POC succeeds

We have the option of HuMo for "talking close-up" shots in the OTR
pipeline. Combine with **Wan 2.2 + SVI Pro** (per
`workflows/external_examples/wan22_SVI_Pro_native_example_KJ.json`) for
landscape atmospheric shots.

But: per your "no talking heads" preference, HuMo may stay parked even
after a successful POC. The POC just confirms it's an OPTION.

## Sources

- [HuMo & Chroma1-Radiance Native Support — ComfyUI blog](https://blog.comfy.org/p/humo-and-chroma1-radiance-support)
- [bytedance-research/HuMo — Hugging Face](https://huggingface.co/bytedance-research/HuMo)
- [HuMo paper (arXiv 2509.08519)](https://arxiv.org/html/2509.08519v1)
- [Comfy-Org/HuMo_ComfyUI — model files](https://huggingface.co/Comfy-Org/HuMo_ComfyUI)
- Local file: `comfyui_workflow_templates_media_video/templates/video_humo.json`
