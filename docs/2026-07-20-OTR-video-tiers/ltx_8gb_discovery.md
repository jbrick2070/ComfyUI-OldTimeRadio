# LTX 0.9.8 distilled 2B -- C0 discovery receipt (2026-07-20)

Live in-process `/object_info` probe on the running OTR headless ComfyUI
(`:54684`, `_otr_headless_model_paths.yaml`). Raw + normalized contracts are in
`ltx_8gb_discovery.json` (same directory). This receipt is the go/no-go basis for
`eng_ltx_8gb` (C1); the throwaway functional smoke (2c) confirms the graph renders.

- Total registered node classes: **1653**
- Normalized-contract SHA-256: `4bde945eade1cf8f57519eecfd74ac64e1f2272a4ba4aeadfccb3f4594d87eb4`
- Comfy code root: `C:\Users\jeffr\ComfyUI-Installs\ComfyUI\ComfyUI`
- venv python: `C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe`
- All 24 probed target classes PRESENT; 0 missing.

## Asset (checkpoint = all-in-one; VAE embedded)
- `CheckpointLoaderSimple.ckpt_name` choices INCLUDE `ltxv-2b-0.9.8-distilled.safetensors`
  (and the FORBIDDEN `ltx-video-2b-v0.9.safetensors`, which must NOT be the 8GB path).
- `CheckpointLoaderSimple` outputs `MODEL, CLIP, VAE`. The 0.9.8 all-in-one
  checkpoint carries the **VAE embedded** (slot 2) -- confirmed by the smoke decode.
  No separate 0.9.x VAE fetch required (assert NO LTX-2.3 VAE enters the graph).
- On disk: `C:\ComfyUI-Models\checkpoints\ltxv-2b-0.9.8-distilled.safetensors`
  size **6340744492** bytes, SHA256
  `76aa8c4786af752fa6f951947129d5290c3c6c0b2fadcadea6b5e114ae2cad8f` (Test-Path OK).

## Text encoder (T5) -- separate, CPU-offloadable
- `CLIPLoader` type token for the LTX family = **`ltxv`** (in the `type` COMBO).
- `CLIPLoader.clip_name` choices INCLUDE `t5xxl_fp16.safetensors` (on disk).
- `CLIPLoader.device` optional COMBO = `['default','cpu']` -> the **T5 offload route
  settled at discovery = a CPU-device CLIPLoader** (`device='cpu'`), no second graph.

## Node contracts (the distilled I2V recipe)
| node | key inputs (discovered) | outputs |
| --- | --- | --- |
| `CheckpointLoaderSimple` | `ckpt_name` | MODEL, CLIP, VAE |
| `CLIPLoader` | `clip_name`, `type='ltxv'`, opt `device='cpu'` | CLIP |
| `CLIPTextEncode` | `text`, `clip` | CONDITIONING |
| `ModelSamplingLTXV` | `model`, `max_shift=2.05`, `base_shift=0.95` | MODEL |
| `LTXVImgToVideo` (I2V) | `positive,negative,vae,image,width,height,length,batch_size,strength` | positive, negative, latent |
| `LTXVConditioning` | `positive,negative,frame_rate=25.0` | positive, negative |
| `LTXVScheduler` | `steps`, `max_shift=2.05`, `base_shift=0.95`, `stretch=True`, `terminal=0.1`, opt `latent` | SIGMAS |
| `KSamplerSelect` | `sampler_name` (incl. `euler`) | SAMPLER |
| `SamplerCustom` | `model,add_noise,noise_seed,cfg,positive,negative,sampler,sigmas,latent_image` | output, denoised |
| `VAEDecode` / `VAEDecodeTiled` | `samples`, `vae` (+tile/temporal on tiled) | IMAGE |

`LTXVImgToVideoConditionOnly` (VAE+image+latent+strength -> LATENT) is the
alternative I2V anchor used by `eng_ltx_video`/`otr_ltx_motion_smoke`; the C1 engine
uses `LTXVImgToVideo` (bundles pos/neg/latent) for a single clean I2V node.

## Legal dimensions + frame rule
- Width/height: `INT` min 64, **step 32** (`EmptyLTXVLatentVideo` / `LTXVImgToVideo`).
- Length (frames): **8n+1** (`length` step 8; `LTXVImgToVideo.length` **min 9**).
  Quantizer = `((L-1)//8)*8+1`, floor 9 (mirrors `eng_ltx_video._ltx_frame_length`).
- Frame rate: `LTXVConditioning.frame_rate` default 25.0 (OTR target_fps=25).

## Distilled sampler/scheduler/cfg/steps (smoke-confirmed)
- sampler `euler` (via `KSamplerSelect`) + `SamplerCustom` + `LTXVScheduler` sigmas.
- Distilled = few-step, **cfg 1.0**, **steps 8** (distilled default). `ModelSamplingLTXV`
  + `LTXVScheduler` shift 2.05/0.95, stretch True, terminal 0.1.

## Functional smoke (2c) -- GO
In-process `wrapper_bridge.run_graph` probe (box reset first; NOT an ad-hoc /prompt
graph). Graph = the recipe above at the legal minimum **512x288 x 9 frames, 8 steps,
cfg 1.0, euler**, T5 on `device='cpu'`. Result: **PASS** -- decoded IMAGE batch shape
`(9, 288, 512, 3)` -> silent mp4 `otr/episodes/ltx098_smoke/ltx098_smoke.mp4` (9 frames).
- CONFIRMED: the 0.9.8 all-in-one checkpoint has **no embedded text encoder**
  (ComfyUI logged "no CLIP/text encoder weights in checkpoint") -> T5 MUST be the
  separate `CLIPLoader(t5xxl_fp16, type='ltxv')`; the **VAE IS embedded** (the decode
  succeeded off `CheckpointLoaderSimple` slot 2 -- no separate VAE fetch).
- CONFIRMED: 1056 node classes resolved in-process; all 12 recipe nodes present;
  distilled 8-step schedule sampled clean; T5 CPU offload route works.

## Excluded / do-not-adopt
- **`LTXQ8Patch` EXCLUDED**: it owns the `quantization_preset` COMBO
  (`['0.9.8','ltxv2','full_bf16','custom']`) -- an fp8/Q8 quantizer patch, NOT the
  loader. Not adopted this sprint (no fp8/NVFP4/Q8 fork; directive s13).

## License / commercial_clean
- Model: Lightricks LTX-Video 0.9.x (`Lightricks/LTX-Video`, HF tag **`license:other`**
  = the Lightricks LTXV License). Commercial use is permitted below Lightricks'
  revenue threshold -- the SAME revenue-capped community model the repo already
  treats as commercial-clean (e.g. `eng_stable_audio` "Stability Community license
  (revenue-capped)" -> `commercial_clean=True`), and the same LTX family basis as the
  sibling `ltx_video` / `ltx_av` engines (both `commercial_clean=True`).
- **Decision: `commercial_clean=True`** for the C1 `ltx_8gb` engine attribute, with an
  honest comment naming `license:other` / the LTXV License so the operator confirms
  provenance at license review (directive s4). NOTE: `commercial_clean` is NOT a
  selection gate -- it only drives the release-gate NON-blocking warning + the release
  filename clean/gated tag (`_otr_release_gate.py`); it never hides or blocks an engine.
