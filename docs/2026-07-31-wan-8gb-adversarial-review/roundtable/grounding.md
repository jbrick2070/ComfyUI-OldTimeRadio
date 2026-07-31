# Grounding ledger

This is a compact evidence bundle for the panel. The Codex judge checked each
repository claim against the real Windows files and each upstream claim against
the linked primary source on 2026-07-31.

## Current OTR code and production evidence

- `nodes/_otr_video_engines/motion_common.py:253-363` uses reference pixels
  `1472*832`, `(7000, 185)` for Wan, margin `0.85`, and floor 17. At 832x480 it
  requires about 9,442 MB free for 17 frames; overhead alone requires about
  8,235 MB free. Thus the current guard rejects every honest `<=8192 MB` free
  report independently of whether execution would fit.
- `free_vram_mb()` uses `torch.cuda.mem_get_info`. `VramPeakProbe` samples
  machine-wide NVML across the render, so its absolute peak includes baseline
  and unrelated GPU use.
- `nodes/_otr_video_engines/eng_wan_ti2v.py:806-897` builds one graph in which
  the VAE feeds the pre-sampler image-to-video latent node and the post-sampler
  decode. The sampler consumes model, positive/negative conditioning, and the
  latent. The graph therefore does not itself establish three disjoint
  encode/sample/decode residency phases.
- `nodes/_otr_video_engines/wrapper_bridge.py:250-263,322-425` drops Python
  source results after their last consumer, then calls `gc.collect()` and
  ComfyUI `soft_empty_cache()`. It deliberately does not unload resident model
  patchers. The render keeps the UNet/VAE/decode nodes, and ComfyUI defaults to
  two async offload streams where supported. `max(stages)` is therefore a
  hypothesis, not a description of this graph's proven lifetimes.
- The Wan engine has no static `render_canvas`. In
  `render_driver.py:2494-2557`, a static engine declaration wins; absent one,
  the shared landscape environment value defaults to 1472x832. The profile's
  832x480 values reach the director/ledger but are not authoritative for each
  Wan clip. `launch.env` affects only a process booted through that profile.
- The current declaration validator requires both dimensions divisible by 32.
  Consequently the separate proposal to declare 768x432 would currently fail
  validation because 432 is not divisible by 32. The existing 832x480 contract
  is legal under that validator but is 26:15, not exact 16:9.
- `docs/PROD_BUG_LOG.md` entry `PBUG-20260723-02` records a production-named
  `wan_8gb` profile run on the 16 GB dev RTX 5080. The engine was asked for 177
  frames and found 30 affordable. The verified bug was that a launch-time 17
  frame ceiling did not reach an already-running server; it was fixed through
  the request ledger. This is not a physical-8-GB, 17-frame failure artifact.
- `BUG_BIBLE.yaml` rule 07.22 records a separate live low-VRAM failure where a
  VAE shared across encode and decode remained live through sampling and caused
  system-memory spill. Its portable rule is to split heavy resources when
  phase lifetimes do not overlap, then verify the actual graph and telemetry.
- The current engine exposes no independent `t5_device`/CPU-versus-GPU knob.
  `CLIPLoaderGGUF` accepts model name and type, not device. The proposed four
  cells are therefore not executable as an isolated factorial without adding a
  qualification-only placement control or using global launch policy that may
  confound other patchers.

## Installed ComfyUI and GGUF path

- The installed ComfyUI is commit
  [`34d0629452cac83dc20aa3d84e45c9b60d9e36b3`](https://github.com/Comfy-Org/ComfyUI/tree/34d0629452cac83dc20aa3d84e45c9b60d9e36b3),
  version 0.28.3 with `comfy-aimdo` 0.4.10.
- At that commit, `main.py:245-269` selects `ModelPatcherDynamic` only when
  Dynamic VRAM is supported and initialized; it is default-on for supported
  configurations, not universal. `cli_args.py` says `--lowvram` does nothing
  under Dynamic VRAM and otherwise puts text encoders on CPU.
- `model_management.py:text_encoder_device()` under Dynamic VRAM chooses GPU
  only when the dtype/device predicate supports fp16; otherwise it returns CPU.
  The statement that it returns GPU regardless is false.
- Installed/current ComfyUI-GGUF
  [`nodes.py`](https://github.com/city96/ComfyUI-GGUF/blob/main/nodes.py)
  subclasses legacy `ModelPatcher` and restores that class on clone, so GGUF
  does not use the AIMDO/VBAR dynamic patcher.
- That does not mean GGUF has no low-VRAM mechanism. The legacy patcher supports
  partial model loading/offload, and GGUF operations can dequantize/move modules
  as needed. [ComfyUI issue 11081](https://github.com/Comfy-Org/ComfyUI/issues/11081)
  includes a user log showing a GGUF model loaded partially with bytes offloaded.
  [Issue 13953](https://github.com/Comfy-Org/ComfyUI/issues/13953) is an open
  feature request for Dynamic VRAM support, not a maintainer benchmark or a
  proof that legacy offload cannot fit 8 GB.
- In mixed loading, ComfyUI treats non-dynamic patchers differently and may
  account them as pinned requirements. Exact behavior must be benchmarked for
  the OTR graph; class inheritance alone does not identify peak memory.

## Official Wan support statement and actual artifacts

- [ComfyUI's official Wan 2.2 guide](https://docs.comfy.org/tutorials/video/wan/wan2_2)
  says TI2V-5B should fit well on 8 GB with native offloading. It does not
  publish resolution, frame count, step count, peak metric, repeats, or an
  acceptance envelope.
- The official 5B workflow uses an FP16 TI2V-5B diffusion model plus the
  `umt5_xxl_fp8_e4m3fn_scaled.safetensors` text encoder and Wan VAE. The
  [day-zero post](https://blog.comfy.org/p/wan22-day-0-support-in-comfyui)
  lists 5B FP16, while FP8 is listed for 14B. Calling the official 5B UNet an
  FP8-scaled safetensors model is incorrect; only the listed text encoder is
  FP8-scaled. There is no official 5B FP8 UNet in that workflow.
- The [Wan 2.2 TI2V-5B model card](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)
  is Apache-2.0. The official support statement is useful prior evidence but is
  not certification of OTR's GGUF graph or a generic 8 GB Windows card.

## Alternatives and embedding cache

- [Lightx2v Wan2.2-Lightning](https://huggingface.co/lightx2v/Wan2.2-Lightning)
  currently releases A14B variants. TI2V-5B four-step support remains an
  unchecked TODO. A 14B+Lightning option changes topology, weights, quality,
  system-RAM traffic, and steps simultaneously; community anecdotes do not
  establish an 8 GB OTR tier.
- [Motif-Video-2B](https://huggingface.co/Motif-Technologies/Motif-Video-2B)
  publishes measured peaks of 15.12 GB BF16 and 12.53 GB Q4_K_M under its listed
  settings, so it is not a demonstrated 8 GB replacement.
- [MobileWan](https://huggingface.co/Qualcomm-AI-Research/mobilewan) carries a
  BSD-3-Clause-Clear file plus Qualcomm Responsible AI terms and lacks the
  shipped OTR/ComfyUI integration, so it does not clear the stated product gate.
- `WanVideoTextEncodeCached` belongs to ComfyUI-WanVideoWrapper and returns the
  wrapper's `WANVIDEOTEXTEMBEDS`, not native ComfyUI `CONDITIONING`; the wrapper
  is not installed in this OTR stack. Its persistent cache key hashes only the
  stripped prompt, omitting encoder artifact/version, tokenizer/config,
  precision/quantization, dtype, and schema. It can therefore collide across
  models and is not a drop-in correctness-safe cache for the native graph.
- OTR's positive prompts are shot-derived and often unique. Reuse inside one
  episode and reuse of the negative prompt are credible; broad cross-episode
  positive-cache hit rate is an assertion that needs trace data.

## Measurement caveats

- Reserving 8 GiB on a 16 GiB card constrains loader policy but does not recreate
  an 8 GiB physical address space, WDDM behavior, display baseline, allocator
  fragmentation, PCIe/topology, or host paging. It can prequalify a policy; it
  cannot certify generic 8 GB hardware.
- Absolute machine-wide NVML peak should be accompanied by pre-cell baseline,
  peak delta, PyTorch allocator data where meaningful, system RAM/pagefile,
  wall time, phase markers, artifact verification, and server logs.
- One canvas and two or four clip lengths cannot independently identify model
  residency, latent/activation scaling, decoder tiling, async transition
  overlap, and fixed reserve. At least repeated cold cells and separate mechanism
  comparisons are needed before fitting an admission envelope.
