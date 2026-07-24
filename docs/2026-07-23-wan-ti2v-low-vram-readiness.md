# WAN TI2V low-VRAM readiness review

## Decision under review

Before any new GPU render, determine whether the existing `wan_ti2v` path is correctly wired and ready for a controlled low-VRAM smoke:

- public selection: `wan_8gb` (legacy/internal engine id: `wan_ti2v`)
- model: Wan 2.2 TI2V-5B Q5_K_M GGUF
- input: one still image plus a motion/text prompt
- no audio input, no lip-sync, no HuMo routing
- canvas: 832x480
- native length: 17 frames (`4n+1`)
- expected output: one silent H.264/yuv420p/bt709 MP4 at 25 fps
- expected peak: approximately 8 GB at this low-VRAM canvas, subject to live measurement

## Grounding facts to verify

The review must inspect the real Windows checkout, not a generated workflow or stale Linux mount:

1. `nodes/_otr_video_engines/eng_wan_ti2v.py` uses the TI2V-5B graph and requires the main UNET, `umt5-xxl-encoder-Q5_K_M.gguf`, and `wan2.2_vae.safetensors`.
2. The TI2V graph uses `Wan22ImageToVideoLatent`, `UnetLoaderGGUF`, `CLIPLoaderGGUF`, `ModelSamplingSD3`, tiled VAE decode, and a silent MP4 contract.
3. The canonical workflow is `workflows/otr_canonical.json`; any production wiring claim must be checked against its actual nodes, links, widgets, and validator.
4. `wan_8gb` is a public alias for `wan_ti2v`; `humo` remains a separate audio-driven-face engine and must not be substituted.
5. The prior WAN failures were an upstream ledger freeze and an overlarge 177-frame budget; neither is a valid TI2V quality result.
6. The current ComfyUI campaign may be resident on another port. Do not kill or edit it during this review; the GPU test requires a selective reset after it completes.

## Questions for the reviewers

- Is the public alias routed to the intended TI2V-5B adapter without changing HuMo or the non-audio LTX lane?
- Are the canonical workflow links/widgets and runtime adapter graph consistent, including init-image input and silent output?
- Is 832x480/17 frames actually enforced or merely an external test intention?
- Are the three required model assets and GGUF/ComfyUI node classes fail-closed before forward?
- Does the VRAM budgeting path permit this test without silently resizing or silently falling back?
- What is the smallest preflight/test sequence before the first GPU render?
- Identify any must-fix wiring or readiness bug. Do not propose a shim or edit files.

## Acceptance gate

Do not call the test green unless the run uses the canonical workflow/adapter path, records the actual model ids and dimensions, emits a canonical asset, reports peak VRAM, and ffprobe proves exactly one silent video stream with the required codec/color contract.

