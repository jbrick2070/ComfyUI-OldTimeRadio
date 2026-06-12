# LTX-2.3 audio-conditioning research notes (operator-supplied, 2026-06-10)

> STATUS: UNVERIFIED third-party research pasted by the operator. The
> planning roundtable MUST verify every load-bearing claim (model files,
> node names, PR numbers, VRAM figures) before the plan relies on it.
> Companion doc: docs/2026-06-10-ltx2-a2v-poc__plan.md (the small PoC plan).

## Claims to verify (verbatim from the research)

- LTX-2.3 ships an official `A2VidPipelineTwoStage` -- audio-to-video
  conditioned on an input audio file; you never use LTX's generated audio.
- The 2B v0.9 checkpoint our `eng_ltx_video.py` drives has NO audio pathway;
  audio conditioning exists only in the LTX-2/2.3 generation.
- Architecture: asymmetric dual-stream transformer (14B video + 5B audio)
  with bidirectional audio-video cross-attention; the WAV is encoded by the
  LTX audio VAE into latents the video stream cross-attends to during
  denoising -- sync is produced IN generation, not as a post-process.
- V-1 fit: decode only the video latents, discard the audio side entirely;
  the mux drops the beat's master `audio_ref` on top. Byte-identical gate
  preserved by construction.
- 16GB paths:
  - GGUF: QuantStack/LTX-2.3-GGUF in `models/unet` (NOT diffusion_models) +
    Unsloth gemma-3-12b GGUF text encoder; Q4_K_M recommended for 8-16GB;
    one 16GB report of Q3_K_S doing 10s 1280x896 in ~10-15 min.
  - NVFP4/NVFP8: ComfyUI + NVIDIA Blackwell support; NVFP4 on RTX 50-series
    cuts VRAM ~60% and runs ~3x faster; checkpoints downloadable in ComfyUI;
    sm_120 is the target.
  - NVIDIA 16GB guidance: ~4s clips at 720p; weight streaming to system RAM
    at a perf cost.
- Required files (Kijai splits): ltx-2.3-22b-distilled transformer
  (fp8_scaled or GGUF), MelBandRoformer_fp16, gemma_3_12B text encoder +
  ltx-2.3 text projection, LTX23 audio VAE + video VAE + taeltx2_3,
  spatial upscaler.
- Graph gotcha: width/height must be divisible by 32 PLUS 1, frame count
  divisible by 8 PLUS 1; invalid values silently round (fail-loud validator
  needed).
- ComfyUI support is NATIVE (built-in templates); Kijai's LTXVReferenceAudio
  (reference-audio speaker identity) merged upstream in PR #13111.
- Role fit: announcer/characters = I2V + custom audio -> true lip-sync from
  a FLUX still (an in-LTX alternative to the HuMo slot; IndexTTS2 output is
  a clean phoneme signal; MelBand Roformer isolates vocals from a music
  bed). music/background = rhythm/tempo alignment -- "visuals breathe with
  the track", NOT Resolume-grade FFT precision; speech->face is the
  best-documented behavior.
- Zero-new-model fallback: Yvann-Nodes audio analysis (per-frame reactive
  weights, peak detection, stem separation, prompt scheduling) modulating
  the EXISTING 2B stack via guide frames / prompt intensity / cut timing --
  "modulating around the audio" vs LTX-2.3 "genuinely hearing it".
- Suggested integration: a NEW adapter (`eng_ltx_av`, family ~
  audio_conditioned_video, required_inputs (text_prompt, audio_ref)) --
  do NOT mutate eng_ltx_video; the 2B engine stays GPU-proven as-is;
  ltx_motion.py's C7 no-audio-imports constraint untouched on its own path.
