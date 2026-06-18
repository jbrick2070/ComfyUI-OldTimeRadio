# flux2_klein -- verify-on-5080 recipe (the last step: promote into the dropdown)

flux2_klein is BUILT (`36dc01a`) + the image-model dropdown de-dup shipped
(`b8bb388`). Weights are on disk. It is still HIDDEN from the dropdown
(`_HIDDEN_IMAGE`, not in `ireg.VALIDATED_ENGINES`) until a green GPU render -- the
same promotion bar z_image_turbo passed. This file is the exact verify recipe.

## Files on disk (downloaded 2026-06-18)
- diffusion: `C:\Users\jeffr\Documents\ComfyUI\models\diffusion_models\flux-2-klein-4b-Q4_K_M.gguf` (2.6 GB)
- text enc:  `C:\Users\jeffr\Documents\ComfyUI\models\text_encoders\mistral_3_small_flux2_fp4_mixed.safetensors` (12.3 GB, NVFP4)
- vae:       `C:\Users\jeffr\Documents\ComfyUI\models\vae\flux2-vae.safetensors` (0.34 GB)

## Env (operator contract)
```
OTR_ENABLE_FLUX2_KLEIN=1
OTR_FLUX2_KLEIN_CKPT=C:\Users\jeffr\Documents\ComfyUI\models\diffusion_models\flux-2-klein-4b-Q4_K_M.gguf
OTR_FLUX2_KLEIN_TE=mistral_3_small_flux2_fp4_mixed.safetensors
OTR_FLUX2_KLEIN_VAE=flux2-vae.safetensors
```
(TE/VAE are basenames resolved by ComfyUI folder_paths; CKPT is the full-path gate.)

## Recipe (built from the official ComfyUI flux2 template + live node schemas)
UnetLoaderGGUF(klein.gguf) -> CLIPLoader(mistral fp4, type=flux2) -> CLIPTextEncode
-> FluxGuidance(4) -> BasicGuider; EmptyFlux2LatentImage(128-ch) + Flux2Scheduler(steps)
+ KSamplerSelect(euler) + RandomNoise -> SamplerCustomAdvanced -> VAEDecode.

## Steps
1. RESET (CLAUDE.md sec 4): kill resident comfy/soak by CommandLine, free :8000,
   confirm `nvidia-smi` ~baseline.
2. Boot the FLOOR/headless server (`scripts/_otr_soak_server_launch.cmd`) with the
   env above in `_marathon_extra_env.cmd` (PYTHONUTF8=1).
3. Force flux2_klein into an image slot. SOURCE-OF-TRUTH RULE (CLAUDE.md sec 0): do
   NOT run against a stale copy of the workflow. Either (a) edit the REAL
   `workflows/otr_scifi_16gb_full.json` -- set OTR_VideoDirector `music_image_model`
   (or all three) to `flux2_klein`, run, then REVERT; or (b) queue a standalone
   minimal flux2 probe graph via /prompt (UnetLoaderGGUF...VAEDecode + SaveImage) --
   a unit probe, not the pipeline, so it does not touch the canonical JSON.
4. Run a short render (30-60w smoke is enough -- only need ONE flux2_klein still).
5. PASS criteria:
   - LOG: `[OTR.image.flux2_klein] minted still WxH seed=... steps=... guidance=...`
   - SageAttention NOT patched onto the flux2 attention before the first forward
     (BUG-070, sm_120) -- the boot/gate must confirm this.
   - VRAM stays within the ceiling with the fp4 TE (it must fit on the GPU during the
     encode pass; 63 GB RAM absorbs offload). Watch `nvidia-smi` peak.
   - the still is a real image (not a degenerate/black frame).

## On GREEN -> promote (one commit)
- add `"flux2_klein"` to `nodes/_otr_image_engines/registry.py` `VALIDATED_ENGINES`.
- remove `"flux2_klein"` from `_HIDDEN_IMAGE` in `tests/test_tested_only_dropdown_gate.py`.
- update `test_image_validated_set` to expect `{flux_gen1, z_image_turbo, flux2_klein}`.
- decide default-OFF (stays opt-in via the flag) vs default-ON. Recommend opt-in
  (it needs ~15 GB of weights the floor user may not have) -- keep `requires_flag`.
- re-run the affected suite + Bug Bible; commit + push.

## VERIFY RUN 2026-06-18 -- FAILED (not yet promotable); two issues found
The flux2 + LTX full episode (announcer/beats=ltx_av_talk, music=ltx_av_music, all
image=flux2_klein, 30w, bark) surfaced two real issues:
1. FIXED -- weights were downloaded to `C:\Users\jeffr\Documents\ComfyUI\models` but
   the headless server loads from `C:\ComfyUI-Models` (via
   `scripts/_otr_headless_model_paths.yaml`). Moved all 3 files there; the
   CLIPLoader then found the TE. (Downloader should target C:\ComfyUI-Models.)
2. OPEN -- the sampler raises `RuntimeError: mat1 and mat2 shapes cannot be
   multiplied (512x15360 and 7680x3072)` in SamplerCustomAdvanced/guider.sample.
   The TE output width (15360) is exactly 2x the klein-4B UNet's expected (7680).
   ROOT CAUSE HYPOTHESIS: the GGUF klein **4B** is paired with the wrong text
   encoder -- I reused flux2-**dev**'s `mistral_3_small_flux2_fp4_mixed` (full
   Mistral-3). ComfyUI's `sd.py` distinguishes `MISTRAL3_24B` vs
   `MISTRAL3_24B_PRUNED_FLUX2`; klein-4B likely needs the PRUNED TE (or its own
   matched TE/quant). NEXT: read the unsloth/FLUX.2-klein-4B-GGUF + BFL klein-4B
   model cards for the EXACT klein-4B TE + ComfyUI recipe (do NOT assume dev's
   recipe transfers); the 4B is size-distilled and may not share dev's encoder
   config. Then re-pair the TE and re-run this verify.

## Risks to watch (why this is a real verify, not a rubber-stamp)
- fp4 (NVFP4) TE load on the torch 2.10 / cu130 / sm_120 stack -- if it does not
  load, fall back to the fp8 TE (18 GB; needs layerwise CPU offload to fit 16 GB) or
  a GGUF TE quant.
- the GGUF flux2 UNet via ComfyUI-GGUF UnetLoaderGGUF (arch=flux) decoding correctly.
- SageAttention/BUG-070 on the FLUX-style attention.
