# OTR Model Inventory

Generated 2026-08-26, after the disk-reclaim pass recorded at the bottom of this file.

Models root is `C:\ComfyUI-Models`, resolved through
`nodes/_otr_gguf_backend.py::_models_root()`. Never assume a path under the repo or the
ComfyUI tree -- see CLAUDE.md section 6A.

**This file is the standing record of what is on disk and what uses it**, so a future
space audit does not have to re-derive it. Regenerate it when the models root changes
materially.

## Where models actually live (three roots, and all three matter)

| Root | Role |
| --- | --- |
| `C:\ComfyUI-Models\` | The store. Registered in ComfyUI's `extra_model_paths.yaml` with `is_default: true`. Nearly everything lives here. |
| `C:\Users\jeffr\ComfyUI-Installs\ComfyUI\ComfyUI\models\` | ComfyUI core's own models dir (core lives here, not under `Documents`). Effectively empty and should stay that way. |
| `C:\Users\jeffr\Documents\ComfyUI\models\upscale_models\` | **LIVE AND LOAD-BEARING.** `_otr_upscale_engines/eng_spandrel_esrgan.py:151` resolves `here.parents[4] / "models" / "upscale_models"`, which lands exactly here. `RealESRGAN_x2plus.pth` is read from this path, and `C:\ComfyUI-Models\upscale_models\` is empty by design (`otr_silent_composite.py:1702` documents that). **Do not delete it during a cleanup.** |

## How the USED-BY column was derived

Every weight file was matched by basename and by stem against a text corpus built from
the OTR repo (`.py`, `.json`, `.md`, `.yaml`, `.txt`, `.toml`, `.ps1`, `.cmd`), and
separately against every other pack under `custom_nodes\`.

| Tag | Meaning |
| --- | --- |
| `OTR` | Named somewhere in the ComfyUI-OldTimeRadio tree. |
| `OTHER` | Named only by a different custom-node pack (LTXVideo, MiniMax-H3-Turbo, KokoroTTS, ...). |
| `NONE` | Named by nothing in any pack. |

**A name match is necessary, not sufficient, and neither is its absence fatal.** Some
`OTR` rows are named only in a handoff doc or a retired-catalog comment, and a `NONE` row
can still be reached through a path the grep cannot see. Every deletion below was
justified by reading the loader, not by the tag.

**A hit inside `.claude\worktrees\` does not count as live.** Those worktrees hold the v1
`visual/backends/` code that no longer executes; treating them as live is how a dead model
looks used.

## Directory totals

`du` counts a hardlinked file only once per run, so a directory whose only content is a
hardlink to a file counted elsewhere reports ~0. `LMStudio\` is the example -- see the
hardlink note below.

| Directory | GB |
| --- | ---: |
| `diffusion_models` | 195.4 |
| `text_encoders` | 116.6 |
| `huggingface` | 91.9 |
| `checkpoints` | 86.9 |
| `unet` | 52.8 |
| `LLM` | 38.2 |
| `diffusers` | 19.4 |
| `loras` | 18.8 |
| `vae` | 13.0 |
| `controlnet` | 7.0 |
| `audio_encoders` | 2.9 |
| `latent_upscale_models` | 2.8 |
| `florence2` | 1.5 |
| `tools` | 1.3 |
| `pulid` | 1.1 |
| `musicgen_cache` | 0.4 |
| `TTS` | 0.3 |

## Hugging Face cache

`HF_HOME` = `C:\ComfyUI-Models\huggingface` (set in the launch bats and resolved by
`_otr_hf_env.py` from `HKCU\Environment`), and `_otr_hf_env.py` sets
`HF_HUB_CACHE` = `<HF_HOME>\hub`. **`resolve_snapshot_dir` reads `hub\models--*` and nothing
else** -- lines 121-125 call out that exact mismatch as a hazard.

A repo sitting at the `huggingface\` root instead of `huggingface\hub\` is therefore
invisible to OTR. That whole stale layer was cleared below, and `Dia-1.6B-0626` -- which
existed ONLY at the root and so had never been reachable -- was moved into `hub\` rather
than deleted.

| Cached repo | GB |
| --- | ---: |
| `models--mistralai--Mistral-Nemo-Instruct-2407` | 22.84 |
| `models--google--gemma-4-12b-it` | 22.31 |
| `models--google--gemma-4-E4B-it` | 14.92 |
| `models--google--gemma-4-E2B-it` | 9.57 |
| `models--nari-labs--Dia-1.6B-0626` | 6.00 |
| `models--google--gemma-2-2b-it` | 4.89 |
| `models--suno--bark` | 4.18 |
| `models--ResembleAI--chatterbox` | 2.97 |
| `models--facebook--musicgen-small` | 2.21 |
| `models--depth-anything--Depth-Anything-V2-Large-hf` | 1.25 |
| `models--stabilityai--sd-vae-ft-mse` | 0.31 |
| `models--hexgrad--Kokoro-82M` | 0.31 |
| `models--depth-anything--Depth-Anything-V2-Small-hf` | 0.09 |

## Full weight-file list

All 131 weight files (`.safetensors` `.gguf` `.ckpt` `.pth` `.pt` `.bin` `.onnx` `.sft`)
under the models root, largest first. Paths are relative to `C:\ComfyUI-Models`.

| GB | Used by | File |
| ---: | --- | --- |
| 42.98 | OTR | `checkpoints/ltx-2.3-22b-dev.safetensors` |
| 23.49 | OTR | `diffusion_models/ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors` |
| 20.03 | OTR | `diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors` |
| 19.53 | OTR | `diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors` |
| 19.53 | OTR | `diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors` |
| 16.66 | OTR | `diffusion_models/Wan2_1-HuMo-14B_fp8_e4m3fn_scaled_KJ.safetensors` |
| 16.06 | OTR | `checkpoints/flux1-dev-fp8.safetensors` |
| 15.89 | OTR | `diffusion_models/humo_17B_fp8_e4m3fn.safetensors` |
| 14.61 | OTR | `text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` |
| 14.32 | OTR | `text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` |
| 13.31 | OTR | `diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors` |
| 13.22 | OTR | `unet/distilled-1.1/ltx-2.3-22b-distilled-1.1-Q4_K_M.gguf` |
| 12.22 | OTR | `unet/ltx-2.3-22b-dev-Q4_K_S.gguf` |
| 11.86 | OTR | `diffusion_models/Wan2_1-HuMo-17B_Q5_K_M.gguf` |
| 11.80 | OTR | `LLM/converted/gemma-4-12b-it/gemma-4-12b-it-Q8_0.gguf` |
| 11.43 | OTR | `text_encoders/mistral_3_small_flux2_fp4_mixed.safetensors` |
| 10.73 | OTR | `diffusion_models/LTX-2.5-Distilled-Q3_K_M.gguf` |
| 10.03 | OTR | `unet/ltx-2.3-22b-dev-Q3_K_M.gguf` |
| 9.90 | OTR | `unet/distilled-1.1/ltx-2.3-22b-distilled-1.1-Q3_K_M.gguf` |
| 9.57 | OTR | `text_encoders/gemma4_e2b_it_bf16.safetensors` |
| 9.56 | OTR | `diffusers/stable-diffusion-xl-1.0-inpainting-0.1/unet/diffusion_pytorch_model.safetensors` |
| 9.12 | OTR | `text_encoders/t5xxl_fp16.safetensors` |
| 8.86 | OTR | `text_encoders/gemma4-12b-with-proj-ltx-2.5-Q5_K_M.gguf` |
| 8.80 | OTR | `text_encoders/gemma_3_12B_it_fp4_mixed.safetensors` |
| 8.73 | OTR | `checkpoints/ltx-video-2b-v0.9.safetensors` |
| 8.40 | OTR | `diffusion_models/HuMo-17b-Q3_K_M.gguf` |
| 7.49 | OTR | `text_encoders/qwen_3_4b.safetensors` |
| 7.39 | OTR | `unet/distilled-1.1/ltx-2.3-22b-distilled-1.1-Q2_K.gguf` |
| 7.08 | OTR | `loras/ltxv/ltx2/ltx-2.3-22b-distilled-lora-384.safetensors` |
| 7.08 | OTR | `loras/ltxv/ltx2/ltx-2.3-22b-distilled-lora-384-1.1.safetensors` |
| 6.96 | OTR | `LLM/converted/Mistral-Nemo-Instruct-2407/Mistral-Nemo-Instruct-2407-Q4_K_M.gguf` |
| 6.63 | OTR | `unsloth/gemma-4-12b-it-GGUF/gemma-4-12b-it-Q4_K_M.gguf` |
| 6.63 | OTR | `LMStudio/unsloth/gemma-4-12b-it-GGUF/gemma-4-12b-it-Q4_K_M.gguf` |
| 6.63 | OTR | `LLM/converted/gemma-4-12b-it/gemma-4-12b-it-Q4_K_M.gguf` |
| 6.27 | OTR | `text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors` |
| 5.91 | OTR | `checkpoints/ltxv-2b-0.9.8-distilled.safetensors` |
| 5.87 | OTR | `text_encoders/qwen3vl_8b_nvfp4.safetensors` |
| 5.25 | OTR | `text_encoders/qwen_3_4b_fp8_mixed.safetensors` |
| 5.11 | OTR | `diffusion_models/ideogram4_unconditional_nvfp4_mixed.safetensors` |
| 5.11 | OTR | `diffusion_models/ideogram4_nvfp4_mixed.safetensors` |
| 4.97 | OTR | `LLM/converted/gemma-4-E4B-it/gemma-4-E4B-it-Q4_K_M.gguf` |
| 4.87 | OTR | `text_encoders/gemma_2_2b_fp16.safetensors` |
| 4.86 | OTR | `diffusion_models/lumina_2_model_bf16.safetensors` |
| 4.85 | OTR | `vae/minimax_h3_video_vae_fp16.safetensors` |
| 4.78 | OTR | `diffusers/stable-diffusion-xl-1.0-inpainting-0.1/unet/diffusion_pytorch_model.fp16.safetensors` |
| 4.68 | OTR | `LLM/converted/Qwen3-8B/Qwen3-8B-Q4_K_M.gguf` |
| 4.59 | OTR | `checkpoints/hunyuan3d-dit-v2-mv.safetensors` |
| 4.52 | OTR | `checkpoints/stable-audio-open-1.0.safetensors` |
| 4.20 | OTR | `diffusion_models/z_image_turbo_nvfp4.safetensors` |
| 3.99 | OTR | `controlnet/FLUX.1-dev-ControlNet-Union-Pro-2.0/diffusion_pytorch_model.safetensors` |
| 3.92 | OTR | `diffusion_models/FastWan2.2-TI2V-5B-q6_k.gguf` |
| 3.86 | OTR | `text_encoders/umt5-xxl-encoder-Q5_K_M.gguf` |
| 3.55 | OTR | `diffusion_models/Wan2_2-TI2V-5B-Turbo-Q5_K_M.gguf` |
| 3.55 | OTR | `diffusion_models/Wan2.2-TI2V-5B-Q5_K_M.gguf` |
| 3.24 | OTR | `diffusion_models/humo_1.7B_fp16.safetensors` |
| 3.19 | OTR | `LLM/converted/gemma-4-E2B-it/gemma-4-E2B-it-Q4_K_M.gguf` |
| 2.97 | OTR | `controlnet/FLUX.1-dev-ControlNet-Depth/diffusion_pytorch_model.safetensors` |
| 2.88 | OTR | `audio_encoders/whisper_large_v3_fp16.safetensors` |
| 2.59 | OTR | `diffusers/stable-diffusion-xl-1.0-inpainting-0.1/text_encoder_2/model.safetensors` |
| 2.55 | OTR | `loras/ltxv/ltx2/ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors` |
| 2.43 | OTR | `diffusion_models/flux-2-klein-4b-Q4_K_M.gguf` |
| 2.15 | OTR | `text_encoders/text_encoders/ltx-2.3-22b-distilled_embeddings_connectors.safetensors` |
| 2.15 | OTR | `text_encoders/ltx-2.3-22b-dev_embeddings_connectors.safetensors` |
| 2.11 | OTR | `checkpoints/stable_audio_3_small_music.safetensors` |
| 1.99 | OTR | `checkpoints/v1-5-pruned-emaonly-fp16.safetensors` |
| 1.45 | OTR | `florence2/Florence-2-large/model.safetensors` |
| 1.37 | OTR | `vae/ltx-2.5-video-vae-bf16.safetensors` |
| 1.35 | OTR | `vae/vae/ltx-2.3-22b-distilled_video_vae.safetensors` |
| 1.35 | OTR | `vae/ltx-2.3-22b-dev_video_vae.safetensors` |
| 1.31 | OTR | `vae/wan2.2_vae.safetensors` |
| 1.29 | OTR | `diffusers/stable-diffusion-xl-1.0-inpainting-0.1/text_encoder_2/model.fp16.safetensors` |
| 1.11 | OTR | `text_encoders/t5gemma_b_b_ul2.safetensors` |
| 1.06 | OTR | `pulid/pulid_flux_v0.9.1.safetensors` |
| 1.06 | OTR | `pulid/pulid_flux.safetensors` |
| 0.93 | OTR | `latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` |
| 0.93 | OTR | `latent_upscale_models/ltx-2-spatial-upscaler-x2-1.0.safetensors` |
| 0.93 | OTR | `latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors` |
| 0.83 | OTR | `text_encoders/t5-base.safetensors` |
| 0.73 | OTR | `loras/h3-turbo-larry-v4/minimax_h3_turbo_v4_step600_ema.safetensors` |
| 0.69 | OTR | `loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors` |
| 0.62 | OTR | `loras/Wan2_2_5B_FastWanFullAttn_lora_rank_128_bf16.safetensors` |
| 0.56 | OTR | `vae/minimax_h3_audio_vae_fp32.safetensors` |
| 0.46 | OTR | `diffusers/stable-diffusion-xl-1.0-inpainting-0.1/text_encoder/model.safetensors` |
| 0.34 | OTR | `vae/ltx-2.5-audio-vae-bf16.safetensors` |
| 0.34 | OTR | `vae/vae/ltx-2.3-22b-distilled_audio_vae.safetensors` |
| 0.34 | OTR | `vae/ltx-2.3-22b-dev_audio_vae.safetensors` |
| 0.31 | OTR | `vae/flux2-vae.safetensors` |
| 0.31 | OTR | `vae/lumina2_ae.safetensors` |
| 0.31 | OTR | `vae/ae.safetensors` |
| 0.31 | OTR | `diffusers/stable-diffusion-xl-1.0-inpainting-0.1/vae/diffusion_pytorch_model.safetensors` |
| 0.30 | OTR | `TTS/KokoroTTS/kokoro-v1_0.pth` |
| 0.24 | OTR | `vae/wan_2.1_vae.safetensors` |
| 0.23 | OTR | `diffusers/stable-diffusion-xl-1.0-inpainting-0.1/text_encoder/model.fp16.safetensors` |
| 0.16 | OTR | `diffusers/stable-diffusion-xl-1.0-inpainting-0.1/vae/diffusion_pytorch_model.fp16.safetensors` |
| 0.10 | OTR | `loras/v3_sd15_adapter.ckpt` |
| 0.01 | OTR | `custom_node_assets/ComfyUI-MiniMax-H3-Turbo/h3_silu_temb_grid.safetensors` |
| 0.00 | OTR | `TTS/KokoroTTS/voices/bf_isabella.pt` |
| 0.00 | OTR | `TTS/KokoroTTS/voices/am_michael.pt` |
| 0.00 | OTR | `TTS/KokoroTTS/voices/af_jessica.pt` |
| 0.00 | OTR | `TTS/KokoroTTS/voices/bm_george.pt` |
| 0.00 | OTR | `TTS/KokoroTTS/voices/bm_daniel.pt` |
| 0.00 | OTR | `TTS/KokoroTTS/voices/am_fenrir.pt` |
| 0.00 | OTR | `TTS/KokoroTTS/voices/af_nicole.pt` |
| 0.00 | OTR | `TTS/KokoroTTS/voices/bm_lewis.pt` |
| 0.00 | OTR | `TTS/KokoroTTS/voices/bm_fable.pt` |
| 0.00 | OTR | `TTS/KokoroTTS/voices/bf_alice.pt` |
| 0.00 | OTR | `TTS/KokoroTTS/voices/am_santa.pt` |
| 0.00 | OTR | `TTS/KokoroTTS/voices/af_sarah.pt` |
| 0.00 | OTR | `TTS/KokoroTTS/voices/af_river.pt` |
| 0.00 | OTR | `TTS/KokoroTTS/voices/af_heart.pt` |
| 0.00 | OTR | `TTS/KokoroTTS/voices/af_bella.pt` |
| 0.00 | OTR | `TTS/KokoroTTS/voices/af_aoede.pt` |
| 0.00 | OTR | `TTS/KokoroTTS/voices/af_alloy.pt` |
| 0.00 | OTR | `TTS/KokoroTTS/voices/bf_lily.pt` |
| 0.00 | OTR | `TTS/KokoroTTS/voices/bf_emma.pt` |
| 0.00 | OTR | `TTS/KokoroTTS/voices/am_puck.pt` |
| 0.00 | OTR | `TTS/KokoroTTS/voices/am_onyx.pt` |
| 0.00 | OTR | `TTS/KokoroTTS/voices/am_liam.pt` |
| 0.00 | OTR | `TTS/KokoroTTS/voices/am_eric.pt` |
| 0.00 | OTR | `TTS/KokoroTTS/voices/am_echo.pt` |
| 0.00 | OTR | `TTS/KokoroTTS/voices/am_adam.pt` |
| 0.00 | OTR | `TTS/KokoroTTS/voices/af_nova.pt` |
| 0.00 | OTR | `TTS/KokoroTTS/voices/af_kore.pt` |
| 0.00 | OTR | `TTS/KokoroTTS/voices/af_sky.pt` |
| 0.00 | OTR | `tools/blender-4.5.10/4.5/python/lib/site-packages/distutils-precedence.pth` |
| 0.00 | OTR | `huggingface/hub/models--suno--bark/.no_exist/70a8a7d34168586dc5d028fa9666aceade177992/model.safetensors` |
| 0.00 | OTR | `huggingface/hub/models--nari-labs--Dia-1.6B-0626/.no_exist/ef2795fcc29c5abe6ffc91fd33808588b49bbc66/model.safetensors` |
| 0.00 | OTR | `huggingface/hub/models--mistralai--Mistral-Nemo-Instruct-2407/.no_exist/04d8a90549d23fc6bd7f642064003592df51e9b3/model.safetensors` |
| 0.00 | OTR | `huggingface/hub/models--hexgrad--Kokoro-82M/.no_exist/f3ff3571791e39611d31c381e3a41a3af07b4987/voices/vz_donor_darya_khan.pt` |
| 0.00 | OTR | `huggingface/hub/models--hexgrad--Kokoro-82M/.no_exist/f3ff3571791e39611d31c381e3a41a3af07b4987/voices/vz_bill_boerst.pt` |
| 0.00 | OTR | `huggingface/hub/models--google--gemma-2-2b-it/.no_exist/299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8/model.safetensors` |
## Disk-reclaim pass, 2026-08-26

C: was at 286 GB free (93% used). After this pass it is at **710 GB free (81% used)** --
about **424 GB** recovered, measured as the free-space delta rather than as the sum of the
`du` figures, because hardlinks and Windows' deferred deletes make that sum a poor estimate.

Operator scope, in his words: *"any models outside my models folder can be deleted -- all
good models should be in my models folder, not a cache."* All three tiers below were
approved, plus HY-WorldMirror ("was a bad experiment").

### Two things that nearly went wrong, and the rule each one leaves behind

**1. `du` is not evidence of reclaimable space -- check the link count.**
`gemma-4-12b-it-Q4_K_M.gguf` appears at three paths and `du` reported 6.63 GB at each, which
reads as 13.3 GB of recoverable duplication. `stat` says otherwise: all three share inode
`7318349396575578` with `links=3`. They are NTFS hardlinks to one extent. **Deleting the
mirrors would have freed exactly zero bytes.** Only the fourth copy -- an Ollama-layout blob
at `blobs\sha256-43fec98c...` with `links=1` and a different inode -- was real, and only that
one was removed. Before claiming a duplicate is reclaimable, run
`stat -c '%i links=%h' <path>` on every copy.

**2. An empty directory can mean the loader looks somewhere else on purpose.**
`C:\ComfyUI-Models\upscale_models\` is empty, which invites the conclusion that the
RealESRGAN files under `Documents\ComfyUI\models\upscale_models\` are a stray leftover. They
are not: `eng_spandrel_esrgan.py:151` deliberately walks `here.parents[4]` to that exact
directory, and `otr_silent_composite.py:1702` documents the empty one. Deleting those three
files would have broken the upscale lane. **They stay where they are.**

### Removed -- orphaned, no live reference

| GB | Item | Why it was dead |
| ---: | --- | --- |
| 27.6 | `diffusers\Wan2.1-I2V-1.3B\` | named only by a downloader script and the retired v1 `visual/backends/wan21_loop.py` |
| 23.4 | HF cache `Captain-Eris_Violet-V0.420-12B` | catalog row removed 2026-05-23 (`_otr_model_catalog.py:234`); `resolve_context_cap` can only reach a repo_id a curated row names |
| 22.8 | Mistral-Nemo `consolidated.safetensors` (+ its blob) | Mistral-native single-file format; transformers loads the five shards via `model.safetensors.index.json` and never opens it. All 5 shards verified present afterward. |
| 17.4 | `diffusers\FLUX.1-dev-torchao-fp8\` | zero `torchao` hits in the live tree |
| 16.9 | `quarantine\` | quarantined by name |
| 15.3 | HF cache `musicgen-medium` | `eng_musicgen.py:17` pins `facebook/musicgen-small`; medium was never wired |
| 4.7 | `WorldMirror-V2\` | zero live references; operator confirmed it was an abandoned experiment |

### Removed -- genuine duplicates and dead caches

| GB | Item | Note |
| ---: | --- | --- |
| 6.6 | `blobs\` + `manifests\` | Ollama-layout store. The sha in the blob name matches `GGUF_ARTIFACTS["Q4_K_M"]` exactly, and no live code reads either directory. |
| 10.0 | HF cache `Comfy-Org--z_image_turbo` | the engine loads `z_image_turbo_nvfp4.safetensors` from `diffusion_models\` via UNETLoader, which reads the flat dir only -- a cache copy cannot serve a render |
| 10.3 | HF cache `Comfy-Org--Lumina_Image_2.0_Repackaged` | same shape; `lumina_image.py:67` loads from the flat dir |
| ~7.3 | stale `huggingface\models--*` at the cache root | pre-`HF_HUB_CACHE` layout, not on any read path; every entry except Dia was already duplicated in `hub\` |

### Removed -- outside the models root

| GB | Item | Note |
| ---: | --- | --- |
| 43.1 | `~\.cache\huggingface\hub\models--mistralai--Mistral-Nemo-Instruct-2407` | a THIRD copy of Mistral-Nemo |
| 3.0 | `~\.cache\huggingface\hub\models--ResembleAI--chatterbox` | duplicate of the models-root copy |
| 5.1 | `~\.cache\whisper\` | `large-v3/medium/small/base/tiny.pt` from the `openai-whisper` package. No `import whisper` in ANY pack -- OTR uses `audio_encoders\whisper_large_v3_fp16.safetensors` through `AudioEncoderLoader`. |
| 14.2 | `Documents\ComfyUI\models\_flux2_dl_scratch\` | held `mistral_3_small_flux2_fp4_mixed.safetensors`, `flux-2-klein-4b-Q4_K_M.gguf`, `flux2-vae.safetensors` -- all three already in the models root at identical byte sizes and separate inodes. The "parked in scratch" anti-pattern CLAUDE.md section 6 warns about. |
| 17.5 | `ComfyUI-Installs\...\models\huggingface\` | `.incomplete` blobs from an aborted Mistral-Nemo download |

### Kept, and why

* **Every Gemma.** `gemma-4-12b-it` (Q8_0 + Q4_K_M), `gemma-4-E4B-it`, `gemma-4-E2B-it`,
  `gemma-2-2b-it`, `gemma-3-12b-it`, `gemma4-12b-with-proj-ltx-2.5` (both quants),
  `gemma4_e2b_it_bf16`, `gemma_2_2b_fp16`, `gemma_3_12B_it_fp4_mixed`, `t5gemma_b_b_ul2`.
  OTR uses several Gemmas across the writer, technical and LTX text-encoder slots.
* **`Dia-1.6B-0626` -- moved, not deleted.** It was the only copy and it sat at the cache
  root where the resolver never looks, so it was unreachable dead weight. Moving it into
  `hub\` made it loadable for the first time.
* `hunyuan3d-dit-v2-mv.safetensors` -- live via `_otr_video_engines/eng_mesh_stage.py`.
* `Depth-Anything-V2-Large/Small` -- live reference in `_otr_model_catalog.py`.
* `diffusers\stable-diffusion-xl-1.0-inpainting-0.1\` (19.9 GB) -- carries both fp32 and
  fp16 variants and is still catalog-referenced. The fp16 half looks redundant but that was
  not proven, and a guess is not a reason to cut 6.5 GB.
* `Documents\ComfyUI\models\upscale_models\` -- see the trap above.
* `~\.lmstudio\extensions\` (2.9 GB) -- LM Studio runtime, not model weights.

### Verification after the pass

`pytest tests/test_model_catalog_scan.py tests/test_model_slot_audit.py
tests/test_image_gen_preflight_matrix.py tests/test_lane_preflight_matrix.py
tests/test_tts_voice_preflight_matrix.py` -- **155 passed**. Spot-checked on disk afterward:
both gemma-4-12b GGUFs, `z_image_turbo_nvfp4`, `lumina_2_model_bf16`, `hunyuan3d-dit-v2-mv`,
all five Mistral-Nemo shards, the three RealESRGAN upscalers, and Dia in its new `hub\`
location.

This pass deleted weights only. No render has been run since; the suites above are a
static gate, not proof of a live leg.
