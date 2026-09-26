# Which machine runs what

Every figure here is read from the code that uses it, so this page cannot drift from what the pack actually does.

Three questions, in the order people ask them.

## Which workflow do I open?

**On a first run, open the canonical.** It names no vendor anywhere and resolves your device at run time, it is the one workflow in Browse Templates, and it fetches no video or image weights -- which is why every machine class below can run it. The per-machine rows are a STEP UP once that has worked: each one is the canonical with its dropdowns pinned to heavier lanes, so the first queue on one of them downloads whatever those lanes need.

| Your machine | Open this | Also install |
|---|---|---|
| Anything, to start | `workflows/otr_canonical.json` (Workflow &rarr; Browse Templates &rarr; Extensions &rarr; Old-Time Radio) | nothing |
| 8 GB NVIDIA -- RTX 4060 / 3070 / 2080 class | `workflows/otr_8gb_video.json` | nothing |
| 16 GB+ NVIDIA -- RTX 5080 / 4080 / 3090 class | `workflows/otr_16gb_video.json` | nothing |
| Mac 16 GB -- Apple Silicon, unified memory | `workflows/otr_mac16_video.json` | nothing |
| AMD ROCm -- Windows or Linux -- and read "What the words mean" at the foot of this page before trusting any AMD cell | `workflows/otr_amd_still.json` &mdash; in the shipping set, and an outside tester published an episode from it on a Radeon AI PRO R9700 under ROCm 7.2 (commit 0fc0fb90) -- see the AMD note at the foot | nothing |
| CPU only -- no GPU at all | `workflows/otr_cloud_low.json` | nothing |

Every machine needs **ffmpeg and ffprobe** on PATH, and Linux needs one monospace TTF installed for burned captions.

If you change a dropdown yourself, these are the only picks that need anything beyond this pack. Everything not listed here runs on what you already have.

| If you select | Install |
|---|---|
| `animatediff15_lightning_video` | ComfyUI-AnimateDiff-Evolved |
| `animatediff15_v3_haunted_video` | ComfyUI-AnimateDiff-Evolved |
| `animatediff15_v3_stillin_lab_video` | ComfyUI-AnimateDiff-Evolved |

## Will this engine run on my machine?

**Video -- procedural, no video weights**

| dropdown | how you get it | size | 8 GB NVIDIA | 16 GB+ NVIDIA | Mac 16 GB | AMD ROCm | CPU only |
|---|---|---|---|---|---|---|---|
| `still_flat` | nothing | -- | fits | **proven** | **proven** | ? | ? |
| `still_motion` | nothing | -- | **proven** | **proven** | **proven** | **proven** | ? |
| `still_pan` | nothing | -- | **proven** | **proven** | **proven** | ? | ? |
| `still_word` | nothing | -- | fits | measured | **proven** | ? | ? |
| `viz_camera` | nothing | -- | fits | **proven** | **proven** | ? | ? |
| `viz_green` | nothing | -- | fits | **proven** | **proven** | ? | ? |
| `viz_mxc_cpu` | nothing | -- | **proven** | **proven** | **proven** | **proven** | ? |
| `viz_mxc_mandala` | nothing | -- | fits | **proven** | fits | ? | ? |

**Video -- hosted, no weights but you supply the key**

| dropdown | how you get it | size | 8 GB NVIDIA | 16 GB+ NVIDIA | Mac 16 GB | AMD ROCm | CPU only |
|---|---|---|---|---|---|---|---|
| `cloud_ltx25_audio_in` | none | -- | key | key | key | key | key |
| `cloud_ltx25_foley_plus` | none | -- | key | key | key | key | key |
| `cloud_seedance_2` | none, **but see below** | -- | key | key | key | key | key |
| `cloud_vidu_q2_pro_fast_720p` | none | -- | key | key | key | key | key |
| `cloud_wan_i2v` | none | -- | key | key | key | key | key |
| `cloud_wan_i2v_audio` | none | -- | key | key | key | key | key |
| `google_omni_video` | none | -- | key | key | key | key | key |
| `google_veo_video` | none | -- | key | key | key | key | key |

**Video -- local diffusion**

| dropdown | how you get it | size | 8 GB NVIDIA | 16 GB+ NVIDIA | Mac 16 GB | AMD ROCm | CPU only |
|---|---|---|---|---|---|---|---|
| `animatediff15_lightning_video` | **auto** | 3.1 GiB | fits | fits | **proven** | ? | not offered |
| `animatediff15_v3_haunted_video` | **auto** | 3.6 GiB | **proven** | **proven** | not offered | ? | not offered |
| `animatediff15_v3_stillin_lab_video` | **auto** | 3.6 GiB | fits | fits | not offered | ? | not offered |
| `mesh_stage` | manual | 4.6 GiB | fits | fits | not offered | ? | not offered |
| `humo17_high_audio_in_portrait` | manual | 12.6 GiB | **OOM** | **proven** | not offered | ? | not offered |
| `humo17_high_audio_in_wide` | manual | 12.6 GiB | **OOM** | **proven** | not offered | ? | not offered |
| `ltx098_low_video` | **auto** | 16.1 GiB | **proven** | **proven** | **proven** | ? | not offered |
| `razzle_ltx_8gb` | **auto** | 16.1 GiB | fits | fits | fits | ? | not offered |
| `ltx25_foley_blackwell` | **auto** | 24.2 GiB | ? | ? | not offered | ? | not offered |
| `ltx25_audio_in_16gb` | **auto** | 25.4 GiB | ? | ? | not offered | ? | not offered |
| `ltx25_foley_16gb` | **auto** | 25.4 GiB | ? | ? | not offered | ? | not offered |
| `ltx25_high_video` | **auto** | 25.4 GiB | measured | measured | not offered | ? | not offered |
| `ltx25_mime_16gb` | **auto** | 25.4 GiB | ? | ? | not offered | ? | not offered |
| `humo14_high_audio_in_portrait` | manual | 26.7 GiB | **OOM** | **proven** | not offered | ? | not offered |
| `humo14_high_audio_in_wide` | manual | 26.7 GiB | **OOM** | **proven** | not offered | ? | not offered |
| `ltx25_audio_in_24gb` | **auto** | 32.5 GiB | ? | ? | not offered | ? | not offered |
| `ltx25_foley_24gb` | **auto** | 32.5 GiB | ? | ? | not offered | ? | not offered |
| `ltx25_mime_24gb` | **auto** | 32.5 GiB | ? | ? | not offered | ? | not offered |
| `h3_low_video` | manual | 41.9 GiB | **OOM** | **proven** | not offered | ? | not offered |
| `h3_low_audio_in` | manual | 42.5 GiB | **OOM** | fits | not offered | ? | not offered |

**Image -- local**

| dropdown | how you get it | size | 8 GB NVIDIA | 16 GB+ NVIDIA | Mac 16 GB | AMD ROCm | CPU only |
|---|---|---|---|---|---|---|---|
| `sd15` | **auto** | 2.0 GiB | fits | **proven** | **proven** | ? | too slow |
| `lumina_image` | **auto** | 10.4 GiB | **OOM** | **proven** | not offered | ? | not offered |
| `flux_gen1` | manual | 13.0 GiB | **OOM** | **proven** | not offered | ? | not offered |
| `ideogram4_local` | manual | 17.3 GiB | **no** | **proven** | not offered | not offered | not offered |
| `z_image_turbo` | **auto** | 19.3 GiB | **proven** | **proven** | not offered | **proven** | not offered |

**Image -- hosted**

| dropdown | how you get it | size | 8 GB NVIDIA | 16 GB+ NVIDIA | Mac 16 GB | AMD ROCm | CPU only |
|---|---|---|---|---|---|---|---|
| `cloud_flux_pro` | none | -- | key | key | key | key | key |
| `cloud_krea_2_turbo` | none | -- | key | key | key | key | key |
| `cloud_luma_photon_flash` | none | -- | key | key | key | key | key |
| `cloud_nano_banana_2` | none | -- | key | key | key | key | key |
| `cloud_seedream_2` | none | -- | key | key | key | key | key |
| `google_image` | none | -- | key | key | key | key | key |
| `ideo` | none | -- | key | key | key | key | key |

**Voice and music -- local**

| dropdown | how you get it | size | 8 GB NVIDIA | 16 GB+ NVIDIA | Mac 16 GB | AMD ROCm | CPU only |
|---|---|---|---|---|---|---|---|
| `kokoro` | **auto** | 0.3 GiB | **proven** | **proven** | **proven** | **proven** | **proven** |
| `musicgen` | **auto** | 2.2 GiB | **proven** | **proven** | measured | ? | **proven** |
| `chatterbox` | own installer (Windows) | 3.0 GiB | not offered | fits | not offered | not offered | not offered |
| `stable_audio_3` | **auto** | 3.5 GiB | **proven** | **proven** | **proven** | ? | not offered |
| `bark` | **auto** | 4.2 GiB | **proven** | **proven** | **OOM** | ? | too slow |
| `stable_audio_music` | GATED | 4.5 GiB | fits | fits | not offered | ? | not offered |
| `dia` | own installer (Windows) | 6.0 GiB | not offered | fits | not offered | not offered | not offered |
| `indextts2` | own installer (Windows) | 11.1 GiB | not offered | **proven** | not offered | not offered | not offered |

**Voice and music -- hosted**

| dropdown | how you get it | size | 8 GB NVIDIA | 16 GB+ NVIDIA | Mac 16 GB | AMD ROCm | CPU only |
|---|---|---|---|---|---|---|---|
| `cloud_elevenlabs` | none | -- | key | key | key | key | key |
| `google_lyria` | none | -- | key | key | key | key | key |
| `google_tts` | none | -- | key | key | key | key | key |
| `sonilo` | none | -- | key | key | key | key | key |

**Upscale**

| dropdown | how you get it | size | 8 GB NVIDIA | 16 GB+ NVIDIA | Mac 16 GB | AMD ROCm | CPU only |
|---|---|---|---|---|---|---|---|
| `off` | nothing | -- | **proven** | **proven** | **proven** | ? | ? |
| `spandrel_esrgan` | manual | 0.1 GiB | fits | **proven** | measured | ? | ? |

**Writer (the LLM that writes the script)**

| dropdown | how you get it | size | 8 GB NVIDIA | 16 GB+ NVIDIA | Mac 16 GB | AMD ROCm | CPU only |
|---|---|---|---|---|---|---|---|
| `google/gemma-2-2b-it` | GATED | 5.2 GiB | **proven** | fits | fits | ? | ? |
| `google/gemma-4-E2B-it` | **auto** | 6.0 GiB | **proven** | **proven** | **OOM** | ? | ? |
| `unsloth/Llama-3.2-3B-Instruct` | **auto** | 6.4 GiB | fits | fits | fits | ? | ? |
| `Qwen/Qwen3.5-4B` | **auto** | 8.7 GiB | **proven** | **proven** | **proven** | ? | ? |
| `google/gemma-4-E4B-it` | **auto** | 9.0 GiB | measured | **proven** | **tight** | ? | ? |
| `google/gemma-4-12b-it` | **auto** | 23.9 GiB | measured | **proven** | **no** | ? | ? |
| `mistralai/Mistral-Nemo-Instruct-2407` | **auto** | 24.0 GiB | **no** | **proven** | **no** | ? | ? |
| `Qwen/Qwen3.8-27B` | **auto** | 51.8 GiB | **no** | **24 GB+** | **no** | ? | ? |

## Where do the manual weights come from?

Every file a **manual** row needs: the repository to download it from, and the folder under your ComfyUI `models/` directory to put it in. `gated` means you must accept the model's licence on Hugging Face first, while signed in.

Two engines can share one group and still download different amounts, because they draw different files from it. **The size in the machine grid above is what YOUR pick costs**; the total on a heading here is the whole group. A heading with no total means that group's manifest predates byte receipts -- the machine grid still has the figure.

### h3_operator_only &mdash; 59.1 GiB total

Selected by: `h3_low_audio_in`, `h3_low_video`

| File | From | Put it in | Size | Gated |
|---|---|---|---|---|
| `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | [`Comfy-Org/MiniMax-H3`](https://huggingface.co/Comfy-Org/MiniMax-H3) | `models/diffusion_models/` | 19.53 GiB | no |
| `minimax_h3_ref2va_pruned_int8_convrot.safetensors` | [`Comfy-Org/MiniMax-H3`](https://huggingface.co/Comfy-Org/MiniMax-H3) | `models/diffusion_models/` | 19.53 GiB | no |
| `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | [`Comfy-Org/MiniMax-H3`](https://huggingface.co/Comfy-Org/MiniMax-H3) | `models/text_encoders/` | 14.61 GiB | no |
| `minimax_h3_video_vae_fp16.safetensors` | [`Comfy-Org/MiniMax-H3`](https://huggingface.co/Comfy-Org/MiniMax-H3) | `models/vae/` | 4.85 GiB | no |
| `minimax_h3_audio_vae_fp32.safetensors` | [`Comfy-Org/MiniMax-H3`](https://huggingface.co/Comfy-Org/MiniMax-H3) | `models/vae/` | 0.56 GiB | no |

### humo &mdash; 26.7 GiB total

Selected by: `humo14_high_audio_in_portrait`, `humo14_high_audio_in_wide`

| File | From | Put it in | Size | Gated |
|---|---|---|---|---|
| `Wan2_1-HuMo-14B_fp8_e4m3fn_scaled_KJ.safetensors` | [`Kijai/WanVideo_comfy_fp8_scaled`](https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled) | `models/diffusion_models/` | 16.66 GiB | no |
| `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | [`Comfy-Org/Wan_2.1_ComfyUI_repackaged`](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged) | `models/text_encoders/` | 6.27 GiB | no |
| `whisper_large_v3_fp16.safetensors` | [`Comfy-Org/HuMo_ComfyUI`](https://huggingface.co/Comfy-Org/HuMo_ComfyUI) | `models/audio_encoders/` | 2.88 GiB | no |
| `wan_2.1_vae.safetensors` | [`Comfy-Org/Wan_2.2_ComfyUI_Repackaged`](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged) | `models/vae/` | 0.24 GiB | no |
| `lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors` | [`Kijai/WanVideo_comfy`](https://huggingface.co/Kijai/WanVideo_comfy) | `models/loras/` | 0.69 GiB | no |

### humo_1_7b &mdash; 12.6 GiB total

Selected by: `humo17_high_audio_in_portrait`, `humo17_high_audio_in_wide`

| File | From | Put it in | Size | Gated |
|---|---|---|---|---|
| `humo_1.7B_fp16.safetensors` | [`Comfy-Org/HuMo_ComfyUI`](https://huggingface.co/Comfy-Org/HuMo_ComfyUI) | `models/diffusion_models/` | 3.24 GiB | no |
| `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | [`Comfy-Org/Wan_2.1_ComfyUI_repackaged`](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged) | `models/text_encoders/` | 6.27 GiB | no |
| `whisper_large_v3_fp16.safetensors` | [`Comfy-Org/HuMo_ComfyUI`](https://huggingface.co/Comfy-Org/HuMo_ComfyUI) | `models/audio_encoders/` | 2.88 GiB | no |
| `wan_2.1_vae.safetensors` | [`Comfy-Org/Wan_2.2_ComfyUI_Repackaged`](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged) | `models/vae/` | 0.24 GiB | no |

### No manifest ships for these

This pack cannot fetch them, and no table here can tell you the filename, because the engine chooses it. Select one anyway and it refuses by name before anything else runs -- **that refusal is the install instruction**: it prints the exact file it wants and the folder it expects. It never quietly substitutes another.

* `flux_gen1`
* `ideogram4_local`
* `mesh_stage`
* `spandrel_esrgan`

## What the words mean

**How you get the weights.** Two things do the fetching for an **auto** row, and
neither of them is a script you have to run: the engine's own library pulls it
through the Hugging Face cache, or `OTR_WorkflowValidator` -- a node inside the
workflow -- downloads it at queue time. A **manual** row may still have a helper in
`scripts/`, but `scripts/` is not in the registry bundle, so from a normal
install it is a step you take by hand and it is labelled as one.

**auto** -- fetched on first use, no account and no
token; just pick it and run. **GATED** -- fetches itself, but only after you
accept a licence on the model page and set `HF_TOKEN`. **manual** -- you fetch
it yourself; the manual-weights table above names every file, the repository it comes from and the folder it goes in.
**none** -- no weights at all. *no lane* -- the engine is registered but no
provisioning lane is declared for it, so nothing will fetch it for you.

**own installer** -- installs through its own script rather than the model
provisioner; **(Windows)** marks the three whose installer is PowerShell with no
`.sh` twin, so on Linux and macOS there is no install path today. That is
packaging, not hardware -- writing the shell installer is what clears it. **nothing** -- pure code; there is nothing to obtain.

Sizes are GiB, summed from the real artifact bytes in the fetch manifests where
a lane carries them, otherwise the figure the fetcher's own pick list states.

**What a machine cell means, and read this before you read one.** Each cell
answers TWO questions in order.

* **not offered** -- this engine's declaration does not list that backend, so
  nothing here claims it works there. It is a statement about RECEIPTS, **not
  about your hardware and not about the menu**: several of these have run on
  that hardware, and the dropdown is built from the whole registry, so the
  engine IS still selectable on that machine. What protects you is not a hidden
  filter -- it is that an engine which cannot run stops the render with a named
  error instead of quietly substituting something else. Moving a cell out of
  this state is a declaration change plus a receipt, not a purchase.
* **too slow** -- offered on a CPU-only box in principle, kept off it because it
  is not practical there.
* Otherwise the engine IS offered, and the word is the memory verdict:
  **proven** (a PUBLISHED EPISODE used it), measured (it ran on that hardware
  in a lab test and worked, but no episode has ever used it), fits (nothing
  blocks it and the arithmetic says it fits -- nobody has run it at all),
  **OOM** (expect to exhaust memory), **?** (offered, nobody has measured it).
  **24 GB+** appears only in the "16 GB+ NVIDIA" column and means what it
  says: too big for a 16 GB card, fine on the 24 GB end of that same column.
  It exists because that column spans the 5080 and the 3090, and a writer can
  land between them.

**The proven/measured split IS the test plan.** "measured" is precisely the list
of engines to close next, and the distinction was earned: a first pass called
both states "proven", which put engines in the same column as ones that had
carried a whole episode. Note also that an episode's FILENAME records only its
dominant video lane, so counting receipts from filenames under-reports -- one
published episode here ran viz_camera, viz_mxc_cpu and viz_green together.

**The AMD column is the weakest one here, and it is weak by construction.** The
ROCm workflows declare `device_backend: "cuda"`, because that is how ROCm
presents itself to torch -- so every CUDA lane reads as offered there, and the
column is really answering "is this vendor-locked or sidecar-locked?" rather
than "has this been run on AMD?". So treat an unmarked AMD cell as the absence
of a hard blocker, nothing more.

**AMD HAS A RECEIPT, and this paragraph used to deny it.** An outside tester ran
`otr_amd_still.json` end to end on a Radeon AI PRO R9700
(32 GB, RDNA4 / gfx1201) under ROCm 7.2 on Ubuntu 24.04 and published a finished
episode, with no edits to the workflow -- commit `0fc0fb90`, 2026-09-14, pack commit
`0b38424`. Four engines are marked **proven** there from their own artifacts
rather than their summary: `viz_mxc_cpu`, `still_motion`, `kokoro` and
`z_image_turbo`. Their music line was ambiguous, so no music engine is marked.
What that receipt does NOT cover, and still reads **?**: RDNA3, Windows, the
8 GB AMD workflow, and every lane past the still tier. It is also 100+ commits
old and the widget tier has regenerated the workflows since, so what is proven
is that the PIPELINE and the engines that workflow selects run on ROCm -- not
that
today's file byte-for-byte has been through a Radeon.

**On a Mac, OOM is a HARD MACHINE REBOOT, not a failed render** -- unified
memory has no separate pool to exhaust. That is why the Mac column is worth
reading before you pick, and why an unmeasured **?** there deserves more caution
than the same mark on a discrete card.

**Where a hosted key goes.** The cell says **key**. How you enter it is the
heading of the README. Google and OpenRouter: `google.secret` /
`openrouter.secret`, or a path in the matching `*_api_key.location` file.
Comfy Cloud: sign into the app. A Comfy key file is only for a headless
box. Never paste a key into a workflow.

---

*This page is generated. To change it, edit `scripts/otr_dropdown_matrix.py` in the GitHub tree and run it -- `scripts/` is not part of an installed copy, so there is nothing here to hand-edit.*
