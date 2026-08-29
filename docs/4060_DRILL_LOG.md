# 4060 DRILL LOG -- real box, real state (restarted 2026-08-29 ~02:00)

Written by the working session ON the 4060 laptop (MRKT). Earlier relayed
entries claiming "all models present" and pushed squawks were from a session
that never verified against this disk; treat everything below as the ground
truth baseline. Central tracking file per operator order -- every step lands
here and is pushed.

## Box fingerprint

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB VRAM, driver 616.56
- RAM 32 GB; C: 550 GB free at start
- ComfyUI: Desktop install backend at
  `C:\Users\jeffr\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI`
  (venv Python 3.13.12, torch 2.12.1+cu130, CUDA OK)
- Pack: was registry alpha.9 (2d42d09f) with uncommitted local edits ->
  stashed (`git stash list`: "pre-update local changes 2026-08-29") ->
  pulled to origin/v2.0-alpha HEAD 9a9e3aaf
- `C:\ComfyUI-Models` DOES NOT EXIST on this box. Models root is the
  install's own `models\` tree; HF cache at `models\huggingface\hub`
  (gemma-4-12b/E4B/E2B, gemma-2-2b, bark, musicgen-small -- 73 GB already
  present). ffmpeg 9.0 on PATH.

## Step 1 -- model gap analysis + downloads (all verified byte-exact)

Missing was exactly the video/image stack. Fetched:

| file | GB | dest | source |
| --- | ---: | --- | --- |
| ltxv-2b-0.9.8-distilled.safetensors | 6.34 | checkpoints | Lightricks/LTX-Video |
| t5xxl_fp16.safetensors | 9.79 | text_encoders | comfyanonymous/flux_text_encoders |
| z_image_turbo_int8_convrot.safetensors | 6.20 | diffusion_models | Comfy-Org/z_image_turbo |
| qwen_3_4b_fp8_mixed.safetensors | 5.63 | text_encoders | Comfy-Org/z_image_turbo |
| ae.safetensors | 0.34 | vae | Comfy-Org/z_image_turbo |
| kokoro-v1_0.pth | 0.33 | TTS/KokoroTTS | hexgrad/Kokoro-82M |

int8_convrot (not nvfp4) chosen for the image UNET: nvfp4 is
Blackwell-native, this card is Ada. LTX 2.5 / MiniMax H3 ruled out on this
box (14.5+ GB VRAM class).

## Step 2 -- profile + launch

- New profile `config/profiles/otr_4060_nano_local.json`: otr_4060_nano with
  `music_engine: musicgen` (stable_audio_3 ckpt not on disk; musicgen-small
  is, in the HF cache). Video ltx_8gb, image z_image_turbo, writer
  google/gemma-4-E2B-it, voices kokoro.
- Headless launch (localized from `_otr_soak_server_launch.cmd`): port 8000,
  `HF_HOME=<install>\models\huggingface`, PYTHONUTF8=1,
  `OTR_ZIMAGE_UNET=z_image_turbo_int8_convrot.safetensors`,
  `OTR_ZIMAGE_CLIP=qwen_3_4b_fp8_mixed.safetensors`,
  output pinned to `C:\Users\jeffr\Documents\ComfyUI\output`.
- Boot clean, 25 OTR nodes registered.

## Step 3 -- leg 1: FAIL at first TTS clip (new portability bug, root-fixed)

`--profile otr_4060_nano_local --act-count 1`, prompt_id 82e70344. Writer
(E2B) wrote 6 lines / 70 words, ledger froze `frozen_with_warns`, casting
assigned kokoro voices -- then:

    TypeError: KPipeline.__init__() got an unexpected keyword argument 'repo_id'

`eng_kokoro.py` passes `repo_id=` unconditionally; kokoro 0.7.16 (the
NEWEST PyPI release) has `KPipeline(lang_code, model, trf, device)` -- no
such kwarg. Any stock `pip install kokoro` hits this, so every clean
install with the kokoro lane does. Fix (this commit): pass `repo_id` only
when `inspect.signature` says the installed KPipeline accepts it.
Prompt executed in 439.35s; no obs publish (correct -- it failed).

## Step 4 -- leg 2: FAIL at image-UNET load (DynamicVRAM native abort on 8 GB)

Kokoro fix HELD: writer wrote (6 lines / 150 words), freeze landed, casting
assigned, all 6 voice clips generated, visual-direction pass completed. Then at
the z_image sampler's step 0/8 ("Model Initializing"):

    aimdo: src/hostbuf.c:283:ERROR:hostbuf_read_file_slice: device copy
    failed result=2 ... size=39321600
    Fatal Python error: Aborted

CUDA error 2 = out of memory, hit while comfy_aimdo (DynamicVRAM) streamed the
6.2 GB image UNET onto a card still holding OTR's HF-side residents (gemma
writer et al). Two ship-relevant findings, only findable on a small card:

1. The pack's residency discipline does not evict the writer before the image
   phase; fine at 16 GB, fatal at 8 GB.
2. DynamicVRAM's failure mode is a NATIVE PROCESS ABORT, not a Python
   exception -- the whole server dies, nothing can catch or retry it. The
   legacy loader raises a catchable OOM instead.

## Step 5 -- leg 3 in flight (legacy loader)

Server relaunched with `--disable-dynamic-vram`; episode re-queued
(`--profile otr_4060_nano_local --act-count 1`). If leg 3 passes only with the
legacy loader, the 4060 profile (or docs) must carry that flag -- or the pack
must evict the writer before the image phase -- before an 8 GB card is a
supported target. Target unchanged: RESULT SUCCESS + obs_publish OK + mp4 in
`output\otr\obs`.
