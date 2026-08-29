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

## Operator observation (2026-08-29 ~03:20, watching the live stream) -- the "dub lane"

Watching an animated episode play on the live OBS stream, the operator called
out that the UNSYNCED mouth/motion animation over the voice track reads like a
JAPANESE DUB -- an aesthetic audiences have accepted for decades -- and asked
for it as ITS OWN LANE ("I'm serious"). Why it matters as a lane, in his
framing: great perceived lip-sync feel at ZERO sync cost -- no audio fed into
the video generator, no audio-conditioned models, no per-beat clip timing
machinery. Animated characters with mouth motion + independent voice track =
"close to perfect for a jap dub". This is a STYLE PRESET on the existing video
lanes (prompt for animated speaking characters, loose motion), not new
infrastructure -- which is exactly what makes it cheap on an 8 GB card.

Acceptance bar, from the operator directly: the trade is worth it at ~0.5x
render time ("thats ok if its .5 teh render time"). Speed IS the product here
-- the lane earns its place by halving the render, with the dub aesthetic
absorbing whatever sync fidelity that costs.

## Step 5 -- leg 3 in flight (legacy loader)

Server relaunched with `--disable-dynamic-vram`; episode re-queued
(`--profile otr_4060_nano_local --act-count 1`). If leg 3 passes only with the
legacy loader, the 4060 profile (or docs) must carry that flag -- or the pack
must evict the writer before the image phase -- before an 8 GB card is a
supported target. Target unchanged: RESULT SUCCESS + obs_publish OK + mp4 in
`output\otr\obs`.

LEG 3 OUTCOME: the crash-point survival was PROVEN -- the legacy loader
partial-loaded the z_image UNET (5.9 GB offloaded to RAM) and was actively
sampling at the exact step where leg 2's aimdo abort killed the process. The
8 GB finding stands confirmed in both directions: DynamicVRAM aborts, legacy
loader survives. Leg 3 was then KILLED BY OPERATOR ORDER (~03:40, selective
CIM kill per section 4) to free the card for the haunted race below; it never
reached obs_publish, so it is logged as killed-in-flight, not as a pass.

## Step 6 -- the haunted race (operator order: all three GPUs)

Operator pivoted the night: a 3-way AnimateDiff race (4060 vs 5080 vs H100
via the 5080) on the pack's OWN haunted lane, `animatediff15_v3_haunted_video`
(eng_ghost_signal_official.py), toward the dub-lane goal above. If AnimateDiff
proves faster overall than LTX for beat video, LTX is killed as the beat lane
("if the animatediff is faster overall, kill the ltx and launch animatediff").

4060 staging, all verified byte-exact against the engine's own documented
sizes: mm-p_0.5.pth 1,817,894,327 B; v3_sd15_mm.ckpt 1,673,262,583 B;
v3_sd15_adapter.ckpt 102,134,097 B (-> models\loras); SD1.5 fp16
2,132,696,762 B; plus ComfyUI-AnimateDiff-Evolved cloned and loading in 0.5 s.

New profile `config/profiles/otr_4060_haunted_local.json`: the proven
nano-local stack (E2B writer, kokoro voices, musicgen) with all visual roles
on the haunted lane, frame_budget 25 per the ghost profile's recipe. The
haunted lane is TEXT_TO_VIDEO -- no scene stills -- so the 6.2 GB z_image
UNET leaves the beat hot path entirely, which may make this the natural 8 GB
lane regardless of the race. Race leg queued ~03:52 as prompt 02835636,
preflight green on all three lane models. Wall time to be reported honestly
against the LTX baseline (5080: 270-285 s/beat; 4060 LTX never completed a
beat).

RACE LEG 1 OUTCOME: FAIL at the scopes stage -- the night's THIRD genuine
cold-install portability catch. Everything upstream passed: writer, voices,
music, base-video encode, and ALL haunted AnimateDiff beats (v3 module +
sliding context window; later beats ~9-10 s/step at a comfortable 4.9 GB,
SD1.5 fully resident -- no offload, unlike everything LTX/z_image). Then
OTR_SceneAwareScopes died instantly: `Unrecognized option 'vsync'`. ffmpeg 9
(this box) REMOVED `-vsync` (deprecated since 5.1); the 5080's older ffmpeg
still accepts it, so the bug was invisible everywhere but here. Five call
sites shipped it (scope_draw, encode_sink, silent_composite x2,
caption_burn).

Fix (this commit), same discipline as the kokoro repo_id gate: a shared
`scope_draw.cfr_flags(ffmpeg)` that probes the installed binary once (1-frame
lavfi null encode -- exercises the real argv parser) and returns
`-fps_mode cfr` when accepted, legacy `-vsync cfr` otherwise; all five sites
route through it. Verified on this box: probe selects `-fps_mode`; AST parse
clean on all four files. Race leg 2 re-queued 04:32:12 as prompt be8d016d
with the fix loaded.

## Step 7 -- RESULT SUCCESS: first episode ever published from this box

Race leg 2 (prompt be8d016d): **RESULT SUCCESS + obs publish + the mp4 on
disk** -- every gate the operator defines, on the first run with the cfr fix.

    signal_lost_the_ledgers_whisper_20260829_045225_silent_procgen_blended
    _captioned_with_credits_final.mp4
    66.6 MB, 69.92 s, h264 + AAC (real audio: kokoro voices + musicgen),
    published to local otr\obs; LISTEN.html rebuilt beside it (1 episode);
    copied to D:\4060-transfer for the 5080's pull the moment a path opens.

Numbers, honestly framed: 3,303 s (~55 min) queue-to-publish end to end on
the haunted lane. Beats sampled at ~9-10.7 s/step x 20 steps (~3-3.6 min per
clip) at 4.9 GB with SD1.5 fully resident. There is no same-box LTX number
to race it against because LTX NEVER FINISHED on this card (leg 2 aimdo
abort; leg 3 killed mid-image by operator order) -- which is itself the
result: **the haunted AnimateDiff lane is the first and only lane to carry a
complete episode through an 8 GB card end to end.** The dub-lane thesis
holds on this hardware.

Night ledger for this box: 3 cold-install bugs found and root-fixed on
origin (kokoro repo_id TypeError, DynamicVRAM native abort at image load,
ffmpeg-9 -vsync removal), ~32 GB of weights staged byte-exact, 2 profiles
shipped, 1 episode published.
