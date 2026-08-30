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

CLAIM SCOPE (verified against the leg-2 log, so nobody overclaims): every
stage ran end to end; the two degradations were LOUD, never silent -- (a)
Ghost Prompt v2 used its deterministic prompt author on all 8 beats (the E2B
writer's leaf attempts were rejected by the guards), (b) pyloudnorm is
missing so mastering fell back from LUFS to legacy peak. "procgen_blended"
in the filename is the canonical compose chain, not a lane fallback. The
honest sentence: **the canonical workflow is 4060-proven end to end on the
otr_4060_haunted_local profile with --disable-dynamic-vram** -- not bare
"4060-proof". Unproven on this box: LTX (never finished), the z_image stills
lane (survived sampling once, never completed an episode), multi-act
episodes, and the haunted lane under stock DynamicVRAM.

FINDING #4 (cold-install): pyloudnorm is declared in NEITHER
requirements.txt NOR pyproject -- every registry install masters by peak,
not LUFS. Rides the next deliberate pyproject bump alongside kokoro.

## Step 8 -- OPERATOR DIRECTIVE (2026-08-29 ~10:00) + the writer-size answer

Directive, his words: "I don't want guards to kill anything -- an OOM is the
only killer." Guards may fall back LOUDLY (the ghost-prompt fallback and the
peak-master fallback last night are the model); they must not abort a render
on an estimate or a quality judgment. Reconciling this with the standing
fail-loud/no-fallback guards (miscast voices, scope NO FALLBACK raises,
VRAMFit) is a DESIGN ITEM for the dev box, not a dawn rewrite from here.

The directive got a live test within minutes. He asked why the writer was
E2B and not gemma-4-12b (which is fully cached here -- not gated, never
selected; every 4060 profile pins E2B for headroom). Measured answers:

- 12B attempt 1: KILLED BY A GUARD, not memory -- VRAMFitFailedError,
  "estimated 11.9 GB peak vs 6.8 ceiling", 0.10 s. Exactly the class the
  directive outlaws. Ceiling raised to 12.0 to let physics judge.
- 12B attempt 2: the memory judge ruled -- bnb 4-bit validate_environment:
  "Some modules are dispatched on the CPU... Make sure you have enough GPU
  RAM to fit the quantized model." 12B nf4 does not fit this card. Accepted.
- E4B attempt (otr_4060_haunted_e4b, nf4, ceiling 6.8): SAME CPU-dispatch
  refusal in 2.17 s -- suspicious, because ~4-5 GB of nf4 weights against
  ~7.1 GB free should fit. CANDIDATE FINDING #5: the loader's
  device_map/max_memory derivation may be over-reserving on small cards
  (or MatFormer per-layer-embedding modules are planned to CPU by design and
  trip the bnb refusal). Two attempts spent; per the two-strikes rule the
  third swing belongs to a panel/dev-box review, not this window.

Standing writer verdict for MRKT until that lands: E2B (unquantized) is the
qualified writer; 12B nf4 measured out; E4B nf4 blocked pending the loader
question.

LOADER FIX LANDED (operator order "fix the auto-loader so it doesn't
reject"): load_llm now retries a refused NF4 load ONCE with
llm_int8_enable_fp32_cpu_offload (the same permission the 8-bit branch has
always had), behind a loud warning -- fitting models take the unchanged
first-attempt path; oversized ones now ATTEMPT instead of being refused.
Verified live: the 12B streamed its weights for the first time all night
(677 shards, retry warning in the log). It then failed DEEPER in
transformers' offload machinery -- "Tensor.item() cannot be called on meta
tensors" -- which is a second, separate defect in the bnb-4bit CPU-offload
path, handed to the dev box with finding #5. The wrap that mislabeled these
refusals as cache errors now names the underlying exception. GGUF lane
checked as the designed alternative (model id -> unsloth/gemma-4-12b-it-GGUF,
native partial offload): blocked on MRKT because llama-cpp-python is not in
this venv; installing the CUDA build on Windows is a dev-box decision.

FINDING #5 CONFIRMED AND PINNED (operator's viz+12B experiment, profile
`otr_4060_viz_12b`: viz_camera on all four video roles to free the whole
card, fresh server, ~7.4 GB free -- SAME refusal):
`_otr_model_loader.py::_plan_max_memory` hardcodes, for bnb quant on a
sub-12GB card, `{0: "6.8GiB", "cpu": "32GiB"}` for any model id tagged
9b/12b/e4b/4b-it -- regardless of actual free VRAM. The explicit cpu lane
invites accelerate to plan offload for anything over 6.8 GiB on-GPU, and
bnb-4bit's validate_environment refuses CPU-dispatched modules. The video
lanes are irrelevant; the cap fires first. E4B dies the same way because
its MatFormer per-layer embeddings stay unquantized and clear 6.8 GiB even
though its linears fit. Third swing handed to the dev box with the
operator's guard directive attached (derive budget from live free VRAM
and/or drop the cpu key so a genuine OOM speaks instead of a guard).

Night ledger for this box: 4 confirmed cold-install findings (kokoro repo_id
TypeError, DynamicVRAM native abort at image load, ffmpeg-9 -vsync removal
-- all three root-fixed on origin -- plus the undeclared pyloudnorm), 1
candidate finding (E4B/nf4 loader fit), ~32 GB of weights staged byte-exact,
4 profiles shipped, 1 episode published. Operator's framing to carry
forward: this box is rung one of a PORTABILITY PROVING GROUND -- rented GPU
classes as a qualification matrix for the workflow, not as render farms.

## Step 9 -- THE ASTERISK IS GONE: the SHIPPING default passes on the STOCK path

The claim's biggest caveat was that every 4060 success used
`--disable-dynamic-vram`, a flag no ordinary user passes. That caveat is now
retired by measurement.

**Leg: `--profile otr_nvidia_8gb_haunted` (the 5080's SHIPPING default, not a
tuned local profile), prompt `814d4e4d`, server booted WITHOUT the flag.**
DynamicVRAM confirmed genuinely ACTIVE at boot rather than assumed:
comfy-aimdo 0.4.15, 6 CUDA hooks installed, NVML pressure enabled, WDDM
adapter matched, "DynamicVRAM support detected and enabled".

    RESULT SUCCESS + obs_publish OK + mp4 on disk
    signal_lost_shadows_lengthening_on_the_heath_20260829_104142
      _silent_procgen_blended_captioned_with_credits_final.mp4
    60.4 MB, 75.32 s, h264 + AAC
    wall: 2322 s (38.7 min) queue-to-publish
    aimdo aborts: 0    (grep: Fatal Python error / hostbuf_read_file_slice = 0)

**THREE FINDINGS BEYOND THE PASS:**

1. **PBUG-20260829-03 does NOT reach the shipping haunted default.** The abort
   fired while DynamicVRAM streamed the 6.2 GB z_image UNET; this lane is
   text_to_video and loads no image model at all, so the hazard is absent by
   construction. `--disable-dynamic-vram` is therefore NOT a requirement for
   `otr_nvidia_8gb_haunted` -- it remains required only for lanes that load a
   large image UNET after the writer. Scope the user-facing docs accordingly:
   the 8 GB haunted path works out of the box.
2. **The stock path was FASTER, which nobody predicted:** 2322 s here versus
   3303 s (55 min) for the byte-identical profile WITH the flag last night --
   about 30% quicker. DynamicVRAM is not merely survivable on this lane, it is
   an improvement. Peak VRAM during beats sat at ~4.2 GB of 8.0 GB.
3. **The writer authored every visual prompt this time:** `Ghost Prompt v2:
   8 beat(s) authored (writer_llm=8)`, zero `deterministic_fallback` -- against
   8-of-8 fallback on the previous leg. So the E2B fallback recorded in step 7
   is NOT a fixed property of the small writer; it varied with the style pack
   (`visual_storybased` here vs `shakespeare_stage_realism` there). Do not
   quote "E2B cannot author ghost prompts" as a finding; it can.

**Degradations: all LOUD, none silent.** Per-beat VRAM reclaim (x8 pairs, by
design), two ghost-prompt info warnings, one stale-master-path re-resolve that
self-healed, one `ledger_clean_line_judge` retry, and the pyloudnorm fallback
below.

**AUDIO PROVENANCE, and it invalidates cross-box mix comparisons before now:**
`pyloudnorm` was absent from this venv, so BOTH 4060 episodes
(`the_ledgers_whisper`, `shadows_lengthening_on_the_heath`) were mastered by
the legacy PEAK path while every 5080 episode was mastered to -14 LUFS. Same
folder, same broadcast, two different masterers, evidence limited to one
warning line. `pyloudnorm 0.2.0` is now installed here (matching the 5080), so
episodes from this box are comparable from the NEXT leg onward -- these two are
not. See PBUG-20260829-04.

**THE HONEST CLAIM NOW:** *the canonical workflow is 4060-proven end to end on
`otr_nvidia_8gb_haunted`, out of the box, with no launch flags* -- twice, on
two different profiles that are byte-identical apart from their names. Still
unproven on this box: LTX (never finished), the z_image stills lane (never
completed an episode), multi-act episodes, and any quantized writer (see
-05/-06/-07).

## Step 10 -- the 12B leg against the -07 fix: the tag fix WORKS and does NOT rescue 12B

Leg: `--profile otr_4060_haunted_12b`, prompt `5c509116`, on origin `5987b336`
(carrying the token-boundary fix `da54ee9d`), fresh server so the new loader
was actually imported.

**The -07 fix is confirmed working, measured two ways.** Executing the shipped
post-fix `_plan_max_memory` (AST-extracted, run in the venv):
`google/gemma-4-12b-it @ 8.00 GB -> {0: '6.8GiB', 'cpu': '32GiB'}` -- up from
`3.2GiB` -- while `gemma-4-2b-it`, `gemma-2-2b-it` and `gemma-4-E2B-it` all
still correctly get `3.2GiB`, and the 16 GB path is unchanged at `13.5GiB`.
Corroborated at runtime: VRAM reached **6676 MiB** during this load, consistent
with a ~6.8 GiB budget being filled, where the pre-fix 3.2 GiB cap could not
have.

**And the leg still FAILED -- shape read from the log, not from the prediction:**

    line  65  Loading LLM model: google/gemma-4-12b-it (quantized=True)
    line 214  hf_quantizer.validate_environment(device_map=device_map)
              ValueError: Some modules are dispatched on the CPU or the disk
    line  69  WARNING [StoryOrchestrator] ... exceeds the GPU budget for a
              full NF4 load -> the 22975e1c retry fires (LOUD, as designed)
    line 242  dispatch_model(model, **device_map_kwargs)
    line 329  RuntimeError: Tensor.item() cannot be called on meta tensors
    Prompt executed in 38.53 s; ModelLoaderError, no publish (correct).

**THE FAILURE MATCHES -06, NOT -05.** -05's defect was the budget; the budget
is now correct and the run still spilled, because 12B NF4 (~6.99 GiB planned,
plus Gemma's bf16 embedding table that NF4 never touches) genuinely does not
fit 6.8 GiB on an 8188 MiB card. So `infer_auto_device_map` legitimately
assigns a CPU tail, bnb refuses it, the retry permits it, and -06's mechanism
then kills it at dispatch -- exactly the panel's Test A -> Test B sequence,
reproduced here at the corrected budget.

**What this settles:** the tag collision was real and is fixed, but it was
never what stopped 12B on this card. **12B NF4 does not fit 8 GB by
arithmetic**, and no budget change reaches that. The remaining routes are the
explicit-dict device_map (panel Test C: loads, forwards, generates) or the GGUF
lane -- and per -07's correction, neither the GGUF weights nor
`llama-cpp-python` exist on this box today. E2B remains the qualified writer
here.

## Step 11 -- the GGUF route is BLOCKED ON MRKT by a CUDA-major mismatch

Four shipped profiles (`8gb_lite`, `otr_8gb_ltx`, `otr_8gb_wan`,
`otr_8gb_fastwan`) pin `unsloth/gemma-4-12b-it-GGUF` at `quant_policy: "none"`
-- i.e. the repo already asserts 12B-on-8GB via llama.cpp, which splits GPU/CPU
natively with no meta-tensor round trip, so none of step 10's arithmetic
applies. **None of those four has ever been run on this box.** Probed the
cheapest disqualifying step first, before spending a ~7 GB model download:

- venv python is **3.13.12**. PyPI has NO `llama-cpp-python` wheel for 3.13
  (`--only-binary=:all:` -> "No matching distribution found").
- The project's own CUDA wheel index DOES have one: `llama_cpp_python 0.3.35`
  installs cleanly from `abetlen.github.io/llama-cpp-python/whl/cu124`.
- **But it cannot load.** `RuntimeError: Failed to load shared library
  'llama_cpp\lib\llama.dll' ... or one of its dependencies`. Cause identified
  from the bundled files rather than guessed: the wheel ships a CUDA-**12**
  build (`ggml-cuda.dll`, 819 MB) and NO CUDA runtime, while this box provides
  only **`cudart64_13.dll`** (CUDA 13.0, via torch 2.12.1+cu130). Major-version
  mismatch, not a missing DLL.

**Left the box exactly as found:** the wheel is UNINSTALLED. An importable-but-
broken `llama_cpp` is worse than an absent one -- any availability probe would
report the lane usable and then fail at load, which is precisely the
"documented path is not a working path" class this drill keeps catching.

**The remaining step is an operator decision, not a window's:** supplying a
CUDA 12 runtime alongside torch's 13 (e.g. `pip install nvidia-cuda-runtime-cu12`
plus its bin dir on PATH at launch) would likely satisfy it, but it mixes CUDA
majors inside a process that also loads torch, on the one box with a PROVEN
shipping render path. Not worth risking that unilaterally for a writer upgrade
when E2B already ships. **Recorded, not attempted.** The four GGUF profiles
therefore remain UNVERIFIED on 8 GB hardware -- a real gap in the shipping
story, and the natural next rung of the proving ground whenever the operator
wants it.

**CORRECTION TO MY OWN CONCERN, from the 5080's audit:** I flagged that those
four GGUF profiles might be asserting `shipping` while unrun. They are all
`draft`. The repo had already declined to make the claim, so there was nothing
to demote and my worry was unfounded -- recorded because a concern raised in
this log should be answered in it.

**The audit's real result is worth more than my question was.** Across all 38
`shipping` profiles: structural validation plus every named visual engine
resolved against the live registry returns ZERO broken. The tier split:

    vram_ceiling 14.5   ->  37 profiles   (the 5080's tier: the dev box)
    vram_ceiling  6.8   ->   1 profile    otr_nvidia_8gb_haunted

**Exactly one shipping profile targets hardware that is not the development
box, and it is the one this box's leg proved.** Every other shipping claim is
a claim about the machine it was written on. That is not dishonest -- nothing
overstates itself -- but it is the precise shape of the ship story: the pack is
broadly proven on a 16 GB 5080 and, as of 2026-08-29, at exactly one point on
8 GB consumer hardware. What evidence `shipping` ought to REQUIRE is an
operator question and has been put to him, not decided by either window.

## OPEN ACTION -- THE REPO IS CURRENT; THE REGISTRY IS NOT. SOMEONE MUST OWN THE PUBLISH.

Operator asked, 2026-08-29, whether anyone is making sure the repo AND the Comfy
registry carry the latest work. Checked both rather than answering from memory:

**REPO: DONE.** `HEAD == origin/v2.0-alpha`, zero uncommitted files. Every fix
and every log entry from both boxes is pushed.

**REGISTRY: NOT DONE, and it needs MORE than the flag clearing.** Two separate
blockers, and the second one is easy to miss:

1. `latest_version` = **alpha.8** (2026-08-25). alpha.9/.10/.11/.12 are all
   Flagged. Proven not to be our doing: alpha.8 shipped 814 files including 28
   `.ps1`/`.bat` installers and PASSED; alpha.12 shipped 715 files with ZERO
   scripts and FAILED, with identical trigger-pattern counts. Deleting 135
   script files changed nothing. Only unblock: **the operator asks Comfy-Org
   directly** -- their scan findings go to a private Discord that publishers
   cannot see.
2. **EVEN IF THE FLAG CLEARED, alpha.12 WOULD NOT FIX A COLD INSTALL.**
   `pyproject.toml` -- the ONLY dependency list the registry reads -- still
   declares neither `kokoro` nor `pyloudnorm` (verified by grep just now;
   `requirements.txt` carries them, and the registry does not read it). So a
   user installing an Active alpha.12 would still hit the kokoro cold-install
   crash on their first spoken line. **The fix requires a NEW version, not a
   promotion of an existing one.**

**RELEASE CHECKLIST for whoever does the bump, so it is not reconstructed:**
- Add `kokoro` and `pyloudnorm` to `[project] dependencies` in `pyproject.toml`
  (static literal list -- the registry does NOT evaluate setuptools' dynamic
  form; proven on alpha.3 vs alpha.4).
- Set a NEW version string. `2.0.0-alpha.12` is BURNED -- `(node_id, version)`
  is uniquely indexed server-side and version-delete is a SOFT delete that
  permanently consumes the string.
- Editing `pyproject.toml` AUTO-FIRES the publish workflow on push. That is the
  trigger, so make it one deliberate commit carrying BOTH declarations.
- After publishing, verify with
  `curl https://api.comfy.org/nodes/comfyui-old-time-radio/versions` that the
  dependency count is non-zero -- do not assume.
- Do NOT bump while a version is Pending, and do not bump merely to show motion:
  a bump into the flag wall publishes a version nobody can install.

**OWNER: the 5080 window** (it owns `pyproject.toml`, profiles and the
ship-facing install story). **GATE: the operator's Comfy-Org question.** Neither
window can clear the gate; both fixes above are worthless to users until it is.

## Step 12 -- the loudness fix proven end to end; shipping profile passes a SECOND time

Leg: `--profile otr_nvidia_8gb_haunted`, prompt `e9799331`, stock path (no
`--disable-dynamic-vram`), pyloudnorm 0.2.0 now present.

    RESULT SUCCESS + obs publish + mp4
    signal_lost_whispers_over_brittle_pages_20260829_113929_...final.mp4
    53.3 MB, 67.72 s, h264 + AAC
    wall: 2121 s (35.4 min)  -- third consecutive pass, and the fastest yet

**PBUG-20260829-04 is now CLOSED ON THIS BOX, proven by the artifact rather
than by the install succeeding:**

    [EpisodeAssembler] Final loudness master: measured -12.01 LUFS ->
    target -14.0 LUFS (gain -1.99 dB), true-peak ceiling -1.0 dBFS
    [peak-limited] (post-crossfade)

`legacy peak master` fallbacks this leg: **0**. So the LUFS path genuinely ran,
and this is the FIRST episode from MRKT that is loudness-comparable to any 5080
episode in the shared obs folder. The two earlier ones
(`the_ledgers_whisper`, `shadows_lengthening_on_the_heath`) remain peak-mastered
and are not comparable -- that gap is historical now, not ongoing. The
`pyproject` declaration is still owed and still rides the next deliberate bump.

**Shipping-profile tally on 8 GB hardware: 3 for 3** (haunted_local, then
`otr_nvidia_8gb_haunted` twice), all publishing, at 55 / 38.7 / 35.4 min.

**Also checked, and my DLL worry was unfounded:** `_import_llama_cpp()`
(`_otr_gguf_backend.py:939`) wraps preparation and import in `except
Exception`, not `except ImportError`, so the shared-library RuntimeError I hit
is caught and converted to a `GGUFNativeConfigError` that already names
`nvidia-cuda-runtime-cu12`/`nvidia-cublas-cu12`; and
`validate_gemma_gguf_ready()` ATTEMPTS the import rather than probing for the
module, so a broken binding reports `binding_available: False`. The uninstall
was still right -- leave the box as found -- but the availability probe would
not have lied.

## Step 13 -- CORRECTION: step 11 was WRONG. The GGUF binding WORKS on MRKT.

**Retracting my own finding, because the test that produced it was invalid.**
Step 11 concluded the GGUF lane was blocked on this box by a CUDA-major
mismatch. The CAUSE named there (no CUDA 12 runtime present) was real, but the
EVIDENCE was worthless: I tested with a bare `import llama_cpp`, which BYPASSES
`_prepare_windows_llama_dll_runtime()` -- the function that adds the DLL
directories and preloads the CUDA dependencies. A bare import fails even on a
fully working install. The 5080 hit the identical error the same way and nearly
declared its own working lane broken, which is how the flaw surfaced.

**Re-tested through OTR's REAL path (`_import_llama_cpp()`), after installing
the two pip packages the code expects:**

    pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12
    pip install --extra-index-url .../whl/cu124 llama-cpp-python

    [loaded _otr_gguf_backend]
    _import_llama_cpp() -> <class 'llama_cpp.llama.Llama'>
    RESULT: GGUF BINDING USABLE

**And the coexistence risk I declined to take is measured safe HERE, not just
on Blackwell.** Tested in the RISKY order -- llama_cpp resident FIRST, then
torch:

    torch 2.12.1+cu130 cuda 13.0
    cuda available after llama_cpp load: True
    real CUDA matmul after llama_cpp load: OK  (1.5147e+08)
    device: NVIDIA GeForce RTX 4060 Laptop GPU

So a CUDA-12 runtime (cudart64_12 / cublas64_12, from pip) and torch's CUDA 13
coexist in one process on Ada as well as Blackwell, and torch keeps driving the
GPU with llama.cpp resident. Verified BEFORE anything else that the render path
was unharmed: a clean torch CUDA matmul immediately after install. **No CUDA
toolkit install was ever required -- it was two pip packages.**

**THE ONLY REMAINING BLOCKER IS THE MODEL FILE -- a path problem plus a
download:**

    validate_gemma_gguf_ready() -> ok: False,
      model_path: C:\ComfyUI-Models\LLM\converted\gemma-4-12b-it\
                  gemma-4-12b-it-Q8_0.gguf
      model_exists: False, expected_size: 12,669,646,240

`C:\ComfyUI-Models` does not exist on MRKT (see the top of this log), so the
GGUF row resolves to a root this box has never had. It needs
`OTR_COMFYUI_MODELS_ROOT` pointed at the real tree, plus the weight itself.

**WHY THIS MATTERS BEYOND ONE WRITER:** every non-NVIDIA profile in the repo
(`otr_mac_mps`, `otr_amd16_rocm`, `otr_amd8_rocm`, `cpu_floor`) runs GGUF at
`quant_policy: "none"`, because bitsandbytes NF4 is CUDA-only. GGUF is not a
niche lane -- it is the entire Mac / AMD / CPU story, and its binding is now
known-good here. Also: GGUF quantizes the embedding table while NF4 leaves it
bf16, which is precisely the 1.88 GiB that put 12B out of reach in step 10.
That reopens the writer question on 8 GB entirely.

**Status: binding PROVEN, lane UNPROVEN.** No GGUF weight has been run on this
box. Which model and which quant is a product decision with quality
implications, not a bug fix, so it is surfaced rather than taken.

**Blast radius (per CLAUDE.md 0B): 4060 ONLY.** No shared code, no `nodes/`, no
profile, no workflow JSON was touched -- this is three pip installs in the MRKT
venv and one log entry. The 5080 is provably untouched because nothing left
this box's environment.

## THE FRICTIONLESS-INSTALL ANSWER: LOW and HIGH on 8 GB, every value measured

Operator's ask (2026-08-29): *"how can we get Claude to see how frictionless our
best frictionless setup low and high capabilities are on the 4060 and decide
what dropdowns those JSONs should have."* Friction is defined as six observable
numbers per candidate, none of them a judgement call. Answers below are from
this box's runs, not from argument. **Where a value is unproven it says
UNPROVEN.**

### LOW -- `otr_nvidia_8gb_haunted` -- CONFIRMED, with one asterisk

The incumbent's job was to be confirmed or beaten. It is **confirmed**: three
published episodes, and it is the only profile in the repo whose evidence comes
from hardware other than the machine it was written on.

| # | friction question | measured answer |
|---|---|---|
| 1 | GB before first render | **16.03 GB** (not the ~3.7 the label used to claim) |
| 2 | HF token required? | **No.** Every artifact fetched anonymously |
| 3 | auto-download or manual? | **MIXED** -- writer/voices/music auto; SD1.5 + motion module + adapter are manual placements |
| 4 | undeclared dependency? | **YES, two.** `ComfyUI-AnimateDiff-Evolved` (PBUG-09) and `pyloudnorm` (PBUG-04) |
| 5 | does it LOAD | **Yes** -- through the real path, three times |
| 6 | wall / peak VRAM | **55 / 38.7 / 35.4 min**, ~4.2 GB peak of 8.0 |

The 16.03 GB breaks down as 3.94 GB of explicit placements (SD1.5 1.99,
`v3_sd15_mm` 1.56, `v3_sd15_adapter` 0.10, `kokoro-v1_0` 0.30) plus HF-cache
pulls the old label omitted: **gemma-4-E2B-it 9.57**, musicgen-small 2.21,
Kokoro-82M 0.31.

**THE ASTERISK, and it is the honest limit of all three episodes:** this box was
hand-prepared. The node pack arrived by `git clone` at 03:00, not by any
documented step. The runs prove the LANE; they do not prove an INSTALL. That
gap closes only on a box that has never been touched -- the clean-room test,
which is parked with the operator.

**Defended dropdown values for LOW** -- every one carries a reason:

    role_overrides  announcer/music/character_visual : animatediff15_v3_haunted_video
                    -- text_to_video, so NO image model enters the beat path.
                       Verified: zero z_image/Lumina2 loads across a whole episode.
                    announcer/music/character_image  : z_image_turbo
                    -- DECLARED BUT NEVER INVOKED on this lane. Inert, kept only
                       for role completeness. UNPROVEN as an 8 GB image lane:
                       z_image never completed an episode here.
    slot_overrides  voice_bank kokoro_builtin | char+announcer kokoro
                    -- 3 episodes; needs the repo_id gate (PBUG-02) to exist at all.
                    music_engine musicgen -- stable_audio_3 ckpt is not on disk.
                    video_render_engine animatediff15_v3_haunted_video -- as above.
    llm             creative+technical google/gemma-4-E2B-it, quant_policy "none"
                    -- PASS/PASS on the fit gate at both ceilings; 3 published
                       episodes; ungated; anonymous. THE proven 8 GB writer.
                       quant_policy "none" also means _plan_max_memory returns
                       None, so the tag-table class of defect (PBUG-05/-07)
                       cannot reach this profile at all.
                    vram_ceiling_gb 6.8 -- the tier value; admits E2B, refuses 12B.
    render          512x288, fps 25, frame_budget 25
                    -- 25 is REQUIRED, not stylistic: mm-p_0.5 has a hard 32-frame
                       ceiling without a context window, and the v3 lane uses a
                       sliding context window to exceed it. 49 frames crashed.
    launch          NO --disable-dynamic-vram. Measured: stock DynamicVRAM is
                    ~30% FASTER here (2322 s vs 3303 s) and does not abort,
                    because this lane never loads the image UNET that triggered
                    PBUG-03.

### HIGH -- the best an 8 GB card can do, still frictionless -- **UNPROVEN**

HIGH's whole premise was a better writer in the same 8 GB. The candidate was
`unsloth/Qwen3-4B-Instruct-2507-GGUF`: 2.33 GiB against E2B's 9.57 GB on disk,
Apache-2.0, ungated, anonymous fetch in **46 s**, byte- and sha256-exact against
the pinned row. Its VRAM fit is **proven with headroom on this card**:

    n_ctx 8192 -> REFUSED  Free 6.94 GB < Needed 8.03 (weights 2.33 + kv 5.70)
    n_ctx 4096 -> ADMITTED Free 6.94 GB   Needed 5.23 (weights 2.33 + kv 2.80)
                 then llama.cpp init: n_ctx=4096, n_gpu_layers=-1  (FULL GPU)

Measured `kv_gb_per_1k`: 0.684 @ 4096 and 0.696 @ 8192 -- not perfectly linear,
so a small fixed term exists; pinned conservatively at 0.70.

**And then it hard-faulted:** `OSError [WinError -1073741795]` =
`STATUS_ILLEGAL_INSTRUCTION`, at `llama_init_from_model`, 2.58 s in. Not VRAM
(1.7 GB headroom), not the artifact (hash-exact). **HIGH therefore has no
proven writer today and I will not ship an unproven one as a default.**

Root cause is NOT settled and my first answer was wrong. I blamed the CPU's
lack of AVX-512; the 5080 lacks it too and works, so that is dead. The live
suspect is the WHEEL VERSION -- and the binaries genuinely differ:

    4060  llama_cpp_python 0.3.35  ggml-cuda.dll 819.27 MB  86555e1c0b39d826...  FAULTS
    5080  llama_cpp_python 0.3.33  ggml-cuda.dll 945.37 MB  715bf1e45e9ff80e...  WORKS

Identical wheel tags (`py3-none-win_amd64`), both from an index, all four DLLs
different. The decisive test -- install 0.3.33, verify by HASH not version
string, load through `_import_llama_cpp()` on a fresh process -- is queued
behind live legs. **Until it resolves, HIGH's writer row is UNPROVEN.**

### WHAT I REJECTED, and why

| rejected | why, measured |
|---|---|
| `gemma-4-12b-it` (transformers) | 11.9 GB; FAIL at 6.8. Floor is 7.46 GB (5.59 NF4 + 1.88 bf16 embeddings NF4 never touches) on an 8.0 GB card |
| `gemma-4-12b-it` **as the shipped default** | it IS the shipped graph default and it FAILS here -- PBUG-13, operator's call |
| `gemma-4-E4B-it` NF4 | refused by the same CPU-dispatch path as 12B before the -07 fix; unretested since |
| LTX video lane | never completed an episode on this card, two attempts |
| `z_image_turbo` stills | survived sampling once under the legacy loader; never completed an episode |
| `--disable-dynamic-vram` as a default | measured 30% SLOWER and unnecessary on this lane |
| `n_ctx` 8192 for GGUF | refused on 8 GB by the backend's own physical-free preflight |
| multi-act episodes | never run here; `--act-count 1` only |

### THE HONEST HEADLINE

**LOW ships today and is proven three times. HIGH does not exist yet** -- not
because 8 GB cannot host a better writer (the fit is proven with headroom) but
because the binding that would run it faults on this machine. One version test
stands between HIGH being real and HIGH being a plan, and it is queued.

### QUALIFICATION of the LOW writer recommendation -- "it renders" was the only thing I measured

**My friction table calls `google/gemma-4-E2B-it` "THE proven 8 GB writer" on
the strength of three published episodes. That endorsement is now qualified,
and the qualification came from an instrument I was not using.**

The 5080 surveyed **339 frozen ledgers** against the `_otr_ledger_clean` stage,
which asks a model per line "is every word of this something the character says
out loud?" The number that matters is not how often a model complains, but
whether it can say WHICH PART of the line it is complaining about:

    model                        eps   flag rate   unclean   whole-line share
    google/gemma-4-E2B-it         28      73%        59%      119/131 =  91%
    google/gemma-2-2b-it           3      42%        17%        1/20  =   5%
    mistralai/Mistral-Nemo       243      83%        23%     191/1643 =  12%
    google/gemma-4-12b-it         57      19%         1%       4/122  =   3%
    unsloth/gemma-4-12b-it-GGUF    7      24%         0%        1/14  =   7%

E2B quotes the ENTIRE LINE as the offending segment in 91% of its flags; every
other model does so 3-12% of the time. A whole-line quote is the judge asserting
that a line of dialogue is stage business in its entirety.

**The harm is in the repairs it COMMITS, not the ones it abandons.** An unclean
row fails safe and the original text reaches TTS untouched. A repaired row is
committed -- so a false positive that the repair believes it resolved rewrites
correct dialogue. Measured: 24 committed repairs across 28 episodes, i.e.
**roughly one damaged line per episode**. A real example from the 5080's leg 3:
*"The clasp is loose, little bird; show me how you keep it fastened."* became
*"Show me how you keep it fastened"* -- in an episode titled *The Loose Clasp*.
Another dropped an apostrophe (`You've` -> `Youve`), which is a TTS
pronunciation defect rather than a cosmetic one.

**WHY MY THREE EPISODES COULD NOT HAVE FOUND THIS, which is the lesson for this
log:** every leg I ran was scored pass/fail on `RESULT SUCCESS + obs_publish +
mp4 on disk`. All three passed. A silently rewritten line of dialogue passes
every one of those gates. **I was measuring whether the pipeline completes, not
whether the artifact is correct** -- and I would have gone on recommending E2B
indefinitely, because more legs of the same kind produce more of the same
evidence. This is the same failure shape as PBUG-04 (peak-vs-LUFS mastering):
the defect is invisible to the check that was being run, so running the check
harder never surfaces it.

**TWO CORRECTIONS I WOULD HAVE INHERITED, both caught on the other side before
they reached me,** recorded because either would have produced a plausible and
wrong fix:
1. It is NOT "the small model flags too much" -- Mistral-Nemo, a 12B, flags MORE
   (83% vs 73%). Flag volume is not the defect; failure to LOCALIZE is.
2. It is NOT a parameter-count law -- `gemma-2-2b-it` is SMALLER (2.6 GB vs
   3.0 GB) and sits at a 5% whole-line share, indistinguishable from the 12B
   rows. "Do not run the clean stage below N parameters" would have been the
   wrong rule.

**STANDING RECOMMENDATION, revised and narrower than what I wrote before:**
`gemma-4-E2B-it` remains the only writer with PUBLISHED EPISODES on this card
and remains what LOW ships today -- it is proven to RENDER. It is NOT
established as the best 8 GB writer, and on this evidence it is probably not.
`google/gemma-2-2b-it` is the leading candidate: 2.6 GB, PASS/PASS at both
ceilings, ungated, already cached here (4.89 GB on disk), and clean on the
localization measure -- but at 3 episodes of one bank it is under-sampled and I
will not promote it on that.

**TEST RUNNING:** prompt `81ad671f` -- `otr_nvidia_8gb_haunted`, `--act-count 1`,
source bank **shakespeare**, `google/gemma-2-2b-it` in both writer slots.
Shakespeare chosen deliberately: verbatim classical source text gives a
mislocalizing judge the most opportunities to be wrong, so it is the bank where
the difference between the two writers should be largest. **It will be reported
with the clean-stage instrument -- flag rate, unclean rate, whole-line share --
not merely pass/fail**, so one 8 GB leg is comparable to the 339.

### FALSIFIED: gemma-2-2b is NOT a replacement writer -- and my leg helped point the wrong way

My shakespeare leg (prompt 81ad671f, published) reported 1 flag, 2 segments
named, 0 committed repairs, and I called it "encouraging, not a property". The
5080 then ran the same model across FOUR banks and the property does not exist:

    whole-line share     gemma-2-2b      gemma-4-E2B-it
      media_archive        2/24 =   8%       93%
      original             2/6  =  33%       86%
      public_domain        4/4  = 100%       94%
      shakespeare          3/3  = 100%      100%

The 8% that made it look like a replacement was a MEDIA_ARCHIVE ARTIFACT.
Aggregate moved 8% -> 30% once the other banks landed, and on the fidelity
lanes it converges with E2B at 100%.

**It is WORSE than E2B, and the damage mode differs in kind.** On shakespeare
it flagged 3 and COMMITTED 3 (E2B commits 12.5% of its shakespeare flags and
fails safe on the rest). And it does not truncate, it SUBSTITUTES:

    before : 'A most excellent way to make an impression, my lady.'
    after  : 'I must truly master this...this...grace'
    before : 'Fancy that.'
    produced: "Maria's voice, I must truly master this grace"

The output bears no relation to the input -- a small model handed a rewrite
instruction invents a replacement line, which is then committed and SPOKEN.
That is worse than truncation because nothing downstream can catch it: the row
is well-formed, in character, and wrong.

**WHAT MY OWN LEG ACTUALLY CONTRIBUTED, stated plainly:** one episode, one
bank, one flag -- and the direction it suggested was wrong. I did label the
denominator before anyone challenged it, and that caveat is the only reason
this did not become a recommendation. It is not enough to be right about the
uncertainty if the number still gets quoted; **a single-leg result on a new
model should not be reported as a signal at all, only as a leg that ran.**

**STANDING WRITER ANSWER FOR 8 GB, unchanged and now better tested:**
google/gemma-4-E2B-it remains the proven renderer and still damages roughly a
line per episode. There is no drop-in small replacement. The 5080's conclusion
is the right one and my data does not dent it: the fix belongs in the REPAIR's
contract -- *may not alter anything outside the segment the judge named* --
because that single invariant blocks all three of gemma-2-2b's substitutions
and E2B's truncations alike. A better judge was never going to be the answer.

### THE FENCE DIVERGENCE IS STILL UNEXPLAINED -- and the stacks differ

Eliminated on BOTH boxes: quantization (quantized=False both sides), snapshot
revision (299a8560... identical), judge temperature (0.200), node, prompt
template, and now the BANK -- the 5080's shakespeare leg quoted
You speak of "excellent" as if you had no hand in the matter. (embedded double
quotes, the exact fence-inducing material) across 146 judge calls with ZERO
failures, against my 3 failures in 9 calls.

Remaining difference, and it is the fourth instance of tonight's pattern:
**the two installs are not running the same stack.**

    4060 (fences):  transformers 5.14.1  tokenizers 0.22.2  torch 2.12.1+cu130
    5080 (clean) :  torch 2.10.0+cu130   (transformers/tokenizers to be confirmed)

The snapshot's own generation_config.json declares
"transformers_version": "4.42.4" -- i.e. these weights were published against
a transformers a full major line older than what this box runs. Whatever the
cause, the parser fix stands on its own: accepting a fenced payload is correct
defensive behaviour regardless of who emits one.