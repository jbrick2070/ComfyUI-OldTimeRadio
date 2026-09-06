# 4060 DRILL LOG -- real box, real state (restarted 2026-08-29 ~02:00)

Written by the working session ON the 4060 laptop (MRKT). Earlier relayed
entries claiming "all models present" and pushed squawks were from a session
that never verified against this disk. This is a chronological lab log: later
dated corrections supersede earlier entries. Central tracking file per operator
order -- every step lands here and is pushed.

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
Blackwell-native, this card is Ada. **SUPERSEDED 2026-09-01:** the original
drill inference that this also ruled out LTX 2.5 / MiniMax H3 was not a physical
test of either current lane; use the correction below.

**2026-09-01 correction:** the NVFP4 sentence above describes the selected
Z-Image artifact, not H3's Qwen encoder. Comfy-Org explicitly documents the H3
NVFP4 encoder as usable without Blackwell. No canonical H3 episode was run in
this drill. The separate `vram-recipe-lab/eightgb_bench` did run physical-4060
H3 cells: a hash-bound 864x480x90 cold/warm/warm ladder at 7.21/6.79/6.79 GiB
VRAM, plus exported 864x480x124 Ref2VA A/V artifacts. That is isolated-clip
proof, not `otr_4060_h3_nano` `RESULT SUCCESS + obs_publish OK` proof; the full
OTR H3 episode remains unqualified on this card. Sixteen LTX 2.5 runs on the
5080 stayed near 15.47-15.60 GiB under reserve/clamp pressure, but that is not a
physical-8GB surrogate: GPU allocators can change behavior with real capacity.
The physical-4060 LTX 2.5 plan is staged but has no completed receipt, so its
status on this card remains UNKNOWN/unqualified.

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
## Step 14 -- 12B RUNS ON THE 8 GB CARD. Full 48-layer GPU offload, measured.

**Operator's question ("I want to try to get 4-12b on the 4060 to work",
"prove me wrong with an OOM but don't put an artificial gate") is ANSWERED:
YES, and with more headroom than anyone predicted.**

Artifact: `gemma-4-12b-it-Q4_K_M.gguf`, 7,121,861,440 bytes, sha256
`0a270ec9fe6b34f4a0d33992b6135117b484ebc4766ab76b51d4ae8c457e4c42` -- fetched
anonymously in 228 s, and independently confirmed by the 5080 against the HF
API's LFS oid as the CURRENT upstream blob. Binding: llama-cpp-python 0.3.33,
DLL hashes byte-identical to the 5080's working build.

Measured by loading through `Llama(model_path=...)` DIRECTLY -- bypassing
`GGUF_ROWS` because the pin is stale (PBUG-19) and would have rejected the
file on size before llama.cpp was ever reached:

    n_gpu_layers  n_ctx   load    generate   VRAM peak / 8188 MiB   result
        35        2048    4.5 s     3.7 s          6585            PASS
        48        2048    5.3 s     2.4 s          7841            PASS
        48        4096    5.3 s     2.8 s          7751            PASS

**ALL 48 LAYERS FIT.** llama.cpp's own line: `load_tensors: offloaded 48/49
layers to GPU`. The estimate going in was ~35 of 48 at 2048 and ~23 at 4096;
the truth is the whole model at 4096, which is why the operator's "no
artificial gate" instruction mattered -- an estimate-based refusal would have
stopped this at 8192 and never learned that 4096 with full offload is
comfortable.

**Doubling the context cost essentially nothing** (7841 -> 7751 MiB, i.e. within
noise and slightly LOWER). That is consistent with gemma-4's hybrid/sliding KV
cache -- its `generation_config.json` declares
`"cache_implementation": "hybrid"` -- so KV does not scale linearly with
n_ctx here the way it does for the Qwen row (0.70 GB per 1k, measured earlier).
**Do not extrapolate KV cost across model families.**

Generation is coherent 1950s radio narration and FASTER at full offload than at
35 layers (2.4 s vs 3.7 s), which is the expected sign that the 13 CPU-resident
layers were the bottleneck.

**Profile shipped: `config/profiles/otr_4060_12b_gguf_offload.json`** --
Q4_K_M, `gguf_n_ctx` 4096, `quant_policy: "none"`, default
`n_gpu_layers` (-1 = all). **No `OTR_GGUF_N_GPU_LAYERS` override is
needed**: the pack's existing default is already the correct setting for this
card, which is a better outcome than a tuned value because a fresh install gets
it without knowing anything.

**WHY THIS IS THE PORTABLE ANSWER, not just a 4060 one:** bitsandbytes NF4 is
CUDA-only, so `otr_mac_mps`, `otr_amd8_rocm`, `otr_amd16_rocm` and
`cpu_floor` can ONLY run GGUF. This is the same artifact and the same lane
they use. The operator asked for "preferably the one that AMD and Mac can also
use" and that is exactly what was tested.

**COST RECORDED, per the operator's own kill order:** the `scifi_news_pro`
re-run (prompt `40d5b422`) was killed at t=740 s to free the GPU. That
forfeits the first live test of the PBUG-16 `anchor_line_id` fix; the fix
itself is unaffected and still needs one leg to confirm.

**STILL UNPROVEN AND NOT CLAIMED:** no EPISODE has been rendered with this
writer. Load and generate are proven; a full canonical leg is not. The 12B is
a ~2x slower writer per token than E2B on this card and the render is long, so
that is the next leg, not a footnote.
## Step 15 -- the 12B EPISODE FAILED, and it qualifies my own step-14 claim

**Step 14 said "12B RUNS ON THE 8 GB CARD". That is true STANDALONE and NOT
true inside the pipeline. Recording the distinction because I stated the
stronger version.**

Leg: prompt `93d53b06`, profile `otr_4060_12b_gguf_offload` exactly as
committed (ceiling 6.8, Q4_K_M, n_ctx 4096), `--source-bank media_archive`,
on 219dab79 with the 5080's gate fix.

**THE GATE FIX WORKS -- confirmed live**, and this is the first thing to say
because it was the blocker I handed over:

    [Selector] proceeding with caution: ctx_cap=UNKNOWN@8192, vram_fit=WARN@9.4 GB

9.4 GB WARN, admitted, exactly as predicted. The old code priced it at 12.2 GB
and FAILED it. My committed ceiling of 6.8 needed no change.

**THEN THE LOAD FAILED, 27 TIMES:**

    [OTR_LineComposer] generate_fn raised: ValueError:
      Failed to load model from file: ...\gemma-4-12b-it-Q4_K_M.gguf
    successful GGUF loads in the whole leg: 0
    RESULT FAIL at t=640 s, node 80 (OTR_CastLock) raised ValueError

**The file is not the problem** -- the identical path loaded three times in my
standalone probe minutes earlier. **The difference is what else is resident.**
The probe ran on an idle card (~500 MiB used). In-pipeline the ComfyUI process
holds VRAM, and although the backend ran its eviction every time --
`[GGUFNative] Running pre-load VRAM eviction` then `[VRAMLevers]
free_otr_pipeline_residue ... OK: unload_llm, _unload_bark, gc.collect,
soft_empty_cache, cuda.empty_cache` -- the load still failed on all 27
attempts.

**THE HONEST ARITHMETIC:** the model needs 7,751 MiB of an 8,188 MiB card. That
leaves ~437 MiB. Torch's caching allocator reserves VRAM it does not return to
the driver, and ComfyUI's own residents sit in the same process. **A model that
needs 95% of the card cannot share a process with anything else, no matter how
well the eviction works.** Standalone success and in-process failure are both
correct results about different situations.

**WHAT IS PROVEN, precisely:**
  * gemma-4-12b-it Q4_K_M loads and generates on this 8 GB card in a CLEAN
    process -- 48/48 layers, 7,751 MiB, three configurations. Unchanged.
  * The current upstream artifact works (the 5080's re-pin evidence). Unchanged.
  * The gate fix admits it. Confirmed live.
  * **It does NOT currently produce an episode**, because the canonical pipeline
    is one process and the writer wants the whole card.

**WHAT WOULD MAKE IT WORK, not attempted and not mine to choose:** the writer
would have to run OUT OF PROCESS (a separate llama.cpp process the node talks
to), or the pipeline would have to fully release the CUDA context before the
writer loads rather than emptying a cache inside a process that keeps its
reservations. Both are architecture calls for the shipping surface.

**Still standing for 8 GB TODAY: `gemma-4-E2B-it`** -- four published
episodes, and the only writer that renders start to finish on this card.
## Step 16 -- LOW_VRAM did NOT fix it, and I reported "0 failures" too early

**Correcting myself before anything else: I told the operator "zero load
failures -- your flag worked" at t=100 s. The final count was 27, the same as
without the flag.** I sampled a counter 100 seconds into a 12-minute leg and
reported it as an outcome. The flag was real and it did change behaviour, but
my headline was measured before the thing it claimed to measure had happened.

Leg: prompt `8ea60a1c`, `--disable-dynamic-vram --lowvram` (both required --
`cli_args.py:170` says `--lowvram` "doesn't do anything if dynamic vram is
enabled", and `enables_dynamic_vram()` is true by default, so the flag alone
is a silent no-op). Boot confirmed `Set vram state to: LOW_VRAM`.

**WHAT LOW_VRAM DID FIX:** the FIRST load succeeded, and the 12B actually wrote
a script --

    [OTR_LedgerScriptWriter] DONE: episode_id=pending_20260829_204814,
      lines=6, words=59, est_minutes=1

That is the first script ever written by a 12B model on this 8 GB card. It also
logged the operator's directive working exactly as intended:

    [GGUFNative] VRAM estimate EXCEEDS free: free 6.94 GB < needed 9.53 GB
      (n_ctx=4096). PROCEEDING ANYWAY -- an OOM is the only authority here.

**WHAT IT DID NOT FIX:** every SUBSEQUENT load. `OTR_LineComposer` reloads the
writer per line and hit `ValueError: Failed to load model from file` 27 times.
The unfilled lines left the ledger structurally incomplete, and the freeze
cascade correctly stamped `needs_full_rerun`, so `OTR_CastLock` refused to
render:

    ValueError: OTR_CastLock: freeze cascade stamped
      freeze_verdict='needs_full_rerun' for structural ledger corruption.
      Refusing to cast/render.

**THE REAL SHAPE OF THE PROBLEM, now that two configurations have shown it:**
this is not "does the 12B fit" -- it fits, measured three ways at 7,751 MiB.
It is that **the pipeline loads and unloads the writer repeatedly**, and a
model needing 95% of the card can only win that race the first time. Once any
allocation lands between reloads, every later load fails. LOW_VRAM raised the
success rate from zero loads to one; it cannot make the repeated case work.

**CastLock's refusal is correct behaviour and should not be "fixed".** A ledger
with unfilled lines must not reach TTS. The defect is upstream of it.

**WHAT WOULD ACTUALLY WORK, unchanged from step 15 and now better evidenced:**
the writer must persist across calls instead of being reloaded per line, or run
out of process. Both are architecture calls on the shipping surface. A flag
cannot reach this.

**8 GB writer answer, unchanged: `gemma-4-E2B-it`,** four published episodes.
## Step 17 -- PBUG-11 reproduced in 16 SECONDS, in isolation

The operator suggested testing the GGUF lane on its own rather than through a
whole episode. That was the right call and it produced the cleanest artifact of
the night. `scripts/repro_pbug11_gguf_cache.py` calls OTR's own
`request_slot` TWICE for the same GGUF row -- no ComfyUI, no graph, no
episode, no competing VRAM consumer, one thread:

    CALL 1: OK in 12.30s   epoch 0 -> 1   LLM_CACHE.model_id = None
    CALL 2: OK in  4.01s   epoch 1 -> 2   LLM_CACHE.model_id = None
    both:   "[Selector] slot=creative GGUF load ... completed after this call
             was abandoned (cache epoch advanced) -- NOT adopting"

**The cache is None after two SUCCESSFUL loads.** The epoch increments once per
call because each load bumps the very counter it then checks itself against.
Call 2's 4.01 s is a real reload (a cache hit would be milliseconds); it is
faster than call 1 only because the OS file cache is warm.

**Why this beats another episode as evidence:** every alternative explanation
is eliminated BY CONSTRUCTION rather than by argument. No second model to
collide with, no pipeline stage, nothing that can time out, no concurrency, and
`n_gpu_layers=35` deliberately chosen because it is the known-good setting on
this card -- so an OOM cannot muddy the result. It also reproduces on the FIRST
call of a cold process, with an empty cache and epoch 0, which kills any fix
keyed on "has something been cached yet".

16 seconds and deterministic, against ~40 minutes for a render that could fail
for six other reasons. Handed to the 5080 so the third fix attempt can be
verified in seconds instead of by burning a leg.

**Scope, not overstated:** this reproduces the CACHING defect only. The memory
collision is a separate, downstream consequence and is not exercised here.
## Step 18 -- public_domain PASSES: third bank proven on this card

Leg 96b37d22, otr_nvidia_8gb_haunted, --source-bank public_domain,
--act-count 1. **RESULT SUCCESS + obs publish**:
signal_lost_the_thickening_dread_20260829_235812_...final.mp4, 48.4 MB.
Fifth episode from this box, staged to D:\4060-transfer.

**Bank coverage on 8 GB hardware is now three distinct banks**, which matters
because "3 for 3" earlier today turned out to be three legs of ONE bank family
and the first new bank crashed it (PBUG-16). Current honest tally for
otr_nvidia_8gb_haunted on this card:

    shakespeare       PASS  (olivias_unsettled_unease, gemma-2-2b writer)
    public_domain     PASS  (the_thickening_dread, E2B writer)
    media_archive     PASS  (earlier legs)
    scifi_news_pro    FAIL  x2 -- PBUG-16 (video stage, 72 min) then
                             PBUG-20 (news-read validator, 5.7 min)

So the profile is proven on three banks and fails on one, for two DIFFERENT
reasons, both now logged and both on the shipping surface. That is a more
useful sentence than "3 for 3" ever was.

**Transport, resolved to a single human action:** the direct cable is dead at
the 5080's end (0 bps, no 10.55.x bound there), so this box's 10.55.0.2 and
its successful pings were never reaching IDREAM -- that traffic was Wi-Fi.
Irrelevant, because Wi-Fi works: 445 is reachable both directions, and
4060-transfer is an exported share with Everyone/Full at
192.168.26.146. The 5080 read Test-Path returning False as "share does not
exist"; a bare 
et use PROMPTED for a username, which proves the path is
fine and SMB is refusing an unauthenticated session before consulting the ACL.
**Blocker is one credential prompt on the 5080, which only the operator can
answer.** Both windows declined to enter credentials or weaken the
guest-logon policy to avoid it.
## Step 19 -- the last failing bank publishes; 5 of 5 proven on 8 GB

`scifi_news_pro` against b2f36242. RESULT SUCCESS, 121 min wall clock.
`signal_lost_the_engineered_bloom_20260830_044129`, 131 MB, 155.28 s.

The number that mattered: **excess -0.03 s** (delta 24.57 s vs a 24.6 s
credits roll), against 18.93 s before. Under one frame at 25fps.

Two process notes worth keeping.

**I verified the fix was in the process before trusting the result.** Server
PID 24536 booted 04:03:07; b2f36242 landed 03:56:42. Three times earlier in
this drill I reported a success signal before confirming the thing it claimed,
so the boot-time check is now a precondition, not a courtesy.

**I did not call it fixed on the RESULT line.** SUCCESS + published is not the
same claim as "the overshoot is gone". I measured both streams with ffprobe
and found delta 24.57 s -- *larger* than the 18.93 s I was hoping to see go to
zero -- and only after pulling the declared credits length (24.6 s) did the
number resolve. Had I stopped at "PUBLISHED" I would have been right by luck;
had I stopped at "delta is 24.57 s" I would have reported a regression. The
tail had to be decomposed before either reading meant anything.

**Signal 1 is unanswered and I said so.** `foley_unpositioned=` never appears
in the log. Absence of a line is not a zero, and I reported it to the 5080 as
absence. That is the same error class as the one I made earlier in this drill
with `llama_perf_context_print`, where I read a missing log line as proof the
12B produced nothing -- it was gated behind an env var and the model had in
fact generated. Missing output means the emit site is silent, nothing more.

**Thermal note for anyone reading 8 GB timings:** beats ran ~20 s/step against
a ~9 s cold baseline. Multi-hour sessions on this card are throttled and their
timings are not comparable to cold-start ones. The 121 minutes is not a
per-episode cost figure.

Open on this box: PBUG-20 (news-read validator rejects real people named in
the source) is independent of this fix and still kills scifi_news_pro legs at
the writer. PBUG-11 (GGUF cache epoch) still blocks the 12B in-pipeline;
repro at `scripts/repro_pbug11_gguf_cache.py`, 16 s, deterministic.

## Step20 -- 2026-09-05 alpha.24 GUI test fails; workflow ownership moves to MRKT

Explicit alpha.24 Pending selection,4.59M ZIP, Manager Completed, Apply Changes
restart, console all25 nodes loaded. Manager retained a disabled old commit
card; subsequent read-only files confirm alpha.24 and loader/canonical hash
parity with development f727a5c4. No reinstall or manual repair.

Loaded shipped canonical from Extensions templates: LTX098-low16:9 x3,
ZImageTurbo x3, Kokoro/Kokoro, MusicGen. Only act_count3->1 for the requested
original run. One Run click22:03:28 PDT;12B failed44.94s (PBUG-20260905-01).
User later chose E2B and submitted another run; failed16.39s (-02).
These do not establish that every Gemma model fails.

Correction: Normal node mode is not the blend bypass. Blend93.bypass=true,
composite84.upscale_engine=off. SignalLostVideo's false widget is draw_scopes,
NOT a render bypass; procedural source rendering remains enabled. No modes
were changed. This does not explain writer loading failures.

Existing environment: cached Gemma, preexisting HF_TOKEN resolution and enabled
AnimateDiff/auto-messaging packs. Agent installed no extra packs, seeded no
cache, entered no token, used no SSH/backend API/install repair commands.
Not pristine cold-install or unauthenticated-gating proof. Application auto-swept
three empty stale pending directories; no agent cleanup, recovery unverified.
Original verdict FAIL stays unchanged: no success/obs_publish/episode receipt.
Private evidence148 events/56 captures plus6 follow-up captures stays in the
Codex output directory, not published with machine-private diagnostics.

User requested /kibitz and transferred workflow ownership to4060. Fresh isolated
source checkout D:\otr-4060-testing\ComfyUI-OldTimeRadio, branchv2.0-alpha,
baselinef727a5c4 (ownership ruling). No installed source edits. Nodes/scripts/
tests/pyproject/registry remain5080-owned. Both reviewer CLIs available.
Optional ComfyUI profile missing in installed older kibitz and repository;
doctor NOT READY for that missing optional file. Four generic prompts/script
are intact; running supplied fan-out script unchanged. Driver is Codex, not
Claude; report actual roster. Plan docs/4060-gemma-canonical-review-plan.md.

Peer substring-match theory is future fragility, not this12B cause: retry line
is explicit. llama_cpp is isolated to selected nativeGGUF path, not the NF4
load that failed. It is undeclared and must be checked before proposing GGUF
as a zero-hand-step alternative; no missing-binding failure was reproduced.

Current step: complete/ground four-round review before any canonical decision.
No source-code fix, registry publish, process restart or new render undertaken.

### Step20 addendum -- 22:20 PDT, user cancels Kibitz and questions cache state

User explicitly requested proceeding without Kibitz. Campaign stopped, not
completed: Codex gpt-5.5/high r1 review produced a substantive file; initial
Antigravity call rejected gemini-3.5-pro as unknown; retry with the installed
gemini-3.1-pro-high was canceled. Only the verified reviewer process tree was
stopped. ComfyUI/model processes were not stopped or restarted.

User reports deleting a models folder and suspects surviving links. Read-only
inspection finds the active installation's models directory and its immediate
subdirectories are ordinary directories, not reparse points. Captured run logs
resolve HF_HOME to that installation's models/huggingface, not evidence of an
empty cache. Gemma12B/E2B/E4B directories still exist there with August creation
timestamps. This does not identify which folder the user deleted or prove all
snapshot files are intact. More precise bounded cache inventory is in progress.

The user is considering deleting models/workflow/links for a fresh start.
No deletion, link repair, dependency change, cache seeding or new render has
been performed. Exact targets, shared-model impact and recovery must be
confirmed before any destructive reset. Existing logs and workflow source
remain preserved. A cache purge is not a demonstrated fix for either trace.

At22:21 PDT, bounded read-only inventory found115.114GiB (123602286958 bytes,
271 ordinary files) under the active install-local models tree, including
78695817400 bytes in its HF cache. The registered ComfyUI-Shared/models tree
contains32 empty directories and zero files. No reparse points were found
anywhere under either model tree. The captured run's HF cache is the populated
install-local tree; surviving ordinary cached files explain rapid loading
without establishing which separate folder the user previously deleted.

User now explicitly authorizes deleting all models. Latest scope is models
only: preserve saved workflows, development/installed code and evidence.
No files have yet been removed. Computer Use deletion requires action-time
confirmation of the now-resolved115.114GiB target. This is a destructive
clean-reset request, not an implemented loader fix or a passing portability run.

### Step21 -- 22:24 PDT, user-confirmed model purge completed

User confirmed deletion after the115.114GiB scope was stated. Computer Use
observed0 active ComfyUI jobs and closed the app. File Explorer opened the
exact install-local models directory; selected all31 child folders; Delete
reported: These31 items are too big to recycle. Do you want to permanently
delete them? The automation's Yes attempt returned window bounds changed,
then Explorer windows disappeared. User subsequently stated deleted. Do NOT
claim an unambiguously successful automation click: completion is user-reported
and independently verified by read-only filesystem inspection at22:24:07 PDT.

Active models directory now contains0 immediate items. The registered shared
models tree still contains32 empty category directories, no model files.
The instance-generated model-path YAML names only that shared root. ComfyUI
Desktop/backend processes were absent in the bounded process check. Model
files were permanently removed rather than recycled; no recovery copy made.
Workflows, installed/development code, outputs and prior evidence preserved.
This reset includes a user hand step and cannot change the original FAIL.
Next attempt must be separately identified as a fresh-model-cache run, not
a clean application/dependency installation or an unauthenticated gating test.

### Step21 correction -- 22:27 PDT, additional user HF cache discovered

Reopened existing instance via GUI after the two model stores verified0 files.
Startup automatically contactedHF for Kokoro voices before any Run click and
reported a separate user-level .cache/huggingface/hub location. That location
was omitted by the first bounded two-store inventory. It is NOT a stale
symlink: all six model trees and their parent directories are ordinary dirs.
Read-only totals at22:27:04 PDT:76 files,7408357092 bytes. Model caches:
1038lab/KokoroTTS40 bytes; hexgrad/Kokoro-82M341870448 bytes;
latent-consistency/lcm-lora-sdv1-5 134621596 bytes;
lllyasviel/control_v11p_sd15_normalbae1445158164 bytes;
stable-diffusion-v1-5/stable-diffusion-v1-5 5482652240 bytes;
suno/bark4054604 bytes. All directory creation dates predate startup; Kokoro
contents partly changed during startup, so its total is not all preexisting.
No Gemma directories were present in that user hub.

The fresh-cache run label is withdrawn pending completion of reset: no Run
was clicked, no workflow execution occurred. Restored editedE2B tab was NOT
run; opened Templates, but selecting the off-screen OTR extension returned
point outside window bounds. Closed ComfyUI again. No new install, model
repair, workflow edit, or token action. Additional shared-per-user cache
deletion needs exact-scope action-time confirmation. Original failure evidence
and the completed115.114GiB model removal remain valid, but ALL model caches
are not yet cleared. Reset GUI evidence:4060-model-reset.html in private Codex
output folder; startup/cache-discovery evidence kept separately.

### Step22 -- 22:30-22:34 PDT, remaining user cache removed; first-time scope clarified

User confirmed the six additional model-cache directory deletions at action
time. Explorer selected exactly those six directories, leaving .locks and
the three hub metadata files unselected. Standard Delete completed; settled
Explorer view contains four items and no models-- directories. Independent
read-only verification at22:32:42 PDT confirmed all six exact paths absent
and matching original-path Recycle Bin records present. These six folders
are recoverable through Recycle Bin; payload integrity was not checked.
The earlier115.114GiB permanent deletion remains a separate operation.

The same verification found29 files,15039463 bytes,10 directories and zero
reparse points in the OLD instance-local models tree after its earlier
startup. Therefore that tree is no longer empty. Shared models remains
zero files/bytes with32 empty category directories and no reparse points.
Do not claim a machine-wide zero-cache state or silently delete this new
material. The old instance is not the intended next test environment.
Bounded metadata inventory at22:33:18 PDT identifies28 Kokoro voice .pt files
under TTS (14655831 bytes) and one Xet log under huggingface (383632 bytes).
All creation/write times are after the22:26:07 startup. No contents read.

User clarified the purpose: a person who has never used ComfyUI should be
able to download it and start OTR with the least friction. Model deletion
alone does not test that. Next attempt uses Desktop New Instance rather
than reusing the old dependency environment, saved workflow or node packs.
Existing source, workflows, outputs and failed-run evidence remain intact.
The existing Desktop application and Windows user profile remain; this
is not a fresh OS, Desktop-installer test, or unauthenticated gating test.

GUI launched existing Comfy Desktop1.0.46 and clicked New Instance. Setup
default: ComfyUI(1), Standalone, Stable recommended v0.34.5, Python3.13.12,
NVIDIA detected/recommended,2256MB package, ~4.96GB required,583.08GB free.
No settings changed; Continue not clicked. Paused for the Computer Use
skill's action-time software-install confirmation. This confirmation is
automation-policy friction, not ComfyUI product friction. No new install,
render, loader fix, workflow edit, token action or extra-node install yet.

Private timestamped screenshot artifacts:4060-additional-cache-reset.html
and4060-first-time-setup.html alongside the prior private evidence. These
artifact writes and read-only verification are documentation/diagnostic
operations, not hidden installation steps. Prior FAIL verdict unchanged.

### Step22 correction -- 22:39 PDT, unnecessary new-instance proposal withdrawn

User points out ComfyUI is already installed and asks whether uninstalling
and reinstalling is necessary. Agent had overinterpreted first-time-user
experience as permission/need to create a second ComfyUI environment.
That proposal is withdrawn. No Continue/install action was submitted.
GUI clicked Back to Dashboard and a settled screenshot confirms only the
existing local ComfyUI v0.34.5 instance remains. No new environment created,
no uninstall, no render and no further deletion in this correction.

Keep existing ComfyUI for the OTR fresh-model startup test. Do not describe
that as a clean dependency installation or a demonstrated Gemma-loader fix.
The extra-instance setup detour and associated confirmation were agent/
automation overhead, not required beginner-guide steps. UI capture first
showed an occluding non-target window; no click was made from that image.
Target Comfy window was activated and reobserved before cancellation.
Timestamped screenshot evidence:4060-setup-correction.html, private output.

### Step23 -- 22:40-22:43 PDT, existing-instance one-act test submitted

User explicitly requests launch existing ComfyUI, open workflow and run;
reiterates1 act. GUI launched existing instance, not New Instance. Startup
22:40:13.031; console at22:40:26.674 confirms all25 OTR nodes loaded. Restored
editedE2B tab was not run. Templates > scroll left categories twice >
EXTENSIONS > comfyui-old-time-radio > otr_canonical loaded shipped template.
Both writer roles visibly returned to Gemma4-12b-it. Two decrement clicks
changed act_count3 to2 to1; settled screenshot confirms1, batch count1.
No other workflow value or node mode changed.

Run clicked exactly once22:41:32.554 PDT. Console got prompt22:41:32.640;
writer start22:41:32.787 explicitly act_count=1. Canonical validator OK:
23 nodes,61 links,widget_vector_drift=0. Source roll chose Folger Shakespeare;
its noncommercial-use warning is preserved verbatim in private evidence.
Only a pending skeleton ledger (0 lines/words) exists at this checkpoint;
not an episode-success artifact.

At22:41:32.909 OTR announced automatic Gemma4-12b-it download23.9GB. GUI
file counter reached7/8; that is not88 percent of bytes. HF warned this
cache does not support symlinks and caching still works with possible
additional disk use. No Developer Mode, administrator launch, environment
change, manual download, extra pack installation or other workaround.
Existing AnimateDiff startup missing-motion-model warning is environmental,
not evidence canonical requires it. Existing extra packs and authentication
state remain: do not claim a pristine dependency install or no-auth test.

GUI LOGS bottom panel renders console pixels without accessibility text.
Read-only current comfyui.log snapshot supplemented timestamps; then Open
logs in a new window exposed full console text directly through GUI. No
further filesystem polling needed. This diagnostic read and artifact writes
are disclosed evidence collection, not hidden installation/run commands.
At22:43 checkpoint run IN PROGRESS downloading; no fatal error, OOM,401,
RESULT SUCCESS or obs_publish OK observed. No rerun submitted.
Private evidence:4060-post-reset-run-part1.html (40 timestamped events).

### Step23 result -- 22:48-22:50 PDT, FAIL WITH FINDINGS; user takes over

Automatic Gemma download completed8/8 files after06:16. Persisted log records
completion22:47:49.447, model loading22:47:49.450, NF4 enable22:47:49.891 and
the CPU/disk-placement refusal plus automatic FP32-CPU-offload retry at
22:47:54.119. GUI then shows Loading weights0/677 and:
Windows fatal exception: access violation.

Captured stack begins torch/storage.py471 __getitem__, Transformers
core_model_loading.py1215 _materialize_copy,1239 _job,955 materialize_tensors,
990 convert,1695 convert_and_load_state_dict_in_model; from_pretrained
reaches OTR nodes/_otr_model_loader.py1020 load_llm. Complete native console
accessibility elements preserve the rest through threading._bootstrap.
The document_text field was capped at20000characters; the full tree was
captured, not silently treated as a complete truncated document.

Fatal exception first observed22:48:36 PDT. Desktop subsequently confirms
ComfyUI exited unexpectedly, exit3221225477 /0xC0000005. Its generic native-
library explanation is application wording, NOT a verified root diagnosis.
No explicit CUDA OOM or writer401 observed. No restart/rerun/tuning/fix.
Only the user's original one-act Run was submitted. No RESULT SUCCESS or
obs_publish OK; acceptance FAIL. This is a new native crash boundary, not
proof the earlier meta-tensor exception was repaired by deleting caches.

Final read-only receipt22:49:27-22:49:39: comfyui.log99lines ends at the
22:47:54.119 retry and omits fatal native stderr; GUI evidence is essential.
Exact pending_20260905_224132 episode directory contains only audio/
pending_20260905_224132_ledger.json7451bytes, written22:41:32.8646833.
No final episode or media file anywhere inside that exact directory;
zero reparse points/read errors. No broad output scan or weight-content read.

Waiting friction: GUI remaining-file counter offered no useful byte progress
or ETA for roughly six minutes. Considered manually inspecting cache growth
but did not. Recorded bounded waits and screenshots; no recurring automation
created. All initial warnings and automatic download messages are retained.

One optional final log-window activation returned foreground window did not
report a process id. User then took over for a fresh install; agent stopped
all GUI input immediately, with no recovery/restart attempt. Evidence saved
outside the install folder:4060-post-reset-run-part2.html and
4060-post-reset-crash.txt. Parts1+2 contain61 timestamped events; original
failed run and model-reset logs are preserved. Next installation is a new,
user-performed setup, not an agent-completed reinstall or passing result.

### Step24 -- 22:51-23:03 PDT, user-return failure and authorized loader repair

User returns control and explicitly asks4060 to fix, log and diagnose. The
original GUI-only qualification remains FAIL; hand repair is a separate phase.
Same-instance startup22:51:46.831; all25nodes22:51:56.104. User prompt
22:52:07.194: Gemma4-12b-it bothslots, act_count1, media_archive_rss.
Retry22:52:10.644; failure22:52:50.997:
load_llm failed for model_id='google/gemma-4-12b-it': Tensor.item() cannot be called on meta tensors
Prompt executed in43.82seconds follows the exception; not SUCCESS. GUI0active,
Failed, View details clicked once; full error/screenshots preserved privately
in4060-user-return-diagnosis.html. No agent Run before the repair.

Read-only audit22:55: loader/canonical match alpha.24 and development baseline;
same instance/Python, old dependency metadata timestamps. Fresh reinstall NOT
established. Stack: Accelerate hooks.py471 state_dict -> bnb modules.py606 ->
functional.py565 nested_offset=self.offset.item(). No explicit OOM or401.
No valid episode reported. Source/config/dependency/log reads were manual
diagnostics, not part of a mouse-only install claim.

Fetched branch: no incoming commits. Pull --rebase refused, verbatim:
error: cannot pull with rebase: You have unstaged changes.
error: Please commit or stash them.
Preserved dirty logs, no stash/reset/commit/push. No remoteGPU/Kibitz.

Candidate correction: NF4 refusal retry only creates an empty skeleton from
the same config/class, ties weights, infers conservative unquantized-size
placement, and passes a concrete map BEFORE bnb conversion. CPU layers stay
ordinary at load dtype, not guaranteedFP32. Disk/allCPU plans fail explicitly.
No double-quant toggle, guard deletion, model swap, dependency/cache change,
workflow widget edit or additional automatic retry. Successful first-load and
fullGPU call paths unchanged in executed tests, not physical5080measurement.

Six stdlib contract tests PASS. Installed-runtime offline Gemma config/meta
probe succeeds: embedding/tiedhead+10decoderlayersGPU, remainderCPU; noCUDA
initialization, weights, downloads or generation. Installed HF/bnb tiny-meta
check PASS:7GPU-planned NF4linears and7CPU-planned ordinarylinears. Completed
independent diff review: no blocking findings. Fullsuite attempted but cannot
run: No module named pytest. Configured BugBiblecheckout absent. No dependency
install, shipping approval or claim of full regression verification.

23:00:07 HAND REPAIR: targeted installed-loader patch matches devcandidate
SHA25691D74E2C2BCA6D237FCCF2E71ACE6C69A1BBC56CED2141463E2FCCCB4B805B47.
Exact private4060-loader-candidate.patch preserves the reversible diff;
original tracked source is recoverable from baseline. Runtime still unproven.
GUIrestart: Menu > OpenDashboard > existingComfyUI Moreactions > Stop.
Generic unsaved-work warning led to Cancel; Ctrl+S had no observable effect,
so workflow Graphmenu > Save preserved existing1actvalues and removed dirty
marker. ReturnedDashboard > Moreactions > Stop > confirmStop. User-input
interruption/occludingCodex was handled with fresh observation, no blind
click retry or Codex input. Full click timestamps/screenshots in private log.

### Step25 -- 23:03-23:07 PDT, patched diagnostic act progresses past loader

Restart console23:03:13.492 confirms all25nodes. Startup also logs
ConnectionResetError: [WinError 10054] An existing connection was forcibly closed by the remote host
in asyncio _ProactorBasePipeTransport._call_connection_lost; server proceeds.
Preserved, not attributed to OTR. Existing extra-pack warnings remain.

Agent Run exactly once23:04:44.645 (server gotprompt23:04:44.736), after
GUI verifies act_count1, Gemma12Bbothslots, batch1,0active. Restored user
workflow saved without widget changes; zoom/minimap navigation only.
Zoom UIA set_value failed verbatim: read UIA value read-only state: Requested property was not in the CacheRequest (0x80070057)
Keyboard entry then minimap drag succeeded. This is automation overhead.

Validator23nodes61links, widget_vector_drift0. Cached model, no modeldownload
this attempt. ExplicitCPUoffloadretry23:04:50.447; weights677/677 complete
23:04:55.486 in4seconds. Subsequent meta-device warning describes ordinary
CPU-offloaded parameters; it is not the earlier nested quant-state exception.
CUDA warmup completes20.4seconds. Live generation: newsranking14tokens in
35.1seconds at0.4tok/s, then cached-model bodyrerank2tokens in5.4seconds.
This is physical evidence of load AND generation past PBUG-20260905-01.
It is not a full episode or performance qualification.

23:05:57.286 Writer starts1act on RSS/Scifi route; selected source body9678
characters. New skeleton pending_20260905_230557_ledger.json,0lines/0words.
23:05:58.021 scifi_news_pro_dossier attempt1/3 starts and reuses modelcache.
No OOM401, no secondRun, no active-run edits. Acceptance remains INPROGRESS;
no RESULTSUCCESS/obs_publish/final episode verified. Evidence through68events
saved4060-diagnostic-repair-part1.html andpart2.html outside installation.
Later monitoring follows separately; user-return log preserves first6events.

### Step26 -- 23:08-23:10 PDT, workflow audit and continued writer activity

Read-only comparison completed 23:08:40: the saved user workflow differs
behaviorally from installed and development canonicals only in act_count,
3 to 1. max_new_tokens_cap is 200 in all three; both Gemma12B selections,
procgen mode and other widget values are unchanged. All 23 node identities,
types and modes and all 61 named connections match. Frontend Save reordered
input arrays on nodes 1,7,85,86 and consistently adjusted eight numeric
target-slot indices; no named connection was changed. Layout/viewport and
frontend state were excluded. This checks saved files, not the queued payload.

Passive console capture 23:10:08 shows writer heartbeat 64 tokens, 0.4 tok/s,
164.0 seconds during scifi_news_pro_dossier attempt 1/3. This is actual writer
output, not only warmup/news ranking. Same act remains active; no additional
Run, widget change, cancellation or patch. Full result still unverified.

Asked user whether to continue the same-act checks every five minutes as a
thread monitor or remain in this turn. No answer/automation at this checkpoint;
active-turn observation continues. Slow CPU offload is recorded as friction,
not called a hang. No duration-based timeout or tuning introduced.

23:12:14 console capture reaches 128 tokens, 0.4 tok/s, 327.6 seconds.
Private 4060-diagnostic-repair-part3.html saves events 68-74 inclusive;
checkpoints together preserve events 0-74 without replacing earlier files.
No download was observed in this cached-model diagnostic attempt.

Duration/UI finding: max_new_tokens_cap=200 is the legacy per-line composer
ceiling, not a global ceiling on the scifi_news_pro custom runner. The dossier
explicitly requests 700 tokens per attempt (_otr_scifi_news_pro.py:265,1886).
At the observed 0.4 tok/s, a full 700-token decode would take about 29 minutes
for that one call, excluding overhead; EOS may occur earlier and rates vary.
This is not an episode ETA, and no token cap/route was modified. A read-only
search initially used incorrect lowercase writer filename and reported
The system cannot find the file specified. (os error 2); corrected via rg.

Read-only stage audit: dossier is up to three attempts per source window;
pitch, treatment, cast aliases and newsread follow with up to three attempts
each, whole-play script up to four, then voice casting up to three. These
later calls use remaining configured context capacity, not the legacy 200
widget. Retries are conditional, not required calls. Shared-tail reflection,
summary and data-dependent ledger judging/repair also remain. A multi-hour
writer phase is plausible at the current rate but no duration is established.
No optimizer, model substitution, token-limit change or cancellation applied.

Source pointers: OTR_LedgerScriptWriter.py:2257,3417; _otr_scifi_news_pro.py:
1576,1868,1915,2913,3615,4704; _otr_structured_call.py:89;
_otr_story_brief.py:570,837; _otr_ledger_clean.py:620,1580.
Post-edit git diff --check passes. Changed code remains loader plus the six
focused regression tests; documentation changes are preserved and uncommitted.

23:14:23 PDT checkpoint: dossier attempt 1/3 has reached 192 tokens,
0.4 tok/s, 491.0 seconds, with no new error. Same single act remains active.
No RESULT SUCCESS, obs_publish OK or final episode is verified. Background
checks have not been enabled: five-minute cadence confirmation is pending.
Stopping this chat turn would not stop the render, but would end active-turn
observation; do not describe an unscheduled watcher as running. Preserve the
current act and inspect its continuation before any subsequent run action.

### Step27 -- 23:20-23:22 PDT, overnight monitoring enabled

User explicitly approved five-minute checks while sleeping and autonomous
overnight continuation. Subsequent user authorization adds other plausible
8 GB model trials ONLY IF this current act fully succeeds. Native same-task
heartbeat 4060-overnight-otr-trials was created ACTIVE at 23:20:26.010 PDT,
then updated with exact-trial success matching. App tool confirms creation,
view and update; persisted configuration verifies five-minute cadence and
this task as target. No standalone task, alternate model or external handoff.
Earlier statements that monitoring was pending are historical checkpoints,
superseded by this entry. No first scheduled execution is claimed yet.

Active identity: GUI Run 2026-09-05 23:04:44.645 PDT, server acceptance
23:04:44.736; writer start 23:05:57.286; pending_20260905_230557. The later
pending timestamp reflects prior warmup/news ranking, not a second Run.
Do not confuse this with pending_20260905_224132 (native crash) or the
22:52 user-run meta-tensor failure. Completion must belong to this exact
trial: RESULT SUCCESS, obs_publish OK, and the corresponding final episode
on disk, following an explicit episode rename if recorded. Historical
success entries and a skeleton ledger are not completion evidence.

23:20 initial capture showed 256 tokens/649.6s at 0.4 tok/s; screenshot was
occluded by Codex, while accessibility belonged to the Comfy console. One
Comfy Logs activation restored a useful image. Refreshed 23:21:09 capture
confirms 320 tokens/810.4s at 0.4 tok/s, dossier attempt 1/3. No new error,
download, Run, model switch, cancellation or installed-code edit. Healthy
slow generation is left alone; unchanged output alone never triggers retry.

Monitoring rules: save timestamped screenshots/full accessibility states,
append compact drill entries, record clicks, download sizes or unknown size,
waits, exact errors and hand-step friction. Report meaningful changes only.
Original GUI-only result remains FAIL; patched results remain separately
labeled. On OOM, writer401 or a required extra-pack request: preserve failure,
stop active testing and diagnose safely; no tuning, credentials or installs.
If baseline fails, conditional other-model campaign does not start. After
failure, source-grounded findings and recoverable candidate tests/fixes may
be prepared in the development checkout under existing fix authority, then
pause the monitor once useful authorized work is exhausted.

After full baseline success, choose a finite prioritized UI-offered model
checklist, beginning with alternative writer models plausible on 8 GB. Use
separate named workflow copies, one act per trial, serially. Preserve the
baseline, change only the selected model, keep downloads app-managed, check
free disk capacity before large downloads, and record all setup friction.
No cache purge/seeding, new node packs, authentication change, broad upgrades,
publishing or external messages. Pause after the feasible checklist and
morning evidence summary, rather than launching endless repeats.

Evidence continuity: part4 preserves events75-79 inclusive and exists on
disk (536330bytes). New overnight-setup part5 begins event80. One diagnostic
read attempted CODEX_HOME, which was unset: Join-Path reported Cannot bind
argument to parameter 'Path' because it is null. Test-Path then reported a
null path. The resulting 'No automations directory exists' message was not
valid evidence; explicit C:\Users\jeffr\.codex\automations verification
was used instead. Creation subsequently produced the verified config there.
These are scheduling diagnostics, not install/run steps. No OS power or
security settings changed; local execution needs the host on and app running,
and GUI interaction needs an unlocked desktop.

Part5 saved and verified at 23:22:18 PDT: 764898bytes, events80-83 inclusive.
All events0-83 now have durable evidence checkpoints. git diff --check passes.
Automation remains ACTIVE on this task; no first scheduled execution or
future success is asserted by this setup receipt.

### Step28 -- 23:27-23:30 PDT, canonical repository synchronization

User directs that patches live in the canonical repository and be uploaded
to Git, explicitly retaining4060 ownership of canonical changes. User also
raises5080 regression risk and requests the term separate JSON for any future
machine-specific workflow. This loader correction needs no workflow-value,
node-link or schema change; the canonical JSON remains untouched. Future
hardware-specific settings will be considered as a separate JSON only when
evidence establishes the need, not as a substitute for repairing shared code.

Verified local checkout origin jbrick2070/ComfyUI-OldTimeRadio, branch
v2.0-alpha, baseline f727a5c4. Fresh fetch found no incoming commits. Installed
loader and source loader remain byte-identical, SHA256
91D74E2C2BCA6D237FCCF2E71ACE6C69A1BBC56CED2141463E2FCCCB4B805B47.
No additional installed-code edit, restart, model call, queue action or
dependency change. This phase only prepares source/test/documentation Git work.

Before/after numeric proof executes the real memory-planning and device-map
selection AST from baseline and candidate for Gemma12B NF4:
- 8.00GiB: both GPU6.8GiB/CPU32GiB, initial device_map auto.
- 15.99GiB: both GPU13.5GiB/CPU32GiB, device_map {"":0}, allGPU.
Those extracted ASTs are identical. Private check-branch-parity.py contains
the reproducible comparison; driver read and reran it successfully. No model
library imports or GPU calls. This is code-path parity, NOT a physical5080
performance measurement. The changed retry is conditional on the specific
NF4 CPU-dispatch refusal, not hardcoded to a4060; any device reaching that
same refusal gets the correction. Successful allGPU loading bypasses it.

Added a permanent regression for both hardware-selection cases to
tests/test_nf4_explicit_cpu_offload.py. All seven focused tests PASS in0.006s;
the six original tests remain intact. Loader itself is unchanged since the
live diagnostic started. Prior independent finished-diff review was clean.
AST parsing, nonempty/noBOM checks and git diff --check pass. No token-pattern
matches in the source/documentation diff. Fullpytest and BugBible remain
unavailable; full episode and physical5080 qualification remain unverified.

Prepare one candidate commit containing only loader, focused tests, drill log
and appended production findings. Preserve the unrelated untracked historical
review-plan document. Private HTML/screenshots stay outside Git. Do not change
pyproject.toml, registry version, tags or release settings: this is a source
branch push, not a registry release or a clean-install PASS declaration.

### Step29 -- 23:37-23:40 PDT, first scheduled overnight check

Heartbeat4060-overnight-otr-trials first observed execution began at
23:37:51.646 PDT. Read latest drill/bug entries and private push receipt;
fresh fetch/pull reports already up to date. Source commit c61ac222 is
confirmed pushed: localHEAD, origin/v2.0-alpha and remote branch matched
at23:31:21. Seven focused tests passed after that push. It included loader,
tests and both logs, not the canonical JSON, pyproject or private screenshots.
This closes Step28's prepared-push action; registry/episode qualification
remains unverified. Untracked historical review-plan document is preserved.

Same active identity: single GUI Run23:04:44.645, writer pending_20260905_230557.
Persisted log shows first dossier call ending with427tokens/1095.5s at
23:24:13.607, followed by another dossier attempt1/3 at23:24:13.613. During
this check the second call reaches320tokens/846.3s at23:38:19.950, then ends
with368tokens/971.6s at23:40:25.303. Another dossier attempt1/3 begins at
23:40:25.316 and reuses the same model cache. These are successive internal
calls within the same act, not agent resubmissions. Rate stays around0.4tok/s.
No new error, OOM,401 or download. No RESULT SUCCESS/obs_publish/final episode
verified; conditional other-model trials have not started.

UI friction: initial Comfy Logs screenshot was occluded by Codex and its
inactive accessibility tree lagged the persisted log. One Comfy Logs window
activation restored its visible console; a second observation resolved the
one-frame accessibility lag. Full trees and images are preserved rather than
treating capped document_text as the complete log. Read-only comfyui.log tail
provided exact message timestamps; this is disclosed diagnostic collection.
No Run click, widget/model change, cancellation, restart or installed edit.
No manual wait/sleep was inserted; the scheduled interval and check duration
are timestamped. Healthy slow generation remains untouched.

Private evidence4060-overnight-20260905-233751.html contains5 event records,
including3 screenshot/full-tree captures. This separate checkpoint follows
diagnostic-repair parts1-5 without overwriting them. Only documentation is
updated for this heartbeat; source candidate and active runtime remain fixed.

### Step30 -- 23:48-23:50 PDT, third dossier call continues

Scheduled wake23:48:51.833; previous check verification23:43:09. Same single
Run23:04:44.645 and pending_20260905_230557. Third dossier call remains
attempt1/3, with heartbeats64tokens/161.0s at23:43:06.352,128/310.4s at
23:45:35.783 and192/460.3s at23:48:05.594; rate remains0.4tok/s. No new
error, OOM,401, download or completion. No RESULT SUCCESS, obs_publish OK
or final episode verified. Other-model campaign remains conditional and idle.

One passive Comfy Logs screenshot/full-tree capture was already visible and
current; no activation, click, keypress or other UI input was needed. Exact
times supplemented from a read-only persisted-log tail. No manual sleep,
installed edit, dependency/cache/model change, restart, cancellation or Run.
Fresh fetch/pull reports already up to date. Updated only this drill entry;
no new source/JSON changes or defect diagnosis. Preserve quiet monitoring
while the same writer stage is healthy and unchanged.

Private4060-overnight-20260905-234851.html stores3 timestamped event records
and the screenshot/full accessibility tree, without replacing prior evidence.

### Step31 -- 23:56 PDT, third dossier call completes and fourth begins

Scheduled wake23:56:21.963. Same single Run23:04:44.645 and episode identity
pending_20260905_230557. Third dossier call reaches256tokens/611.1s at
23:50:36.485,320/763.2s at23:53:08.594 and ends366/870.1s at23:54:55.430.
A fourth internal dossier attempt1/3 begins23:54:55.434 using the same cache.
No new error, OOM,401, download or episode completion. No RESULT SUCCESS,
obs_publish OK or final episode verified; no other-model trial started.

One passive console screenshot/full accessibility tree capture agreed with
the persisted-log tail; no activation, click, keypress or queue action.
No manual sleep or active-run modification. Fresh fetch/pull was up to date.
Private4060-overnight-20260905-235621.html preserves3 event records and one
screenshot/full-tree capture. A bounded read-only source-window audit was
requested to determine expected dossier call count; repeated helper labels
alone are not evidence of a retry loop, and no intervention was made.

Bounded source audit completed at00:02 PDT on September6: the finite window
loop invokes a fresh _pass_dossier for every source window, resetting its
attempt counter. The fourth attempt1/3 is consistent with the fourth source
window, not retry/thrash evidence. Each digest caps at3600 characters,
including repeated framing; body windows overlap239 characters and may
rewind their cut to a sentence boundary, making stride variable. The pending
ledger records9678 body characters but not the full source/framing or window
coordinates. Exact total/remaining calls cannot be determined from that
metadata; do not assume the fourth is last. Coverage receipts are assembled
after the loop. No runtime interaction, model imports/calls or source edits
were used for this audit; article/full-ledger content was not printed.

Audit-tool friction: one read-only rg call used invalid Windows wildcard
file arguments and returned os error123; directory plus -g corrected it,
without runtime effect. Final persisted-log check at00:02 PDT shows fourth
call heartbeats64tokens/154.4s at23:57:29.803 and128/313.6s at00:00:09.084,
still0.4tok/s, with no new failure. Private evidence file verified253243bytes.
No hand repair or tuning was attempted. Commit only this documentation;
preserve the unrelated historical review-plan document and all JSONs.

### Step32 -- 2026-09-06 00:08-00:10 PDT, dossier complete; pitch underway

Scheduled wake00:08:22.165. Same single GUI Run2026-09-05 23:04:44.645 and
pending_20260905_230557; no additional queue. Fourth dossier call finished
187tokens/459.6s at00:02:35.070. Pending ledger saved00:02:35.078, then
scifi_news_pro_pitch attempt1/3 began00:02:35.091 with creative-slot cache
reuse. Pitch heartbeats64tokens/149.6s at00:05:04.712 and128/297.3s at
00:07:32.383 remain0.4tok/s. No new error, OOM,401 or download. No matching
RESULT SUCCESS, obs_publish OK or final episode verified; other-model trials
remain conditional and have not started.

Bounded read-only pending-ledger metadata check now resolves Step31's unknown
window count: coverage_complete=true and news_seed_receipt_match=true;
four windows, each exactly one attempt, covering body ranges0-3227,
2988-6200,5961-8906,8667-9678. Body9678 characters; repeated header295;
overlap239; digest cap3600. All dossier windows finished, with no dossier
retries. This is a completed intermediate stage, not full writer/act success.

One passive visible Comfy Logs screenshot/full accessibility tree capture;
zero activation/click/key input. Exact timestamps supplemented from a
read-only persisted-log tail; metadata audit printed no article/full ledger.
No manual wait, hand repair/tuning, installed edit, JSON/model/dependency/cache
change, restart or cancellation. Fresh fetch/pull reports already up to date.
Previous Step31 documentation commit203a196 was pushed and local/tracking/
remote heads matched at00:02:53. Preserve the untracked historical review plan.

Private4060-overnight-20260906-000822.html preserves4 timestamped events and
one screenshot/full-tree capture, verified256474bytes. No checkpoint is
overwritten; documentation-only checkpoint follows, with runtime unchanged.

### Step33 -- 2026-09-06 00:16 PDT, pitch complete; treatment underway

Scheduled wake00:16:22.328. Same single GUI Run23:04:44.645 on September5,
pending_20260905_230557. Pitch finished174tokens/406.5s at00:09:21.634;
scifi_news_pro_treatment attempt1/3 began00:09:21.645 with creative-slot
cache reuse. Treatment heartbeats64tokens/148.7s at00:11:50.323,
128/292.0s at00:14:13.700 and192/435.8s at00:16:37.438 remain0.4tok/s.
No new error, OOM,401, download or verified episode completion. Matching
RESULT SUCCESS, obs_publish OK and final episode remain unverified. No
conditional other-model trial started; this is intermediate writer progress.

One passive visible console screenshot/full accessibility tree, supplemented
by a read-only persisted-log tail. No activation, click, keypress, queue,
manual wait, active installed edit, restart, cancellation or JSON/model change.
No hand repair/tuning needed or attempted. Fresh fetch/pull already up to date;
previous Step32 documentation push3af8154 was remote-verified. Preserve the
untracked historical review-plan document. No new source/JSON patch justified.

Private4060-overnight-20260906-001622.html preserves3 timestamped events,
one screenshot/full-tree capture, verified255434bytes. Previous checkpoints
remain intact. Documentation-only checkpoint; healthy slow run left untouched.

### Step34 -- 2026-09-06 00:23 PDT, treatment continues without intervention

Scheduled wake00:23:22.429. Same single GUI RunSeptember5 23:04:44.645 and
pending_20260905_230557. Treatment attempt1/3 continues:256tokens/588.2s
at00:19:09.909,320/740.0s at00:21:41.669, still0.4tok/s. No new error,
OOM,401, download, stage transition or episode completion. Matching RESULT
SUCCESS, obs_publish OK and final episode remain unverified. No new trial.

One passive visible console screenshot/full accessibility tree, with exact
times supplemented by read-only persisted-log inspection. Zero UI input;
no manual wait, installed/JSON/model change, restart, cancel or hand tuning.
Fresh fetch/pull already up to date; Step33 documentation push89c95b8 was
remote-verified. Preserve the untracked historical review plan. No source
fix is justified by healthy slow progress; stay quiet while stage unchanged.

Private4060-overnight-20260906-002322.html preserves3 timestamped events,
one screenshot/full-tree capture, verified253687bytes. Prior evidence intact.
This heartbeat changes documentation only; the existing act remains active.

### Step35 -- 2026-09-06 00:29 PDT, treatment and cast aliases complete

Scheduled wake00:29:52.535. Same single GUI RunSeptember5 23:04:44.645 and
pending_20260905_230557. Treatment reached384tokens/891.7s at00:24:13.424,
then finished397/923.6s at00:24:45.270. Cast-aliases attempt1/3 began
00:24:45.273, reached64/152.1s at00:27:17.353, and finished105/249.6s
at00:28:54.930. News-read attempt1/3 began00:28:54.938 with technical-slot
cache reuse. Rate remains0.4tok/s. No new error, OOM,401 or download.
No matching RESULT SUCCESS, obs_publish OK or final episode verified;
other-model trials remain conditional and idle. Intermediate stages only.

One passive visible console screenshot/full accessibility tree; read-only
persisted-log tail supplies exact timestamps. No activation/click/key input,
manual wait, new Run, installed edit, JSON/model/cache/dependency change,
restart/cancellation or hand tuning. Fresh fetch/pull already up to date.
Step34 documentation push90c6b9b was remote-verified. Historical untracked
review-plan remains untouched; no source/JSON change justified by this check.

Private4060-overnight-20260906-002952.html preserves3 timestamped events,
one screenshot/full-tree capture, verified249808bytes. Prior evidence intact.
Documentation-only checkpoint; active single act remains untouched.

### Step36 -- 2026-09-06 00:36 PDT, news-read complete; creative call underway

Scheduled wake00:36:52.612. Same GUI RunSeptember5 23:04:44.645 and episode
pending_20260905_230557; no additional queue. News-read reached64tokens/
150.2s at00:31:25.195, then finished78/182.9s at00:31:57.866. Creative-slot
cache reuse at00:31:57.876 was followed by64/148.4s at00:34:26.253 and
128/294.2s at00:36:52.055, still0.4tok/s. No explicit subsequent StructuredCall
stage label appeared. A bounded read-only source lookup was requested to
identify that call without runtime interaction or guessing from output text.
No new error, OOM,401 or download; no matching RESULT SUCCESS, obs_publish OK
or final episode verified. Conditional other-model trials remain idle.

One passive visible console screenshot/full accessibility capture at00:37:12.822;
read-only persisted-log inspection supplies exact timestamps. Zero UI input,
manual wait, new Run, installed edit, JSON/model/cache/dependency change,
restart/cancellation or hand tuning. Fresh fetch/pull already up to date.
Step35 documentation pushd1bc55d was remote-verified; historical untracked
review plan remains untouched. No active-run intervention is justified.

Source-only audit identifies the next call as high-confidence script-stage
inference: immediately after news_read the runner enters scifi_news_pro_script
and _pass_script with creative_fn, with no intervening model call. That markup
ladder invokes creative_fn directly rather than StructuredCall, explaining
the absent label. Generic heartbeats alone do not prove the exact live stage
or attempt number. Later pass_receipts/script and parse attempt_trace can
confirm; required receipt persistence follows downstream casting/assembly.
No runtime/model calls or source changes were used for this interpretation.

Private4060-overnight-20260906-003652.html preserves4 timestamped events,
one screenshot/full-tree capture, verified247391bytes. Prior evidence intact.
Documentation-only checkpoint; no source/JSON modification or new trial.

### Step37 -- 2026-09-06 00:44-00:46 PDT, run advances; GUI capture unavailable

Scheduled wake00:44:22.764. Same single GUI RunSeptember5 23:04:44.645 and
pending_20260905_230557. Persisted-log heartbeats for the same creative call
(script-stage inference from Step36):192tokens/443.3s at00:39:21.191,
256/590.5s at00:41:48.381 and320/737.1s at00:44:15.017, still0.4tok/s.
No new runtime error, OOM,401, download or completion. Matching RESULT SUCCESS,
obs_publish OK and final episode remain unverified; no other-model trial.

New evidence-capture friction: one passive console capture is black except
cursor, and its full accessibility tree lags at192tokens. Read-only window
inventory still returned Comfy Logs, workflow and dashboard, with no lock or
security surface. This does NOT prove the desktop is unlocked. One console
activation attempt failed verbatim: `failed to activate captured window`.
Refreshed returned-window selection and rehydrated the unique console; the
single bounded activation retry failed verbatim with the same message.
Stopped GUI input after those two failures. No click/key/Run action, restart,
cancellation, OS power/security setting change or lock-screen interaction.
Focus outcome is unverified. Root cause of black capture is undetermined;
do not relabel it a render failure or assume that Windows locked itself.

Read-only log checks remain available and confirm later progress than the
stale GUI tree. Subsequent checks may request passive captures, but must not
repeat activation while this visibility condition persists. Preserve missing
visual evidence honestly; do not infer completion from an inaccessible view.
No hand repair/tuning, installed edit, JSON/model/cache/dependency change or
manual sleep. Fresh fetch/pull already up to date; Step36 push5557945 was
remote-verified. Historical untracked review plan remains untouched.

Private4060-overnight-20260906-004422.html preserves8 timestamped events,
one black screenshot/full-tree capture and both activation errors, verified
104692bytes. Prior screenshots remain intact. Documentation-only checkpoint;
healthy slow runtime is not interrupted by this GUI-observation limitation.

### Step38 -- 2026-09-06 00:52 PDT, generation continues; console still obscured

Scheduled wake00:52:52.904. Same single GUI RunSeptember5 23:04:44.645 and
pending_20260905_230557. Read-only persisted log advances within the same
creative call (script-stage inference):384tokens/888.6s at00:46:46.517,
448/1039.2s at00:49:17.087 and512/1190.0s at00:51:47.889, still0.4tok/s.
No new runtime error, OOM,401, download, stage transition or completion.
Matching RESULT SUCCESS, obs_publish OK and final episode remain unverified;
no conditional other-model trial started.

One passive capture now shows a scenic background rather than black, but not
the console; full accessibility tree remains stale at192tokens. This is NOT
recovered visual access or proof of a locked/unlocked state. No activation
retry or input was attempted after Step37's exhausted bounded recovery.
Keep GUI input stopped while console visibility remains unavailable; preserve
the evidence limitation and continue authorized read-only log observation.
No security/power/lock-screen interaction, Run, installed/JSON/model/cache/
dependency change, manual wait or hand repair/tuning. Fresh fetch/pull already
up to date; Step37 pushf3f9b4e was remote-verified. Historical review plan intact.

Private4060-overnight-20260906-005252.html preserves3 timestamped events,
one background screenshot/full stale tree, verified260658bytes. Prior evidence
intact. Documentation-only checkpoint; runtime remains active and untouched.

### Step39 -- 2026-09-06 00:59 PDT, script call finished; voice casting begins

Scheduled wake00:59:53.006. Same single GUI RunSeptember5 23:04:44.645 and
pending_20260905_230557. Read-only persisted-log progress:576tokens/1339.7s
at00:54:17.653,640/1486.4s at00:56:44.276; script call finishes701/1625.4s
at00:59:03.273, around0.4tok/s. At00:59:03.274 the app reports verbatim:
`[scifi_news_pro] stripped a conversational wrapper: 1 line(s) before TITLE:, 0 after END. (the script itself is untouched)`
This is app-managed normalization, not an agent hand step. Casting-voices
attempt1/3 starts00:59:03.279 and technical-slot cache reuse follows
00:59:03.280. The observed transition moves beyond the previously inferred
script call; full writer/episode acceptance is still not established.

No new runtime error, OOM,401 or download. Matching RESULT SUCCESS,
obs_publish OK and final episode remain unverified; no other-model trial.
One passive screenshot remains background-only and full accessibility tree
stale; GUI input remains stopped, with no activation recovery repeated.
No Run, installed/JSON/model/cache/dependency change, restart/cancel, manual
wait or hand tuning. Fresh fetch/pull already up to date; Step38 push13f2a4c
was remote-verified. Preserve the historical untracked review-plan document.

Private4060-overnight-20260906-005953.html preserves3 timestamped events,
one background screenshot/full stale tree, verified260681bytes. Prior evidence
intact. Documentation-only checkpoint; existing single act left running.

### Step40 -- 2026-09-06 01:06 PDT, voice casting continues

Scheduled wake01:06:53.076. Same single GUI RunSeptember5 23:04:44.645 and
pending_20260905_230557. Read-only persisted log shows casting-voices
attempt1/3 progressing:64tokens/154.3s at01:01:37.641,128/306.4s at
01:04:09.725 and192/457.8s at01:06:41.048, still0.4tok/s. No new runtime
error, OOM,401, download, stage transition or completion. Matching RESULT
SUCCESS, obs_publish OK and final episode remain unverified; no new trial.

One passive screenshot is background-only; full accessibility tree remains
stale. No activation/input/security interaction or additional recovery attempt.
No Run, installed/JSON/model/cache/dependency change, restart/cancel, manual
wait or hand tuning. Read-only persisted log supplies actual current progress;
do not infer it from the stale tree. Fresh fetch/pull already up to date;
Step39 pusha6f197c was remote-verified. Historical untracked review plan intact.

Private4060-overnight-20260906-010653.html preserves3 timestamped events,
one background screenshot/full stale tree, verified260529bytes. Prior evidence
intact. Documentation-only checkpoint; healthy slow act remains untouched.

### Step41 -- 2026-09-06 01:13 PDT, fixed story saved; reflection reload succeeds

Scheduled wake01:13:53.167. Same single GUI RunSeptember5 23:04:44.645 and
pending_20260905_230557. Voice casting reached256tokens/611.7s at01:09:14.966,
then finished307/732.5s at01:11:15.764. App saved successive ledger states:
2lines/39words01:11:15.792;14/36401:11:15.806;18/45601:11:16.315 and
01:11:16.330. Delivery stamp16, delivery-differs0; cast requested2/locked2.
At01:11:16.330 the app reports fixed story complete, cast2/scenes1/
character_words325. Episode canon written01:11:16.333 with script title.
These are app-managed outputs, not manual changes or full episode acceptance.

At01:11:16.338 the app warns verbatim:
`You shouldn't move a model that is dispatched using accelerate hooks.`
Then run_story_brief_reflection attempt1/3 begins01:11:17.652, automatically
reloads the same Gemma12B from its existing snapshot, and enables NF4. The
usual CPU/disk-dispatch refusal at01:11:21.967 triggers the candidate's explicit
unquantized CPU-map retry. Full third-party refusal text is retained in the
private checkpoint. Weights677/677 complete01:11:26.675 in4s; at01:11:26.778:
`Some parameters are on the meta device because they were offloaded to the cpu.`
CUDA warmup completes01:11:45.178 in18.4s; reflection produces64tokens/147.9s
at01:14:13.202. Warning/refusal did not terminate this reload: subsequent
generation is observed. Do not conflate expected offload metadata with the
earlier fatal nested meta-tensor exception. No model download reported.

One passive screenshot remains background-only and full tree stale; no
activation/input or new recovery attempt. Read-only persisted-log collection
preserves exact warnings and timing; a bounded metadata/source audit was
requested for saved pass receipts and automatic reload context. No active
installed/JSON/model/cache/dependency edit, Run, restart/cancel, manual wait,
hand tuning or security/power interaction. Fresh fetch/pull already up to date;
Step40 pushd96ebde was remote-verified. Historical untracked review plan intact.
No OOM,401, requested extra pack, RESULT SUCCESS, obs_publish OK or final
episode verified. Original GUI-only FAIL stands; other-model trials remain idle.

Saved metadata confirms10 completed news-pro runner calls: dossier4, then
pitch/treatment/cast_aliases/news_read/script/casting1 each. Earlier ranking
and later reflection are excluded. Script attempt1 accepted and selected,
structural_retries0,salvaged=false; casting1 completed. Act_count1,18lines/
456words,castlocked,sourcecoveragecomplete; final audio/video path fields
still empty. This qualifies intermediate story receipt only, not the episode.

Source ordering explains the reload: lane returns run_story_spine=False;
shared tail explicitly unloads writer before story-brief reflection, then
reflection reacquires the technical slot. Teardown calls model.to(cpu), which
is wrapped by Accelerate's warning emitter; OTR catches teardown exceptions.
This matches observed warning timing, not proof of a fatal error or a reason
to change the active run. Bounded audit used metadata/source only.
Audit-tool friction: malformed rg regex returned `error: unclosed group`;
guessed nodes/_otr_slot_scheduler.py returned os error2. Both complete error
texts are in the private checkpoint; corrected lookup found scheduler in
OTR_LedgerScriptWriter.py. No runtime effect and no source edit.

Private4060-overnight-20260906-011353.html preserves5 timestamped events,
one background screenshot/full stale tree, exact runtime warning/refusal and
both audit-tool errors, verified265526bytes. Prior evidence intact. Only drill/
bug documentation updated; the same active runtime and canonical JSON stay fixed.

### Step42 -- 2026-09-06 01:18 PDT, user reaffirms full4060 test continuation

User requests: keep going; a full4060 test is required. Confirmed existing
same-thread heartbeat4060-overnight-otr-trials remains ACTIVE with five-minute
interval; no duplicate schedule or schedule modification. Same single GUI
RunSeptember5 23:04:44.645 and pending_20260905_230557. Read-only persisted
log at01:18:42 shows story-brief reflection128tokens/293.7s at01:16:38.920,
still0.4tok/s. No new runtime failure, download or episode completion.

One passive capture remains background-only with stale full accessibility
tree. No GUI activation/input, Run, restart/cancel, installed/JSON/model/cache/
dependency change, manual wait or hand repair. Fresh fetch/pull already up to
date; Step41 source-branch documentation push1575c15 was remote-verified.
Historical untracked review plan preserved. User reaffirmation does not make
this a second act or erase original GUI-only FAIL. Continue until the same
trial has matching RESULT SUCCESS, obs_publish OK and a final episode file,
or a recorded terminal failure; intermediate writer success is insufficient.

Private4060-overnight-20260906-011842.html preserves3 timestamped events,
one background screenshot/full stale tree, verified260680bytes. Prior evidence
intact. Documentation-only checkpoint; overnight monitoring remains active.

### Step43 -- 2026-09-06 01:24 PDT, reflection complete; story summary underway

Scheduled wake01:24:53.338. Same single GUI RunSeptember5 23:04:44.645 and
pending_20260905_230557. Read-only persisted log: story-brief reflection
192tokens/445.5s at01:19:10.709,256/596.5s at01:21:41.769, finishes
281/656.0s at01:22:41.296. run_produced_story_summary attempt1/3 begins
01:22:41.323, technical-slot cache reuse01:22:41.327, then64tokens/151.9s
at01:25:13.198, still0.4tok/s. No new runtime error, OOM,401 or download.
Matching RESULT SUCCESS, obs_publish OK and final episode remain unverified;
no other-model trial. This is another intermediate writing step, not completion.

One passive screenshot remains background-only/full accessibility tree stale;
no GUI activation/input or recovery retry. No Run, restart/cancel, installed/
JSON/model/cache/dependency change, manual wait or hand tuning. Fresh fetch/
pull already up to date; Step42 push022e541 was remote-verified. Historical
untracked review plan preserved. Actual progress comes from read-only log.

Private4060-overnight-20260906-012453.html preserves3 timestamped events,
one background screenshot/full stale tree, verified260590bytes. Prior evidence
intact. Documentation-only checkpoint; same single act remains untouched.
