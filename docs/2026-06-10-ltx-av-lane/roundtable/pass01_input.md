# PASS 01 REVIEW FOCUS: ARCHITECTURE

You are one panelist in an adversarial review of the plan below. THIS pass is
the ARCHITECTURE pass. Confine your must-fix findings to architecture; note
other issues in one line each at most.

Pressure-test exactly these:
1. Family token (Q1): new `audio_conditioned_video` family vs reusing
   `audio_driven_face` vs TWO adapters (talking-head vs music-reactive)
   sharing one engine core. Judge against the registry protocol and
   role_compat in the grounding. One engine spanning lip-sync AND
   music-reactive motion is the design tension -- is it a smell?
2. Isolation (Q4): in-process (the eng_ltx_video precedent) vs cu128 sidecar
   (the latentsync precedent). What is the explicit STOP rule if the LTX-2.3
   audio-conditioned path drags deps beyond ComfyUI-native into the cu130
   venv (V-12)?
3. Fallback chain (Q8): ltx_av -> humo -> latentsync -> still_kenburns is 4
   hops and humo pillarboxes a portrait while ltx_av renders full-frame
   landscape -- is a mid-chain ASPECT change an acceptable LOUD degrade, or
   should the chain be ltx_av -> still_kenburns direct? Check
   fallback.py's cycle/hop machinery in the grounding.
4. Adapter shape: does MotionEngineBase fit an audio-conditioned engine
   without mutation? Does anything in the render lifecycle
   (assert_usable/prepare/render_clip/canonicalize/teardown) need a NEW
   member for audio conditioning, and is that additive?
5. The Yvann-Nodes zero-new-model lane: parallel cheap lane for music_visual
   only, or distraction? Decide and defend in 5 lines.
6. Registry/schemas touch list: is the additive file list complete and
   minimal? Reject any delta that mutates existing engine behavior.

Rules: every claim about the code must cite the grounding excerpts; flag
anything you cannot verify as VERIFY-AT-BUILD; do not parrot the corrected
dims claim ("/32+1" is wrong for W/H); the V-1 audio invariant and the 14.5GB
NVML ceiling are non-negotiable; eng_ltx_video.py and the 13 unpushed commits
are out of scope for edits. Output: numbered MUST-FIX items (with the file
they touch), then SHOULD-CONSIDER, then OPEN-QUESTIONS. Be specific and
terse.


# LTX-AV lane (eng_ltx_av) -- problem statement + draft sprint plan (pass00)

> PLANNER-WINDOW artifact, 2026-06-10. NO production code this window.
> Roundtable campaign: 8 themed passes (architecture, inputs/outputs, prompts,
> wiring, ComfyUI-native testing, hardware, pre-mortem, finishing). Claude is
> a PANELIST and the FINAL JUDGE; the OpenRouter panel (GPT/Gemini/DeepSeek)
> critiques; every claim is grounded against the repo before acceptance.

## Mission

ADD an additive, audio-conditioned LTX lane (`eng_ltx_av`, LTX-2.3 A2V) to the
OTR video-engine registry, selectable from the OTR_VideoDirector dropdowns for
THREE roles: `announcer_visual` (lip-sync from a FLUX still via I2V+audio),
`character_video` (same), `music_visual` (audio-reactive scene motion --
"visuals breathe with the track"; FFT precision NOT expected). The existing
`ltx_video` 2B engine stays EXACTLY as-is (GPU-proven, shipped, production
default for announcer/music radio opens).

## Non-negotiable constraints (encode in every milestone)

- V-1: the new engine DISCARDS LTX's audio side entirely. Only
  OTR_MasterAudioMux emits audio; `test_audio_byte_identical` stays green.
  The per-beat `audio_ref` (already sliced from the frozen master) is the
  conditioning input; the engine's clip is canonicalized `has_audio=False`
  (the eng_humo precedent).
- 16GB RTX 5080 laptop: quantization lane picked by a MEASURED P0 probe gate;
  single-resident <= 14.5 GB machine-NVML; weight-streaming fallback
  documented.
- Registry-clean: new adapter + (probably) new family; required_inputs
  (text_prompt, audio_ref); default-OFF behind `OTR_ENABLE_LTX_AV`; full-frame
  landscape canvas; LOUD fallback chain; dropdown-selectable. NO mutation of
  existing engines; NO workflow-JSON surgery beyond what the Director already
  supports.
- Fail-loud dims validator (see corrected rule in the claims ledger).
- eng_ltx_video.py untouched. The kickoff's "ltx_motion.py C7 no-audio-imports"
  maps in the actual repo to: V-12 cold-import cleanliness (no audio-package
  imports in video modules), the V-1 silent-clip contract, and the b7
  forbidden-sweep bans. There is no ltx_motion.py; the motion base is
  `nodes/_otr_video_engines/motion_common.py` (zero audio mentions today --
  keep it that way).

## Claims ledger (verified 2026-06-10 by the judge; web + HF + repo)

CONFIRMED:
- `A2VidPipelineTwoStage` is real: Lightricks/LTX-2 monorepo,
  packages/ltx-pipelines; two-stage = base generation + latent upscaler.
  ComfyUI shipped native LTX-2.3 support day-zero; an official
  "LTX-2.3 Image Audio to Video" (IA2V) workflow template exists -- exactly
  the announcer/character shape (image + audio -> video).
- PR #13111 (Comfy-Org/ComfyUI) "LTX2: Support reference audio (ID-LoRA)
  (CORE-16)" by kijai, merged 2026-03-24. CAVEAT: Desktop/portable builds LAG
  core -- issues #13194 and #13308 report LTXVReferenceAudio MISSING in
  ComfyUI Desktop. Jeffrey runs Comfy Desktop -> M0 must check the node
  exists in HIS build before anything depends on it.
- QuantStack/LTX-2.3-GGUF exists incl. `LTX-2.3-distilled/
  LTX-2.3-distilled-Q4_K_M.gguf` (also unsloth/LTX-2.3-GGUF, 310K downloads).
  GGUF unet placement: ComfyUI-GGUF loader convention (`models/unet`).
  Exact file size: inventory at P0 (one community report says ~17.8 GB
  VRAM-class for Q4_K_M -- treat as UNMEASURED until P0).
- Kijai/LTX2.3_comfy splits (4M downloads, file list verified on HF):
  22B distilled-1.1 fp8_scaled transformer = 25,226,571,988 B (~23.5 GiB);
  LTX23_audio_vae_bf16 = 365 MB; LTX23_video_vae_bf16 = 1.45 GB;
  taeltx2_3 = 23.5 MB; ltx-2.3_text_projection_bf16 = 2.3 GB; distilled-1.1
  dynamic LoRA = 2.7 GB. NOT in this repo: MelBandRoformer, gemma text
  encoder, spatial upscaler -- they live elsewhere (P0 locates or drops).
- NVFP4: `ltx-2.3-22b-dev-nvfp4.safetensors` ~21.7 GB; native nvfp4 matmul
  needs Blackwell + PyTorch cu130 (Jeffrey's venv IS torch 2.10/cu130,
  sm_120). ~2-3x speed + large VRAM cut claimed. OPEN RISK: Comfy issue
  #11864 "Native NVFP4 (Blackwell) Loading Failure on RTX 5090" -- loading
  bugs exist in the wild. NVFP4 file is DEV (not distilled) -> more steps.
- The operator's repo ALREADY carries `scripts/download_ltx_2_3.ps1` (pulls
  the Kijai 22B distilled-1.1 fp8_scaled into C:\ComfyUI-Models with a
  symlink into `diffusion_models/`). Whether it has been RUN = P0 disk
  inventory. eng_ltx_video's docstring says the box runs the LTX 2.3 wrapper
  stack (gemma encoder + distilled LoRA + audio nodes) -- the PLUGIN is
  present; the OTR adapter graph itself drives the 2B v0.9 + T5 path
  (GPU-verified probe_f3). Research claim "current engine = 2B v0.9"
  CONFIRMED.

CORRECTED (the panel must NOT parrot the original):
- Dims rule. The research said "W/H divisible by 32 PLUS 1". Upstream issue
  #347 (Lightricks/ComfyUI-LTXVideo) calls the "+1" a documentation error for
  width/height. Correct rule: W and H divisible by 32 (silent rounding to the
  nearest valid value confirmed); FRAME COUNT is 8n+1 (9, 17, ..., 121).
  The fail-loud validator enforces: W % 32 == 0, H % 32 == 0,
  frames % 8 == 1, and raises (never rounds) on violation, logging the
  nearest valid values. Landscape canvas 1472x832: 1472 % 32 == 0,
  832 % 32 == 0 -> valid as-is.

UNVERIFIED -> verify-at-build (P0 gates):
- "FP8 runs in ~12GB" / "Q3_K_S 10s 1280x896 in 10-15 min on 16GB" -- single
  community reports; OUR NVML measurement decides the lane.
- Output audio of the A2V pipeline is the input audio bit-exact ("returns the
  original audio untouched"). We DISCARD the output audio track regardless
  (V-1 by construction), but P0 hashes it anyway as a model-behavior probe.
- gemma text-encoder requirement for the 2.3 graph: which file, what size,
  CPU-offloadable? (Unsloth gemma-3-12b GGUF claim -- locate exact repo/file
  at P0.) MelBandRoformer: likely UNNEEDED for v1 (announcer/character
  audio_refs are clean TTS slices; no vocal isolation required) -- panel
  judges whether music_visual needs it or whether the raw music slice
  conditions fine.
- 20s LTX-2.3 clip cap vs OTR beats (~4-10s): beats fit; opening-music beats
  b000 may run longer -> clamp policy needed.

## Repo grounding (read 2026-06-10; line refs as of HEAD 56caa5b)

- Registry protocol (`nodes/_otr_video_engines/registry.py`): family is one
  of `audio_driven_face | lipsync_overlay | image_to_video | text_to_video |
  static_image_gen | static_motion | abstract` (docstring list; `character_3d`
  ships with B). A NEW family token (working name `audio_conditioned_video`)
  touches: registry.py docstring, `schemas.py` (family validation -- check),
  role_compat reasoning. ALTERNATIVE the panel must weigh: reuse
  `audio_driven_face` (HuMo's family) for the talking-head roles -- but
  music_visual is NOT a face; one engine spanning face-sync AND
  music-reactive scene motion is the design tension.
- role_compat (`nodes/_otr_shared/role_compat.py`): supplied-inputs per role:
  ANNOUNCER_VISUAL {text_prompt, init_image, audio_ref, base_clip_ref};
  CHARACTER_VIDEO {text_prompt, init_image, audio_ref, base_clip_ref};
  MUSIC_VISUAL {text_prompt, init_image, base_clip_ref} -- **NO audio_ref**.
  An engine with required_inputs (text_prompt, audio_ref) FAILS CLOSED on
  music_visual today. Options for the panel: (a) add audio_ref to
  MUSIC_VISUAL's supplied set (one-line additive change + tests -- the music
  beat's slice exists; confirm the driver actually populates it), (b) split
  the lane into two adapters (talking vs reactive), (c) drop music_visual
  from v1. The driver side (render_driver.py, 42KB) must be grounded for
  where audio_ref is attached to requests.
- Director (`nodes/otr_video_director.py`): V-6 -- per-role COMBO = FULL
  static registry + "+ Add Custom Model" sentinel; compatibility filtered at
  execute time; fail-closed named error. A new registered adapter
  auto-appears in all three slots. VIDEO_SLOT_ROLES: announcer_video_model ->
  announcer_visual; music_video_model -> music_visual; other_beats ->
  character_video/scene_broll/background_abstract. NO Director code change
  expected.
- Fallback (`nodes/_otr_shared/fallback.py`): single-linked fallback_engine
  chain, cycle-checked, must terminate at the still_kenburns floor. Draft:
  ltx_av -> humo -> (humo's existing latentsync) -> still_kenburns (4 hops,
  acyclic). LOUD: log swap + ledger restamp (operator directive; never
  silent).
- eng_humo.py is the SHAPE TEMPLATE: audio-driven family, default-OFF dark
  registration, requires_flag, V-1 has_audio=False silent clip, AS-3 single
  -resident lease, BUG-291 reclaim_idle_models, V-12 lazy heavy imports,
  in-process forward in the EXECUTOR THREAD (A-S7.5 finding).
- eng_ltx_video.py: BUG-070 SageAttention gate (int8-PV process-aborts LTX;
  assert_sage_not_patched BEFORE first forward) -- eng_ltx_av inherits this
  gate. PROMOTED DEFAULT-ON 2026-06-10 for announcer/music radio opens
  (default_roles); the new lane must NOT disturb those defaults.
- Sandbox gotcha (re-confirmed today): the cowork sandbox mount showed 12
  phantom trailing bytes on workflows/otr_scifi_16gb_full.json (30,650 vs
  Windows-true 30,638). All build-window file verification goes through
  Desktop Commander, not the sandbox mount.

## Draft design (the thing the 8 passes harden)

Adapter: `nodes/_otr_video_engines/eng_ltx_av.py`
- class LtxAvEngine(MotionEngineBase), name "ltx_av",
  family "audio_conditioned_video" (panel: vs audio_driven_face reuse),
  roles ("announcer_visual", "character_video", "music_visual"),
  default_roles () -- dark, requires_flag "OTR_ENABLE_LTX_AV",
  required_inputs ("text_prompt", "audio_ref") with init_image consumed when
  the role supplies it (announcer/character I2V; panel: is init_image
  REQUIRED for the talking roles -- two-tier required_inputs?),
  fallback_engine "humo", declared_isolation IN_PROCESS (panel: the 2.3
  audio-conditioned graph pulls no new pip deps if ComfyUI-native nodes ARE
  the dependency -- V-12 check), target_fps 25, engine_version "1".
- Graph: ComfyUI-native LTX-2.3 IA2V topology (core nodes post-#13111 or the
  Lightricks plugin equivalents -- P0 inventories which exist in the
  installed build): audio_ref -> audio encode/conditioning, FLUX still ->
  I2V conditioning (announcer/character), text prompt via the 2.3 encoder
  stack, two-stage optional (base + latent upscale) -- v1 may run base-only
  at 1472x832; sampler per the distilled-1.1 schedule; VAEDecode VIDEO ONLY;
  the audio latent side is never decoded (V-1) -- if the installed node API
  forces a joint decode, drop the audio track at canonicalize and document.
- Dims validator: shared helper (motion_common or _otr_shared), raises on
  W%32 / H%32 / frames%8!=1 with nearest-valid hint; unit tests; wired into
  eng_ltx_av request prep ONLY (additive -- existing engines unchanged this
  sprint; flagged as a cheap follow-up for ltx_video/wan).
- Quantization lanes (P0 measures, plan documents all three):
  L1 22B distilled-1.1 fp8_scaled (23.5 GB file; ComfyUI block-swap /
  weight-streaming; download script already in repo);
  L2 GGUF Q4_K_M (QuantStack; ComfyUI-GGUF loader; models/unet);
  L3 NVFP4 (dev-only, cu130 native, issue #11864 risk; likely L3 = stretch).
  Gate: clip renders at target canvas; NVML peak <= 14.5 GB; wall time per
  ~6s clip recorded; eyeball quality vs the 2B baseline.
- VRAM: AS-3 single-resident heavy lease via wrapper_bridge; BUG-291
  reclaim_idle_models; free_after_use like LTX/Wan today.

## Milestones (draft -- the passes refine)

- M0 PROBE (= the operator's P0/P1/P2 PoC, folded in): disk inventory
  (which Kijai/GGUF files present; LTXVReferenceAudio / IA2V nodes present in
  the INSTALLED Desktop + headless builds); scratch A2V graph OUTSIDE OTR
  (V-12: stop if deps escape ComfyUI-native); render ~5s @ 720p-class with a
  real per-beat slice; hash the output audio track (probe); NVML peak + wall
  time per lane L1/L2(/L3); P1 eyeball matrix (a/b/c/d) -> verdict LIPSYNC |
  STYLIZED | INERT. INERT on all lanes = write the finding, close the lane,
  ship nothing.
- M1 ADAPTER (CPU-safe): eng_ltx_av.py skeleton + registration + family/
  schemas/role_compat additive edits + dims validator + unit tests; suite +
  Bug Bible green; byte-identical untouched (engine dark).
- M2 GRAPH + LANE: in-process graph spec from M0's winning lane; Sage gate;
  lease; canonicalize video-only; LOUD fallback chain registered + cycle
  test.
- M3 WIRING: Director pick-through (no code change expected -- prove with the
  existing policy tests); audio_ref supply for music_visual (per the panel's
  chosen option); ledger stamps (engine identity per clip -- the H4/P0-zero
  pattern from the audio side); OTR_FORCE_ENGINE_MAP support.
- M4 GATES: full suite + Bug Bible + test_audio_byte_identical + the live
  30-word smoke (scripts/queue_smoke.py) with ltx_av forced on each of the
  three roles; acceptance greps (engine-identity lines, fallback restamps,
  NVML ceiling); obs gains playable AAC finals only.
- M5 LOOK-QA + DOCS: operator eyeball; README/handoff/tracker rows; the
  Yvann-Nodes appendix verdict recorded.

## The zero-new-model comparison lane (panel judges: parallel cheap lane or distraction)

Yvann-Nodes audio-feature scheduling driving the EXISTING 2B ltx_video stack:
per-frame reactive weights / peak detection / prompt-intensity scheduling /
cut timing. Pros: zero new weights, zero quant risk, works today, music_visual
-only scope. Cons: "modulating around the audio" not "hearing it"; new custom
-node dependency (b7 sweep / license / V-12 review); does NOTHING for
lip-sync roles. Panel question: keep as a parallel music_visual-only lane,
fold behind the same flag, or cut.

## Open questions for the panel (each pass picks its own)

Q1 family token: new `audio_conditioned_video` vs reuse `audio_driven_face`
   vs TWO adapters (talking vs reactive) sharing one engine core.
Q2 music_visual audio_ref: role_compat supply-set addition vs adapter-level
   optionality vs drop the role from v1.
Q3 init_image: hard-require for announcer/character (true I2V lip-sync) --
   how does required_inputs express role-dependent requirements?
Q4 isolation: in-process (ltx_video precedent) vs cu128 sidecar (latentsync
   precedent) -- decided by P0 dep inventory; what's the STOP rule?
Q5 text encoder: exact gemma file/size/placement for the 2.3 graph; CPU
   offload; GGUF encoder for L2.
Q6 two-stage: base-only at 1472x832 vs base+latent-upscale; cost per clip vs
   the ~6 min/clip LTX opens today; episode time budget.
Q7 distilled-1.1 vs dev weights (NVFP4 is dev-only -- step count + LoRA
   compat).
Q8 fallback chain: ltx_av -> humo (portrait pillarbox mismatch with
   full-frame landscape -- acceptable degrade?) vs ltx_av -> ltx_video
   (text-only, loses sync) vs ltx_av -> still_kenburns direct.
Q9 clamp policy for beats > 20s (b000 music opens).
Q10 Desktop-vs-headless node availability skew (PR #13111 lag) -- which
    build gates M0.

## Pass plan (operator-ordered, 2026-06-10)

pass01 architecture; pass02 inputs/outputs; pass03 prompts; pass04 wiring;
pass05 ComfyUI-native testing; pass06 hardware; pass07 pre-mortem/red-team
(judge's pick); pass08 finishing/convergence. Panel: ~openai/gpt-latest +
~google/gemini-pro-latest + deepseek/deepseek-v4-pro, max_tokens 12000,
reasoning-effort none, temperature 0.5, budget $3.00/pass. Claude writes its
OWN panelist critique each pass BEFORE reading the panel's, then grounds all
four against the repo and synthesizes. Convergence = a pass yields no new
must-fix items.

## Hard exclusions (this sprint)

No 3D, no switchable-workflow, no whiny-voice, no edits to the 13 unpushed
commits (3f55ef9..56caa5b), no GPU work from the planner window, no
production code in the planner window, no workflow-JSON surgery, no mutation
of eng_ltx_video.py / motion_common.py behavior paths.
