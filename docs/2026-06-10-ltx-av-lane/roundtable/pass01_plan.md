# LTX-AV lane -- sprint plan after pass01 (architecture CONVERGED)

> Campaign: docs/2026-06-10-ltx-av-lane/. Pass01 (architecture) synthesized
> from GPT-5.5 + Gemini 3.1 Pro + DeepSeek v4 + Claude panelist, every claim
> grounded vs HEAD 56caa5b. See pass01_judgment.md. Passes remaining:
> inputs/outputs, prompts, wiring, testing, hardware, pre-mortem, finishing.

## Mission (unchanged)

Additive audio-conditioned LTX-2.3 lane for three Director roles --
announcer_visual + character_video (lip-sync from a FLUX still, I2V+audio)
and music_visual (audio-reactive scene motion). `ltx_video` (2B) stays
exactly as-is. NO production code from the planner window.

## ARCHITECTURE (LOCKED in pass01)

ONE new file `nodes/_otr_video_engines/eng_ltx_av.py`: a private shared core
(graph spec, audio-latent discard, dims validation, lane config) + TWO thin
registered adapters on MotionEngineBase:

- `ltx_av_talk` -- roles ("announcer_visual", "character_video");
  family "audio_driven_face" (REUSED; schemas maps it to
  (audio_ref, init_image) -- exactly the contract);
  required_inputs ("text_prompt", "audio_ref", "init_image");
  fallback_engine "humo" -> real chain humo -> humo_1.7B -> latentsync ->
  still_kenburns (5 engines from ltx_av_talk; eng_humo.py:99).
  Degrade aspect change (landscape -> HuMo 480x832 pillarbox) is an ACCEPTED
  LOUD policy: log swap + ledger restamp; reason string names the aspect
  change.
- `ltx_av_music` -- roles ("music_visual",);
  family "audio_conditioned_video" (NEW);
  required_inputs ("text_prompt", "audio_ref");
  fallback_engine "ltx_video" -> still_kenburns (role-valid: ltx_video
  carries music_visual natively; aspect-stable landscape; ZERO edits to
  eng_ltx_video.py -- the attr lives on the new adapter).
- Both: default_roles () (dark), requires_flag "OTR_ENABLE_LTX_AV" (ONE flag
  gates the lane), declared_isolation ISOLATION_IN_PROCESS, target_fps 25,
  engine_version "1", BUG-070 assert_sage_not_patched, AS-3 single-resident
  lease via wrapper_bridge, BUG-291 reclaim_idle_models, V-12 lazy heavy
  imports, executor-thread forward (A-S7.5).
- assert_usable pre-flight (fail closed, named errors): flag gate -> Sage
  gate -> NODE-AVAILABILITY check (every required ComfyUI node class
  resolves in NODE_CLASS_MAPPINGS; missing classes listed by name) ->
  checkpoint/encoder presence on disk.
- V-1: the audio side of LTX-2.3 is DISCARDED (video-only decode if the node
  API allows; otherwise drop the audio track at canonicalize). Clips
  canonicalize has_audio=False (humo pattern). Only OTR_MasterAudioMux emits
  audio. test_audio_byte_identical green at every milestone.

### Isolation STOP rule (LOCKED)

In-process IFF: (a) pip freeze of the cu130 venv is IDENTICAL before/after
M0 setup (zero new packages, zero version changes), AND (b) all required
node classes resolve in the installed build(s). Either violation -> declare
ISOLATION_SIDECAR_REQUIRED (token exists, motion_common.py:46-48), STOP the
sprint, write the finding (latentsync precedent). No "temporary" installs
into cu130, ever.

### Additive touch list (complete; nothing else moves)

- NEW  nodes/_otr_video_engines/eng_ltx_av.py (core + 2 adapters)
- NEW  nodes/_otr_shared/av_dims.py (fail-loud dims validator, dep-free)
- EDIT nodes/_otr_video_engines/__init__.py (guarded import, existing
       pattern lines 22-65)
- EDIT nodes/_otr_video_engines/schemas.py (FAMILIES +=
       "audio_conditioned_video"; FAMILY_REQUIRED_INPUTS entry
       ("text_prompt","audio_ref"); the sync assert then passes)
- EDIT nodes/_otr_video_engines/registry.py (docstring family list ONLY)
- EDIT nodes/_otr_shared/role_compat.py (MUSIC_VISUAL supply +=
       "audio_ref") -- CONDITIONAL on pass04 driver verification
- NEW  tests (pass05 enumerates; engine-count/dropdown tests update expected)
- Docs: this campaign dir + handoff/tracker rows at ship.

### Dims validator (corrected rule -- do not regress to "/32+1")

nodes/_otr_shared/av_dims.py: assert_ltx_dims(width, height, frames):
W % 32 == 0, H % 32 == 0, frames % 8 == 1; on violation RAISE (never round)
naming the nearest valid values. Landscape 1472x832 passes as-is. Upstream
silently rounds (Lightricks/ComfyUI-LTXVideo issue #347: the "+1" on W/H is
a doc error; frames are 8n+1). Both adapters call it in prepare. Existing
engines adopt it in a LATER sprint (no touch now).

## Claims ledger (verified; carry forward unchanged unless re-verified)

CONFIRMED: A2VidPipelineTwoStage (Lightricks/LTX-2 ltx-pipelines; two-stage
= base + latent upscale). ComfyUI native LTX-2.3 day-zero; official IA2V
template (image+audio->video). PR #13111 LTXVReferenceAudio (kijai, merged
2026-03-24) -- Desktop/portable builds LAG core (issues #13194/#13308):
Jeffrey runs Comfy Desktop -> M0 gates on HIS builds. QuantStack
LTX-2.3-GGUF incl. distilled Q4_K_M (+ unsloth mirror); GGUF unet ->
models/unet (ComfyUI-GGUF convention). Kijai/LTX2.3_comfy file inventory:
22B distilled-1.1 fp8_scaled = 23.5 GiB; LTX23 audio VAE 365 MB; video VAE
1.45 GB; taeltx2_3 23.5 MB; text_projection 2.3 GB; distilled-1.1 dynamic
LoRA 2.7 GB; NO MelBandRoformer / gemma encoder / spatial upscaler in that
repo (P0 locates or drops). NVFP4 = ltx-2.3-22b-DEV-nvfp4 ~21.7 GB, needs
cu130 (Jeffrey HAS torch 2.10/cu130/sm_120), known loading bug report
(Comfy issue #11864). repo already carries scripts/download_ltx_2_3.ps1
(Kijai fp8 -> C:\ComfyUI-Models symlink); the LTX-2.3 wrapper stack is
installed per eng_ltx_video.py's docstring while the OTR adapter drives the
2B v0.9 + T5 graph (probe_f3).

CORRECTED: dims rule (above). HuMo fallback chain (humo -> humo_1.7B ->
latentsync -> still_kenburns).

UNVERIFIED -> P0 gates: 16GB VRAM community reports (fp8 ~12GB; Q3_K_S
timing); output-audio bit-exactness probe (we discard regardless); gemma
text-encoder file/size/offload (which exact artifact the IA2V graph needs);
MelBandRoformer NOT needed for v1 (clean TTS slices) -- music slice
conditioning quality is the M0 P1c cell; >20s beat clamp policy (pass02).

## Repo grounding (pass01-refreshed)

schemas.py:30-68 FAMILIES (8 incl. character_3d) + FAMILY_REQUIRED_INPUTS +
sync assert; family_hint validated at request build (:162-176).
role_compat.py ROLE_AVAILABLE_INPUTS: ANNOUNCER_VISUAL/CHARACTER_VIDEO
supply {text_prompt, init_image, audio_ref, base_clip_ref}; MUSIC_VISUAL
{text_prompt, init_image, base_clip_ref} (no audio_ref yet);
engine_fits_role callers = the two Directors (policy/dropdown), NOT the
render path. fallback.py: single-linked chain walk, cycle/hop guarded
(DEFAULT_MAX_HOPS=16). resolver.py: execution-group DAG validation +
orphaned-PROVIDER pruning on fallback restamp (AS-2) -- no role-aware chain
pruning exists; the split-adapter design is what makes every chain
role-valid by construction. eng_humo.py: required_inputs
("audio_ref","init_image"), render hard-fails without both (:320-323),
fallback_engine "humo_1.7B" (:99), has_audio=False silent-clip precedent.
otr_video_director.py: V-6 full-registry COMBO + ADD_CUSTOM sentinel;
VIDEO_SLOT_ROLES routes announcer/music/other_beats; new adapters
auto-appear; NO Director edits. __init__.py: guarded per-adapter imports.
Sandbox mount showed phantom trailing bytes on the saved workflow JSON
(30,650 vs Windows-true 30,638) -- build-window file verification goes
through Desktop Commander.

## Milestones

- M0 PROBE (operator P0/P1/P2 folded in; GPU evening, AFTER the acceptance
  -test window): disk inventory (Kijai/GGUF/encoder files present?);
  LTXVReferenceAudio + IA2V node presence in BOTH Comfy Desktop and the
  headless launcher build (Q10: headless is authoritative for OTR renders;
  Desktop is the operator's eyeball surface -- BOTH must pass or the lane
  waits on a Comfy update); pip-freeze snapshot -> scratch IA2V graph
  OUTSIDE OTR -> render ~5s with a real per-beat slice; pip-freeze diff ==
  empty (STOP rule); hash output audio track (probe); NVML peak + wall time
  per lane L1 (fp8_scaled 23.5GB + block swap) / L2 (GGUF Q4_K_M) / L3
  (NVFP4, stretch); P1 eyeball matrix a/b/c/d -> verdict LIPSYNC | STYLIZED
  | INERT per role-shape. INERT everywhere = write the finding, close the
  lane, ship nothing.
- M1 ADAPTERS (CPU-safe, coder window): eng_ltx_av.py skeleton (2 dark
  adapters + core), av_dims.py, schemas/role_compat/__init__/registry
  deltas, unit tests; suite + Bug Bible green; byte-identical untouched.
- M2 GRAPH + LANE: in-process graph spec from M0's winning lane; node
  pre-flight; lease + reclaim; canonicalize video-only; chain registration +
  termination tests.
- M3 WIRING: Director pick-through proof (policy tests only);
  render_driver music-beat audio_ref attachment (per pass04); ledger
  engine-identity stamps; OTR_FORCE_ENGINE_MAP entries for both names.
- M4 GATES: full suite + Bug Bible + test_audio_byte_identical + live
  30-word smoke with each role forced to the new adapters; acceptance greps
  (identity lines, LOUD restamps incl. aspect-change reason, NVML <= 14.5);
  obs gains playable AAC finals only.
- M5 LOOK-QA + DOCS: operator eyeball; README/handoff/tracker; appendix
  verdicts recorded.

## Appendix: cut lanes

- Yvann-Nodes audio-feature scheduling: CUT from this sprint (4/4 panel).
  Music-only payoff, new custom-node dep (b7/license/V-12 review), competes
  with the probe. Revisit ONLY if M0 returns INERT for music conditioning.

## Open questions (assigned)

- pass02: asset_refs shapes (audio_ref/init_image extraction, humo
  _ref_path pattern); canonicalize contract details (bt709/yuv420p, ffprobe
  asserts); frames derivation from beat duration vs 8n+1 (+ sync with the
  slice length); clamp policy for >20s beats (b000 music opens); canvas/fps
  plumbing (OTR_VIDEO_LANDSCAPE_CANVAS).
- pass03: prompt composition for the three roles (story-brief
  finish_visual_prompt path, no-text clause, era tails, person guard
  interplay for portraits).
- pass04: driver attachment of music audio_ref; execution-group/provider
  effects on degrade; restamp wording; FORCE map; dropdown policy tests.
- pass05: full test list + Desktop-vs-headless gate mechanics.
- pass06: encoder artifact + size; per-clip wall time budget vs ~6 min LTX
  opens; L1/L2/L3 decision gate numbers; weight-streaming fallback.
- pass07: failure modes (OOM mid-episode, fallback storms, partial
  downloads, Comfy restarts, zombie VRAM, caption/credits interplay).
- pass08: convergence + coder-window tickets.
