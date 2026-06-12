# PASS 04 REVIEW FOCUS: WIRING

You are one panelist in an adversarial review of the plan below. THIS pass
is the WIRING pass: how the two new engines thread through Director ->
ShotLock -> render driver -> fallback -> ledger. Architecture, I/O, and
prompts are LOCKED (pass01-03) -- one-line flags only.

Pressure-test exactly these against the grounding:

1. MUSIC audio_ref ATTACH: the slice mechanism exists
   (render_driver._slice_master_audio; the announcer/talking path fills
   audio_ref by slicing [start_s, start_s+dur_s] from the frozen master).
   Find the exact call site / branch where MUSIC beats' requests are
   assembled, and specify the additive change that attaches the per-beat
   slice for engine_id == "ltx_av_music" ONLY (existing engines must see
   IDENTICAL requests to today -- byte-identical ledger/request hashing
   concerns?). Does request hashing / render_request_hash change when
   audio_ref is added, and does that invalidate caches or seeds for
   EXISTING engines? This is the pass's most dangerous edge -- be precise.
2. DIRECTOR -> SHOTLOCK: V-6 says the dropdown auto-includes new
   registered engines and role compat filters at execute. Verify
   ltx_av_talk appears in announcer_video_model AND other_beats slots,
   ltx_av_music in music_video_model; what does OTR_ShotLock's
   assert_usable path do when the flag is off (GATED_BY_FLAG) -- clean
   fail-closed to the role's default engine, or episode abort? Specify
   expected behavior + the test that proves it.
3. FALLBACK MECHANICS: on a render-time ltx_av_talk failure, who walks
   the chain (driver? render batch node?), what EXACTLY is restamped in
   the ledger (engine_id? degradation_trail? group restamp via
   resolver.prune_orphaned_groups?), and what log line proves the swap is
   LOUD? Write the exact restamp wording for (a) the talk aspect-change
   degrade, (b) the music ltx_video degrade, (c) the pad-tail >2s case.
4. ENGINE-IDENTITY LEDGER STAMPS: the audio side proved per-line engine
   identity is unprovable without stamps (H4/P0-zero). What is the video
   side's current per-clip identity stamp (engine_id on shot rows?
   degradation_trail?), and what must the new lane add so an acceptance
   grep can PROVE which engine rendered every clip (incl. after
   fallback)?
5. OTR_FORCE_ENGINE_MAP: how does it interact with role compat + the
   flag gate for the new names -- can the operator force ltx_av_talk on
   announcer beats for the M4 smoke with one env, and does forcing bypass
   assert_usable (it must NOT)?
6. PORTRAIT / init_image SUPPLY for announcer: portrait_ledger +
   announcer alias behavior -- does an announcer beat reliably get an
   init_image today (in-character portraits shipped 435ba0a), and what
   happens to ltx_av_talk when the portrait is missing (fail-closed ->
   fallback, or starve)? Specify expected behavior.
7. SEEDS: request_seed derives from render_request_hash (build_request).
   Confirm the new engines inherit deterministic per-shot seeds with no
   extra work, and that the C7 env overrides (OTR_CAST_SEED/...) are
   irrelevant here.

Rules: cite grounding or VERIFY-AT-BUILD; existing engines' requests/
hashes/ledgers must be BIT-IDENTICAL to today when the new lane is dark
(default-OFF) -- any wiring change that alters dark-lane behavior is a
MUST-FIX against itself. Output: numbered MUST-FIX (file + what),
SHOULD-CONSIDER, OPEN-QUESTIONS. Terse.


# LTX-AV lane -- sprint plan after pass03 (architecture + I/O + prompts LOCKED)

> Campaign docs/2026-06-10-ltx-av-lane/. Judgments: pass01/02/03_judgment.md.
> Remaining passes: wiring, testing, hardware, pre-mortem, finishing.
> NO production code from the planner window.

## Mission

Additive audio-conditioned LTX-2.3 lane: `ltx_av_talk` (announcer_visual,
character_video -- lip-sync from the FLUX still, I2V + per-beat audio) and
`ltx_av_music` (music_visual -- audio-reactive scene motion). `ltx_video`
(2B) and every other shipped engine stay untouched.

## ARCHITECTURE (LOCKED pass01)

ONE new file `nodes/_otr_video_engines/eng_ltx_av.py`: private shared core +
two thin MotionEngineBase adapters.

- ltx_av_talk: roles (announcer_visual, character_video); family
  audio_driven_face (REUSED; schemas requires (audio_ref, init_image));
  required_inputs (text_prompt, audio_ref, init_image); fallback "humo" ->
  humo -> humo_1.7B -> latentsync -> still_kenburns; degrade aspect change
  landscape -> pillarbox is a LOUD restamped policy.
- ltx_av_music: roles (music_visual,); family audio_conditioned_video (NEW);
  required_inputs (text_prompt, audio_ref); fallback "ltx_video" ->
  still_kenburns (role-valid, aspect-stable, zero ltx_video edits).
- Both: default_roles () dark; ONE flag OTR_ENABLE_LTX_AV;
  ISOLATION_IN_PROCESS; target_fps 25; engine_version "1"; BUG-070
  assert_sage_not_patched; AS-3 lease; BUG-291 reclaim_idle_models; V-12
  lazy imports; executor-thread forward.
- assert_usable order: flag -> Sage -> NODE-AVAILABILITY (graph node
  classes resolve in NODE_CLASS_MAPPINGS; missing named) -> weights on
  disk -> av_dims on the request_template canvas (CPU-fail BEFORE lease).
- Isolation STOP rule: in-process IFF pip freeze identical before/after M0
  AND nodes resolve; else ISOLATION_SIDECAR_REQUIRED + STOP + finding.

## I/O CONTRACTS (LOCKED pass02)

- EXTRACTION: audio = tolerant _ref_path(request.audio_ref) (AudioRef
  {"path":...} | str | .path obj; eng_humo.py:366-383); init_image =
  asset_refs.get("init_image",""); conditioning_refs never satisfies
  required inputs; base_clip_ref ignored (documented). ltx_av_talk fails
  closed pre-render if either path empty (humo :320-323 pattern).
- AUDIO NORMALIZE: sliced-master slices are s16le/44.1k/mono WAV
  (_slice_master_audio); other sources unnormalized -> the core ALWAYS
  ffmpeg-normalizes audio_ref to s16le/44.1k/mono in the episode temp dir.
  LTX node accepted formats: M0 sheet.
- FRAMES: T = timing.target_frame_count (authority). next_8n1(n) =
  ((n+6)//8)*8+1 (snap UP; legacy :281 snaps DOWN -- do not copy).
  render_frames = min(next_8n1(T), LTX_AV_MAX_FRAMES [M0-measured, init
  497]). canonicalize TRIMS to T, or PADS-BY-LAST-FRAME to T (cap case),
  LOUD when padding > 2s. No compositor hold assumed.
- OUTPUT: graph terminates at video VAEDecode -> IMAGE batch ->
  wrapper_bridge.encode_frames_to_silent_mp4() (-an; :446/:512). No
  audio-bearing container ever on disk; joint-AV fallback strip
  `-map 0:v:0 -an` guarded by a fake-AV-mp4 unit test. canonicalize
  returns CanonicalClip (has_audio=False, yuv420p, bt709, fps 25, integer
  frame_count; eng_humo._clip_from_raw precedent) + ffprobe-asserts ZERO
  audio streams.
- CANVAS: request.canvas.w/h renders; av_dims.assert_ltx_dims (W%32, H%32,
  frames%8==1, RAISE naming nearest valid; "+1 on W/H" is upstream doc
  error, Lightricks #347) in assert_usable + prepare.
- INIT IMAGE (talk): in-graph core-node preprocessing (ImageScale + crop)
  from resolve_aspect_transform math; v1 = uniform COVER + center-crop
  (no pad bars); pad+outpaint = M0 experiment cell only.

## PROMPTS (LOCKED pass03)

- ADAPTER-THIN: adapters read ONLY request.text_prompt /
  request.negative_prompt; NO brief-helper imports in eng_ltx_av.py
  (AST-test enforced). Prompt composition lives in the DRIVER, joining the
  gap-audit pipeline (M4 creative > operator override > brief-composed).
- DRIVER DELTAS at render_driver.py:418 (the no-creative compose gate;
  CHARACTER_BEARING_ROLES = {character_video} in ShotLock, so announcer/
  music NEVER carry M4 creative -- this gate is their only prompt source):
  - ltx_av_music JOINS the existing tuple ("ltx_video","wan_i2v") --
    verbatim reuse: OTR_LTX_RADIO_PROMPT honored on opens, "a vintage
    radio set glowing in the scene" open clause, brief compose via
    get_story_brief_ltx + finish_visual_prompt (240 cap, no-text clause).
  - ltx_av_talk gets a SIBLING branch (same precedence, NO radio
    override): no-creative fallback = subject (character_description when
    present, else "a 1940s radio announcer") + "head and shoulders at a
    period microphone" (setting noun, NO speech verbs -- double-driving
    risk) finished via finish_visual_prompt (240, style_tail=False) +
    "no on-screen text". FORBIDDEN content: quoted dialogue/beat text,
    stage directions, vocative character names, caption text.
  - Character beats arrive with M4 creative (ShotLock derives for
    character_video) and pass through unchanged.
- OTR_LTX_RADIO_PROMPT asymmetry (grounded :418-427, engine-gated not
  role-scoped): music honors, talk skips, NO new env vars.
- NEGATIVE: _LTX_DEFAULT_NEGATIVE verbatim, one shared core constant.
  Conditional extension ONLY on M0 evidence of audio-inert renders
  (pre-agreed string ", frozen pose, still image", music-only).
- MUSIC TAIL: none in v1 (no rhythm vocabulary -- strobing/over-constraint
  risk; audio conditioning carries motion). One M0 P1 cell tests a single
  motion-energy clause; a win lands it in the DRIVER music branch.
- CAP: 240 chars everywhere (consistency + hash stability).
- M0 P1 prompt cells: +/- speech verb (talk); +/- motion clause (music);
  optional 240-vs-long probe.

## Additive touch list (updated pass03)

- NEW  nodes/_otr_video_engines/eng_ltx_av.py (core + 2 adapters)
- NEW  nodes/_otr_shared/av_dims.py (+ next_8n1)
- EDIT nodes/_otr_video_engines/__init__.py (guarded import)
- EDIT nodes/_otr_video_engines/schemas.py ("audio_conditioned_video" in
       FAMILIES + FAMILY_REQUIRED_INPUTS ("text_prompt","audio_ref"))
- EDIT nodes/_otr_video_engines/registry.py (docstring family list only)
- EDIT nodes/_otr_shared/role_compat.py (MUSIC_VISUAL += "audio_ref",
       unconditional, M1)
- EDIT nodes/_otr_video_engines/render_driver.py:
       (a) :387 canvas tuple += ltx_av_talk, ltx_av_music
       (b) :418 prompt gate: tuple += ltx_av_music; sibling ltx_av_talk
           branch with the talk fallback template (NO radio override)
       (c) music-beat audio_ref attach (pass04 specifies)
- NEW  tests (pass05 enumerates)
- Docs/tracker at ship.

## Claims ledger (cumulative deltas)

CONFIRMED p03: prompt-compose gate :418 is engine-tuple-gated; the radio
override lives INSIDE it; CHARACTER_BEARING_ROLES = {character_video};
finish_visual_prompt/get_story_brief_ltx are driver-side helpers (helpers
module is pure, consumer->helper direction locked by AST test precedent).
CONFIRMED p02: snap-DOWN legacy formula :281; landscape tuple :387; slice
format; encode_frames_to_silent_mp4 -an :446/:512; Timing fields.
MISREADS DISCARDED: timing.dur_s schema violation (p02); role-scoped radio
override (p03).
UNVERIFIED -> M0 sheet: node IO shapes; temporal ceiling; accepted audio
formats; IA2V resize convention; FLUX portrait dims; M4 creative
suitability for talk beats; speech-verb / motion-clause / long-prompt
cells.

## Milestones

- M0 PROBE: disk inventory; node presence in BOTH Desktop + headless;
  pip-freeze sandwich (STOP rule); scratch IA2V render with a real slice;
  output-audio hash probe; NVML peak + wall time per lane L1 (fp8_scaled
  23.5GB block-swap) / L2 (GGUF Q4_K_M) / L3 (NVFP4 stretch); RECORD the
  M0 sheet (node IO shapes, ceiling, audio formats, resize convention,
  portrait dims, pad-vs-crop, prompt cells); P1 eyeball matrix ->
  LIPSYNC | STYLIZED | INERT per role-shape; INERT everywhere = close the
  lane with a finding.
- M1 ADAPTERS (CPU): eng_ltx_av.py dark skeleton + av_dims + schemas/
  role_compat/__init__/registry deltas + render_driver tuple/branch edits
  + unit tests; suite + Bug Bible green; byte-identical untouched.
- M2 GRAPH + LANE: winning-lane graph; pre-flight; lease; silent encode;
  trim/pad; chain registration + termination tests.
- M3 WIRING: Director pick-through proof; music-beat audio_ref attach;
  ledger engine-identity stamps; OTR_FORCE_ENGINE_MAP entries.
- M4 GATES: full suite + Bug Bible + byte-identical + live 30-word smoke
  per role forced; acceptance greps (identity lines, LOUD restamps incl.
  aspect-change + pad-tail reasons, NVML <= 14.5); obs playable AAC only.
- M5 LOOK-QA + DOCS.

## Appendix: cut lanes

Yvann-Nodes: CUT (pass01, 4/4); revisit only on INERT-for-music.
OTR_LTX_AV_PROMPT / OTR_LTX_AV_NEGATIVE env vars: CUT (pass03).

## Open questions (assigned)

- pass04 WIRING: music-beat audio_ref attach point (the slice mechanism
  exists at _slice_master_audio + :650; specify the music-branch call
  site); announcer-with-description subject source; ShotLock
  execution-group/provider effects on degrade; restamp wording (aspect
  change, pad-tail); OTR_FORCE_ENGINE_MAP; dropdown policy tests; ledger
  engine-identity stamps (H4 pattern).
- pass05 TESTING: full test list; Desktop-vs-headless gate mechanics;
  fake-AV-mp4 strip test; AST no-brief-import test; cold-import; chain
  termination; engine-count updates; M0 sheet as a checklist artifact.
- pass06 HARDWARE: gemma encoder artifact + size + placement; per-clip
  wall time vs ~6 min LTX opens; L1/L2/L3 decision numbers; weight
  streaming; FLUX co-residency.
- pass07 PRE-MORTEM: OOM mid-episode; fallback storms; partial downloads;
  Comfy restart staleness; zombie VRAM on cancel; slice-cache key;
  caption/credits interplay; Desktop node-lag (#13194/#13308).
- pass08 FINISHING: convergence + coder-window tickets.
