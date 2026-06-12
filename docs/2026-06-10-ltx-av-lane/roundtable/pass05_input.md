# PASS 05 REVIEW FOCUS: COMFYUI-NATIVE TESTING

You are one panelist in an adversarial review of the plan below. THIS pass
is the TESTING pass. Pass01-04 decisions are LOCKED -- one-line flags only.

The suite is currently 3815/0 with Bug Bible green at every commit; the
new lane must land the same way: every M1-M3 commit keeps the FULL suite
green, byte-identical green, and the engine dark by default.

Pressure-test exactly these against the grounding (existing test patterns
are in the grounding files -- mirror them, do not invent new frameworks):

1. TEST ENUMERATION: produce the DEFINITIVE list of new test files /
   cases for M1-M3, each with (a) what it proves, (b) the pattern file it
   mirrors. Cover at minimum: registry/dropdown additive presence; role
   compat (music audio_ref supply; talk slot fit); schema family
   round-trip; av_dims unit cases (W/H/frames, nearest-valid hints,
   1472x832 passes, 1450x832 raises, frames snap-up cases incl. T=25,
   T=26, cap); chain termination (5-hop talk chain, 3-hop music chain,
   trail retention); dark-lane GOLDEN FIXTURES (existing-engine requests
   bit-identical with the lane registered-but-dark); flag-off render
   -time degrade; force-map role guard; announcer portrait alias; the
   fake-AV-mp4 strip + zero-audio-stream ffprobe assert; AST
   no-brief-import; cold-import (V-12) with the new module; identity
   stamps in CanonicalClip/manifest; pad-tail marker emission;
   _render_one request_template pass-through (TypeError guard).
2. EXISTING-TEST FALLOUT: which existing tests MUST change when two
   engines register (engine counts, dropdown enumerations, fallback
   -chain sweeps, ENGINE_FAMILY assertions, b7 forbidden sweep)? Name
   the files from grounding where possible; the b7 sweep's AST loop var
   must be `imp` (repo gotcha). What existing test would FAIL TODAY if
   the coder forgets each touch-list edit (one per edit -- the "forgot
   it" detector matrix)?
3. GPU-GATED VS CPU TESTS: the suite runs headless/CPU; HuMo/LTX forwards
   are GPU-smoke scripts, not pytest. Define the exact split for the new
   lane: what is CPU-provable (everything above) vs what lives in the M0
   /M4 GPU scripts (real render, NVML ceiling, wall time, lip-sync
   eyeball). Should the M0 sheet be a CHECKED-IN artifact (e.g.
   docs/.../M0_RESULTS.md) that a later test asserts exists + parses?
4. BYTE-IDENTICAL GUARD: test_audio_byte_identical is the crown jewel.
   Does the new lane need a DEDICATED variant (episode rendered with
   ltx_av forced -> master hash unchanged), and can that run CPU-only
   via the existing prune-to-node-7 trick (audio path without video
   cost), or is it M4-GPU-only? Specify.
5. DESKTOP-VS-HEADLESS NODE GATE: PR #13111 nodes may exist in one build
   and not the other. Where is that gate TESTED -- assert_usable unit
   with a mocked NODE_CLASS_MAPPINGS missing one class (CPU), plus an M0
   checklist row per build? Anything else?
6. REGRESSION DISCIPLINE: Bug Bible flow for this lane -- which existing
   BUG-IDs are at risk of regression (BUG-070 Sage, BUG-291 reclaim,
   BUG-265 family) and does any new lane behavior deserve a NEW Bug
   Bible row at ship (e.g. the silent-rounding dims trap)?

Rules: cite grounding or VERIFY-AT-BUILD; mirror existing patterns; no
new test frameworks; CPU determinism (no network, no GPU in pytest).
Output: numbered MUST-FIX (file + what), SHOULD-CONSIDER, OPEN-QUESTIONS.
Terse.


# LTX-AV lane -- sprint plan after pass04 (arch + I/O + prompts + wiring LOCKED)

> Campaign docs/2026-06-10-ltx-av-lane/. Judgments pass01-04. Remaining:
> testing, hardware, pre-mortem, finishing. NO production code this window.

## Mission

Additive audio-conditioned LTX-2.3 lane: `ltx_av_talk` (announcer_visual,
character_video; I2V lip-sync from the FLUX still + per-beat audio) and
`ltx_av_music` (music_visual; audio-reactive scene motion). All shipped
engines untouched.

## ARCHITECTURE (LOCKED pass01)

ONE new file `nodes/_otr_video_engines/eng_ltx_av.py`: private shared core +
two thin MotionEngineBase adapters.

- ltx_av_talk: roles (announcer_visual, character_video); family
  audio_driven_face (reused); required_inputs (text_prompt, audio_ref,
  init_image); fallback "humo" -> humo_1.7B -> latentsync -> still_kenburns
  (aspect change on degrade = LOUD, documented).
- ltx_av_music: roles (music_visual,); family audio_conditioned_video
  (NEW); required_inputs (text_prompt, audio_ref); fallback "ltx_video" ->
  still_kenburns (eng_ltx_video.py:70 grounds the floor hop).
- Both: dark (default_roles ()); ONE flag OTR_ENABLE_LTX_AV (gates
  USABILITY; @register at import is unconditional -- the dark lane is
  registered, selectable in the dropdown, and fails closed at render);
  ISOLATION_IN_PROCESS; fps 25; BUG-070 Sage gate; AS-3 lease; BUG-291
  reclaim; V-12 lazy imports; executor-thread forward.
- assert_usable(host_caps, profile, request_template=None) order: flag ->
  Sage -> NODE-AVAILABILITY (NODE_CLASS_MAPPINGS, missing named) ->
  weights on disk -> av_dims on request_template.canvas (CPU-fail BEFORE
  the lease).
- Isolation STOP rule: in-process IFF pip freeze identical before/after M0
  AND nodes resolve; else ISOLATION_SIDECAR_REQUIRED + STOP + finding.

## I/O CONTRACTS (LOCKED pass02)

- EXTRACTION: audio = tolerant _ref_path(request.audio_ref)
  (eng_humo.py:366-383); init_image = asset_refs.get("init_image","");
  conditioning_refs never counts; base_clip_ref ignored. ltx_av_talk
  fails closed pre-render if either is empty (classified hard failure
  BEFORE GPU work -> render_shot walks the chain; never starve/hang).
- AUDIO NORMALIZE: core always ffmpeg-normalizes audio_ref to
  s16le/44.1k/mono WAV in the episode temp dir (master slices already
  are; per-line variants are not). LTX node format tolerance: M0 sheet.
- FRAMES: T = timing.target_frame_count (authority); next_8n1(n) =
  ((n+6)//8)*8+1 (snap UP; legacy :281 snaps DOWN -- do not copy);
  render_frames = min(next_8n1(T), LTX_AV_MAX_FRAMES [M0, init 497]);
  canonicalize TRIMS to T or PADS-BY-LAST-FRAME to T, LOUD marker
  "[ltx_av] pad-tail rendered=<n> target=<T>" when padding > 2s.
- OUTPUT: graph ends at video VAEDecode -> IMAGE batch ->
  wrapper_bridge.encode_frames_to_silent_mp4() (-an); joint-AV fallback
  strip `-map 0:v:0 -an` + fake-AV-mp4 unit test; canonicalize returns
  CanonicalClip (has_audio=False, yuv420p, bt709, fps 25, integer
  frame_count) AND STAMPS engine_id (+family) -- the manifest's per-clip
  identity column; ffprobe-asserts zero audio streams.
- CANVAS: request.canvas.w/h; av_dims (W%32, H%32, frames%8==1, RAISE
  with nearest-valid; "+1 on W/H" is upstream doc error #347).
- INIT IMAGE (talk): in-graph core-node preprocess (ImageScale+crop),
  uniform COVER + center-crop; pad+outpaint = M0 cell only.

## PROMPTS (LOCKED pass03)

- Adapter-thin: request.text_prompt / request.negative_prompt only; no
  brief-helper imports (AST test). Driver owns composition (M4 creative >
  override > brief-composed).
- Driver: ltx_av_music JOINS the :418 tuple verbatim (radio override +
  open clause + brief compose); ltx_av_talk gets a SIBLING branch, NO
  radio override, fallback prompt = subject (character_description else
  "a 1940s radio announcer") + "head and shoulders at a period
  microphone" (no speech verbs) via finish_visual_prompt (240,
  style_tail=False) + no-text clause. Forbidden: dialogue, stage
  directions, vocatives, caption text.
- NEGATIVE: _LTX_DEFAULT_NEGATIVE verbatim (shared constant); extension
  only on M0 inert evidence (", frozen pose, still image", music-only).
- MUSIC TAIL: none in v1; one M0 cell tests a motion-energy clause.
- CAP: 240 everywhere. M0 prompt cells: +/- speech verb, +/- motion
  clause, optional long-prompt probe.

## WIRING (LOCKED pass04)

- MUSIC AUDIO: the universal master-slice fallback (:351-378) already
  feeds line-backed beats for ANY engine. ONE engine-gated delta: when
  the line has no start_s/dur_s AND engine_id == "ltx_av_music", slice
  from the SHOT row's synthetic timing (covers b000 opening-music).
  Dark-lane golden-fixture test: existing-engine requests bit-identical.
- HASH/SEEDS: render_request_hash is ShotLock-stamped, driver-read-only;
  request_seed = _seed_from_hash(hash) on the episode path; attaching
  audio_ref moves NOTHING (test asserts).
- assert_usable PLUMBING: _render_one (:490) gains
  request_template=request (Protocol already declares the kwarg;
  TypeError guard for any legacy adapter -- VERIFY cheap_families).
- ENGINE_FAMILY (:53-63) += both new names; SYNTH_FALLBACKS += 
  {"ltx_av_talk": "humo", "ltx_av_music": "ltx_video"} as one-line
  belt-and-braces for the guarded-import edge.
- FLAG-OFF: enforcement is RENDER-TIME (ShotLock never calls
  assert_usable; "an episode NEVER aborts, a beat is NEVER dropped").
  Dark pick -> gated EngineUnusable -> restamp -> chain -> completes.
  registry.py's stale "ShotLock calls assert_usable" docstring line is
  corrected in the same M1 docstring touch.
- FORCE MAP: apply_engine_override validates (role, engine) via
  engine_fits_role; incompatible entries IGNORED with LOUD warning;
  forcing never bypasses render-time asserts. M4 smoke env documented.
- ANNOUNCER PORTRAIT: engine-gated alias -- engine_id == "ltx_av_talk",
  empty char_id, role announcer_visual -> resolve the shipped non-cast
  announcer portrait from ledger["images"] (object id VERIFY-AT-BUILD).
  Missing portrait -> classified pre-render fail -> humo (also starved,
  LOUD) -> floor; trail records every hop.
- IDENTITY: final shot engine_id = post-restamp engine;
  degradation_trail keeps ltx_av_* origin; acceptance greps =
  _rt.format_swap_log lines + manifest engine_id + the pad-tail marker.
  No group pruning (lane has no provider groups; prune_orphaned_groups
  is not wired in run_episode today -- claim removed).

## Additive touch list (consolidated pass04)

- NEW  nodes/_otr_video_engines/eng_ltx_av.py (core + 2 adapters)
- NEW  nodes/_otr_shared/av_dims.py (assert_ltx_dims + next_8n1)
- EDIT nodes/_otr_video_engines/__init__.py (guarded import)
- EDIT nodes/_otr_video_engines/schemas.py (family + required-inputs map)
- EDIT nodes/_otr_video_engines/registry.py (docstring: family list +
       correct the ShotLock-assert claim)
- EDIT nodes/_otr_shared/role_compat.py (MUSIC_VISUAL += "audio_ref")
- EDIT nodes/_otr_video_engines/render_driver.py -- SEVEN additive deltas:
       (a) :387 canvas tuple += ltx_av_talk, ltx_av_music
       (b) :418 prompt gate += ltx_av_music; sibling ltx_av_talk branch
       (c) synthetic-timing slice fallback (ltx_av_music only)
       (d) _render_one passes request_template=request
       (e) ENGINE_FAMILY += both names
       (f) apply_engine_override role-compat guard (ignore+warn)
       (g) announcer portrait alias (ltx_av_talk-gated)
       (+) SYNTH_FALLBACKS belt-and-braces entries
- NEW  tests (pass05 enumerates; patterns exist:
       test_video_engine_registry_base_additive.py,
       test_video_fallback_chain_additive.py, test_video_humo*.py,
       test_brief_prompt_finishing.py)
- Docs/tracker at ship.

## Claims ledger (cumulative; deltas p04)

CONFIRMED: universal master-slice fallback :351-378; _render_one
assert_usable without request :490; ENGINE_FAMILY :53-63; SYNTH_FALLBACKS
:53 {hunyuan3d_talk}; registration unconditional (@register import-time);
ShotLock never calls assert_usable; CanonicalClip carries engine_id;
ltx_video.fallback_engine still_kenburns :70.
MISREADS DISCARDED: dark-lane unregistered floor-bypass; engine-gating
the existing slice path; ASPECT_CHANGE failure-kind; lock-time abort;
role-scoped radio override (p03); timing.dur_s violation (p02).
UNVERIFIED -> M0/M1: cheap_families assert_usable signatures; restamp/
swap-log exact formats; announcer portrait object id; CanonicalClip
field list for a pad note; node IO shapes; temporal ceiling; audio
formats; resize convention; portrait dims; prompt cells.

## Milestones

- M0 PROBE: disk inventory; node presence (Desktop + headless);
  pip-freeze sandwich; scratch IA2V render w/ real slice; output-audio
  hash probe; NVML + wall time L1 fp8_scaled / L2 GGUF Q4_K_M / L3 NVFP4;
  record the M0 sheet; P1 matrix -> LIPSYNC | STYLIZED | INERT; INERT
  everywhere closes the lane with a finding.
- M1 ADAPTERS (CPU): eng_ltx_av.py dark + av_dims + schemas/role_compat/
  __init__/registry + driver deltas (a-g) + unit tests + golden dark-lane
  fixtures; suite + Bug Bible green; byte-identical untouched.
- M2 GRAPH + LANE: winning-lane graph; pre-flight; lease; silent encode;
  trim/pad; chain registration + termination tests.
- M3 WIRING PROOF: Director slot asserts; flag-off degrade test; force
  -map guard test; announcer alias test; identity stamps/manifest test.
- M4 GATES: full suite + Bug Bible + byte-identical + live 30-word smoke
  (flag ON + force map per role); acceptance greps (swap-log lines,
  manifest engine_id, pad-tail marker, NVML <= 14.5); obs playable AAC
  only.
- M5 LOOK-QA + DOCS.

## Appendix: cut lanes

Yvann-Nodes (pass01). OTR_LTX_AV_PROMPT / _NEGATIVE envs (pass03).
ASPECT_CHANGE FailureKind (pass04). Group-prune wiring (pass04 -- not
needed for this lane).

## Open questions (assigned)

- pass05 TESTING: enumerate the full test list (golden fixtures, AST
  no-brief-import, fake-AV strip, chain termination incl. 5-hop talk
  chain, dropdown/slot asserts, flag-off degrade, force guard, announcer
  alias, identity stamps, av_dims unit, cold-import, engine-count
  updates, Desktop-vs-headless node gate mechanics, M0-sheet-as-checklist).
- pass06 HARDWARE: encoder artifact/size/placement; per-clip wall time vs
  ~6 min LTX opens; L1/L2/L3 decision numbers; weight streaming; FLUX
  co-residency; lease behavior with 23.5GB fp8 file.
- pass07 PRE-MORTEM: OOM mid-episode; fallback storms; partial downloads;
  Comfy restart staleness; zombie VRAM on cancel; slice-cache key;
  caption/credits interplay; Desktop node-lag (#13194/#13308); golden
  -fixture rot.
- pass08 FINISHING: convergence + coder-window tickets.
