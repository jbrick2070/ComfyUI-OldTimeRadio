# PASS 08 REVIEW FOCUS: FINISHING / CONVERGENCE SWEEP

You are one panelist on the FINAL pass. The plan below is the product of
seven grounded passes (architecture, I/O, prompts, wiring, testing,
hardware, pre-mortem). Your ONLY job: find anything BUILD-BLOCKING that
survived -- a contradiction between locked sections, a missing
coder-actionable detail that would stall a coder window mid-ticket, a
gate that cannot be evaluated as written, or a sequencing error in the
milestones.

Explicitly NOT wanted: relitigating locked decisions, style opinions,
nice-to-haves, new features, more detectors, more tests beyond gaps in
the stated matrix. If a finding is not build-blocking, put it in ONE
line under SHOULD-CONSIDER at most.

Checklist to sweep:
1. CONTRADICTIONS: do any two sections disagree (names, env vars, file
   paths, thresholds, chain orders, role lists, tuple contents)?
2. CODER-STALLERS: for each touch-list item, could a competent coder
   start it from this document alone (plus the named grounding files)
   without a question back to the operator? Name the first question
   they would be forced to ask, if any.
3. GATE EVALUABILITY: is every PASS/FAIL bar measurable as written
   (units, tools, thresholds)? Any gate that depends on an undefined
   artifact?
4. MILESTONE ORDER: M0 before M1 is intentional (probe gates the lane;
   M1 is CPU-safe and COULD start in parallel) -- is the dependency
   structure stated correctly? Any milestone consuming an output that
   does not exist yet at its start?
5. TICKET BOUNDARIES: propose the cleanest cut of coder-window tickets
   (2-4 tickets) over M1-M4 with explicit done-criteria each -- or
   endorse a cut if one is implicit. One ticket must never span a
   suite-red intermediate state.
6. ANYTHING ELSE that would make you say "no" to "build-ready as-is?"

Output format: VERDICT line ("CONVERGED -- build-ready" or "NO --
build-blocking items remain"); then numbered BUILD-BLOCKING items (file/
section + the exact fix), then SHOULD-CONSIDER one-liners, then the
ticket-cut proposal. Terse. Cite grounding or mark VERIFY-AT-BUILD.


# LTX-AV lane -- sprint plan after pass07 (ALL themed passes folded; finishing next)

> Campaign docs/2026-06-10-ltx-av-lane/. Judgments pass01-07. pass08 =
> final convergence sweep. NO production code from the planner window.

## Mission

Additive audio-conditioned LTX-2.3 lane: `ltx_av_talk` (announcer_visual,
character_video; I2V lip-sync from the FLUX still + per-beat audio) and
`ltx_av_music` (music_visual; audio-reactive scene motion). All shipped
engines untouched. Dark by default behind OTR_ENABLE_LTX_AV.

## ARCHITECTURE (pass01)

eng_ltx_av.py: private shared core + 2 thin MotionEngineBase adapters.
- ltx_av_talk: roles (announcer_visual, character_video); family
  audio_driven_face; required (text_prompt, audio_ref, init_image);
  fallback humo -> humo_1.7B -> latentsync -> still_kenburns; degrade
  aspect change = LOUD documented policy.
- ltx_av_music: roles (music_visual,); family audio_conditioned_video
  (NEW); required (text_prompt, audio_ref); fallback ltx_video ->
  still_kenburns.
- Both dark; ONE flag (usability, not registration); IN_PROCESS; fps 25;
  BUG-070 Sage gate; AS-3 lease; BUG-291 reclaim; V-12 lazy imports;
  executor-thread forward. Config envs OTR_LTX_AV_CKPT /
  OTR_LTX_AV_TEXT_ENCODER.
- assert_usable(host, profile, request_template=None) order: flag ->
  Sage -> NVML REQUIRED (nvml_available() False = fail closed, named --
  this lane only) -> node gate (missing classes -> MISSING_MODEL naming
  them; six-reason enum pinned) -> weights (realpath EXISTS + size >=
  per-artifact floor: transformer >= 12 GiB, encoder >= 10 GiB, video
  VAE >= 1 GiB; message names the exact artifact) -> av_dims on
  template canvas when provided (None tolerated).
- STOP rule: pip-freeze sandwich + nodes resolve, else
  ISOLATION_SIDECAR_REQUIRED + STOP + finding.

## I/O CONTRACTS (pass02)

_ref_path(request.audio_ref) tolerant (humo :366-383); init_image =
asset_refs["init_image"]; conditioning_refs never counts; base_clip_ref
ignored; talk fails closed pre-render on missing inputs (classified,
before GPU). Core ffmpeg-normalizes audio to s16le/44.1k/mono (episode
tmp). T = timing.target_frame_count; next_8n1 = ((n+6)//8)*8+1 snap-UP
(legacy :281 snaps DOWN -- never copy); render = min(next_8n1(T),
LTX_AV_MAX_FRAMES [M0 sheet; init 497]); canonicalize trims to T / pads
by last frame, LOUD "[ltx_av] pad-tail rendered=<n> target=<T>" >2s.
Graph ends at video VAEDecode -> IMAGE batch ->
encode_frames_to_silent_mp4 (-an); joint-AV strip -map 0:v:0 -an +
fake-AV test; CanonicalClip silent/yuv420p/bt709/fps25/int frames +
engine_id/family stamps; ffprobe zero audio streams. canvas =
request.canvas; av_dims RAISES with nearest-valid (#347 doc-error).
init image: in-graph ImageScale+crop COVER+center-crop; pad+outpaint =
M0 cell.

## PROMPTS (pass03)

Adapter-thin; driver composes (M4 creative > override > brief).
ltx_av_music JOINS the :418 tuple verbatim (radio override honored);
ltx_av_talk SIBLING branch, NO radio override, fallback prompt =
character_description-or-"a 1940s radio announcer" + "head and
shoulders at a period microphone" (no speech verbs) via
finish_visual_prompt(240, style_tail=False) + no-text clause; forbidden
content list. NEGATIVE = _LTX_DEFAULT_NEGATIVE verbatim (conditional
music-only extension on M0 evidence). No music tail v1. Cap 240. AST
test: no brief-helper imports in the adapter.

## WIRING (pass04)

Universal master-slice already feeds line-backed beats; ONE gated
delta: line-less shots slice from SHOT synthetic timing iff engine_id
== ltx_av_music. Hash/seed safety test-proven. _render_one passes
request_template=request (TypeError guard). ENGINE_FAMILY += both;
SYNTH_FALLBACKS += both. Flag-off = RENDER-TIME degrade (ShotLock never
asserts; registry docstring corrected). Force map role-guarded
(LOUD-ignore), never bypasses asserts. Announcer portrait alias
(ltx_av_talk-gated). Identity: post-restamp engine_id + trail origin;
greps = format_swap_log + manifest engine_id + pad-tail marker. No
group pruning.

## TESTING (pass05)

tests/test_av_dims.py; tests/test_video_ltx_av.py (humo-mirror full
list incl. node gate via mocked NODE_CLASS_MAPPINGS, NVML fail-closed
case, weight-floor case); tests/test_ltx_av_driver_wiring.py (dark-lane
SEMANTIC-PROJECTION goldens; flag-off degrade; force guard; alias;
synthetic slice gating; template pass-through; ENGINE_FAMILY; canvas;
prompt gates) + slice-cache key unit. Fallout: retry-taxonomy sweep +
exact-enumerations -> membership; b7 auto-covers (loop var `imp`).
Forgot-it matrix complete. Byte-identical: CPU structural; forced-lane
master-hash = M4 GPU. Pytest = no network/CUDA/weights/forwards. M0
sheet checked in; parser test post-M0; LTX_AV_MAX_FRAMES == sheet
(drift test). Bug Bible: BUG-070/291 pins; new dims row at ship
(Three-File Contract).

## HARDWARE (pass06)

Verified sizes table (see pass06_plan). DEAD full-residency: Q4_K_S+,
fp8_scaled; candidates Q3_K_S (13.0 GiB) / Q3_K_M (13.7 BORDERLINE);
total NVML decides. M0 TABLE rows: Q3_K_S resident / Q3_K_M resident /
Q4_K_M offloaded / L1 fp8 block-swap / 2-CONSECUTIVE-CLIP marginal-cost
row / taeltx-vs-full-VAE cell / FLUX-ordering verification row /
NEGATIVE-INSTALL drill row. PASS: NVML peak+sustained <= 14500 MB; wall
<= 10 min/clip PASS, 10-15 WARN, > 15 FAIL; episode <= 30 min PASS, >
45 dead; quality >= 2B A/B. Encoder phasing inside ONE lease (encode ->
reclaim_idle_models -> transformer; never release between phases); M0
measures GPU-encode-then-reclaim vs CPU-offloaded encode (default = the
passing mode, prefer GPU). RAM/pagefile/commit rows; >= 32 GB for
block-swap rows. NVFP4 CUT from M0 (stretch later). Two-stage base-only.
download script disk note >= 24 GiB; ComfyUI-GGUF pack = inventory row.

## PRE-MORTEM (pass07) -- detectors + disciplines folded in

- EPISODE SUMMARY block at run_episode end (existing surfaces):
  fallback_counts_by_from_engine, pad_tail_count + padded_s,
  nvml_available, max_vram_mb, final_engine_histogram.
- STORM lines: ">= 2 degrades same ltx_av_* origin" ->
  "[ltx_av] FALLBACK_STORM from=<e> count=<n>/<beats>"; ">= 2 clips
  pad > 2s" -> "[ltx_av] PAD_TAIL_STORM count=<n> padded_s=<s>". M4
  normal smoke FAILS on either line (unless the test forces it).
- NVML fail-closed for this lane (architecture, above). M0 rows with
  probe_used_mb==0 + render work are INVALID.
- Weight floors + realpath in assert_usable (above); full hashes M0.
- Post-cancel/post-OOM discipline: before the next heavy shot, lease
  absent AND wait_until_below_mb(14500), else ONE screaming line
  instructing restart. Operator rule "restart after any mid-render
  cancel" in M0 checklist + adapter docstring + ship notes (lease
  wedges ~120s on a live PID; reclaim only frees dead PIDs).
- RELOAD THRASH accepted v1 (per-clip lifecycle is the contract); the
  M0 2-clip row prices it; keep-resident = future policy question.
- M0 LAUNCHER GATE: :8000 active-job check + no held lease + NVML idle
  before any pull/render; abort during the acceptance window.
- Slice-cache key += master mtime_ns + size (the ONE shared-path
  bugfix; unit-tested; also fixes latent HuMo case).
- Desktop/headless PARITY table in M0_RESULTS.md = SHIP GATE for
  look-QA (runtime self-gate is safety-sufficient; parity is the ship
  bar). Restart discipline in checklist + docstring + error text.
- Captions/credits: LOW (absolute timeline placement); covered by
  existing duration_check/captions greps + one manifest
  frame_count-vs-target note.
- Goldens = semantic projection + regen-note policy.

## Additive touch list (FINAL consolidation)

- NEW  nodes/_otr_video_engines/eng_ltx_av.py (core + 2 adapters)
- NEW  nodes/_otr_shared/av_dims.py
- NEW  tests/test_av_dims.py, tests/test_video_ltx_av.py,
       tests/test_ltx_av_driver_wiring.py (+ semantic-projection
       fixtures under tests/fixtures/ltx_av_dark/)
- EDIT nodes/_otr_video_engines/__init__.py (guarded import)
- EDIT nodes/_otr_video_engines/schemas.py (family + required map)
- EDIT nodes/_otr_video_engines/registry.py (docstrings: family list +
       ShotLock-claim correction)
- EDIT nodes/_otr_shared/role_compat.py (MUSIC_VISUAL += audio_ref)
- EDIT nodes/_otr_video_engines/render_driver.py -- NINE deltas:
       (a) :387 canvas tuple += both names
       (b) :418 prompt gate += ltx_av_music; ltx_av_talk sibling branch
       (c) synthetic-timing slice fallback (ltx_av_music-gated)
       (d) _render_one request_template=request (+TypeError guard)
       (e) ENGINE_FAMILY += both
       (f) apply_engine_override role-compat guard (LOUD-ignore)
       (g) announcer portrait alias (ltx_av_talk-gated)
       (h) run_episode EPISODE SUMMARY + the two STORM lines
       (i) _slice_master_audio cache key += mtime_ns + size [shared
           -path bugfix, own unit test]
       (+) SYNTH_FALLBACKS += both
- EDIT retry-taxonomy sweep + exact-enumeration tests -> membership
- EDIT scripts/download_ltx_2_3.ps1 (disk note >= 24 GiB; sibling pull
       lines for GGUF/encoder as needed)
- NEW  docs/2026-06-10-ltx-av-lane/M0_RESULTS.md (sheet incl. parity
       table) + M0 operator checklist
- Docs/tracker + Bug Bible dims row at ship (Three-File Contract).

## Milestones (FINAL shape)

- M0 PROBE (GPU evening; launcher gate enforces no-acceptance-window):
  inventory (artifacts + ComfyUI-GGUF pack + node parity Desktop/
  headless); pip-freeze sandwich; scratch IA2V render w/ real slice;
  output-audio hash probe; the M0 TABLE incl. 2-clip + encoder-mode +
  VAE cells; prompt cells; NEGATIVE-INSTALL drill; P1 matrix ->
  LIPSYNC | STYLIZED | INERT (INERT everywhere closes the lane).
- M1 ADAPTERS (CPU): eng_ltx_av dark + av_dims + schema/role_compat/
  __init__/registry + driver deltas a-i + 3 test files + goldens +
  fallout edits; suite + Bug Bible green; byte-identical untouched.
- M2 GRAPH + LANE: winning-lane graph; pre-flight; lease+phasing;
  silent encode; trim/pad; LTX_AV_MAX_FRAMES pinned to sheet.
- M3 WIRING PROOF: slot asserts; flag-off degrade; force guard; alias;
  identity/manifest; storm-line emission tests.
- M4 GATES: full suite + Bug Bible + byte-identical + forced-lane
  master-hash + live 30-word smoke (flag ON + force map); greps:
  swap-log lines, manifest engine_id, NO storm lines, duration_check
  OK, captions events, NVML <= 14.5; obs playable AAC only.
- M5 LOOK-QA + DOCS + Bug Bible row + parity gate check.

## Appendix: cut lanes (cumulative)

Yvann-Nodes (p01). New prompt/negative envs (p03). ASPECT_CHANGE kind +
group-prune wiring (p04). New usability reason / GPU pytest / framework
(p05). NVFP4-in-M0 (p06). Timeline-assertion script + keep-resident-v1
+ exact-byte weight checks (p07).

## pass08 (finishing): convergence sweep

Panel reviews THIS document for build-blocking contradictions or
missing coder-actionable detail ONLY. If no new must-fix items survive
grounding, the campaign is CONVERGED and the coder tickets are cut from
the touch list + milestones as written.
