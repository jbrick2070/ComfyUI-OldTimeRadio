# PASS 06 REVIEW FOCUS: HARDWARE

You are one panelist in an adversarial review of the plan below. THIS pass
is the HARDWARE pass. Pass01-05 LOCKED -- one-line flags only.

Box: RTX 5080 Laptop 16GB VRAM (sm_120, Blackwell), torch 2.10/cu130 main
venv, machine-NVML single-resident ceiling 14.5GB, Windows, Comfy Desktop
for the operator + a headless launcher for renders. The existing 2B
ltx_video does 1472x832 opens at ~6 min/clip. System RAM size UNKNOWN to
the plan (the M0 sheet must record it).

JUDGE-VERIFIED file sizes (HF API, 2026-06-10 -- treat as ground truth;
do not parrot community VRAM folklore over these):
- Kijai 22B distilled-1.1 fp8_scaled transformer: 23.5 GiB file
- QuantStack LTX-2.3-distilled GGUF: Q2_K 11.6 / Q3_K_S 13.0 /
  Q3_K_M 13.7 / Q4_K_S 15.6 / Q4_K_M 16.5 / Q5_K_S 17.3 / Q5_K_M 18.1 /
  Q6_K 19.6 / Q8_0 23.7 GiB
- gemma_3_12B_it_fp8_scaled text encoder: 13.2 GB (placement
  models/text_encoders; CPU/RAM-offloaded text encoding is a known
  community lever; a GGUF Q3 encoder variant exists "comfortably in
  12GB"; Lightricks issue #303 = no official smaller encoder yet)
- LTX23 audio VAE 365MB, video VAE 1.45GB, taeltx2_3 23.5MB,
  text_projection 2.3GB, distilled-1.1 dynamic LoRA 2.7GB
- NVFP4: ltx-2.3-22b-DEV-nvfp4 21.7GB, needs cu130 (present), native on
  Blackwell, open loading-failure report (Comfy issue #11864), DEV not
  distilled (more steps).

Pressure-test exactly these:

1. LANE DECISION GATE: define the M0 decision TABLE -- for each lane
   (L1 fp8_scaled 23.5GiB w/ ComfyUI block-swap/weight-streaming; L2
   GGUF per-quant; L3 NVFP4), the measured columns (NVML peak,
   wall-time/clip at 1472x832 x ~6s, quality eyeball vs 2B baseline)
   and the PASS criteria (NVML <= 14.5 sustained; wall time budget --
   propose one vs the ~6 min 2B opens; quality >= 2B). Given Q4_K_M =
   16.5GiB FILE, is full residency arithmetic already dead for Q4 on a
   14.5 ceiling, leaving Q3_K_S/Q3_K_M as the realistic full-resident
   GGUF picks unless ComfyUI-GGUF per-layer offload measures under the
   ceiling? Say so explicitly.
2. ENCODER PHASING: the 13.2GB fp8 gemma encoder + a 12-16GiB
   transformer CANNOT co-reside under 14.5. Specify the phase
   discipline the adapter must rely on (encode -> free encoder ->
   load transformer; ComfyUI model management vs explicit
   reclaim_idle_models), whether text-encode-on-CPU is the v1 default
   for this lane, and what the AS-3 lease wraps (the whole render or
   per-phase?). Check eng_humo/wrapper_bridge/gpu_residency in the
   grounding for the existing lease + BUG-291 reclaim mechanics and
   mirror them.
3. SYSTEM RAM: block-swap/offload eats system RAM (23.5GiB fp8 file
   streamed). The M0 sheet must record RAM size + peak commit; name the
   failure mode if RAM is short (paging -> wall-time blowup) and the
   gate (e.g. wall-time/clip ceiling catches it).
4. EPISODE TIME BUDGET: with ~2 talk beats + 1 music open per 30-word
   episode, propose the per-clip wall-time PASS bar and the episode
   -level budget delta vs today (the 2B opens already cost ~6 min);
   when does the lane become operationally unusable even if it renders
   (e.g. >15 min/clip)?
5. TWO-STAGE: base+latent-upscale doubles cost; v1 = base-only at
   1472x832 was locked in pass02. Confirm no hardware reason to revisit
   (or flag one).
6. FLUX CO-RESIDENCY: portraits render upstream (FLUX) before video;
   confirm sequential phases (image batch THEN video batch) mean no
   co-residency requirement, per the existing pipeline order; flag if
   the grounding suggests otherwise.
7. L3 NVFP4: given DEV-only weights (not distilled -> more steps),
   issue #11864, and 21.7GB file, judge whether L3 stays a stretch
   -goal column in the M0 sheet or gets cut from M0 to save probe time
   (the operator can add it later).

Rules: cite grounding or the judge-verified numbers above; arithmetic
must be explicit (GiB vs GB consistent); the 14.5 NVML ceiling and V-1
are non-negotiable; no new pip into cu130 (STOP rule). Output: numbered
MUST-FIX (file/section + what), SHOULD-CONSIDER, OPEN-QUESTIONS. Terse.


# LTX-AV lane -- sprint plan after pass05 (arch+I/O+prompts+wiring+testing LOCKED)

> Campaign docs/2026-06-10-ltx-av-lane/. Judgments pass01-05. Remaining:
> hardware, pre-mortem, finishing. NO production code this window.

## Mission

Additive audio-conditioned LTX-2.3 lane: `ltx_av_talk` (announcer_visual,
character_video; I2V lip-sync from the FLUX still + per-beat audio) and
`ltx_av_music` (music_visual; audio-reactive scene motion). All shipped
engines untouched.

## ARCHITECTURE (LOCKED pass01)

ONE new file nodes/_otr_video_engines/eng_ltx_av.py: private shared core +
two thin MotionEngineBase adapters.
- ltx_av_talk: roles (announcer_visual, character_video); family
  audio_driven_face; required_inputs (text_prompt, audio_ref, init_image);
  fallback humo -> humo_1.7B -> latentsync -> still_kenburns (aspect
  change on degrade = LOUD documented policy).
- ltx_av_music: roles (music_visual,); family audio_conditioned_video
  (NEW); required_inputs (text_prompt, audio_ref); fallback ltx_video ->
  still_kenburns (:70 grounds the floor hop).
- Both: dark; ONE flag OTR_ENABLE_LTX_AV (usability, not registration --
  @register is import-time); ISOLATION_IN_PROCESS; fps 25; BUG-070 Sage
  gate; AS-3 lease; BUG-291 reclaim; V-12 lazy imports; executor-thread.
- assert_usable(host_caps, profile, request_template=None): flag -> Sage
  -> node gate (missing classes -> MISSING_MODEL naming them; the
  six-reason enum is PINNED) -> weights -> av_dims on template canvas
  when provided (`if request_template:` -- None tolerated).
- Isolation STOP rule: in-process IFF pip freeze identical before/after
  M0 AND nodes resolve; else ISOLATION_SIDECAR_REQUIRED + STOP + finding.

## I/O CONTRACTS (LOCKED pass02)

- audio = _ref_path(request.audio_ref) tolerant (humo :366-383);
  init_image = asset_refs["init_image"]; conditioning_refs never counts;
  base_clip_ref ignored; talk fails closed pre-render on missing either.
- Core ALWAYS ffmpeg-normalizes audio_ref to s16le/44.1k/mono WAV
  (episode temp dir). Node format tolerance: M0 sheet.
- T = timing.target_frame_count; next_8n1(n)=((n+6)//8)*8+1 snap-UP
  (legacy :281 snaps DOWN -- do not copy); render = min(next_8n1(T),
  LTX_AV_MAX_FRAMES [M0; init 497]); canonicalize trims to T / pads by
  last frame, LOUD marker "[ltx_av] pad-tail rendered=<n> target=<T>"
  when >2s.
- Graph ends at video VAEDecode -> IMAGE batch ->
  wrapper_bridge.encode_frames_to_silent_mp4 (-an); joint-AV strip
  `-map 0:v:0 -an` + fake-AV test; CanonicalClip has_audio=False,
  yuv420p, bt709, fps 25, integer frame_count, engine_id+family STAMPED;
  ffprobe zero audio streams.
- canvas = request.canvas; av_dims (W%32, H%32, frames%8==1; RAISE w/
  nearest-valid; "+1 on W/H" is upstream doc error #347).
- init image: in-graph ImageScale+crop, uniform COVER+center-crop; pad+
  outpaint = M0 cell.

## PROMPTS (LOCKED pass03)

- Adapter-thin (request.text_prompt/negative_prompt only; AST-tested no
  brief-helper imports). Driver composes: M4 creative > override >
  brief-composed.
- ltx_av_music JOINS the :418 tuple verbatim (radio override honored);
  ltx_av_talk SIBLING branch, NO radio override, fallback prompt =
  character_description-or-"a 1940s radio announcer" + "head and
  shoulders at a period microphone" (no speech verbs) via
  finish_visual_prompt (240, style_tail=False) + no-text clause.
  Forbidden: dialogue, stage directions, vocatives, captions.
- NEGATIVE: _LTX_DEFAULT_NEGATIVE verbatim shared constant; extension
  only on M0 inert evidence (", frozen pose, still image", music-only).
- MUSIC TAIL none in v1 (M0 cell tests one motion-energy clause).
  CAP 240 everywhere. M0 prompt cells: +/- speech verb, +/- motion
  clause, optional long-prompt.

## WIRING (LOCKED pass04)

- Universal master-slice already feeds line-backed beats; ONE delta:
  line-less shots slice from SHOT synthetic timing when engine_id ==
  ltx_av_music (b000 covered; dark-lane golden-fixture proven).
- render_request_hash ShotLock-stamped, driver-read-only; request_seed =
  _seed_from_hash; audio_ref attach moves nothing (tested).
- _render_one passes request_template=request (Protocol kwarg;
  TypeError guard for legacy adapters).
- ENGINE_FAMILY += both; SYNTH_FALLBACKS += {ltx_av_talk: humo,
  ltx_av_music: ltx_video} (belt-and-braces).
- Flag-off = RENDER-TIME degrade (ShotLock never asserts; episode never
  aborts); registry.py stale ShotLock claim corrected in same docstring
  touch.
- Force map validates (role, engine) via engine_fits_role, ignores
  incompatible LOUD; never bypasses render asserts. M4 smoke:
  OTR_ENABLE_LTX_AV=1 + OTR_FORCE_ENGINE_MAP=announcer_visual=
  ltx_av_talk,character_video=ltx_av_talk,music_visual=ltx_av_music.
- Announcer portrait alias (ltx_av_talk-gated; object id VERIFY);
  missing portrait -> classified pre-render fail -> chain -> floor,
  trail records hops.
- Identity: final engine_id post-restamp; trail keeps origin; greps =
  _rt.format_swap_log + manifest engine_id + pad-tail marker. No group
  pruning (lane has no providers).

## TESTING (LOCKED pass05)

New files (mirror named patterns; membership assertions, never
cardinality):
- tests/test_av_dims.py (pure unit): next_8n1 25->25/26->33/33->33/
  idempotent; 1472x832+f49 pass; 1450x832 raises naming 1440/1472;
  height+frames violations w/ nearest-valid; cap T=520 -> 497 + pad
  flag.
- tests/test_video_ltx_av.py (mirrors test_video_humo.py): registered_
  and_dark x2; role fit (music audio_ref); required_inputs match family
  schema; assert_usable ORDER (flag-first / Sage / node-gate
  MISSING_MODEL naming class via mocked NODE_CLASS_MAPPINGS / weights /
  dims) + request_template=None tolerated; ref extraction str/dict/
  None; deterministic build; canonicalize silent bt709 + identity
  stamps; fake-AV strip + ffprobe zero audio (ffmpeg skip-guard
  VERIFY); pad-tail marker; AST no-brief-import; cold-import; ascii;
  chains 5-hop talk + 3-hop music; SYNTH_FALLBACKS membership.
- tests/test_ltx_av_driver_wiring.py: dark-lane GOLDEN FIXTURES (full
  request dicts vs tests/fixtures/ltx_av_dark/ JSON); flag-off degrade
  completes + trail origin; force-map guard LOUD-ignores; announcer
  alias only for ltx_av_talk; synthetic slice only for ltx_av_music
  (same shot w/ ltx_video stays None); _render_one template
  pass-through + TypeError guard; ENGINE_FAMILY; canvas tuple; prompt
  gate (music joins, talk sibling, no radio override on talk).
Existing fallout: retry-taxonomy chain sweep gains both chains [file
VERIFY]; pre-code literal search (FAMILIES / ENGINE_FAMILY /
all_engine_names / dropdown / chains) converts exact enumerations to
membership in the same commit; b7 sweep auto-covers the new file (loop
var stays `imp` if edited).
Forgot-it matrix: every touch-list edit -> a named failing test (above).
Byte-identical: CPU = structural only (goldens; hashes unchanged;
canonicalize never emits audio); DEDICATED forced-lane master-hash run =
M4 GPU (OTR_REGRESSION_RUNTIME mechanics). Prune-to-node-7 is the soak
harness, not pytest.
GPU/CPU split: pytest = no network/CUDA/weights/forwards (mock node
maps; no Comfy imports at module scope). M0/M4 scripts own real
renders, NVML, wall time, eyeballs, build skew.
M0 sheet: docs/2026-06-10-ltx-av-lane/M0_RESULTS.md, `key: value` rows;
parser test lands AFTER M0; from M2 a test pins LTX_AV_MAX_FRAMES ==
sheet value. Bug Bible: BUG-070 + BUG-291 pins (lease-release mirror
test); BUG-265 scope verified first; NEW row at ship "LTX dims round
silently; OTR fails loud" under the Three-File Contract.

## Additive touch list (consolidated; tests folded in)

- NEW  nodes/_otr_video_engines/eng_ltx_av.py
- NEW  nodes/_otr_shared/av_dims.py
- NEW  tests/test_av_dims.py, tests/test_video_ltx_av.py,
       tests/test_ltx_av_driver_wiring.py (+ fixtures/ltx_av_dark/)
- EDIT nodes/_otr_video_engines/__init__.py (guarded import)
- EDIT nodes/_otr_video_engines/schemas.py (family + required map)
- EDIT nodes/_otr_video_engines/registry.py (docstrings: family list +
       ShotLock claim)
- EDIT nodes/_otr_shared/role_compat.py (MUSIC_VISUAL += audio_ref)
- EDIT nodes/_otr_video_engines/render_driver.py (deltas a-g + SYNTH_
       FALLBACKS)
- EDIT retry-taxonomy chain sweep + any exact-enumeration tests found
- Docs/tracker + Bug Bible row at ship.

## Claims ledger -- see pass04_plan (cumulative) + p05 deltas

CONFIRMED p05: EngineUsabilityReason pinned at six codes; byte-identical
runtime gate (OTR_REGRESSION_RUNTIME=1 + real render); humo test
patterns (registered_and_dark / role_fit / canonicalize_silent_bt709 /
chain_converges / cold_import / ascii / lease-release) exist to mirror.
UNVERIFIED -> M0/M1: retry-taxonomy sweep filename; ffmpeg skip-guard
pattern; #13111 node class names; announcer portrait object id; BUG-265
scope; cheap_families assert_usable signatures; swap-log formats;
node IO shapes; temporal ceiling; audio formats; resize convention;
portrait dims; prompt cells.

## Milestones

- M0 PROBE: disk inventory; node presence (Desktop + headless);
  pip-freeze sandwich; scratch IA2V render w/ real slice; output-audio
  hash probe; NVML + wall per lane L1 fp8_scaled / L2 GGUF Q4_K_M / L3
  NVFP4; M0_RESULTS.md sheet; P1 matrix -> LIPSYNC | STYLIZED | INERT;
  INERT everywhere closes the lane.
- M1 ADAPTERS (CPU): eng_ltx_av dark + av_dims + schemas/role_compat/
  __init__/registry + driver deltas + the three new test files + golden
  fixtures + fallout edits; suite + Bug Bible green.
- M2 GRAPH + LANE: winning-lane graph; pre-flight; lease; silent
  encode; trim/pad; max-frames constant pinned to sheet (drift test).
- M3 WIRING PROOF: slot asserts; flag-off degrade; force guard;
  announcer alias; identity/manifest tests.
- M4 GATES: full suite + Bug Bible + byte-identical + DEDICATED
  forced-lane master-hash run + live 30-word smoke (flag ON + force
  map); greps (swap-log, manifest engine_id, pad-tail, NVML <= 14.5);
  obs playable AAC only.
- M5 LOOK-QA + DOCS + Bug Bible new row (Three-File Contract).

## Appendix: cut lanes

Yvann-Nodes (p01). New env knobs (p03). ASPECT_CHANGE FailureKind (p04).
Group-prune wiring (p04). New usability reason / GPU pytest / framework
additions / b7 loop-var test (p05).

## Open questions (assigned)

- pass06 HARDWARE: text-encoder artifact for the 2.3 graph (which gemma
  file, size, placement, CPU-offload?); L1 (23.5GB fp8_scaled block-swap)
  vs L2 (GGUF Q4_K_M -- exact file size) vs L3 (NVFP4 dev 21.7GB, cu130,
  issue #11864) decision gate NUMBERS on the 14.5GB ceiling; expected
  per-clip wall time vs the ~6 min 1472x832 LTX opens today (episode
  budget); weight-streaming/block-swap mechanics in ComfyUI native;
  co-residency with FLUX portraits + the AS-3 lease cycle; two-stage
  (base+upscale) cost.
- pass07 PRE-MORTEM: OOM mid-episode; fallback storms; partial
  downloads; restart staleness; zombie VRAM on cancel; slice-cache key;
  caption/credits interplay; Desktop node-lag; golden-fixture rot.
- pass08 FINISHING: convergence + coder-window tickets.
