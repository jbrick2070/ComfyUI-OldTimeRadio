# OTR Creative Formats Plan -- F1 Living Evidence Board + F2 Tin-Toy Theatre

Status: r4-CONVERGED (kibitz full arc r1-r4: claude + codex grounded
panels; agy benched after a hang; judgments in
kibitz-runs/2026-07-02-creative-formats/). BUILD-READY pending
prerequisites.
DOCS ONLY -- the coder window currently holds the code baton (LTX
fixes); nothing here builds until the cloud-lane sprints land and the
operator says go. Operator adoption decision 2026-07-02 (ideas #2 + #1
of docs/2026-07-02-cloud-engines/roundtable/ideas_synthesis.md).

## 0. What these are

Two SHOW FORMATS built on the cloud engine lanes (build doc:
docs/2026-07-02-cloud-engines/roundtable/pass04_plan.md; S0 chunks 1-3
shipped @ 5a79a926/7d8c490f/f3a97ea6):

- F1 LIVING EVIDENCE BOARD: a 4K 1940s corkboard (suspect polaroids,
  red string, pinned clues), locally composited. The camera prowls it
  locally; whichever photo the camera finds MOUTHS ITS OWN LINES via
  kling lipsync on the cropped speaker region. Doubles as a complete
  low-budget episode format. Identity locked (base photos never
  change).
- F2 TIN-TOY THEATRE: character beats performed by 1940s tin-toy
  automatons -- STATIC mesh minted per character (re-mint only when
  the character design changes: key = portrait hash + tin_toy profile
  version), rendered in Blender (turntable/dolly -- same no-rig
  pattern as the Prop Shot POC), mouths lip-synced by kling on the
  rendered face. The tin-toy aesthetic designs OUT the uncanny valley.
  Designated character-beat STYLE of the 3D POC line.

## 1. Ground rules established at r1

- TOKEN MAPPING CONTRACT: descriptor tokens (base_clip_ref/audio_ref)
  are role-compat CAPABILITY tokens. Every cloud adapter maps tokens ->
  PINNED field names from partner_nodes.yaml at invoke time
  (kling_lipsync REAL inputs: `video` VIDEO, `audio` AUDIO,
  `voice_language` COMBO). Stated once here; applies everywhere.
- NO UNPINNED DEPENDENCIES: being in the 214-node dump is NOT pinned.
  Any new row (Meshy rig/animate, Ideogram, FluxProFillNode, LTX
  anything) goes through the FULL S0 pipeline (pin script + drift test
  + pricing + ToS) before any plan step may name it. MVP below uses
  ONLY currently-pinned rows + local work.
- FORMAT = ENGINE: formats register as local zero-cloud-cost VIDEO
  ENGINE ROWS (`fmt_evidence_board` in
  nodes/_otr_video_engines/eng_evidence_board.py; `fmt_tin_toy` in
  eng_tin_toy.py). ShotLock/ledger/reactivity policy see every shot --
  no bypass; board pans classify + stamp as format-local shots. UX: one
  `visual_format` widget (standard|evidence_board|tin_toy) APPENDED at
  the END of OTR_VideoDirector's optional widgets (BUG-LOCAL-097
  append-only rule; JSON change in the SAME commit), flipping all three
  per-role defaults to the format row; headless `OTR_VISUAL_FORMAT`;
  explicit per-role picks still win (universal-slot rule).
- LOCAL 3D SCAFFOLD: F2 SUPERSEDES the parked eng_character_3d lane
  (triposg/hunyuan3d/trellis + ARKit/Rhubarb) for character
  presentation. F2 never touches that code; its removal continues
  under open item E1. No coexisting talking-3D paths.
- ASSET PLACEMENT (operator directive): EVERY asset an episode uses
  materializes under `otr\episodes\<ep>\` (format subdirs:
  `evidence_board\`, `tin_toy\`) -- boards, layers, crops, lipsynced
  clips, concept sheets, mesh GLBs, Blender plates, canonicalized
  media. Cross-episode reusables are COPIED into each consuming
  episode (GB-scale mesh duplication = ACCEPTED tradeoff; visibility >
  disk). Only the final published mp4 lives in `otr\obs\`. The global
  cloud cache is billing-dedup plumbing ONLY: nothing serves from it;
  losing it costs money, never episode integrity.

## 1b. Coding contracts (r2-hardened; line-cited against the repo)

- REGISTRATION CHECKLIST per format engine: new family value
  `format_composite`; honest `required_inputs`; `@register`; the
  import line in `_otr_video_engines/__init__.py`; CAPABILITIES row
  (cpu_ok True, vram 0 -- the ENGINE is local; its cloud spend is
  governed per-invoke by the S0 budget machinery); `ENGINE_FAMILY`
  entry in render_driver; capability-set test extension
  (tests/test_capability_profiles.py pattern).
- FORMAT CONTEXT: a planning/stamping phase (ShotLock /
  ImageGenDispatcher) writes format assets + the BOARD MANIFEST
  (`otr/episodes/<ep>/evidence_board/board_manifest.json`:
  cast[{char_id, portrait_hash, x,y,w,h}], layers, z_order,
  layout_seed) BEFORE render. A versioned OPTIONAL `format_ctx` block
  is added to the VideoRequest SCHEMA (episode_dir, manifest path,
  lines[]{speaker,start_s,end_s,audio_path}) -- extras stay forbidden;
  context is schema'd, never smuggled. Engines write STRAIGHT to
  canonical episode paths (no tmp staging).
- visual_format SEMANTICS (r4-corrected): !=standard sets all three
  role-slot DEFAULTS to the format row. Explicitness is measured
  against the EFFECTIVE default source -- the active profile's patched
  role value when a profile applied (profiles patch role widgets
  directly, e.g. 16gb_full sets viz_green/humo_14B_169), else the
  registry default. Widget value == effective default => inheritable
  (format overrides it); != => explicit, preserved. Precedence:
  explicit pick > visual_format (widget/env) > profile default >
  registry default. Build test: 16gb_full + visual_format=
  evidence_board overrides all inheritable role values while an
  explicit per-role override still wins. Append-only widget +
  widget-vector tests.
- CLOUD FROM ENGINES: only via the S0 invoke bridge inside
  render_clip (may block for the async poll; errors classified into
  the render_driver retry taxonomy). render_clip RETURNS one
  composited mp4 path; per-line lipsync happens inside render_clip.
- CACHE KEYS: mesh = eng_mesh_stage pattern (subject id + portrait
  SHA + 3D row id + adapter/export version + tin_toy profile
  version); Blender plate = (mesh_hash, camera_preset_or_path_hash,
  frame_count); duration derived. SEPIA/polaroid styling = LOCAL PIL
  post-processing on the RAW portrait (portrait_hash preserved,
  face-similarity comparable) -- never a re-minted sepia still.
- KLING LATENCY: provider concurrency defaults to 1 -> per-line jobs
  serialize. MVP: lipsync CLOSE-UP lines only (board wides/pans are
  format-local mute shots, stamped); per-episode Kling call count in
  the estimate report; raising concurrency = ToS/pricing verify item.
- FACE-SIMILARITY: reuse the portrait_ledger machinery; on failure =
  LOUD ledger stamp + that line stays still (no paste), never a bad
  paste. BLENDER: version gate (>= 4.5) in assert_usable; per-plate
  render manifest; timeout/corrupt_output per the S0 error taxonomy.

## 1c. Wiring contracts (r3-hardened; line-cited)

- `format_composite` is ADDED to the closed FAMILIES tuple +
  FAMILY_REQUIRED_INPUTS (import-time guard) in the SAME change as the
  engines; ENGINE_FAMILY (render_driver) gains both fmt entries.
- `format_ctx: Optional[FormatContext] = None` declared on
  VideoRequest (extra=forbid respected); FormatContext = _Forbid
  sub-model; no VideoRequest validator reads it. Manifest path stamps
  into the patched ledger BEFORE the video batch node;
  build_request_from_shot copies it into fmt-engine requests.
- fmt rows register ALWAYS; their assert_usable fails CLOSED (named
  error citing visual_format) when format_ctx is absent. Explicit
  per-role fmt picks are honored when the context exists.
- Explicit-vs-default: direct() compares each role widget value to the
  registry default at resolution time (== default => inheritable;
  != => explicit). Edge documented: picking the default value reads
  as inheritable.
- visual_format appends AFTER gate_in (forceInput slot undisturbed);
  widgets_values grows by exactly 1 at the end; saved-workflow
  round-trip test required. Headless `OTR_VISUAL_FORMAT` is read at
  the SAME direct() resolution point (one resolution point, no
  double-apply by the profile applier).
- Bridge converts canonical mp4/wav PATHS -> Comfy VIDEO/AUDIO objects
  (payload test against the pinned cloud_kling_lipsync schema);
  normalize the existing base_clip_ref {"path": ...} shape drift in
  S3. Kling output is stripped (-an): fmt render_clip returns SILENT
  video at the target frame count and obeys the standard post-render
  contract (canonicalize runs after, as for any engine). Per-line
  lipsync failures are handled INSIDE render_clip (stay-still line +
  LOUD stamp); driver-level fallback is not the inner retry path.
- V1 kill-switch DECOUPLED from the 3D pin: runs with a CHECKED-IN
  fixture GLB as soon as the S3 kling adapter exists.
- Headless env preflight in every format acceptance
  (OTR_ENABLE_COMFY_CLOUD_MEDIA, credentials, budget, Kling
  concurrency) + launcher/soak env wiring.
- Board manifest: top-left integer pixel coords on the 4K canvas,
  rounding + paste-scaling declared; layout_seed ->
  random.Random(layout_seed). Estimate report gains per-line lipsync
  rows (cached vs billed visible). Mesh key: tin_toy profile changes
  ride the concept-sheet content hash already in the key.
- r4 FINAL CONTRACTS: (a) THE kling row for BOTH formats is
  `cloud_kling_lipsync` ONLY -- its adapter receives VIDEO, never
  IMAGE (the sibling cloud_kling_avatar takes image+sound_file and
  GENERATES a face rather than syncing the given one; recorded as a
  V1-failure alternative probe, not the default). (b) fmt rows declare
  `required_inputs = ()` and `FAMILY_REQUIRED_INPUTS["format_composite"]
  = ()` -- format_ctx is a SCHEMA field checked by assert_usable, NOT a
  role-compat token (unknown tokens fail closed). (c)
  `FormatContext.lines[]` carries `line_id` + canonical `char_id` (+
  start_s/end_s/audio_path); crops join by char_id, never display
  speaker strings. (d) FormatContext carries `format_ctx_version`;
  assert_usable distinguishes ABSENT (stamping never ran) from STALE
  (version mismatch) -- both fail closed with distinct named errors.
  (e) Existing VideoRequest validators are family-gated and do not
  read format_ctx; NO new validator is added for it. (f) Cast
  polaroids are RAW portraits + LOCAL PIL sepia/border only (no
  prompt-tail re-mint; portrait_hash preserved). (g) Golden-30s
  samples live in tests/goldens/formats/ behind OTR_RUN_CLOUD_SMOKE=1;
  the F1 episode acceptance entrypoint is
  scripts/run_otr_30word_smoke.py in board format, with Test-Path
  canonical-asset checks per repo sec-6. (h) board_manifest.json
  sha stamps into the production ledger (write-once at image phase).
  (i) ADDITIVE SAFETY: format work changes no existing engine, schema
  default, or workflow JSON value until the fmt sprints; the local
  byte-identical baseline holds throughout. (j) F1's true gate
  everywhere: S1 + the cloud_kling_lipsync ADAPTER existing (not full
  S3 matrix acceptance). (k) 1472x832 = the standard landscape render
  canvas (OTR_VIDEO_LANDSCAPE_CANVAS).

## 2. Prerequisites (hard ordering)

1. Cloud S0 remainder (invoke bridge + smokes) -- [ASSUMPTION] neither
   format builds if this stalls.
2. Cloud S1 STILLS lane (+ `portrait_mint_3d` and `tin_toy_v1` prompt
   profiles -- both run on PINNED stills rows: recraft /
   nano_banana_2).
3. Cloud S3 VIDEO lane (kling rows live).
4. F2 additionally: ONE new 3D adapter -- image->multiview->mesh
   (Tripo multiview or MeshyMultiImageToModelNode), PINNED first via
   the S0 pipeline. NO rig/animate in MVP (future-lane, per POC).

## 3. F1 build slices (needs S1 + kling row only)

- F1-a BOARD MINT (two cached layers): (a) CAST LAYER -- polaroid
  stills per cast member (portrait-hash reuse, sepia/polaroid prompt
  tail) + cork backdrop still, composited LOCALLY onto a 4K canvas;
  keyed by the portrait-hash set. (b) EPISODE DRESSING LAYER -- clue
  notes/photos/string minted per episode; NOT cached in MVP
  (regenerate; negligible cost vs Kling lines). Same-cast episodes
  reuse (a). Deterministic z-order via layout_seed in the manifest.
  No outpaint dependency; 4K not 8K (crops deliver at 1472x832).
- F1-b CAMERA DESK: lives in eng_evidence_board.render_clip -- LOCAL
  pan/crop viewport over the 4K board (ffmpeg/PIL; no GPU), camera
  path per beat, cuts on line boundaries from ledger line timings.
- F1-c SPEAKER ANIMATION: per line -- crop speaker polaroid (WE own
  crop coords) -> LOCAL still->silent-clip (ffmpeg loop, role fps) ->
  kling lipsync (video/audio/voice_language per pinned schema) ->
  paste back at exact coords -> composite into the pan. Only the
  speaking crop ever hits the video API.
- F1-d FORMAT SWITCH: per sec 1 FORMAT=ENGINE + visual_format widget.
- Acceptance: GOLDEN 30-SECOND SAMPLE first; then a full episode via
  the 30w smoke script in board format; zero non-kling video spend;
  re-run caching verified as a smoke ASSERTION (guaranteed by the S0
  RequestCacheKey by construction, not a separate gate); post-paste
  FACE-SIMILARITY check vs the portrait
  chain (not just +/-2px geometry); mux-LAST intact; captions/line
  metadata unaffected.

## 4. F2 build slices (needs S1 + S3 + the ONE pinned 3D row)

- F2-a CONCEPT MINT: `tin_toy_v1` prompt profile (trade-catalog
  style) on PINNED stills rows (recraft / nano_banana_2), from
  character_description -- front and 3/4 minted as SEPARATE
  generations (the multi-image mesher consumes multiple stills;
  single-sheet multiview generation is NOT assumed). Mint gate
  (fail-closed) before any mesh credits.
- F2-b MESH MINT: concept sheet -> multiview -> STATIC mesh (the one
  new pinned row). Cached globally keyed by portrait hash + tin_toy
  profile version; COPIED into each consuming episode per the
  placement directive. Re-mint only on character-design change.
- F2-c STAGE RENDER: LOCAL Blender 4.5.10 (shipped + selftested)
  renders per-beat plates: static toy at its mark, turntable/dolly
  camera, noir key light. Cached per (mesh, camera, duration).
  Lives in eng_tin_toy.render_clip.
- F2-d MOUTH PASS: kling lipsync on the rendered face region per line
  (crop->still-clip->sync->paste discipline as F1-c; full-frame when
  the toy IS the shot).
- F2-e FORMAT SWITCH: per sec 1. CHARACTER BEATS ONLY in MVP --
  whole-episode tin-toy mode deferred until the look is proven.
- Acceptance: GOLDEN 30-SECOND SAMPLE (one character: mint -> mesh ->
  plate -> mouth); then a 3-beat segment; design-change re-mint test;
  re-run = zero mesh re-billing.

## 5. Verify probes (ordered; before committing either aesthetic)

V1 (FIRST -- cheap kill-switch): Kling lipsync on a tin-toy rendered
face at REAL shot sizes (crop + full-frame), INCLUDING acceptance of a
still-frame silent input clip -- tests still-video acceptance,
texture-morph, and mouth readability in one probe. (The photoreal-CG
question belongs to the Prop Shot mouth gate, not this format.) If V1
fails: fallback is an OPERATOR DECISION to route toy renders through
the LOCAL audio-driven mouth lane (HuMo / ltx_audio_in RECIPE_IA2V --
a different engine choice, not an in-engine auto-fallback; it changes
the zero-VRAM posture), or F2 parks. V2: the chosen multiview->mesh row's export is Blender-
importable (GLB) -- probe at pin time. V3: 4K local board composite --
seam/perspective sanity of pasted polaroids (no generative stitch in
MVP). V4: Kling output framing vs crop/paste round-trip -- mouth lands
within +/-2px at paste coords AND passes the face-similarity check.

## 6. Cost posture [ASSUMPTION until S0 pricing stamps rows]

F1: image credits at cast-layer mint (amortized until cast changes) +
episode dressing layer + tiny kling crops per line. Cheapest format.
F2: concept stills + ONE mesh mint per character (+ kling per line);
Blender renders free. Both ride the S0 budget guard / billing cache /
ledger; the estimate report gains format-specific rows.

## 7. Sequencing

Coder window (LTX fixes) holds the code baton NOW. Cloud S0 remainder
-> S1 -> S3 land first. Then F1 (mostly local compositor code + one
format engine), then F2 (one pinned 3D row + format engine). Workflow
JSON changes ride their lane sprints (visual_format widget appends in
the SAME change as the format-engine code; validator + widget audit
rerun).
