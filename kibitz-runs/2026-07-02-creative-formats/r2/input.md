# OTR Creative Formats Plan -- F1 Living Evidence Board + F2 Tin-Toy Theatre

Status: r1-hardened (kibitz: claude + codex grounded; agy benched).
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
  notes/photos/string minted per episode, keyed by
  episode_evidence_hash. Same-cast episodes reuse (a), never (b). No
  outpaint dependency; 4K not 8K (crops deliver at 1472x832).
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
  re-run 100% CACHED; post-paste FACE-SIMILARITY check vs the portrait
  chain (not just +/-2px geometry); mux-LAST intact; captions/line
  metadata unaffected.

## 4. F2 build slices (needs S1 + S3 + the ONE pinned 3D row)

- F2-a CONCEPT MINT: `tin_toy_v1` prompt profile (trade-catalog style,
  front + 3/4) on PINNED stills rows (recraft / nano_banana_2), from
  character_description. Mint gate (fail-closed) before any mesh
  credits.
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

V1 (FIRST -- cheap kill-switch, EXPANDED): Kling lipsync on (a) a tin-
toy rendered face AND (b) a photoreal CG render, at REAL shot sizes
(crop + full-frame) -- tests texture-morph AND mouth readability. If
Kling mangles CG faces generally, F2 AND the Prop Shot mouth path die
together; honest fallback = the LOCAL audio-driven mouth lane (HuMo
audio-driven face / ltx_audio_in RECIPE_IA2V) on toy renders, or F2
parks. V2: the chosen multiview->mesh row's export is Blender-
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
