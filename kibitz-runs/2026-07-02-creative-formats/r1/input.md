# OTR Creative Formats Plan -- F1 Living Evidence Board + F2 Tin-Toy Theatre

Status: pass00 for kibitz. DOCS ONLY -- the coder window currently holds
the code baton (LTX fixes); nothing here builds until the cloud-lane
sprints land and the operator says go. Operator adoption decision
2026-07-02: ideas #2 and #1 from the judged ideation synthesis
(docs/2026-07-02-cloud-engines/roundtable/ideas_synthesis.md).

## 0. What these are

Two SHOW FORMATS built on the cloud engine lanes (build doc:
docs/2026-07-02-cloud-engines/roundtable/pass04_plan.md, S0 chunks 1-3
shipped @ 5a79a926/7d8c490f/f3a97ea6):

- F1 LIVING EVIDENCE BOARD: an 8K 1940s corkboard (suspect polaroids,
  red string, pinned clues). The camera prowls it locally; whichever
  photo the camera finds MOUTHS ITS OWN LINES via kling_lipsync on the
  cropped speaker region. Doubles as a complete low-budget episode
  format. Identity is mathematically locked (base photos never change).
- F2 TIN-TOY THEATRE: character beats performed by 1940s tin-toy
  automatons -- mesh-minted once per character per season, rendered in
  Blender under noir lighting, mouths lip-synced by kling_lipsync on
  the rendered face. The tin-toy aesthetic DESIGNS OUT the uncanny
  valley (artifacts read as charm). This is the designated character-
  beat STYLE of the 3D POC line ("The Prop Shot" character variant).

## 1. Prerequisites (hard ordering)

1. Cloud S0 remainder: invoke_partner_node async bridge + smokes #1/#2
   (next coder window; operator env prereqs in GO_FORWARD_PLAN).
2. Cloud S1 STILLS lane -- both formats consume it. F2 additionally
   needs the `portrait_mint_3d` profile (full-frame neutral pose, clean
   backdrop, mint gate before mesh credits) already specced in
   pass04_plan Appendix A.
3. Cloud S3 VIDEO lane -- kling_lipsync row live (pinned:
   KlingLipSyncAudioToVideoNode, lipsync_overlay, base_clip_ref +
   audio_ref).
4. F2 needs ONE new 3D adapter chain (the only net-new engine work in
   this plan): image->multiview->mesh (Tripo multiview or
   MeshyMultiImageToModelNode) + MeshyRigModelNode +
   MeshyAnimateModelNode (all verified present in the live 214-node
   dump) behind the same Surface-A invoke/billing/cache machinery.

## 2. F1 build slices (cheapest first -- needs S1 + kling row only)

- F1-a BOARD MINT: per-episode board assembly. Cast polaroids from the
  existing portrait chain (portrait-hash reuse; sepia/polaroid prompt
  tail) -> Flux fill/expand grows the corkboard field -> LTX 2.3
  outpaint cloud template stitches to ~8K. Board asset cached in the
  GLOBAL cloud cache keyed like any other request (board prompt + cast
  portrait hashes); re-billed only when the cast changes.
- F1-b CAMERA DESK: LOCAL pan/crop viewport renderer (pure
  ffmpeg/PIL/compositor work, no GPU): per-beat camera path over the
  board, cuts on line boundaries from the ledger's line timings.
- F1-c SPEAKER ANIMATION: for each line, crop the active speaker's
  polaroid region (WE control crop coords) -> kling_lipsync with the
  line's audio slice -> paste back at exact coords -> composite into
  the pan. Only the speaking crop ever hits the video API: the
  cheapest possible per-line video spend.
- F1-d FORMAT SWITCH: `evidence_board` becomes a selectable episode
  VISUAL FORMAT (widget on the VideoDirector policy or an episode-
  level toggle -- decide at wiring round): when on, all three video
  roles resolve to the board pipeline for that episode.
- Acceptance: full episode in board format, zero non-kling video
  spend, re-run 100% CACHED board + lipsync crops, mux-LAST intact,
  captions/line metadata unaffected.

## 3. F2 build slices (needs S1 + S3 + the 3D chain)

- F2-a CONCEPT MINT: Ideogram (IdeogramV4 pinned-family) generates the
  tin-toy character sheet from character_description via a
  `tin_toy_v1` prompt profile (trade-catalog style, front + 3/4).
  Mint gate (same fail-closed pattern as portrait_mint_3d).
- F2-b MESH MINT: concept sheet -> multiview -> mesh
  (Tripo/MeshyMultiImage) -> MeshyRigModelNode -> MeshyAnimateModelNode
  idle preset. One mint per character PER SEASON; mesh + rig cached
  globally keyed by character portrait hash + tin_toy profile version.
- F2-c STAGE RENDER: LOCAL Blender 4.5.10 (shipped + selftested, 0-E
  Phase A) renders per-beat plates: idle-animated toy at its mark,
  noir key light, camera per beat class. CPU/GPU-cheap, cached per
  (mesh, camera, duration) locally.
- F2-d MOUTH PASS: kling_lipsync on the rendered face region per line
  (same crop->sync->paste discipline as F1-c where the face is small
  in frame; full-frame lipsync when the toy is the shot).
- F2-e FORMAT SWITCH: `tin_toy` selectable for character beats (or the
  whole episode), same switch mechanism as F1-d.
- Acceptance: one character scene end-to-end (mint -> mesh -> render
  -> mouth) + a 3-beat episode segment; season re-run = zero mesh
  re-billing.

## 4. Verify probes (before committing either aesthetic)

V1 (F2, FIRST -- cheap kill-switch): one Kling lipsync clip on a
rendered tin face -- does Kling force human skin texture onto painted
metal? If yes at default settings, probe style/strength params; if
unfixable, F2 falls back to the Prop Shot photoreal character variant.
V2 (F2): MeshyAnimateModelNode animation inventory -- confirm an
idle/sway preset exists and loops cleanly.
V3 (F1): LTX outpaint stitch quality at 8K board scale (seams,
perspective coherence of pinned items).
V4 (F1): Kling output framing vs crop/paste round-trip -- mouth must
land back on the polaroid within +/-2px at paste coords.

## 5. Invariants (unchanged, guarded) + ASSET PLACEMENT DIRECTIVE

Master audio frozen, mux LAST (all Kling output audio stripped --
must_strip_audio=True per the descriptor table). Per-line granularity
preserved (captions + delivery vectors). Portrait-hash identity chain
is the SOURCE for both formats' character imagery. Fail-closed
everywhere: un-mintable concept/portrait = stay 2D/default format for
that beat, LOUD ledger note. Reactivity policy untouched: both formats
are lipsync_overlay-driven = reactive by construction; wide/mute board
pans are classified and stamped like any auto-mute selection.

ASSET PLACEMENT (operator directive 2026-07-02): EVERY asset an
episode uses -- cloud or local -- MATERIALIZES under
`otr\episodes\<ep>\` (subdirs per format, e.g. `evidence_board\`,
`tin_toy\`): board 8K + per-episode dressing layer, polaroid crops and
their lipsynced clips, tin-toy concept sheets, mesh GLBs + rig/anim
exports, Blender plates, canonicalized audio/stills/video -- ALL of
it, visible, never hidden. Assets reused across episodes (meshes,
cast-polaroid layer) are COPIED into every episode that uses them.
The ONLY asset that lives elsewhere is the final published mp4 ->
`otr\obs\` (obs_publish, unchanged). The global cloud cache
(`otr\cache\cloud_media\`) is BILLING-DEDUP PLUMBING ONLY -- it is
never the serving location, nothing renders from it, and losing it
must cost money only, never episode integrity.

## 6. Cost posture

F1: image credits at board mint (amortized per cast change) + tiny
kling crops per line. Cheapest format in the catalog. F2: 3D mint per
character per season (Tripo/Meshy + rig/animate) + kling per line;
Blender renders free. Both ride the S0 budget guard / billing cache /
ledger; per-episode estimate report includes format-specific rows.

## 7. Sequencing vs the rest of the repo

Coder window (LTX fixes) holds the code baton NOW. Cloud S0 remainder
-> S1 -> S3 land first (separate windows, GO_FORWARD order). Then
F1 (small: b/c/d are mostly local compositor code + one template
lane), then F2 (the 3D chain). Workflow JSON changes only with their
lane sprints per the standing rule; format switches enter the JSON in
the SAME change as their code, validator + widget audit rerun.
