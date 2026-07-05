# Multi-Modal Story Schema -- STAGE 3 SUB-PLAN (v2, post-kibitz r1)

Date: 2026-07-05. Branch: `v2.0-alpha`. Parent: `docs/multimodal-story-schema/BUILD_PLAN.md`
(Stage 3). Predecessors: Stage 1 @36e8b4cb/c8a9be74, Stage 2 @1d06f5c3 + 2C @78bee5d5.
Status: v2 (kibitz r1 folded; r2 coding + r3 wiring w/ Sonnet fan-out + r4 pending).
Arc artifacts: `kibitz-runs/2026-07-05-multimodal-stage3/`.

## 0. Scope + law (narrowed, r1 M2)

`visual_style` rewrites ONLY downstream still/video prompt STYLE LANGUAGE; ledger
content untouched. **JSON owns visual-style DELTAS (tails + declared overrides);
Python owns geometry contracts (headroom/frontal/lip-sync anchors), core prompt
assembly, validation, routing.** Unknown id = hard error, no fallbacks.

v1 SLICE (r1 M3/CUT-3): **tails + allow_radio_tails only.** Subject overrides,
motion-register overrides, and STYLE_ANCHOR styling are NOT in v1 -- they sit on
the Stage-3 checklist (section 8), gated on a geometry-vs-look split design and
full authoring across all packs.

Grounded chokepoints (Explore sweep 2026-07-05, spot-verified):
- `nodes/_otr_story_brief_helpers.py`: ERA_TAIL_DEFAULT :229, STYLE_TAIL_DEFAULT
  :232, IMAGE_GRADE_TAIL :243, RADIO_BROADCAST_TAIL :251; `get_era_tail` :259
  (3 profiles), `compose_still_prompt` :456 (tails at :504/:512/:517),
  `finish_visual_prompt` :524 (style tail :552) = the shared finishing seam.
- `nodes/otr_meta_brief_image_prompt.py`: `build_radio_host_prompt` :297
  (finish+grade :363), `derive_image_prompts` portrait finish :1528-1555,
  `_compose_char_scene_prompt` :1178-1182, `_compose_background_plate_prompt`
  grade append.
- Direct-constant reads OUTSIDE those seams are swept by an AST/grep guard test
  (anchor r1 R3): after 3A, no production module reads STYLE_TAIL_DEFAULT /
  IMAGE_GRADE_TAIL / RADIO_BROADCAST_TAIL / ERA_TAIL_DEFAULT except the style
  module + the byte-identity extraction test.

## 1. Design

- **Packs:** `nodes/visual_styles/<style_id>.json`; header style_id == filename
  (hard error). Lab `sci_fi_radio.json` tails verified byte-identical to the
  live constants.
- **Row schema v1 (exact; unknown key = hard error):** style_id, label,
  positive_tail, image_grade_tail, broadcast_tail (str, may be ""),
  allow_radio_tails (bool), forbidden_terms (list[str]), era_tail (str),
  schema_version. `ledger_directives` + subject/motion/anchor fields are NOT in
  v1 (r1 CUT-1/CUT-3); the archival adaptation STRIPS them (2B pattern).
- **Loader:** `nodes/_otr_visual_styles.py` -- stdlib-only, LAZY (zero
  import-time I/O, test-pinned), sweep (every *.json validates + matches path),
  typed errors (VisualStyleError base + UnknownVisualStyleError,
  VisualStyleValidationError), `_clear_caches()`, `list_style_ids()` (registry
  = the directory, ids sorted deterministically for the dropdown).
- **Threading channel = ledger meta:** writer stamps `meta["visual_style"]`
  (3C); composers resolve `get_visual_style(meta)` =
  meta.get("visual_style", "sci_fi_radio") -> pack, fail-loud on unknown id.
  Default keeps everything byte-identical. VERIFY AT BUILD (anchor R1): every
  visual-prompt composer runs downstream of the meta stamp (portrait minting
  included); if any composer sees meta before the stamp, stamp earlier
  (immediately after `meta` creation next to meta["source_bank"]).
- **era_tail (r1 M4):** the pack string REPLACES only the ERA_TAIL_DEFAULT
  constant; `get_era_tail`'s profile ("full"/"still"/"portrait") logic is
  UNCHANGED Python.
- **forbidden_terms (r1 M1):** NO post-assembly scrub. Load-time lint: the
  pack's own tail fields must not contain its own forbidden terms (hard
  error). Compose-time: WARN-ONLY disposition log when a forbidden term
  appears in a finished prompt (observability; zero mutation; default-off
  noise guard: log once per (style, term, seam) per episode).

## 2. Chunk 3A -- loader + sci_fi_radio + chokepoint routing (byte-identical)

- `_otr_visual_styles.py` + `nodes/visual_styles/sci_fi_radio.json` (tails
  byte-identical extraction).
- Route ALL tail reads through the pack: finish_visual_prompt (:552),
  compose_still_prompt (:504/:512/:517, broadcast gated by
  pack.allow_radio_tails), build_radio_host_prompt (:363), the portrait /
  char-scene / background-plate grade appends. Constants move INTO the style
  module (single owner) or survive only as the extraction-test fixture.
- AST/grep guard test (section 0) pins no stray direct reads.
- Tests: byte-identity MATRIX (r1/anchor R2 -- per seam x era profile x
  role-conditional branch, not one happy path), loader fail-loud matrix,
  lazy-import guard, sweep, _clear_caches, load-time lint.

## 3. Chunk 3B -- author the non-default packs (addressable, DORMANT until 3C)

- `anime`, `cartoon`, `paper_origami`, `archival_documentary` adapted from
  schema-examples, v1 schema only (strip ledger_directives + subject/motion
  fields from archival). allow_radio_tails=false packs: broadcast tail
  appends NOTHING at the compose_still_prompt gate (that IS the behavior
  delta; radio subjects themselves are Python and unchanged in v1).
- Tests: per-pack exact key sets; forced-meta spot tests (stamp
  meta["visual_style"] directly in the test) prove each pack changes the
  finished prompt at every TAIL seam; sci_fi_radio byte-identity holds;
  empty-string tails append nothing (no dangling ", ").
- DORMANT (r1 M6): no production episode can select them until 3C stamps the
  meta; addressability is proven by tests only.

## 4. Chunk 3C (GATED, last) -- selector surface (the 2C playbook)

- `visual_style` widget appended at END of node-1 optional (workflow slot 26,
  default "sci_fi_radio"); choices = list_style_ids(); INPUT_TYPES raises
  LOUD on a broken registry (same deliberate convention exception + comment +
  registration-failure test).
- run() gains `visual_style="sci_fi_radio"` after `source_bank`, before the
  `*` block; `get_style(visual_style)` fail-loud validation FIRST (next to
  require_runnable_bank -- both before any side effect); `_resolve_inputs`
  carries it; stamp `meta["visual_style"]` next to meta["source_bank"].
  (The 2C signature-filtered refine capture auto-carries any new
  positional-default param -- r1 S3 CONFIRMED.)
- Same-commit test updates: guardrails (26->27 + slot-26 pin + registered-id
  cross-check), story_scaffold positional pins (source_bank -> -2,
  visual_style -> -1), openrouter-s2 order test (order[26], len 27),
  api-companions fixtures (+ slot 25 stays source_bank), INPUT_TYPES
  last-optional test, patch_widget_by_name slot-26 test.
- CREATIVE_WHITELIST x2 gains `visual_style`; parity test.
- Workflow JSON same commit; validator + round-trip + widget audit.

## 5. Invariants
Audio spine FROZEN; sci-fi lane byte-identical through 3A/3B at default;
suite + Bug Bible + B7 green per chunk; UTF-8 no BOM; commit AND push per
green chunk (CLAUDE.md section 7); prod/main gated; do NOT clobber the
Sprint-1 radio_object / Sprint-2 still_word prompt work -- still_word
typography/backdrop maps stay Python (operator lettering-consistency
directive 2026-07-04).

## 6. Acceptance
- 3A: fail-loud loader + sweep; byte-identity matrix green; AST guard green;
  zero episode change.
- 3B: 4 packs addressable; forced-meta spot tests show tail deltas at every
  TAIL seam; dormant in production.
- 3C: widget in the real JSON slot 26; meta stamped; gate-first ordering;
  all positional pins updated same commit; whitelists; validator green.

## 7. Verify-at-build
- Meta stamp timing vs first visual-prompt composition (section 1).
- Image dispatcher/cache hash keys include the composed prompt (style change
  => new hash; no stale-cache cross-style reuse) -- r1 S2.
- render_driver :1703-1738 scene-prompt seam uses style_tail=False by design
  (i2v carries the look); confirm the era-tail read there routes through the
  pack without re-adding the style tail.

## 8. Stage-3 checklist (deferred, NOT v1)
- Subject overrides (announcer/music/scene_open), character portrait/scene
  style phrases, motion-register + talking-register overrides: require the
  geometry-vs-look split design (r1 M5) + full authoring across packs.
- still_word typography/backdrop pack ownership (operator directive holds).
- ledger_directives: reintroduce only WITH a consumer.
- forbidden_terms as real enforcement (visual-adaptation subsystem), if ever.
