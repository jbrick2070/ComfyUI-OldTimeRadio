# Multi-Modal Story Schema -- STAGE 3 SUB-PLAN (v4, post-kibitz r3 -- codex + Sonnet 3-lens fan-out)

Date: 2026-07-05. Branch: `v2.0-alpha`. Parent: `docs/multimodal-story-schema/BUILD_PLAN.md`
(Stage 3). Predecessors: Stage 1 @36e8b4cb/c8a9be74, Stage 2 @1d06f5c3 + 2C @78bee5d5.
Status: v4 (kibitz r1+r2+r3 folded; r3 ran codex + a 3-lens Sonnet grounded
fan-out -- all four reviewers independently converged on the mesh_fodder seam;
r4 convergence pending). Full judgment: kibitz-runs/2026-07-05-multimodal-stage3/r3/final.md.
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
- **forbidden_terms (r1 M1 + r2 M3/CUT-1):** NO post-assembly scrub AND no
  compose-time warn in v1 (per-episode/seam warn state invites resident-server
  global-state bugs). v1 = LOAD-TIME LINT ONLY: a pack's own tail fields must
  not contain its own forbidden terms (hard error). The field stays in the
  schema as declared data for the future visual-adaptation subsystem.
- **Schema hygiene (r2 S2/S3/OPT):** schema_version pinned to a
  KNOWN_STYLE_SCHEMA_VERSIONS set = {"v1"} (packs adapted from the lab's
  "v2.0" are re-stamped "v1"); style_id must match `^[a-z0-9_]+$` AND the
  filename; loaded rows surface as a frozen dataclass (attribute access, not
  raw dict keys).
- **NO SWALLOWED STYLE ERRORS (r2 M1 -- the load-bearing wiring rule):** the
  existing composer seams wrap finish/grade work in `except Exception: pass`
  (grounded: otr_meta_brief_image_prompt.py :352-367, :1172-1183, :1258-1269,
  :1532-1555). Once tails are pack-routed, those catches would silently ship
  unstyled prompts on UnknownVisualStyleError -- a hidden fallback. 3A
  restructures every such site: the flat-import shim may catch ImportError
  ONLY; VisualStyleError (and any style-resolution error) PROPAGATES. Add an
  AST pin: no visual-prompt seam wraps a style-module call in a bare
  except-Exception.
- **get_era_tail contract (r2 S1):** stays fail-soft internally; style
  resolution happens ONCE per composer entry via fail-loud
  `get_visual_style(meta)`, and the resolved pack (or its era_tail) is passed
  in -- get_era_tail itself never raises except on an explicitly invalid
  resolved style object. Doc/tests updated accordingly.

## 2. Chunk 3A -- loader + sci_fi_radio + chokepoint routing (byte-identical)

- `_otr_visual_styles.py` + `nodes/visual_styles/sci_fi_radio.json` (tails
  byte-identical extraction).
- Route ALL tail reads through the pack: finish_visual_prompt (:552),
  compose_still_prompt (:504/:512/:517, broadcast gated by
  pack.allow_radio_tails), build_radio_host_prompt (:363), the portrait /
  char-scene / background-plate grade appends, `compose_still_word_prompt`
  (otr_meta_brief_image_prompt.py :808-865 -- era tail + IMAGE_GRADE_TAIL;
  r2 M2; typography/backdrop maps stay Python per section 8), AND
  `_compose_mesh_fodder_prompt` (:1223-1244 -- get_era_tail inside a bare
  except; r3, found independently by all 4 reviewers).
- CONSTANT OWNERSHIP (r3): the 4 tail constants REMAIN as literal
  definitions in _otr_story_brief_helpers.py = the EXTRACTION FIXTURE
  (lazy-safe, no import-time pack I/O); the extraction test pins
  sci_fi_radio.json == constants byte-for-byte; PRODUCTION reads route
  through the pack; the AST guard bans production reads outside the style
  module + the definitions themselves + tests/. The 5 existing tail-pin test
  files (test_still_spine_helpers, test_brief_prompt_finishing,
  test_talking_portrait_s4b, test_video_platform_aseam, test_era_literals_c2a)
  keep importing the constants UNCHANGED -- they double as the byte-identity
  matrix.
- HELPER SIGNATURES (r3 S1): finish_visual_prompt / compose_still_prompt /
  get_era_tail gain `style=None` (None => fail-loud get_visual_style(meta)
  internally); multi-helper composers resolve ONCE and pass down;
  render_driver's style_tail=False path pinned by test.
- De-swallow the seams (r2 M1 + r3): ImportError-only shims stay; style
  errors propagate at ALL SIX sites (:352-367, :1172-1183, :1234-1243
  mesh_fodder, :1258-1269, :1532-1555, + any new). AST pin distinguishes the
  INNER ImportError shim (benign) from the OUTER except-Exception (banned
  around style calls), matches bare `get_era_tail(` too, and carries a
  positive list of confirmed-unwrapped call sites (still_word caller
  :1716-1718, render_driver :1731) so future edits can't silently wrap them.
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
- Same-commit test updates (r3-completed list): guardrails :634-745 (len 27
  + slot-26 block + registered-id cross-check); test_story_scaffold_toggle
  (story_scaffold -> -3, source_bank -> -2, visual_style -> -1);
  test_source_bank_widget_2c TWO pins (:61-64 order[-1] test -- RENAME it --
  and the :312-313 patch test len 26->27, slot 25 unchanged);
  openrouter-s2 (order[26], len 27); api-companions fixture + its MOCK
  INPUT_TYPES schema gains visual_style; new INPUT_TYPES last-optional +
  patch_widget_by_name slot-26 tests; gate-order test for
  get_style(visual_style) beside require_runnable_bank (2C sentinel
  pattern); forced-meta mesh_fodder test. Whitelist parity test
  (test_workflow_apply :258-261) is the tripwire -- no edit, enforcement.
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
- Meta stamp timing: RESOLVED (grounded 2026-07-05) -- all visual composers
  run in downstream nodes off the serialized ledger (OTR_MetaBriefImagePrompt
  .generate reads meta from script_json :1867; render_driver reads ledger
  meta), so the writer's meta stamp always precedes them.
- Image cache/hash: CONFIRMED SAFE for the portrait/still lane (r3 sonnet-3:
  _content_hash of the FINISHED prompt, stamp :1558; ShotLock request_hash is
  content-keyed, prompt_hash separate) -- residual: spot-check
  otr_image_gen_dispatcher's own cache-key uses the post-finish prompt text.
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
