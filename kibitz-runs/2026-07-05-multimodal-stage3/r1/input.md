# Multi-Modal Story Schema -- STAGE 3 SUB-PLAN (v1 DRAFT, pre-kibitz)

Date: 2026-07-05. Branch: `v2.0-alpha`. Parent: `docs/multimodal-story-schema/BUILD_PLAN.md`
(Stage 3). Predecessors: Stage 1 @36e8b4cb/c8a9be74, Stage 2 @1d06f5c3 + 2C @78bee5d5.
Status: DRAFT -- kibitz arc pending (codex panel + a Sonnet grounded fan-out at r3 wiring).

## 0. Scope + the one big difference from Stage 2

`visual_style` rewrites ONLY downstream still/video prompt language; the ledger
CONTENT is untouched. Unlike story banks (which need an execution lane and ship
`runnable:false`), a visual style is FULLY LIVE the moment its pack exists -- no
run gate needed beyond fail-loud resolution. Law unchanged: JSON owns content,
Python owns validation/routing, unknown id = hard error, NO fallbacks.

Grounded site map (Explore agent, 2026-07-05, verified against the real files):
the hard-coded style language concentrates in
- `nodes/_otr_story_brief_helpers.py` -- ERA_TAIL_DEFAULT :229, STYLE_TAIL_DEFAULT
  :232, IMAGE_GRADE_TAIL :243, RADIO_BROADCAST_TAIL :251, NO_TEXT_CLAUSE :256;
  `get_era_tail` :259, `compose_still_prompt` :456 (appends the 3 tails at
  :504/:512/:517), `finish_visual_prompt` :524 (appends style tail at :552) --
  the SHARED finishing seam.
- `nodes/otr_meta_brief_image_prompt.py` -- STYLE_ANCHOR/_WIDE/_TALKING :92-121,
  radio-host subjects :167-210, `build_radio_host_prompt` :297 (finish + grade
  :363), `derive_image_prompts` portrait finish :1528-1555,
  `_compose_char_scene_prompt` :1178-1182, mesh-fodder/background scaffolds
  :485-503, still_word typography/backdrop maps :629-662.
- Motion registers `_LTX_MOTION_PROMPT_BY_ROLE` (`motion_common.py` :529) +
  render_driver scene-prompt seam :1703-1738.
No visual-style widget exists today; the node-1 `style` widget is NARRATIVE
(feeds `_RADIO_FORM_MAP` keywords only) and stays orthogonal.

## 1. Design (adopting the lab schema, adapted)

- **Packs:** `nodes/visual_styles/<style_id>.json` (path IS the coordinate;
  header `style_id` must match filename, hard error). Adapt
  `schema-examples/visual_styles/*` -- the lab `sci_fi_radio.json` tails are
  ALREADY byte-identical to the live constants (verified: positive_tail ==
  STYLE_TAIL_DEFAULT, image_grade_tail == IMAGE_GRADE_TAIL, broadcast_tail ==
  RADIO_BROADCAST_TAIL, era_tail == ERA_TAIL_DEFAULT).
- **Row schema (exact, unknown key = hard error):** style_id, label,
  positive_tail, image_grade_tail, broadcast_tail (each str, may be ""),
  allow_radio_tails (bool), forbidden_terms (list[str]), era_tail (str),
  schema_version; OPTIONAL subject/motion overrides:
  announcer_visual_subject, music_visual_subject, scene_open_subject,
  character_portrait_style, character_scene_style (str),
  motion_prompts (dict role->str, roles must be within the known role set),
  ledger_directives (dict, scalar values, opaque).
- **Loader:** `nodes/_otr_visual_styles.py` -- stdlib-only, LAZY (zero
  import-time I/O), Stage-2 conventions: sweep (every *.json in the dir must
  validate + match path), typed errors (`VisualStyleError` base +
  `UnknownVisualStyleError`, `VisualStyleValidationError`), `_clear_caches()`
  hook, `list_style_ids()` for the dropdown.
- **Threading channel = the ledger meta** (NOT param threading): the writer
  stamps `meta["visual_style"]`; every composer already receives `meta`.
  `get_visual_style(meta)` resolves meta.get("visual_style", "sci_fi_radio")
  -> pack, fail-loud on unknown id. Default keeps every existing episode +
  test byte-identical.
- **forbidden_terms:** deterministic compose-time SCRUB (not an error) --
  content shaping, not a fallback; scrubs log one disposition line per hit
  (LOUD trail, deterministic output). Empty list = no-op.

## 2. Chunk 3A -- loader + sci_fi_radio pack + chokepoint routing (byte-identical)

- `_otr_visual_styles.py` + `nodes/visual_styles/sci_fi_radio.json` (tails
  extracted byte-identical; sci_fi_radio carries NO subject/motion overrides --
  the current Python constants remain the content for those in 3A).
- Route the THREE chokepoints through the pack:
  S1 `finish_visual_prompt` (:552 style tail; era tail via `get_era_tail`),
  S2 `build_radio_host_prompt` (:363 finish + grade),
  S3 `compose_still_prompt` (:504/:512/:517 style/grade/broadcast tails,
  broadcast gated by pack.allow_radio_tails).
  Also the two scattered direct IMAGE_GRADE_TAIL appends
  (`derive_image_prompts` :1528-1555, `_compose_char_scene_prompt` :1179) --
  all reads go through the pack; the module constants survive ONLY inside
  sci_fi_radio.json extraction tests (byte-identity pins).
- Tests: byte-identity (default meta -> every composer output unchanged),
  loader fail-loud matrix, lazy-import guard, sweep, `_clear_caches`.

## 3. Chunk 3B -- style packs live (anime / cartoon / paper_origami / archival_documentary)

- Author the 4 packs from schema-examples (adapted to production role names).
- forbidden_terms scrub wired at the compose seams (post-assembly, pre-return).
- Subject overrides consumed where provided: announcer/music/scene_open
  subjects (SEAM into `get_open_subject` + radio-host subject pick, gated by
  allow_radio_tails=false paths), character portrait/scene style phrases
  (STYLE_ANCHOR family substitution), motion_prompts override
  `_LTX_MOTION_PROMPT_BY_ROLE` per role when present.
- Tests: per-pack exact key sets; a non-default style changes the composed
  prompt (spot prompts per seam); sci_fi_radio stays byte-identical; scrub
  determinism; unknown role in motion_prompts = hard error.

## 4. Chunk 3C (GATED, last) -- selector surface

Reuse the 2C playbook wholesale:
- `visual_style` widget appended at END of node-1 optional (workflow slot 26,
  default "sci_fi_radio"), choices live from `list_style_ids()`, INPUT_TYPES
  raises LOUD on a broken registry (same deliberate convention exception).
- run() gains `visual_style="sci_fi_radio"` before the `*` block (the fixed
  signature-filtered refine capture auto-carries it); resolve/validate the id
  FIRST (fail-loud `get_style(visual_style)` next to require_runnable_bank);
  `_resolve_inputs` carries it; stamp `meta["visual_style"]` (the threading
  channel -- downstream needs no new params).
- Same-commit test updates: guardrails (26->27, slot 26 pin + registered-id
  cross-check), story-scaffold positional test (-2 -> -3 shift... update the
  order pins), openrouter-s2 order test, api-companions fixtures, INPUT_TYPES
  last-optional test, patch test; `source_bank` keeps slot 25.
- CREATIVE_WHITELIST x2 gains `visual_style`.
- Workflow JSON same commit + validator + round-trip + widget audit.

## 5. Invariants
Audio spine FROZEN (visual-only changes; test_audio_byte_identical green);
sci-fi lane byte-identical through 3A/3B at the default style; suite + Bug
Bible + B7 green per chunk; UTF-8 no BOM; commit AND push per green chunk
(CLAUDE.md section 7); prod/main gated; do not clobber the Sprint-1
radio_object + Sprint-2 still_word prompt work (BUILD_PLAN Stage-3 note) --
still_word typography/backdrop maps stay Python in Stage 3 (per-episode
lettering-consistency directive 2026-07-04; checklist item for later).

## 6. Acceptance
- 3A: packs load fail-loud; default output byte-identical across every
  composer (pinned); zero episode change.
- 3B: 4 non-default styles produce changed prompts at every routed seam;
  scrub deterministic + logged; sci-fi untouched.
- 3C: widget in the real JSON slot 26; meta stamped; validator green; all
  positional pins updated same commit; whitelists updated.

## 7. Open questions (for the kibitz arc)
- Q1: era_tail profile variants ("full"/"still"/"portrait" in get_era_tail) --
  one pack era_tail string vs per-profile fields?
- Q2: STYLE_ANCHOR family (portrait framing) -- pack-owned in 3B or deferred?
  Framing is partly a GEOMETRY contract (headroom for HuMo/talking) not just
  style language -- splitting look-words from geometry-words may be needed.
- Q3: motion_prompts override granularity -- whole-register replacement per
  role only, or also the talking (ia2v) register?
- Q4: should ledger_directives be consumed anywhere in Stage 3, or stay
  opaque metadata until a consumer exists (lean = opaque)?
