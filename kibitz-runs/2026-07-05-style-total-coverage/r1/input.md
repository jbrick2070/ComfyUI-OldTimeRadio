# Visual-Style TOTAL COVERAGE -- Stage-3 section-8 slice (DRAFT v1, pre-kibitz)

Date: 2026-07-05. Branch: `v2.0-alpha`. Parent: `docs/multimodal-story-schema/STAGE3_SUBPLAN.md`
section 8 (PROMOTED from deferred by operator directive 2026-07-05: "when a visual_style is
selected it must impact ALL downstream prompts -- stills, video, 3D, every promptable surface,
announcer_visual + music_visual included"). Status: DRAFT -- kibitz r1..r4 pending.

## 0. Problem (grounded fan-out audit 2026-07-05)

The v1 style pack (schema v1: style_id/label/positive_tail/image_grade_tail/broadcast_tail/
era_tail/allow_radio_tails/forbidden_terms) rewrites TAILS only. The anime episode proved the
gap: character portraits went cel-shaded (tails ride every portrait), but announcer/radio and
music visuals stayed classic because their SUBJECT and MOTION content is hard-coded. The
grounded audit found ~21 uncontrolled sites; the load-bearing ones:

- ANNOUNCER SUBJECTS: `otr_meta_brief_image_prompt.py` `_RADIO_CONSOLE_FACE` (:167, the
  dial-face subject), `_RADIO_OBJECT_SUBJECT` (:177); `_otr_story_brief_helpers.py`
  `get_open_subject()` (:447-456, the three "radio warming up / broadcast booth / glowing
  warmly" scene-open phrases).
- PORTRAIT LOOK: `STYLE_ANCHOR` (:92), `STYLE_ANCHOR_WIDE` (:102), `STYLE_ANCHOR_TALKING`
  (:116) each mix GEOMETRY (headroom/face-visible/framing -- engine-safety, NOT styleable)
  with LOOK ("period-accurate costume and environment, dramatic film lighting" -- styleable).
- MOTION REGISTERS: `render_driver.py` `_LTX_MOTION_PROMPT_BY_ROLE` (~:1656, the console
  motion phrase for announcer/music), `_IA2V_TALKING_CLAUSE_CHARACTER` (~:1566),
  `_IA2V_TALKING_PROMPT_ANNOUNCER` (~:1665).
- MUSIC EMBLEM: `_mesh_fodder_subject()` music fallback "a single emblematic object..."
  (:1196-1228).
- NOT COVERABLE BY PROMPTS: procedural viz_* engines (`cheap_families.py` lavfi color slate)
  take NO prompt -- total coverage on those roles comes from SELECTING promptable engines
  (still_flat/still_word/ltx/...), which are already selectable. Documented, not coded.
- OUT OF SCOPE (geometry/architecture, stays Python): headroom/framing/composition clauses,
  NO_TEXT_CLAUSE, the LLM "do not mention radios" guard, mesh/background isolation scaffolds,
  the LTX composition insurance clause, Blender matcap/turntable (3D lane parked), procedural
  palette hex.

## 1. Design -- schema v2: subjects + look + motion registers (geometry-vs-look split)

**The split (r1 M5 design, now concrete):** every anchor/register decomposes into
GEOMETRY (framing, headroom, face-visibility, isolation, composition insurance -- Python-owned
constants, NEVER in packs; they protect engine contracts) and LOOK/SUBJECT (what the thing is
and how it's rendered aesthetically -- pack-owned). Only LOOK/SUBJECT moves.

### 1a. Pack schema v2 (nodes/_otr_visual_styles.py + all 5 packs, SAME COMMIT)

New REQUIRED str fields (schema_version bumps to v2; loader validates the EXACT field set,
fail-loud; sci_fi_radio.json values = the extracted current constants BYTE-IDENTICAL; the
constants stay in Python as extraction fixtures with AST production-read guards, the 3A
pattern):

1. `portrait_look` -- the look segment shared by the three portrait anchors
   ("period-accurate costume and environment, dramatic film lighting" for sci_fi_radio;
   anime pack: cel-shaded equivalent).
2. `announcer_subject_face` -- the dial-face radio-host subject (_RADIO_CONSOLE_FACE).
3. `announcer_subject_object` -- the faceless tabletop-radio subject (_RADIO_OBJECT_SUBJECT).
4. `open_subjects` -- dict with EXACT keys {"table", "booth", "plain"} (the three
   get_open_subject phrases; key names verify-at-build against the real selector logic).
5. `motion_console` -- the announcer/music console motion register (LTX lane).
6. `talking_clause_character` -- the ia2v character talking clause.
7. `talking_prompt_announcer` -- the ia2v announcer talking prompt.
8. `music_emblem_subject` -- the music_visual mesh/still emblem fallback subject.

Composition rule: composers assemble GEOMETRY (Python) + LOOK/SUBJECT (pack) at the existing
seams; the assembled sci_fi_radio output must be byte-identical to today's prompts
(equality tests against the current composed strings, not just the fields).

### 1b. Composer re-routes (all via the EXISTING style= channel; helpers never re-resolve)

- `otr_meta_brief_image_prompt.py`: the three STYLE_ANCHOR* constants become
  GEOMETRY_ANCHOR* (framing only) + `style.portrait_look` appended at the SAME position;
  build_radio_host_prompt takes the two announcer subjects from the pack;
  `_mesh_fodder_subject` music fallback reads `music_emblem_subject`.
- `_otr_story_brief_helpers.py`: `get_open_subject()` gains style= (the callers already
  hold the resolved style or meta) and reads `open_subjects[...]` from the pack.
- `render_driver.py`: the motion-register dict entries for the console + the two talking
  registers read from the pack at render time via `get_visual_style(meta)` (meta is already
  available in render_clip; resolve ONCE per clip, outside any swallow). Character
  motion entries that are beat-derived stay Python.
- Fail-loud law: NO try/except around any new style read; ImportError-only shims where the
  3A pattern already uses them.

### 1c. Explicitly OUT

viz_* procedural engines (promptless by design -- coverage = engine selection, documented in
the style README block); 3D Blender stage (parked); LLM guard phrases; NO_TEXT_CLAUSE;
mesh/background scaffolds beyond the already-routed era tail; ledger_directives;
forbidden_terms enforcement; any workflow-JSON change (the widget shipped in 3C; NO new
widgets -- assert validator no-diff).

## 2. Authoring (the 4 non-default packs)

anime / cartoon / paper_origami / archival_documentary each author the 8 new fields in their
existing voice (adapted from the lab blueprints where present, else written to match the
pack's tails). still_word lettering stays CONSISTENT per episode (operator 2026-07-04) --
the new fields must not introduce per-card randomness.

## 3. Tests

1. Loader v2 matrix: exact field set (missing/extra field fails), open_subjects exact keys,
   all values non-empty str; every shipped pack loads; schema_version pin.
2. BYTE-IDENTITY: for sci_fi_radio, every re-routed composer's OUTPUT equals the pre-change
   composed string (captured as fixtures) -- portrait anchors x3, radio-host (face+object),
   open subjects x3, motion console, talking x2, music emblem.
3. Delta coverage: for EACH non-default pack, each new field visibly changes the composed
   prompt (the 3B forced-meta delta pattern, extended to the new seams).
4. Geometry guard: AST/equality pin that the GEOMETRY_ANCHOR* constants contain no
   look-vocabulary from any pack (no drift of styleable text back into Python), and that no
   production site reads the old STYLE_ANCHOR* constants (extraction-fixture guard, 3A
   pattern).
5. render_driver style threading: resolve-once pin; a broken/unknown style at render time
   raises (no swallow); sci_fi default byte-identical motion prompts.
6. Suite + Bug Bible + B7 green; workflow validator no-diff.

## 4. Acceptance

- sci_fi_radio episodes byte-identical end-to-end (stamps + prompts).
- An anime episode's announcer/radio + music stills AND motion prompts carry the anime
  fields (log/ledger proof; operator eyeball gates the look).
- Full suite green per chunk; commit AND push per green chunk.

## 5. Open questions for the panel

- Q1: one shared `portrait_look` for all three anchors vs per-anchor look fields (draft:
  one shared -- the current look text is identical across the three; per-anchor splits ride
  a future need).
- Q2: should render_driver read the pack at render time (meta->style) or should the
  motion/talking registers be composed into the ledger at write time like text_prompt
  (draft: render-time read -- the registers are engine-lane concerns and the ledger schema
  stays untouched; but panel should check reproducibility/determinism implications).
- Q3: chunking -- one commit (schema+routing+packs) vs two (schema+sci_fi byte-identical,
  then the 4 packs' authoring) (draft: two -- the 3A/3B precedent).
- Q4: `open_subjects` keying -- verify the real get_open_subject selector variants and pick
  key names from the code, not invented ones.
