# Visual-Style TOTAL COVERAGE -- Stage-3 section-8 slice (v2, post-r1)

Date: 2026-07-05. Branch: `v2.0-alpha`. Parent: `docs/multimodal-story-schema/STAGE3_SUBPLAN.md`
section 8 (PROMOTED by operator directive 2026-07-05: a selected visual_style must impact ALL
downstream prompts -- stills, video, 3D, announcer_visual + music_visual included). Status: v2
after kibitz r1 (codex + antigravity, Claude anchor+judge -- judgment
`kibitz-runs/2026-07-05-style-total-coverage/r1/final.md`). r2..r4 pending.

## 0. Problem (grounded)

The v1 style pack rewrites TAILS only (positive/image_grade/broadcast/era +
allow_radio_tails). The anime episode proved the gap: portraits went cel-shaded, but
announcer/radio + music visuals stayed classic -- their SUBJECT/LOOK/MOTION content is
hard-coded. Load-bearing uncontrolled sites (grounded 2026-07-05, r1-corrected):

- ANNOUNCER SUBJECTS (3 dispatch arms of `build_radio_host_prompt`):
  `_RADIO_CONSOLE_FACE` (dial-face), `ltx_radio_mouth` (the talking appliance-face init --
  r1 codex M1), `_RADIO_OBJECT_SUBJECT` (faceless tabletop radio); plus the radio-OBJECT
  anchor look text ("dramatic film lighting" inside _RADIO_OBJECT_ANCHOR*, r1 codex M3).
- SCENE OPENS: `get_open_subject()` -- three TEMPLATE arms interpolating the brief-driven
  radio FORM (`"%s warming up on a table..." % form`); the form itself rides the BRIEF
  axis and stays out of scope (style x form would double the authoring matrix).
- PORTRAIT LOOK: STYLE_ANCHOR / STYLE_ANCHOR_WIDE share the look segment
  "period-accurate costume and environment, dramatic film lighting"; STYLE_ANCHOR_TALKING
  has its own ("period-accurate costume, warm dramatic lighting"). GEOMETRY (framing,
  headroom, face-visibility) is engine-safety and stays Python. Talking portraits SKIP
  era/grade tails BY DESIGN (S4b lip-sync law) -- portrait_look_talking is their ONLY
  style surface; packs author it conservatively.
- LLM INSTRUCTION LOOK: the portrait/char-scene LLM request hard-codes "photographic and
  period-consistent" look language (:1061-1114) that fights non-default styles (r1 codex
  M4).
- MOTION REGISTERS: `_LTX_MOTION_PROMPT_BY_ROLE` carries FOUR console/music entries
  ({announcer, music_open, music_close, music_inter} -- r1 AG M1). The character talking
  clause + announcer talking motion prompt are style-AGNOSTIC lip mechanics (r1 AG CUT)
  and stay Python.
- MESH/PLATE: the general no-character emblem fallback (:1228, "a single emblematic
  object...") is styleable; music_visual's mesh subject is radio_form_from_meta (brief
  axis, NOT the fallback -- r1 codex M2 + AG M3). BACKGROUND_PLATE scaffold contains look
  text ("period-accurate set") -- split.
- STILL_WORD: typography/backdrop/title-mood maps are hard-coded (:631-664) -- IN SCOPE as
  chunk C (r1 codex M5); per-episode lettering consistency (operator 2026-07-04) is
  preserved: selection logic stays Python, packs own vocabulary.
- DOCUMENTED LIMITATION (not coverage): procedural viz_* engines take no prompt; styling
  those roles requires selecting promptable engines. 3D Blender stage parked.

## 1. Design -- schema v2 (geometry-vs-look split, now concrete)

LAW: every anchor/register decomposes into GEOMETRY (framing/headroom/face-visibility/
isolation/mouth-safety/composition insurance -- Python constants named *_GEOMETRY, NEVER
in packs) and LOOK/SUBJECT (pack-owned). Only LOOK/SUBJECT moves. sci_fi_radio.json v2
values = the extracted current literals BYTE-IDENTICAL (constants stay in Python as
extraction fixtures with AST production-read guards -- the 3A pattern; the get_open_subject
inline literals are extracted into constants FIRST, r1 AG S1).

### 1a. New pack fields (typed; ALL REQUIRED; new str fields non-empty -- existing
### empty-tail semantics for the v1 tail fields stay legal; schema_version -> v2)

str fields:
1. `portrait_look` -- shared look segment of STYLE_ANCHOR + STYLE_ANCHOR_WIDE.
2. `portrait_look_talking` -- the talking anchor's look segment.
3. `portrait_instruction_look` -- the LLM-facing look language (photographic/period text
   moves here; face/headroom/gear-guard constraints stay Python).
4. `announcer_subject_face` -- dial-face radio-host subject.
5. `announcer_subject_ltx_mouth` -- the talking radio init-still subject (appliance face);
   mouth-visibility safety language stays Python geometry.
6. `announcer_subject_object` -- faceless tabletop radio subject.
7. `radio_object_look` -- the look/lighting text split out of _RADIO_OBJECT_ANCHOR*.
8. `plate_look` -- the look text split out of the background-plate scaffold.
9. `non_character_emblem_fallback` -- the general no-character mesh/still fallback subject
   (routed at the REAL fallback :1228; music_visual keeps radio_form_from_meta).

dict fields (exact keys, load-validated):
10. `open_subjects` -- keys {synthetic, announcer, default}; each value a TEMPLATE
    containing the `{form}` placeholder EXACTLY ONCE (load lint); composed via
    str.format(form=...) at the existing seams.
11. `motion_registers` -- keys {announcer, music_open, music_close, music_inter}; the
    LTX-lane console/music motion phrases.

chunk-C fields (exact shape designed at build against the real still_word maps :631-664;
episode-level consistency preserved): still_word typography vocabulary + backdrop mood +
music title-mood.

### 1b. Composer re-routes

- `otr_meta_brief_image_prompt.py`: STYLE_ANCHOR* -> PORTRAIT_GEOMETRY /
  WIDE_PORTRAIT_GEOMETRY / TALKING_PORTRAIT_GEOMETRY + the pack look appended at the SAME
  position; build_radio_host_prompt's THREE dispatch arms read pack subjects; radio-object
  anchor + plate scaffold read their look fields; the LLM request reads
  portrait_instruction_look; the general emblem fallback reads
  non_character_emblem_fallback.
- `_otr_story_brief_helpers.py`: get_open_subject gains OPTIONAL style=None (resolves via
  get_visual_style(meta) when absent -- existing tests keep passing, r1 AG S2) and formats
  the pack template with the brief-driven form.
- `_otr_video_engines/render_driver.py`: motion-register substitution happens in
  `build_request_from_shot()` (full ledger/meta access -- render_clip has NO meta, r1 AG
  M4); resolve the style ONCE per shot, OUTSIDE any swallow; the composed prompt metadata
  stamps `visual_style` + the pack field source for every newly routed prompt (r1 codex
  S3).
- Fail-loud law: no try/except around style reads; ImportError-only shims per 3A.

### 1c. Explicitly OUT (with rationale)

Radio FORM map (brief axis); character talking clause + announcer talking motion prompt
(style-agnostic lip mechanics); viz_* procedural engines (promptless -- documented
limitation, config guidance only); 3D Blender stage (parked); LLM gear-guard +
NO_TEXT_CLAUSE + isolation scaffolds' geometry; ledger_directives; forbidden_terms
enforcement; workflow JSON (no new widgets -- validator no-diff asserted).

## 2. Build order (three chunks, commit+push per green chunk)

- **Chunk A:** literal extraction -> schema v2 loader + sci_fi_radio.json v2 ->
  re-routes -> BYTE-IDENTITY equality tests (composed OUTPUT vs pre-change fixtures for
  every re-routed surface). Suite green.
- **Chunk B:** author the 8+2 fields in the 4 non-default packs (adapted to each pack's
  voice); forced-meta delta tests per surface + per-surface negative-vocab smokes (the
  anime assembled prompt contains no photographic/35mm/film/period vocabulary -- r1 codex
  OPT). Suite green.
- **Chunk C:** still_word typography/backdrop/title-mood pack ownership (exact fields vs
  the real maps; per-episode lock preserved). Suite green.

## 3. Tests

1. Loader v2 matrix: exact field set; dict exact keys; {form} placeholder exactly once;
   new fields non-empty; every shipped pack loads; schema pin.
2. BYTE-IDENTITY (sci_fi_radio): every re-routed composer's OUTPUT equals the pre-change
   composed string -- portrait anchors x3, radio-host x3 arms, radio-object anchor, plate,
   open subjects x3 (with a fixed form), motion registers x4, emblem fallback, LLM
   instruction text.
3. Delta + negative-vocab coverage per non-default pack per surface.
4. Geometry guards: *_GEOMETRY constants contain no pack look vocabulary; no production
   reads of the old extracted literals (AST fixture guards).
5. build_request_from_shot style threading: resolve-once pin; unknown style raises (no
   swallow); prompt metadata stamps visual_style + field source; sci_fi default motion
   prompts byte-identical.
6. Suite + Bug Bible + B7 green per chunk; workflow validator no-diff.

## 4. Acceptance

- sci_fi_radio episodes byte-identical end-to-end (prompts + stamps).
- An anime episode's announcer/radio + music stills AND motion prompts provably carry the
  anime fields (ledger prompt-metadata stamps); operator eyeball gates the look.
- Full suite green per chunk; commit AND push per green chunk to v2.0-alpha.

## 5. Verify-at-build

- build_radio_host_prompt dispatch arms (:203/:228/:334-337) + exact scaffold texts.
- _LTX_MOTION_PROMPT_BY_ROLE exact keys + line location (:529-544 vs ~:1656 conflicting
  reports).
- still_word maps :631-664 exact shape; the LLM instruction text :1061-1114.
- Existing 3A/3B test files: which pins re-point vs stay (the 45 3B delta tests must keep
  passing untouched where surfaces are additive).
- image_policy/talking-map availability assumptions in headless lanes (r1 AG assumption).
