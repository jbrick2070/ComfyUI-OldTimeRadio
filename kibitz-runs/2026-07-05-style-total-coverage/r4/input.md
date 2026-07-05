# Visual-Style TOTAL COVERAGE -- Stage-3 section-8 slice (v4, post-r3)

Date: 2026-07-05. Branch: `v2.0-alpha`. Parent: `docs/multimodal-story-schema/STAGE3_SUBPLAN.md`
section 8 (PROMOTED by operator directive 2026-07-05: a selected visual_style must impact ALL
downstream prompts -- stills, video, 3D, announcer_visual + music_visual included). Status: v4
after kibitz r1+r2+r3 (codex + antigravity, Claude anchor+judge -- judgments
`kibitz-runs/2026-07-05-style-total-coverage/{r1,r2,r3}/final.md`). r4 convergence pending.

**r3 STRUCTURAL AMENDMENTS (override anything below that disagrees):**
- **The v2 schema is FINAL from chunk A1 and includes ALL fields** (11 str + 4 dict, incl.
  the still_word fields -- no second schema bump; chunk C only CONSUMES). New str field
  `scene_instruction_look` (the char-scene builder :1094-1110 has its OWN look text --
  portrait_instruction_look stays portrait-only).
- **Chunk A1 upgrades ALL FIVE packs to v2 syntactically** (the loader sweeps the whole
  dir at node registration -- a v1 pack would break ComfyUI startup): non-default packs
  temporarily carry the sci-fi-extracted DEFAULT values for the new fields (behavior
  identical to today's tails-only delta, dormant); chunk B = style-voice authoring +
  delta tests only.
- **build_radio_host_prompt name collision:** its existing `style=` kwarg is the DISPATCH
  arg (console_face/ltx_radio_mouth/radio_object) -- renamed `radio_host_style` with all
  callers (:1439/:1624/:1660) updated explicitly; the resolved pack travels as `vstyle=`.
- **Motion substitution ORDER:** pack motion value first, THEN the _talking_swap override
  (:1663-1665); a test proves pack motion_registers["announcer"] never leaks into the
  probe-locked IA2V talking prompt. `_ltx_motion_role_key`'s env-key membership check
  moves to a STATIC key set (no reference to the retired constant).
- **Threading:** resolve ONCE at the image-prompt entry; compose_image_prompt_fallback
  gains style=None; _compose_char_scene_prompt / derive_image_prompts thread vstyle; no
  helper re-resolve. `prompt_field_source` values specified per arm
  ("motion_registers:<key>", "open_subjects:<key>", "announcer_subject_<arm>", ...).
- **New tests:** INPUT_TYPES() import test after pack edits (the real startup path);
  dormant-field load+lint pin in A1; still_word per-episode determinism; post-re-route
  swallow re-audit. Extraction+re-route per surface in ONE edit; the AST fixture guard
  lands with the LAST surface of each chunk.

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

str fields (9):
1. `portrait_look` -- shared look segment of STYLE_ANCHOR + STYLE_ANCHOR_WIDE.
2. `portrait_look_talking` -- the talking anchor's look segment.
3. `portrait_instruction_look` -- the LLM-facing look language; lives ONLY in
   `_build_char_prompt_request` (:1068, r2 AG S2 corrected); the style-look insertion for
   `_build_char_scene_request` is specified separately at build (byte-identity preserving);
   BOTH builders take the resolved style. Face/headroom/gear-guard constraints stay Python.
4. `announcer_subject_face` -- dial-face radio-host subject.
5. `announcer_subject_ltx_mouth` -- TEMPLATE with `{form}` exactly once (the current
   _RADIO_CONSOLE_MOUTH carries %s form, :203/:335 -- r2 codex M2); the call site moves to
   .format(form=...); load lint additionally requires mouth-prominence vocabulary (the
   ia2v lip-sync contract). The ia2v talking MOTION prompts
   (_IA2V_TALKING_PROMPT_ANNOUNCER / _IA2V_TALKING_CLAUSE_CHARACTER) are PROBE-PROVEN
   VERBATIM constants (P8 2026-07-02: a paraphrase halves articulation) -- NEVER pack
   fields; do not reopen.
6. `announcer_subject_object` -- faceless tabletop radio subject.
7. `radio_object_look` -- the look/lighting text split out of _RADIO_OBJECT_ANCHOR*.
8. `plate_look` -- the look text split out of the background-plate scaffold.
9. `non_character_emblem_fallback` -- TEMPLATE with `{base}` exactly once (the fallback
   interpolates intent/setting: "a single emblematic object representing %s", :1226 -- r2
   codex M3); routed at the REAL fallback (:1228); music_visual keeps radio_form_from_meta.

dict fields (2; exact keys, load-validated; stored as immutable mappings in the frozen
dataclass -- r2 codex OPT):
10. `open_subjects` -- keys {synthetic, announcer, default}; each value a TEMPLATE with
    `{form}` exactly once; composed via str.format(form=...).
11. `motion_registers` -- keys {announcer, music_open, music_close, music_inter}; the
    LTX-lane console/music motion phrases. Role->key SELECTOR + the
    OTR_LTX_OPEN_MOTION_KEY env retarget stay Python (packs own VALUES only); the
    BUG-LOCAL-112 240-char budget is enforced at LOAD on these values; the silent
    `or ...["announcer"]` fallback (:1656) is RETIRED -- exact-key indexing raises on a
    missing console key (non-console roles keep the "" no-op arm) (r2 codex S3 + anchor
    M1/M3).

Load lint: the forbidden-terms lint extends over ALL new string leaves + dict values
(r2 codex S2). New fields non-empty; v1 tail fields keep empty-legal semantics.

chunk-C fields (r2 codex M4, concrete): `still_word_typography: dict` +
`still_word_backdrop: dict` (exact keys {noir, "sci-fi", western, pulp, default} -- keep
the hyphenated on-disk key EXACTLY) + `still_word_title_mood_style: str`
(:631/:642/:662); per-episode lettering consistency preserved (selection stays Python).

### 1b. Composer re-routes

- `otr_meta_brief_image_prompt.py`: STYLE_ANCHOR* -> PORTRAIT_GEOMETRY /
  WIDE_PORTRAIT_GEOMETRY / TALKING_PORTRAIT_GEOMETRY + the pack look appended at the SAME
  position; `_style_anchor_for_aspect(aspect, talking=False, style=None)` threads the
  resolved style through ALL FOUR callers (compose_image_prompt_fallback,
  _build_char_prompt_request, _build_char_scene_request, build_radio_host_prompt -- r2 AG
  M1); build_radio_host_prompt's THREE dispatch arms read pack subjects; radio-object
  anchor + plate scaffold read their look fields; the emblem fallback formats
  non_character_emblem_fallback with {base}.
- `_otr_story_brief_helpers.py`: final signature `get_open_subject(role, synthetic,
  meta=None, style=None)`; compose_still_prompt passes its ALREADY-RESOLVED _style
  (:510 -> :522) -- helpers never re-resolve (r2 codex S1 + AG M3); the pack template is
  formatted with the brief-driven form.
- `_otr_video_engines/render_driver.py`: motion-register substitution happens in
  `build_request_from_shot()` (render_clip has NO meta -- r1 AG M4); resolve the style
  ONCE per shot, OUTSIDE any swallow. PROVENANCE (r2 codex M5): ADDITIVE observability
  keys `visual_style` + `prompt_field_source` on the request, ADDED to the trace-copy
  allowlist (:2033-2035) so they reach the trace rows + node-92 /history; prompt TEXT and
  the existing sha/chars stamps stay byte-identical for sci_fi_radio.
- Fail-loud law: no try/except around style reads; ImportError-only shims per 3A.

### 1c. Explicitly OUT (with rationale)

Radio FORM map (brief axis); character talking clause + announcer talking motion prompt
(style-agnostic lip mechanics); viz_* procedural engines (promptless -- documented
limitation, config guidance only); 3D Blender stage (parked); LLM gear-guard +
NO_TEXT_CLAUSE + isolation scaffolds' geometry; ledger_directives; forbidden_terms
enforcement; workflow JSON (no new widgets -- validator no-diff asserted).

## 2. Build order (four chunks, commit+push per green chunk)

- **Chunk A1 (image lane):** literal extraction -> schema v2 loader (all 11 fields; the
  A2 surfaces load but are not yet consumed) + sci_fi_radio.json v2 -> portrait-look trio
  + announcer subjects + open subjects re-routes -> SEAM-LEVEL string-equality
  byte-identity tests (composed OUTPUT vs pre-change fixtures -- the build gate; full-
  episode identity is operator acceptance, not the gate, r2 codex CUT). Suite green.
  SAME COMMIT: test_visual_styles_3b.py schema/exact-key pins re-pointed v1 -> v2 (r2 AG
  OPT2); a v1 pack fails load with a clear upgrade message (NO back-compat defaults --
  fail-loud law; r2 AG OPT1 rejected).
- **Chunk A2 (video/mesh lane):** motion registers + radio_object/plate look + emblem
  fallback re-routes + provenance keys. Suite green.
- **Chunk B:** author the 9+2 field set in the 4 non-default packs (exact key list from
  1a; adapted to each pack's voice); forced-meta delta tests per surface + per-surface
  negative-vocab smokes (r1 codex OPT). Suite green.
- **Chunk C:** still_word typography/backdrop/title-mood pack ownership (fields per 1a;
  per-episode lock preserved). Suite green.

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
