# r3 JUDGMENT (Cowork Claude, anchor + judge) -- style total-coverage, wiring round

## Accepted
- CODEX M1 = AG M1 (both agents independently; CONFIRMED architecture: _load_all sweeps
  the whole visual_styles/ dir and INPUT_TYPES loads at node registration): chunk A1
  upgrades ALL FIVE packs to v2 SYNTACTICALLY -- non-default packs carry the
  sci-fi-extracted DEFAULT values for the new fields (behavior identical to today's
  tails-only delta, dormant until B); chunk B becomes style-voice authoring + delta
  tests only.
- AG M2: the v2 schema is FINAL from A1 -- ALL fields incl. the chunk-C still_word
  fields (no second schema bump; C consumes what A1 loads). Field inventory: 11 str
  (portrait_look, portrait_look_talking, portrait_instruction_look,
  scene_instruction_look, announcer_subject_face, announcer_subject_ltx_mouth,
  announcer_subject_object, radio_object_look, plate_look,
  non_character_emblem_fallback, still_word_title_mood_style) + 4 dict (open_subjects,
  motion_registers, still_word_typography, still_word_backdrop).
- CODEX M2: `build_radio_host_prompt`'s existing `style=` kwarg is the DISPATCH arg
  (console_face/ltx_radio_mouth/radio_object; :297, callers :1439/:1624/:1660) --
  renamed `radio_host_style`, all callers updated explicitly; the resolved pack travels
  as `vstyle=`.
- CODEX M3 + AG S1: NEW field `scene_instruction_look` for `_build_char_scene_request`'s
  own look text (:1094-1110); portrait_instruction_look stays portrait-only.
- CODEX M4: substitution ORDER pinned -- pack motion value first, THEN the _talking_swap
  override (:1663-1665); test proves pack motion_registers["announcer"] never leaks into
  the IA2V talking prompt.
- CODEX M5 + AG M3: _ltx_motion_role_key membership check moves to a STATIC Python key
  set {announcer, music_open, music_close, music_inter} (no reference to the retired
  constant; music_inter fallback preserved).
- CODEX S1 + AG S2: exact signature threading list -- resolve ONCE at the image-prompt
  entry; `compose_image_prompt_fallback` gains style=None and passes it down;
  `_compose_char_scene_prompt` / `derive_image_prompts` thread vstyle; no helper-level
  re-resolve.
- CODEX S2: stage changes pack JSON only, no node inputs/widgets -- validator no-diff
  stands. CODEX OPT: an INPUT_TYPES() import test after the pack edits (the real startup
  path). AG S3: `prompt_field_source` value map specified per dispatch arm
  ("motion_registers:<key>", "open_subjects:<key>", "announcer_subject_<arm>", etc.).
  AG OPT: MappingProxyType for the dict fields.
- ANCHOR M1 (dormant-fields pin), M2 (validator no-diff per chunk), M3 (extraction+
  re-route per surface in one edit; AST fixture guard lands with the LAST surface of the
  chunk), S4 (post-re-route swallow re-audit), S5 (still_word determinism test).

## Rejected
- None material.

## Verify-at-build
- Exact current texts of :1094-1110 (scene builder) for the scene_instruction_look split.
- The _talking_swap variable/flow names at :1654-1665.
- INPUT_TYPES / list_style_ids load path at writer :2297-2308.
