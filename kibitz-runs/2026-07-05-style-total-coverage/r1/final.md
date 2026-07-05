# r1 JUDGMENT (Cowork Claude, anchor + judge) -- style total-coverage

## Accepted
- ANCHOR M1 = AG M2 (CONFIRMED :450-456): open_subjects values are TEMPLATES with a
  required {form} placeholder (exactly once, load-validated); keys = the real selector
  arms {synthetic, announcer, default}.
- ANCHOR M2: portrait_look (base+wide) + portrait_look_talking (talking anchor's look
  differs: no environment, warm light). ANCHOR M3: talking portraits skip tails BY DESIGN
  -- stated as geometry law.
- CODEX M1 (ltx_radio_mouth third announcer subject): new `announcer_subject_ltx_mouth`;
  mouth-visibility/safety language stays Python geometry. (Verify the dispatch at
  :203/:228/:334-337 at build.)
- CODEX M2 = AG M3 (CONFIRMED by both agents independently): music_visual mesh subject is
  radio_form_from_meta, NOT the emblem fallback. Field renamed
  `non_character_emblem_fallback`, routed at the REAL general fallback (:1228). The radio
  FORM rides the brief axis (anchor S4) -- out of scope with rationale.
- CODEX M3 + S4: _RADIO_OBJECT_ANCHOR* and BACKGROUND_PLATE scaffold contain look text --
  split: `radio_object_look` + `plate_look` pack fields; isolation/geometry stays Python.
- CODEX M4: the LLM instruction's "photographic and period-consistent" look language moves
  to `portrait_instruction_look`; face/headroom/gear-guard constraints stay Python.
- CODEX M5 (still_word): IN scope as its own build chunk (C); exact fields designed
  against the real :631-664 maps at build; per-episode lettering consistency (operator
  2026-07-04) preserved -- selection logic stays Python, packs own the vocabulary.
- AG M1: motion register is FOUR roles ({announcer, music_open, music_close, music_inter})
  -- `motion_registers` dict with exact keys, not one field.
- AG M4 (CONFIRMED contract shape): render_clip has no meta -- style resolution +
  register substitution happen in build_request_from_shot (full ledger/meta access),
  resolve ONCE, outside any swallow.
- AG S1: extract the get_open_subject inline literals into Python constants FIRST
  (extraction fixtures, 3A pattern). AG S2: get_open_subject style param optional
  (style=None -> resolve from meta) so existing tests keep passing.
- CODEX S1/CUT2: non-empty validation applies to NEW fields only; existing empty-tail
  semantics stay legal. CODEX S2: schema stated as typed fields. CODEX S3: prompt metadata
  stamps visual_style + field source for newly routed prompts. CODEX OPTs: geometry
  constants named *_GEOMETRY; per-surface anime negative-vocab smoke tests.
- CODEX CUT3: viz_* promptless = documented limitation, not claimed coverage.
- AG CUT1 (character talking clause is style-agnostic lip mechanics): ACCEPTED --
  `talking_clause_character` and `talking_prompt_announcer` DROPPED as fields; the
  talking-lane style surface = portrait_look_talking + announcer_subject_ltx_mouth; the
  motion mechanics stay Python.

## Rejected
- None outright; scope disagreements resolved above (still_word in, form map out).

## Verify-at-build
- build_radio_host_prompt dispatch arms + exact object/plate scaffold texts;
  _LTX_MOTION_PROMPT_BY_ROLE exact keys (:529-544 per AG vs ~:1656 per audit -- line
  numbers differ, verify); still_word maps :631-664; the LLM instruction text :1061-1114.
