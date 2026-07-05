# ANCHOR REVIEW (Cowork Claude) -- style total-coverage r1 (arc/coherence)

Doc: STAGE3_TOTAL_COVERAGE_SUBPLAN.md v1. Grounding this round: read
_otr_story_brief_helpers.py :430-470 (get_open_subject + radio form) and
otr_meta_brief_image_prompt.py :85-130 (the three anchors) directly.

VERDICT: SHIP WITH FIXES. The geometry-vs-look split is the right organizing law and the
field inventory matches the audit; three of my own draft's details are WRONG against the
code.

MUST-FIX (self-caught, CONFIRMED):
1. **`open_subjects` are TEMPLATES, not flat strings.** get_open_subject interpolates the
   brief-driven radio FORM: `"%s warming up on a table..." % form` (:450-456, form from
   radio_form_from_meta). Pack fields must carry a `{form}` placeholder (validated at load:
   placeholder REQUIRED exactly once; str.format at compose). Keys are
   {"synthetic", "announcer", "default"} (the actual selector arms: synthetic flag,
   role==announcer_visual, else) -- NOT my invented {"table","booth","plain"}.
2. **One shared `portrait_look` does NOT fit the talking anchor.** Base+wide share
   "period-accurate costume and environment, dramatic film lighting" (:92-105) but TALKING
   is "period-accurate costume, warm dramatic lighting" (:116-121, no environment, warm --
   an S4b lip-sync-driven choice). Two fields: `portrait_look` (base+wide) +
   `portrait_look_talking`. Q1 resolved.
3. **Talking portraits SKIP the era/grade/palette tails BY DESIGN** (comment :113-115,
   derive_image_prompts) -- the lip-sync engine needs a bright frontal face. The plan must
   state: `portrait_look_talking` is the ONLY style surface on talking portraits; packs
   must author it conservatively (face-visibility survives the style); the skip is
   geometry-law, not a coverage bug.

SHOULD-FIX:
4. The radio FORM map itself (_RADIO_FORM_MAP / _RADIO_FORM_DEFAULT) is brief-driven
   subject vocabulary -- arguably styleable (an origami radio?). DEFER: the form rides the
   BRIEF axis, not the style axis; mixing them doubles the authoring matrix. State as
   out-of-scope with rationale.
5. Field name `motion_console` -> `motion_register_console` for self-documentation; the
   two talking register fields similarly prefixed.

CONFIRMED-OK: the geometry constants list; render-time style read for motion registers
(meta reaches render_clip); the two-chunk build split (3A/3B precedent); viz_* promptless
documentation stance.
