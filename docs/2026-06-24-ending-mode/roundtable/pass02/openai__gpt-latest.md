<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. Core wiring is underspecified and §2/§3 contradict the shown Path C outline flow and voiced-announcer structure.

MUST-FIX BEFORE BUILD:
1. [§2 / Open] Defect: “final-beat prompt” is ambiguous and currently not enough to code. In `_otr_outline.py`, production Path C uses `_build_beat_user_prompt()` only for Stage 3 intent/mood generation, not dialogue lines, and it receives no `ending_tag`/template fields. Concrete fix: add explicit defaulted fields to `OutlineRequest`, e.g. `ending_tag: str = ""`, `ending_template: str = ""`, and update `_build_beat_user_prompt(..., is_final_character_beat: bool = False)` to append the ending instruction only when `req.ending_tag` and `is_final_character_beat` are set. Keep defaults empty to preserve off-byte identity.

2. [§2 / §6] Defect: “reuse from l12” for final-beat detection cannot be coded as written. `_otr_story_quality_l12.validate_beat_roles()` validates a `roles_by_beat` mapping after character beat IDs are known; it does not identify the final beat during Stage 3 prompt construction. Concrete fix: in `generate_outline()`, after `phase_skeletons` are built, compute the final voiced CHARACTER coordinate once, e.g. `(last_phase_idx, last_beat_idx)` from the last non-empty phase skeleton, and pass `is_final_character_beat=(phase_idx, beat_idx)==that_coord` into `_build_beat_user_prompt()`.

3. [§2 / Announcer-outro fix] Defect: current combiner appends a voiced announcer close after the last character beat. Grounding: `_assemble_outline()` appends an `announcer` beat with intent `Close on a concrete final image showing what changed...`; `stamp_dialogue_slot_ids()` treats announcers as voiced. This directly conflicts with “climax = final voiced beat” and can narrate outcome. Concrete fix: do not claim final voiced beat unless you remove/convert the close, which would break budget validator #7. Smallest safe fix: gate only the announcer close intent under the new flag:
   - flag off: keep the exact existing intent string.
   - flag on: use a generic non-outcome intent such as `Close the episode without explaining the outcome; identify the program only.`
   Also update any tests/metrics to say “last voiced CHARACTER beat,” not “final voiced beat.”

4. [§1 / Open] Defect: enum/data model is not buildable. “Add a field `ending_tag` to every `_otr_style_catalog.py` entry” gives no importable constant, type, validation function, or template lookup shape. Concrete fix: define in the catalog module:
   - `ENDING_TAGS: tuple[str, ...] = (...)`
   - `ENDING_TEMPLATES: dict[str, str] = {...}`
   - each style entry has `ending_tag: str` and existing prose is renamed/aliased as `ending_flavor`.
   - add a catalog self-check function that fails if any style entry has missing/unknown `ending_tag` or no template.
   [ASSUMPTION] actual catalog entry type must be verified.

5. [§1 / §5] Defect: adding `ending_tag` to catalog entries can break byte-identity if the existing picker serializes full catalog entries into prompts or hashes them. The plan asserts byte-identical off, but does not constrain existing style picker usage. Concrete fix: verify current picker input shape. If existing off-path reads full entries, make `ending_tag` excluded from any prompt/stringification used while flag is off, or store the map externally as `ENDING_TAG_BY_SLUG` instead of mutating entry objects