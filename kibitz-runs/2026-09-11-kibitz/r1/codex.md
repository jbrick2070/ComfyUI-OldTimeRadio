VERDICT: no. The ownership diagnosis needs a complete consumer boundary and a story-setting authority before it can become a buildable fix.

MUST-FIX BEFORE BUILD:

1. [§2, §4(b), §5.1] “No owner arbitrates” misses an existing arbitration attempt. In docs/2026-09-11-my-story-5080-qualification/pairlock_09_ledger.json, shot_001_b9 has source_rewrite.status="rewritten", yet its retained_prompt combines a Los Angeles meal with “photoreal Shakespearean actor portrait, period costume.” nodes/otr_meta_brief_image_prompt.py::_rewrite_char_scene_from_source already supplies source corrections and style instructions together. The defect is conflicting authority inside that operation, not simply missing source access. Fix: declare source/current-scene facts authoritative for physical dress and setting, and supply treatment-only style instructions to that existing operation and its retained-template fallback.

2. [§4(b), §5.3–4] Story-time precedence has no defined cross-bank input. The production ledger has no meta.period key; explicit “present day” lives in meta.source_meta.story_input.fields.setting. nodes/_otr_story_brief.py::StoryBriefModel has no era field, and its reflection prompt asks for places and discourages inventing dates. nodes/otr_meta_brief_image_prompt.py::_read_setting reads only two setting terms. Furthermore, nodes/_otr_story_source.py::raw_fields_from_ledger returns None outside My Story. Fix: identify the existing accepted setting/source context supplied to each visual author on each lane, including an unspecified-time case. Do not classify eras from bank names or assume the My Story source path covers adaptations. Verify: the adaptation source/context handoff before claiming Macbeth is protected.

3. [§2, §4(b), §5.1] The proposed boundary is incomplete even within the inspected files. nodes/visual_styles/shakespeare_stage_realism.json also supplies portrait_instruction_look (“period stage actor”) and plate_look (“candlelit period interior”): the pack can furnish the ROOM too. Separately, nodes/_otr_story_brief_helpers.py::compose_still_prompt supplies “a period-dressed character” when appearance is missing. Character descriptions enter through _appearance_for_char and can be re-prepended by _compose_char_scene_prompt in nodes/otr_meta_brief_image_prompt.py. Fix: cover portrait instructions, scene instructions, physical-set instructions, appearance inputs and deterministic fallbacks under the same ownership rule. A five-field pack edit leaves competing owners.

4. [§4, §5.2] The forks conflate a behavior decision with its implementation. Story precedence does not require a new schema or runtime parsing of mixed strings. Choose (b), with explicit editorial changes to the affected checked-in pack fields: a clause describing rendering medium, lighting quality or framing belongs to treatment; a clause specifying garments, historical dates or physical surroundings belongs to story context. For example, retain “candlelit stage light,” remove the pack’s unconditional “period costume,” and preserve costume facts supplied by the story. nodes/_otr_visual_styles.py::validate_pack permits changing existing string values. This avoids a forbidden-word list and preserves the independent roll in nodes/_otr_rolls.py::resolve_style_selection. [ASSUMPTION] The intended style identity can survive as theatrical treatment without mandatory Elizabethan clothing; record that product decision explicitly.

SHOULD-FIX:

1. [§1, §3] Correct the evidence framing. The retained source in docs/2026-09-11-my-story-5080-qualification/pairlock_09_ledger.json places dinner in Los Angeles; the Bay Area is a spoken memory. Its images.images entry identifies shot_000_b1 as announcer_visual/scene_beat, whereas shot_001_b9 is character_video/scene_character. Fix: distinguish those prompt paths and define whether the house frame shares the dramatic world’s dress policy. Do not treat both stills as evidence from one character-scene composer.

2. [§4(c), §5.5] Narrow the hash claim. nodes/_otr_visual_styles.py::get_visual_style verifies embedded bytes specifically for visual_storybased; named styles resolve through the current registry. Fix: state separately what happens to old embedded packs and to rerenders using edited named packs. If schema work becomes necessary, update _KNOWN_FIELDS/validation and use optional fields without modifying stored embedded bytes. Optionality preserves readability; it does not itself fix conflicting historical pack content.

3. [§3, §6] Define acceptance without promising general wardrobe continuity. After the four-machine baseline, inspect delivered prompts and pixels for the present-day dinner, a genuine period adaptation, and a missing-setting/appearance case. Exercise the actual portrait, character-scene and relevant frame paths. docs/2026-09-11-my-story-5080-qualification/pairlock_09_receipt.md explicitly leaves clothing continuity unqualified. Removing contradictory instructions is testable; identical clothing across beats is a separate claim.

OPTIONAL / NICE-TO-HAVE:

[§5.1] Correct stale wardrobe-writer comments during later implementation. nodes/otr_shot_lock.py::_appearance_for_char explicitly records the outfit lock’s removal; _build_subject_sigils still describes a helper that may invoke it. The inspected implementation provides no active wardrobe writer to reuse.

CUT THESE (scope / over-engineering):

1. [§4(a)] Era-based style eligibility. It violates the stated full-pool goal and cannot reconcile contradictory instructions within an explicitly selected style. Keep resolve_style_selection independent.

2. [§4(c)] Mandatory migration of every pack. Existing string edits can express the chosen ownership rule. Defer schema expansion until a concrete consumer needs structured data that existing fields cannot provide.

3. [§1, §3] Any implied outfit-lock or face-consistency subsystem. This build should remove style-versus-story contradictions. Neither the evidence nor the proposed method establishes a need to restore the retired outfit feature.
