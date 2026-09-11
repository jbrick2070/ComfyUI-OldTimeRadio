# r2 judgment -- visual continuity ownership

r2 = cursor (`cursor-grok-4.6-high`, mode ask, logged in). **It responded** -- no
substitution was needed, so the operator's "if cursor doesn't respond, ask Fable"
fallback went unused. Lane health: it returned a full structured review with exact
file:line citations and concrete string rewrites, not a preamble.

Driver = Claude Opus 5. Every claim below was checked against the real files before
being folded in.

## OPERATOR RULING RECEIVED -- the blocking assumption is now explicit

Cursor ended by saying the document "cannot be implemented until that bit is
explicit". The operator made it explicit: **`shakespeare_stage_realism` is JUST the
visual style.** So the pack owns theatrical TREATMENT (candlelight, stage realism,
practical texture, framing) and does NOT own what century anyone's clothes are from.

Consequences, now settled:
- Cursor's M2-M4 stand.
- Era-aware eligibility stays CUT; `resolve_style_selection` stays independent.
- A genuine Shakespeare/public-domain ADAPTATION still deserves authentic period
  dress -- but that must come from the SOURCE, not from a visual-style pack.

## VERIFIED AND FOLDED IN

**1. The pack inventory is larger than seven fields.** Verified in the JSON:
`image_grade_tail` ("crisp **costume** detail"), `announcer_subject_object`
("**period** tabletop radio"), `announcer_subject_ltx_mouth`, and `open_subjects.*`
("period stage radio", "period dials"). So ~12 era touchpoints, not 7.

**2. THE MECHANISM FOR THE ANNOUNCER STILL -- `positive_tail` is SHARED.** It is
appended on scene_beat, scene_open and scene_character
(`_otr_story_brief_helpers.py::compose_still_prompt`), on portraits via
`finish_visual_prompt`, on plates via `_compose_background_plate_prompt`, and on
ShotLock video. One costume clause therefore reaches the house frame, the drama and
the plate. That is why `still_000` -- a faceless *radio* subject -- rendered a
costumed man in a kitchen. This explains the still my own diagnosis could not, and
it means the shared-field rewrite IS the house-frame fix. No second dress policy,
no per-role tail.

**3. A SECOND empty-appearance writer.** `compose_image_prompt_fallback` does not use
the "period-dressed" literal; it degrades to `_style_anchor_for_aspect`, which is the
pack's `portrait_look`. So fixing `_story_brief_helpers.py:567` alone is insufficient.

**4. CORRECTION to the driver's anchor: `_appearance_for_char` is a READER, not a
writer.** It chains portrait_prompt|appearance|description|character_description|name
and invents no era. My r2 question listed it as a candidate writer; cursor is right
to refuse the framing. (It does note the two copies disagree -- the image one lacks
`description` and the name fallback -- which is a separate tidy-up, not this arc.)

**5. THE BIGGEST FINDING, and it is not about Shakespeare at all.** The DEFAULT pack
`nodes/visual_styles/sci_fi_radio.json` asserts costume on every portrait:
`portrait_look` = *"**period-accurate costume** and environment, dramatic film
lighting"*, plus `portrait_look_talking`, `portrait_instruction_look`
("period-consistent"), `plate_look` ("period-accurate set") and
`still_word_title_mood_style` ("atmospheric period illustration"). It is byte-locked
to Python fixtures -- `PORTRAIT_LOOK_DEFAULT` at
`nodes/otr_meta_brief_image_prompt.py:171` is the identical string -- and
`tests/test_visual_styles_a1.py` pins them, so editing the JSON without the fixture
in the same commit fails the suite. `archival_documentary` carries the same pattern.

**This raises a question the driver will NOT answer on the operator's behalf:** for an
OLD-TIME RADIO product, "period-accurate" in the default look may be DELIBERATE -- the
1940s radio-era aesthetic, i.e. the house style, not a defect. If so, the default
pack is correct as written and only `shakespeare_stage_realism` is in scope. If not,
the product's default look asserts an era on every episode that does not roll a
specific style. Carried to r3 as an explicit scope question; not assumed either way.

**6. After the pack edit, no owner writes garments on non-My-Story lanes.** Verified
chain: stills read only `meta.story_brief_terms.setting[:2]`; `StoryBriefModel` has no
era field and its reflection prompt forbids invented dates; `raw_fields_from_ledger`
returns None outside My Story so `_rewrite_char_scene_from_source` never runs; and
casting's `character_description` format requests age/role/Face/Presence/Voice --
**clothing is not requested**. So today the pack IS the adaptation's period-dress
owner, and removing it dresses Macbeth in modern clothes.

Cursor's proposed resolution -- declare unspecified wardrobe modern/model-default on
every lane including Macbeth -- **collides with the operator's own fidelity rule**
("Shakespeare stays authentic") and with CLAUDE.md's adaptation-fidelity directive.
Driver ruling: that resolution is NOT accepted. The adaptation lanes need a garment
owner that is neither the visual-style pack nor bank-name routing. The candidate the
panel has not yet examined is the SOURCE itself -- `nodes/_otr_source_document.py`
already carries `SourceDocument`/`SourceSpan` with a canonical body, and for an
adaptation the source text is the legitimate era authority. **That is r3's first
question.**

**7. Acceptance, accepted as proposed.** Unit assertions on loaded pack objects and
composed strings -- no pixels, no runtime checker, no forbidden-word list at
generation time, and explicitly NOT a claim about identical clothing across beats.
A unit assert on authored config is config, not a gate.

## CUT LIST -- accepted

Era-aware eligibility; pack schema/new keys/`meta.period`/StoryBriefModel era field
(a new era field is a classifier); extending `raw_fields_from_ledger` to adaptation
banks (bank-shaped routing); per-role `positive_tail`; runtime costume checker, retry
or publication veto; wardrobe continuity / outfit lock (ripped 2026-08-27); teaching
casting's Face/Presence/Voice contract to author clothes.

## CARRIED TO r3

1. **Who owns garments on the adaptation lanes once the pack stops?** Examine
   `_otr_source_document.py` as the era authority. Is source-derived era distinct
   from bank-name routing, and is it reachable by the visual authors?
2. **Scope: is the DEFAULT pack's "period-accurate" the house style or a defect?**
   If in scope, the Python fixtures and `test_visual_styles_a1.py` change in the same
   commit or the suite fails.
3. `shot_lock._FALLBACK_SETTING = "a vintage radio studio"` vs the image path's
   empty string -- an era leak on character_video when the brief fails.
4. Exact rewrites: cursor supplied them field by field. Verify each against the
   loaded JSON and confirm `validate_pack` accepts (it rejects empty strings).

## Unchanged

Nothing from this arc ships before tonight's four-machine wave.
