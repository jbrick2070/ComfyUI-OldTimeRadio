<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: yes-with-fixes. The integration sequencing is finally correct, but the plan introduces several build-blocking ambiguities by demanding new text and algorithms without defining them.

MUST-FIX BEFORE BUILD:
1. **[BUILD 3] Missing Intent-Enrichment Copy:** The plan demands a "ROLE-KEYED map covering setup/pressure/personal_stake + every assignable CLIMAX_CLASS_ROLES member, each with class-specific tail text" for `_enrich_intent`, but provides ZERO text for these 11 roles. A builder cannot invent production copy.
   *Fix:* Provide the exact text, or simplify the rule: Do not enrich `setup` or `pressure`. For `personal_stake`, keep the existing text. For ANY `CLIMAX_CLASS_ROLES`, use: `f"on-stage, the {beat_role.replace('_', ' ')} regarding {obj} happens now, costing {fc.get('personal_cost', '')}"`.
2. **[BUILD 2] "Scored Matching" Ambiguity (K10):** "DOMAIN_PALETTE scored matching" is undefined and build-blocking. The implementor has no algorithm for how to score keyword hits.
   *Fix:* Drop scored matching entirely. Fix the "trial" collision trivially by changing the `_DOMAIN_KEYWORDS` entries to `("clinical trial", "medicine")` and `("court trial", "law")`.
3. **[BUILD 1] Prompt De-licensing Ambiguity:** "scope composer 1162 ('mission control ... fine') + 1442 ('Ground this line in the news facts') to compatible styles." There is no definition of which styles are "compatible", making this impossible to implement.
   *Fix:* Make the prompt universally compatible instead of gating it. Change line 1162 to: `Generic roles ("the tech", "the expert", "the director") are fine.` Leave 1442 alone (all episodes are news-grounded, regardless of style).
4. **[BUILD 2] Prompt Block Overlap:** The plan adds `style_grammar` to `LineRequest` and says to "render ONE canonical style-grammar block", but fails to specify what happens to the existing `style_descriptor` field and its render block in `_build_user_prompt`.
   *Fix:* Explicitly state: "Remove `style_descriptor` from `LineRequest`. In `_build_user_prompt`, replace the `STYLE: {req.style_descriptor}` block with the multi-line `req.style_grammar`."

SHOULD-FIX:
1. **[BUILD 1] Constant Name Mismatch:** The plan references `CLIMAX_CLASS_ROLES|{PRESSURE}` for the validator. `PRESSURE` is not defined; the constant in `_otr_story_quality_l12.py` is `BEAT_ROLE_PRESSURE`.
   *Fix:* Use the correct constant name `BEAT_ROLE_PRESSURE`.

OPTIONAL / NICE-TO-HAVE:
- In `BUILD 4`, when forcing `resolved=False` for the unresolved set, consider logging a warning if the `ending_change` string was actually populated, so the operator knows the engine overrode a resolved state.

CUT THESE:
1. **DOMAIN_PALETTE scored matching:** Over-engineered. A simple keyword string change solves the collision.
2. **Style-gating the composer prompt instructions:** Over-engineered and brittle. Changing the literal "mission control" to "the expert" solves the genre-bleed without adding conditional logic to the prompt builder.

VERIFY-AT-BUILD:
- [ ] Verify `_otr_pitch_room.run_pitch_room`'s `dataclasses.replace` preserves the new `style_grammar`, `sound_world`, `story_engine`, and `ending_tag` fields on `OutlineRequest`. [ASSUMPTION: `_otr_pitch_room.py` not in grounding].
- [ ] Verify `_otr_reroll.build_reroll_line_request` correctly threads the new `grounded_nouns` field. [ASSUMPTION: `_otr_reroll.py` not in grounding].
- [ ] Verify first run: dump `resolved["style"]` vs `contract.slug` to confirm the ADD-not-collapse strategy works and downstream consumers don't crash on the dual representation.