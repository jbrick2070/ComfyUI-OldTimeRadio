<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: no. The load-bearing K1 body-gate is entirely missing, K2/K5 StoryContract was ignored, and critical interface mismatches will cause the climax to use a different style than the outline.

MUST-FIX BEFORE BUILD:
1. [OTR_LedgerScriptWriter.py / _otr_story_quality_l12.py] K1 Body Gate Missing. `validate_composed_grounding` is not implemented in `_otr_story_quality_l12.py`, and it is NEVER called in `OTR_LedgerScriptWriter.py` Section I. The composer loop blindly accepts `line_res.text` without validating grounded nouns or triggering the reroll loop.
   - Fix: Implement `validate_composed_grounding`. In `OTR_LedgerScriptWriter.py` Section I, call it on `cleaned` for character beats and trigger the `compose_line` retry ladder if it fails.

2. [OTR_LedgerScriptWriter.py] Style Interface Conflict. Section D.2 selects the episode style (`resolved["style"]`). But Section F2 ignores this and calls `_OTRSTYLE.select_style(_premise_str, meta, cast_seed)` to pick a *new* style, injecting that new style's ending template into the climax beat. The outline and the climax will diverge.
   - Fix: In Section F2, do not call `select_style`. Look up the `ending_tag` and `ending_template` using `_style_slug = resolved["style"]`.

3. [_otr_style_catalog.py] K2/K5 StoryContract Missing. The synthesis mandated a `StoryContract` dataclass and a `build_story_contract` helper. Neither exists in the file; it remains a list of dicts.
   - Fix: Implement the `StoryContract` dataclass and `build_story_contract` as specified in the R2 synthesis.

4. [_otr_line_composer.py] K1 Data Plumbing Missing. `LineRequest` does not have the `grounded_nouns` field required to pass the premise palette to the composer.
   - Fix: Add `grounded_nouns: frozenset = field(default_factory=frozenset)` to `LineRequest`. Compute it via `premise_noun_palette` in the writer and pass it in.

5. [_otr_line_composer.py] Build 4 Announcer Close Unification Ignored. `compose_announcer_outro` lacks the `ending_tag` parameter and the logic to override the `resolved` flag for unresolved/revelation/quiet styles.
   - Fix: Add `ending_tag: str = ""` to `compose_announcer_outro`. If `ending_tag` implies an unresolved ending, force `resolved = False` and append "do not resolve the outcome" to the system prompt.

6. [_otr_story_quality_l12.py] K3 Body Beat Starvation. `build_sq_data` (line 447) still hardcodes `if beat_role in (BEAT_ROLE_PERSONAL_STAKE, BEAT_ROLE_IRREVERSIBLE_CHOICE):`, starving all other beats of fallback enrichment.
   - Fix: Change the condition to `if beat_role in CLIMAX_CLASS_ROLES or beat_role in (BEAT_ROLE_SETUP, BEAT_ROLE_PRESSURE, BEAT_ROLE_CONSEQUENCE):`.

7. [_otr_story_quality_l12.py] K3 Truncation Bug. `_enrich_intent` appends the tail, but `build_sq_data` (line 451) blindly does `new_intent = new_intent.strip()[:_INTENT_MAX].strip()`, truncating the newly added tail instead of the original intent.
   - Fix: In `_enrich_intent`, calculate the remaining length (`_INTENT_MAX - len(sep) - len(tail)`), truncate the original `intent` to that length FIRST, then append the tail.

8. [OTR_LedgerScriptWriter.py] Build 4 Grouped-Exchange Bypass. In Section I, if `_ex_text is not None`, it short-circuits `compose_line`. This means the grouped-exchange text completely bypasses `cast_strip`, `detect_phantom_names`, `strip_announcer_vocative`, and the Stage 3 validators.
   - Fix: Extract the deterministic strip pipeline and Stage 3 validators from `compose_line` into a shared helper, and run it on `_ex_text` before appending to `last_lines`.

9. [_otr_line_composer.py] K7 ARC_PHASE_GUIDANCE Shadowed. In `_build_user_prompt` (line 1195), `req.position` still completely shadows `req.arc_phase`. The guidance is dropped.
   - Fix: In the `if req.position:` block, look up the guidance from `_OTRB.ARC_PHASE_GUIDANCE` and append it: `parts.append(f"POSITION: {req.position} {guidance}")`.

SHOULD-FIX:
1. [OTR_LedgerScriptWriter.py] `_sqv2_raw` Local Import Leak. In Section D.1, `import os as _os_sqv2` is used to check the `OTR_STORY_QUALITY_V2` env var. Later in Section D.2.5 (line 2187), `import os` is called again. This is fragile and clutters the function scope.
   - Fix: Move `import os` to the top of the `run` method and use it consistently.