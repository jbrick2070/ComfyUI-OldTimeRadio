<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: yes-with-fixes. The structural changes are sound, but moving the style selection pre-outline and moving the crisis-noun gate into the line composer both break existing data-flow dependencies that must be patched.

MUST-FIX BEFORE BUILD:
1. [K1] **Missing `grounded` noun palette in `LineRequest`.** The plan proposes running `count_ungrounded_crisis` inside the line composer (`_otr_line_composer.py`). However, `count_ungrounded_crisis` requires a `frozenset` of grounded premise nouns. `LineRequest` only carries `allowed_people` and `allowed_things` (which are just cast names and key terms, not the full premise noun palette). 
   *Fix:* Add `grounded_nouns: frozenset[str] = field(default_factory=frozenset)` to `LineRequest`. In `OTR_LedgerScriptWriter.py` (Section I), compute it once via `_OTRSQL12.premise_noun_palette(allowed_roster, resolved["news_seed"], ...)` and pass it into `_build_line_request_for_beat`.
2. [K2/K5] **Chicken-and-egg in pre-outline `select_style`.** The plan moves catalog style selection BEFORE `generate_outline` so the style grammar can be injected into the outline prompts. But `_otr_style_catalog.select_style` (line 356) requires a `premise` string, which is currently fed from `outline.premise` in Section F2 of the writer. If called pre-outline, `outline.premise` does not exist.
   *Fix:* In `OTR_LedgerScriptWriter.py`, pass `script_brief` (or `resolved["news_seed"]` if the brief is empty) to `select_style` instead of `outline.premise` when moving the call to Section D.2.
3. [K7] **Dead-coded `ARC_PHASE_GUIDANCE` confirmed.** As suspected in the plan, `_position_for` in `OTR_LedgerScriptWriter.py` (lines 1358-1383) returns only "phase, beat N of M. Next phase: X." It drops the dramatic-function directive. Because `req.position` is always truthy, `_otr_line_composer.py`'s `_build_user_prompt` (lines 600-613) takes the `if req.position:` branch and NEVER executes the `elif req.arc_phase:` branch. The model lost the guidance.
   *Fix:* In `_otr_line_composer.py` `_build_user_prompt`, fetch `guidance = _OTRB.ARC_PHASE_GUIDANCE.get(req.arc_phase, "")` unconditionally, and append it to the `POSITION` block: `parts.append(f"POSITION: {req.position} {guidance}")`.

SHOULD-FIX:
1. [Self-Correction / C5] **`compose_announcer_outro` signature mismatch.** The plan states: "move the 'do not resolve outcome' policy INTO `compose_announcer_outro` (pass `ending_tag`/`style_slug`)". `compose_announcer_outro` in `_otr_line_composer.py` (line 1121) currently does not accept `ending_tag`. 
   *Fix:* Add `ending_tag: str = ""` to the `compose_announcer_outro` signature. In the function body, if `ending_tag == "unresolved_final_sound"`, force `resolved = False` and append the "Do NOT resolve" instruction to `system_content` to override the F3 outcome-state logic.
2. [K4] **Climax position shift breaks outro's `final_character_line`.** If the climax is no longer forced to be the LAST voiced character beat, the announcer outro pass in `OTR_LedgerScriptWriter.py` (Section I.5, lines 1629-1634) will grab the wrong line. It blindly searches `reversed(led.data.get("lines"))` for the last character beat, which will now be a `consequence` beat rather than the climax.
   *Fix:* In Section I.5, look up the text of `_climax_beat_id` specifically to pass as `final_character_line`, rather than just the last character line in the ledger.

OPTIONAL / NICE-TO-HAVE:
- [K11] **Style-driven render profile in `eng_ltx_av`.** To make the video render style-driven, `eng_ltx_av.py`'s `_build_render_request` (line 289) needs access to the style slug. Pass `style_slug` down through the `request` dictionary from the orchestrator so the engine can map it to specific CFG/step overrides.

CUT THESE:
1. [K8] **Domain palette fallback.** The plan suggests adding real domain entries to `_PERSONAL_COST` or dropping the domain arg. Since `fallback_content` safely defaults to `"general"` (line 343 in `_otr_story_quality_l12.py`), this is harmless tech debt. Safe to cut for this round to focus on the structural gates.