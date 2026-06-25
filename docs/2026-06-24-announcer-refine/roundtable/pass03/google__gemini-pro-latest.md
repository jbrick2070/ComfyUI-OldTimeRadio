<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: yes-with-fixes. The pipeline sequence is mostly sound, but in-place mutations and missing state-rebuilds will cause prompt poisoning and data loss under the flag.

MUST-FIX BEFORE BUILD:

1. [Section 2 / Section 4] **Enriched Intent Leak into Open (Sequence Bug)**: KILL 4 enriches `BEAT_ROLE_SETUP` via `_otr_story_quality_l12.py`, which mutates the beat's `intent` in-place (`setattr(b, "intent", new_intent)`). If the OPEN reads the setup beat's intent at :4465, it will read the *enriched* intent (containing meta-instructions like "This establishes the normal world"), poisoning the `opening_status_quo`.
   *Fix*: Capture `opening_status_quo` directly from `outline.beats[1].intent` immediately after `generate_outline` (:3101), BEFORE the `_otr_story_quality_l12` pass mutates it, and store it in `SafeOpenBrief` then.

2. [Section 1 / R3 Facts] **Reroll Rebuild Drops Compact Register (State Loss)**: The R3 Wiring Facts explicitly warn that `build_reroll_line_request` (:3922) must rebuild any new line-level fields, or rerolled lines lose them. Section 1 says to pass the compact register tag to body lines, but fails to instruct updating the rebuild function.
   *Fix*: Explicitly update `build_reroll_line_request` to extract the compact register tag from `meta["story_contract"]["label"]` and inject it into the rebuilt `LineRequest`.

3. [Section 3] **Fallback Logic Leaks Fictional Ending (Branching Bug)**: Section 3 says to "SUPPRESS the resolved-fiction branch". However, the existing fallback logic (Grounding :2840) reads: `if not brief and not close: fb = _resolved_outro_fallback(ending_change, close) if resolved else...`. If `news_close_brief` is empty or stripped, this will trigger `_resolved_outro_fallback` and leak the fictional ending into the news coda fallback.
   *Fix*: In `compose_announcer_outro`, update the early-out `if not brief and not close:` block to ALWAYS route to `fallback_news_coda_outro(coda_lead_in, close)` when under the flag, completely bypassing `_resolved_outro_fallback`.

4. [Section 0 / Section 1] **Phase/Beat Prompt Threading Disconnect (Interface Mismatch)**: Section 0 notes phase/beat prompts take `macro`, not `OutlineRequest`, and Section 1 says to "render story_engine + ending_mode in the macro prompt (and phase/beat...)". `macro` is an LLM-generated output dataclass; request fields do not automatically pass through it.
   *Fix*: Modify the Python signatures of `_build_phase_user_prompt` and `_build_beat_user_prompt` to accept `StoryContract` (or `OutlineRequest`) as an explicit additional parameter. Do not attempt to thread it through the parsed `macro` object.

5. [Section 1] **`select_style` Deletion Breaks Downstream State (Variable Init)**: Section 1 says "DELETE the late `select_style(...)` @ :3224". But Section 0 mandates "Do NOT touch `resolved["style"]` / `meta.style`". If the assignment at :3224 is blindly deleted, the variable holding the style slug becomes uninitialized, breaking the exact downstream state assignments you are trying to preserve.
   *Fix*: Do not delete the line at :3224; replace the right-hand side with the already-built `contract.slug` (e.g., `style_slug = contract.slug`) to safely feed the existing downstream assignments.

SHOULD-FIX:

1. [Section 3] **Climax Line Selection Ambiguity**: Section 3 says "find the ledger line where `_ln.get("beat_id") == _climax_beat_id`". A single beat can produce multiple ledger lines (e.g., split sentences).
   *Fix*: Specify "take the LAST ledger line matching `beat_id == _climax_beat_id`" to ensure the final thought of the climax is used, matching the existing `reversed()` scan behavior.

2. [Section 3] **Validator Execution Order**: Section 3 says "The validator asserts the body does NOT already contain a lead-in variant."
   *Fix*: Explicitly state that `validate_news_coda_line` must run on the raw LLM output *before* the composer prepends `coda_lead_in`, otherwise the validator will always fail its own prefix.

3. [Section 2] **`forbidden_tokens` Dependency**: Section 2 requires computing `forbidden_tokens` using `ending_change` at the OPEN call site (:4465).
   *Fix*: Note that the caller at :4465 must explicitly extract `ending_change` from `meta.dramatic_state.ending_change` [ASSUMPTION: it is populated by the outline by this point], as it is not natively in scope for the OPEN.