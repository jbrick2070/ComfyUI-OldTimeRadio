<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: no. The style catalog is completely disconnected from the prompt payloads, and the L12 anti-trope system only grounds the outline intent while leaving the dialogue LLM free to hallucinate its training priors.

MUST-FIX BEFORE BUILD:

1. [_otr_style_catalog.py / OTR_LedgerScriptWriter.py] **Style diversity dies because the rich grammar is never injected.** `_otr_style_catalog.py` defines `render_style_grammar()` to output the `sound_world` and `story_engine`, but `OTR_LedgerScriptWriter.py` never calls it. It passes the bare slug (e.g., `memory_erasure_clinic_session`) to `OutlineRequest.style` and `LineRequest.style_descriptor`. The LLM ignores the slug in favor of the news premise.
   *Fix:* In `OTR_LedgerScriptWriter.py` `_resolve_inputs`, call `_otr_style_catalog.render_style_grammar(resolved_style)` and pass the resulting string to the outline and composer requests, not the bare slug.

2. [_otr_line_composer.py] **Dead code / Cargo-culting of Phase Guidance.** In `_build_user_prompt`, the `POSITION` block (`if req.position:`) completely shadows the `ARC PHASE` block (`elif req.arc_phase:`). Because `OTR_LedgerScriptWriter.py` now always populates `req.position`, the `ARC_PHASE_GUIDANCE` (which tells the LLM what "setup" or "pressure" actually means) is never rendered. This contributes to the body beats collapsing.
   *Fix:* In `_otr_line_composer.py`, fetch `ARC_PHASE_GUIDANCE.get(req.arc_phase)` and append it to the `POSITION` block so the model actually receives the phase instructions.

3. [_otr_story_quality_l12.py / OTR_LedgerScriptWriter.py] **Single-prior trap on Console Standoffs.** `ground_crisis_nouns` replaces `GENERIC_CRISIS_NOUNS` (lever, console, countdown) in the outline's `beat.intent` ONLY. It does nothing to prevent the line composer LLM from generating those exact words in the dialogue, which Gemma will inevitably do because of its sci-fi training priors.
   *Fix:* In `OTR_LedgerScriptWriter.py`, pass the ungrounded `GENERIC_CRISIS_NOUNS` into `compose_line` as `stage3_banned_phrases`, or explicitly append them to the `WRITE LINE` negative constraints in `_otr_line_composer.py`.

4. [_otr_story_quality_l12.py] **False-distinction / Body beats abandoned.** `build_sq_data` explicitly gates its `fallback_content` and `_enrich_intent` logic behind `if beat_role in (BEAT_ROLE_PERSONAL_STAKE, BEAT_ROLE_IRREVERSIBLE_CHOICE):`. The body beats (`setup`, `pressure`, `consequence`) are starved of dramatic framing, leaving the LLM to default to generic tropes.
   *Fix:* Remove the `if beat_role in (...)` gate in `build_sq_data` and apply `fallback_content` and `_enrich_intent` to all dramatic-function beats.

SHOULD-FIX:

1. [_otr_outline.py] **Forced-default hides outline failures.** `Beat.arc_phase` uses `Field(default="setup")`. If a weak LLM omits the phase on a climax beat, it defaults to "setup". While the monotonic validator catches out-of-order phases, a missing phase on the *first* beat silently succeeds but corrupts downstream logic.
   *Fix:* Remove `default="setup"` and make `arc_phase` strictly required, forcing the retry ladder to fix the omission.

2. [_otr_line_composer.py] **Defended-invariant audit on Stage Directions.** The prompt defends `Output ONLY the words... no stage directions anywhere`. But `compose_line_draft` relies on a regex (`detect_stage_business_for_reroll`) to catch failures. If the LLM consistently fails, it burns retries. 
   *Fix:* Add a positive constraint: "Format your output as plain text." Instruct-tuned models often wrap actions in asterisks because they think they are roleplaying in a chat interface.

CUT THESE (scope / over-engineering):

1. `use_exchange` and `_otr_compose_exchange` in `OTR_LedgerScriptWriter.py`. It is a massive, parallel dialogue-generation path ("Build 4 grouped-exchange") that defaults to OFF and falls back to the legacy composer on failure. It bloats the writer and splits the testing surface before the core per-beat composer can even follow the style grammar reliably. Safe to cut.

[ASSUMPTION] The `_otr_stage3_validators.py` module (not provided) actually enforces `stage3_banned_phrases` at the dialogue level. If it only warns, Defect 3's fix must be applied directly to the `_build_user_prompt` system instructions.