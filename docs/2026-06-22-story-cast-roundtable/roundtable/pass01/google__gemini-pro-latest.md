<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: no. The plan relies on micro-managing LLM creativity through dense, per-line JSON constraints, which mathematically guarantees the exact "flat line" and "uneven arc" failures you are trying to prevent, especially on 12B-class models. 

MUST-FIX BEFORE BUILD:
1. [Section 2 & 5.1/5.2] **Cognitive Overload / Constraint Conflict.** The plan forces the LLM to simultaneously optimize prose, JSON schema, `arc_phase`, `trait`, `beat_intent`, `line_job`, and `hidden_pressure` for *every single line*. Small models (Mistral/Gemma) cannot do this; they will output wooden dialogue to satisfy the JSON keys. 
   *Fix:* Decouple the writing from the metadata. Have the LLM write the scene in a standard script format first (focusing purely on the `dramatic_state`), then use a secondary parsing step (or a cheaper model) to map the text into the `lines[]` JSON ledger.
2. [Section 4 & 5.5] **Misattributing Pipeline Bugs to LLM Behavior.** The cast contract failure (`kokoro` or `bark` ending up in the `role` field instead of `character` or `announcer`) is a schema mapping/data-pipeline bug, not a creative prompting issue. You cannot prompt an LLM out of a bad system integration.
   *Fix:* Hardcode the `role` field mapping in your casting script. Do not expose TTS engine names to the LLM's role-assignment prompt at all.
3. [Section 4 & 5.3] **Blind Reroll Loop.** The reroll bounds out because the writer LLM isn't being told *why* a line failed. If the critic just flags "target(s): line_12", the writer has no gradient to improve it and will just hallucinate a lateral change.
   *Fix:* The critic must output a concrete `correction_instruction` string (e.g., "Line 12 needs to show Character A's hidden pressure of guilt") that is explicitly injected into the writer's reroll prompt.
4. [Section 3 & 5.5] **Silent Fallbacks on Voice Binding.** `voice_preset=None` means your casting generation lacks a strict enum validation. 
   *Fix:* Enforce a strict JSON schema validation on the casting output that rejects nulls for `voice_preset` and forces selection from a predefined list of valid IDs before the episode is allowed to proceed to the writer.

SHOULD-FIX:
1. [Section 4 & 5.6] **Fighting Cheap Post-Processing.** You want to stop stage-direction leakage `(whispering)` at generation. While a system prompt ("DO NOT USE PARENTHETICALS") helps, LLMs trained on screenplays will inevitably slip up. 
   *Fix:* Keep the regex post-scrub. It is deterministic, computationally free, and 100% reliable. Do not waste context window or reroll cycles trying to achieve 0% leakage at inference time.
2. [Section 4 & 5.4] **Voice Drift.** Characters lose their voice because the LL