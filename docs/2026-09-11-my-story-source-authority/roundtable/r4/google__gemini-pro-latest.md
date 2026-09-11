<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview -->

Here is the rigorous review of the R4 final convergence plan.

### 1. Defect: Local Endpoint Obfuscation in Author Prompt
**Section:** 4 (`_pass_act` derivation)
**Issue:** The plan states `endpoint = global_ending or plan.ending_state` and instructs to "Use endpoint in the EXISTING user target line". 
If `global_ending` is truthy, the explicit bullet point `- where it should leave the story: %s` will contain *only* the global ending. The `act_scope` text tells the model: *"Its explicit target is the episode conclusion, which supersedes this act's planned ending_state where they conflict."* 
However, because the bullet point replaced `plan.ending_state`, the author model's immediate instruction block lacks the local scene resolution. While the model can technically hunt for `plan.ending_state` inside the raw `THE ACCEPTED TREATMENT` JSON dump earlier in the prompt, removing it from the explicit task list risks the model abandoning the immediate physical scene resolution (e.g., escaping the room) to jump straight to the thematic global ending (e.g., they lived happily ever after).
**Fix:** Do not replace the local endpoint in the bullet list. Instead, append the global ending to the bullet if it is the final act, e.g., `- where it should leave the story: [plan.ending_state] (Global Target: [global_ending])`, so the model can synthesize the immediate scene exit with the final episode conclusion.

### 2. Risk: Unintended Consequence on Frame/Scene Correction
**Section:** 5 (Common correction prompt replacement)
**Issue:** Changing `"An act need not repeat every fact..."` to `"A partial artifact need not repeat source facts outside its scope..."` applies globally to `rewrite_story_source`. 
While the plan acknowledges this affects `spoken`, `scene`, and `frame` consumers, the phrase "outside its scope" is highly abstract for a zero-shot LLM corrector. For the `frame` (which generates announcer intros/outros), the model might misinterpret "outside its scope" and aggressively prune required attribution sentences or requested music cues if it decides they belong in the "acts" rather than the "frame". 
**Mitigation:** Ensure the `instruction` kwarg passed by `_pass_frame` explicitly defines the frame's scope to prevent the corrector from stripping frame-specific requirements under the new "partial artifact" clause.

### Limits & Boundaries
*   **No new architectural defects found:** The routing of `source_rewrite_instruction` through `_call` to `rewrite_story_source` is syntactically sound and respects the existing `**kwargs` expansion.
*   **Data Preservation:** The plan correctly preserves the original `treatment.ending` logic without mutating the underlying Pydantic models or database schemas.