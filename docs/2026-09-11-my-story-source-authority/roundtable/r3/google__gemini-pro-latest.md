<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview -->

This is a highly rigorous and well-constrained plan. The exact seams are correctly identified, and the proposed changes directly address the R2 findings without introducing scope creep. 

Here is the adversarial review of the specific mechanics:

**1. `**source_kwargs` Mutation (Safe, but requires strict adherence)**
*Plan:* "Assign `source_kwargs['source_rewrite_instruction'] = act_scope` in `_pass_act`'s private kwargs dictionary. Pass `**source_kwargs` once to `_call`."
*Critique:* Because `_pass_act` receives `**source_kwargs` in its signature (e.g., `def _pass_act(..., **source_kwargs)`), Python creates a *new* local dictionary for those kwargs. Mutating it locally before passing it to `_call` is perfectly safe and will not leak the `act_scope` to subsequent acts in the loop (which was a potential risk if a shared dictionary were passed). Ensure the implementation strictly mutates this local `source_kwargs` and does not attempt to mutate a higher-level shared config object.

**2. `_call` Signature Modification**
*Plan:* "`_call` signature gains keyword-only `source_rewrite_instruction=""` before `**kwargs`."
*Critique:* This is the correct approach. By making it an explicit keyword argument in `_call` (`def _call(..., source_rewrite_instruction="", **kwargs)`), it safely absorbs the value unpacked from `_pass_act`'s `**source_kwargs` while allowing other callers (`_pass_interpret`, `_pass_treatment`) to omit it and safely default to `""`. 

**3. Exact Prompt Replacement**
*Plan:* Replace `"An act need not repeat every fact; speculation is not a fact, and absence "` with `"A partial artifact need not repeat source facts outside its scope. Speculation is not a fact, and absence "`
*Critique:* Grounding check confirms this exact string exists at `nodes/_otr_story_source.py:182`. The replacement is surgically precise and correctly mitigates the Terra scene-caller risk by removing the word "act" from the shared instruction, replacing it with "partial artifact".

**4. String Interpolation Safety**
*Plan:* `"This is the final act... dialogue here: %s. ... " % global_ending`
*Critique:* Because `global_ending` (which originates from user/model input) is passed as the *argument* to the `%` operator rather than being concatenated into the format string itself, there is no risk of a `ValueError` if the user's ending happens to contain a `%` character. 

**Conclusion:**
No defects, gaps, or over-engineering found. The plan correctly isolates the scope instruction to the act pass, safely pipes it through the existing `_call` boundary, and patches the shared source prompt without breaking other artifact types. Proceed with implementation.