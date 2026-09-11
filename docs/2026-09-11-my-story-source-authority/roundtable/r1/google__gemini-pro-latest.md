<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview -->

Here is a rigorous critique of the R1 proposal, focusing on defects, architectural gaps, and risks based strictly on the provided code and logs.

### 1. Defect in Action 2: Redundant/Conflicting Raw Source Injection
**Proposal:** "...with raw listener source above derived plans."
**Critique:** This is redundant and risks prompt formatting errors. `nodes/_otr_my_story.py:382` inside `_call` *already* prepends the raw source to the final prompt message for every pass:
`prompt[-1]["content"] = _SOURCE.raw_source_block(bundle.fields) + "\n\n" + prompt[-1]["content"]`
If you attempt to manually add the raw source again inside `_pass_act`'s prompt definition (line 606), you will double-inject the raw source, wasting tokens and potentially confusing the model with duplicate blocks. 

### 2. Architectural Gap in Action 3: Corrector Context Blindness
**Proposal:** "Give the existing source corrector appropriate scope instructions: a partial act may defer a fact... the sole/final act and complete spoken ledger cannot defer an ending..."
**Critique:** You cannot currently express this using existing caller data without modifying `_call`. 
Look at `_call` (line 375) and `rewrite_story_source` (line 125). `rewrite_story_source` accepts an `instruction=""` parameter, but `_call` does not expose a way for `_pass_act` to pass a custom `source_rewrite_instruction`. `_call` hardcodes the call to `_SOURCE.rewrite_story_source` (line 395) without passing any act-specific instructions. To implement Action 3, you must wire a new kwarg (e.g., `rewrite_instruction`) through `_call` so `_pass_act` can conditionally pass the "final act cannot defer" rule based on its `is_last` boolean.

### 3. Risk in Action 1: Over-engineering "Incidental" vs "Requirement"
**Proposal:** "Make existing P0 semantics explicit: directed people/actions/relationships/ending are requirements even if brief or nonspeaking..."
**Critique:** The `my_story_interpret_system` prompt already explicitly warns: *"A REQUIREMENT is something they are asking for. An incidental mention is not."* By aggressively redefining "brief or nonspeaking" mentions as mandatory requirements, you risk breaking the model's ability to filter out actual background noise (e.g., "my sister loves lighthouses, anyway the story is about a diver"). The failure in P0/P1 was dropping the *ending/girlfriend*, which are core narrative beats, not just background nouns. Fix the ending obligation in the treatment/act prompts (Action 2) rather than bloating the interpret prompt's definition of a noun, which risks exploding the `conflicts` array.

### Verdict
**CONDITIONAL GO**, but you must:
1. Drop the "raw listener source above derived plans" addition from Action 2, as `_call` already handles it globally.
2. Add the necessary wiring in `_call` to pass `is_last` context to `rewrite_story_source`'s `instruction` parameter for Action 3.
3. Restrict Action 1 to clarifying that *endings and core relationships* are not incidental, rather than making a blanket rule about "brief/nonspeaking" entities.