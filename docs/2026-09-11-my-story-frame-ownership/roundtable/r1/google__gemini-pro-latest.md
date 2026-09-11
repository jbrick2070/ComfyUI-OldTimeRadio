<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview -->

Here is a rigorous critique of the proposed plan, focusing on defects, gaps, and risks in the implementation strategy:

### 1. Defect: Conflicting P0/P1 Instructions Remain Unresolved
The plan states it will "clarify existing phase ownership in P0/P1 pack instructions," but fails to address the root cause of the contradiction. 
In the actual raw source provided, P0 (`StoryInterpretation`) explicitly includes `Announcer` in `named_cast` with `"speaking": true, "required": true`, and justifies it in `cast_plan.reason`. 
Meanwhile, P1 (`my_story_treatment_system`) is instructed: "THE PERSON'S REQUIREMENTS OUTRANK YOUR INVENTION. Preserve their material and names... The cast array contains the story's speaking characters, excluding ANNOUNCER."
**The Risk:** If P0 continues to strictly mandate the Announcer as a required speaking character based on the user's prompt, P1 is trapped in a double-bind. It must obey P0's required cast, but is told to exclude the Announcer. Modifying the repair prompt won't fix the upstream P0 defect where the non-diegetic announcer is being treated as a diegetic requirement. P0's system prompt (`my_story_interpret_system`) must be updated to explicitly exclude house-frame roles from `named_cast`.

### 2. Gap: Lack of Conditional Repair Logic in `_pass_treatment`
The plan proposes to "give the EXISTING reserved-name typed-repair a concrete whole-treatment action: remove house-frame cast, replace frame turns with source-grounded dramatic beats..."
However, looking at `nodes/_otr_my_story.py:510`, `_pass_treatment` currently hardcodes a *single, static* repair instruction string:
`repair_prompt_factory=_full_artifact_repair("Reorganize the treatment into exactly %d acts...")`
**The Defect:** `_full_artifact_repair` does not currently accept conditional logic based on the validation error (unlike the Sci-Fi parser which checks `defects`). If you just append the new Announcer instructions to the static string, you will pollute the repair prompt for *all* treatment failures (e.g., wrong act count) with irrelevant instructions about the Announcer. The plan must specify how `_make_treatment_validator` or `_full_artifact_repair` will dynamically route the specific "Announcer" repair text only when the `ANNOUNCER_NAME` validation fails.

### 3. Risk: Over-reliance on "Global Ending" Semantic Rewriting
The plan requires the repair to "make globalending the characters' realized conclusion". 
In the failed Gemma output, the `ending` field was: `"The ANNOUNCER provides a closing thought..."`
**The Risk:** Instructing the model in a repair turn to semantically rewrite the `ending` field to match the `ending_state` risks hallucinating new plot points that weren't in the original source, especially under the pressure of a repair prompt. The validator (`_make_treatment_validator` at line 494) currently only checks cast names and act counts; it has no programmatic way to verify if the model successfully fixed the `ending` field. If the model removes the Announcer from the cast but leaves the Announcer in the `ending` string, the validator will pass it, and the downstream `_pass_act` (line 588) will feed that broken `global_ending` into the final act's scope, potentially breaking the dialogue generation anyway.