<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview -->

This is a rigorous critique of the R2 proposal. While the architectural decision to rely on existing owners and avoid new gates is sound, the concrete implementation details contain logical defects and over-engineering that risk introducing new failure modes.

### 1. Defect: Over-engineered and Presumptive Validator Error (Point 2)
The proposal dictates changing the reserved-name error in `_make_treatment_validator` (line 494) to a 64-word paragraph:
> *'ANNOUNCER is reserved for the separate frame pass. Rewrite the complete treatment: remove the house ANNOUNCER from cast; replace house-frame openings or closings in act turns with source-grounded dramatic beats; make ending and the final act ending_state the source's realized dramatic conclusion...'*

**The Flaw:** This error is triggered **solely** by `ANNOUNCER_NAME.casefold() in folded` (line 499). The validator does *not* check the `turns`, `ending`, or `ending_state` fields. 
By hardcoding instructions to "replace house-frame openings" and "make ending... the source's realized dramatic conclusion" into an error triggered only by a cast list violation, you are feeding the model a presumptive hallucination. If a model includes "ANNOUNCER" in the cast but writes perfectly valid, drama-only turns and endings, this error message will falsely accuse the model of corrupting the turns/ending. The model will then attempt to "fix" unbroken turns, likely degrading the actual story material to satisfy a misapplied instruction. 

**Resolution:** The error string must remain a statement of the exact validation failure, not a blind macro-repair script for unverified fields. If you want to instruct the model to check the whole treatment *when* the cast fails, phrase it conditionally: *"ANNOUNCER is reserved for the frame. Remove it from the cast, and if you included house-frame introductions or conclusions in the act turns or ending, replace them with the story's actual dramatic events."*

### 2. Risk: Name vs. Profession Collision (Points 1 & 4)
Point 1 and Point 4 emphasize: *"Preserve actual dramatic people, including an announcer as a profession."*
The existing validator strictly checks `if ANNOUNCER_NAME.casefold() in folded:` (line 499). 
If the original source is about a radio station and features a dramatic character literally named "Announcer" (or if the model extracts their profession as their name, which is common), the validator will unconditionally reject it. 

Because Point 4 adds a `source_rewrite_instruction` to P0 explicitly demanding the preservation of "an announcer as a profession", you risk creating an inescapable loop:
1. P0 (Interpret) correctly identifies the dramatic "Announcer" and includes them in `named_cast`.
2. P1 (Treatment) includes "Announcer" in the cast.
3. The validator rejects it.
4. The repair loop fails because the model is torn between the P0/source mandate to keep the character and the validator's mandate to drop the name.

**Resolution:** The proposal claims "No name-based guessing/deleting", but relies on a literal name-based gate. You must ensure the prompt instructions clearly distinguish between the *name* "ANNOUNCER" and the *role/profession* of an announcer, explicitly telling the model to give a dramatic announcer a distinct personal name (e.g., "Bob the Announcer") to survive the validator.

### 3. Gap: `source_rewrite_instruction` Injection (Point 4)
Point 4 proposes passing `source_rewrite_instruction` to P0 (`_pass_interpret`) and P1 (`_pass_treatment`). 
Looking at `_pass_interpret` (line 462) and `_pass_treatment` (line 510), both functions accept `**source_kwargs` and pass them to `_call`. `_call` (line 375) accepts `source_rewrite_instruction=""` and passes it to `rewrite_story_source`. 
This wiring is correct and supported. However, the proposed P0 instruction tells the model to *"explain the separate owner in cast_plan.reason"*. 
`cast_plan.reason` is meant to explain the delta between `requested` and `planned` counts (e.g., "The user asked for 2, but the waitress makes 3"). Forcing the model to use this field to explain system architecture ("The house frame is owned by a separate pass") pollutes the semantic purpose of the field and risks the model dropping the *actual* explanation for the dramatic cast count. 

### 4. Safe Modifications (Point 3)
The changes to `_pass_treatment`'s repair factory (line 510) and `_full_artifact_repair` (line 420) are well-targeted. Removing the unconditional *"Reorganize the treatment into exactly %d acts"* from the P1 repair is a necessary fix. Currently, if the model fails the ANNOUNCER check but has the correct act count, the repair prompt needlessly commands it to reorganize the acts, introducing structural instability. Changing this to *"If the act count is already correct, preserve its grouping"* correctly patches this defect without altering the signature.