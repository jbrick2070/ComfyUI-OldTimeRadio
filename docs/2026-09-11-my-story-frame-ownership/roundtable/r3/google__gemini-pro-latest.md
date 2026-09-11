<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview -->

Here is a rigorous critique of the proposed R3 wiring and sequencing plan, focusing on concrete defects, risks, and over-engineering.

### 1. Over-engineered and Prescriptive Validator Error (Section 2)
**Defect:** The plan proposes changing the `_make_treatment_validator` reserved-name condition to return a massive, highly specific repair prompt: *"ANNOUNCER is reserved for the separate frame pass. Rewrite the complete treatment: remove the house ANNOUNCER from cast; replace house-frame openings or closings in act turns with source-grounded dramatic beats; make ending and the final act ending_state the source's realized dramatic conclusion..."*

**Risk:** This conflates a validation state with a highly prescriptive, scenario-specific repair instruction. It assumes that *every* instance of an "ANNOUNCER" cast leak will perfectly mirror the `terminal08` failure (i.e., that it will always include house-frame openings/closings in the acts and a corrupted global ending). 
If a model simply hallucinates "ANNOUNCER" into the cast list without actually writing them into the act turns, this error message will explicitly instruct the model to "replace house-frame openings... with source-grounded dramatic beats." This risks forcing the model to hallucinate new dramatic beats to fulfill a repair instruction for a symptom that didn't exist in that specific draft. The validator should state the invariant broken; the repair factory should handle the prompt engineering.

### 2. Global Pollution in `_full_artifact_repair` (Section 3)
**Defect:** The plan modifies the shared `_full_artifact_repair` wrapper to include: *"Preserve unaffected story events and relationships and the original source's intended ending; correct any named defect to respect the original source."*

**Risk:** `_full_artifact_repair` is a shared factory used across all four passes: Interpret, Treatment, Act, and Frame (see `nodes/_otr_my_story.py:462, 510, 588, 661`). 
Injecting "preserve... the original source's intended ending" globally pollutes the Frame pass (`_pass_frame`). The Frame pass owns the non-diegetic intro, outro, and coda. Instructing the Frame repair to preserve the "original source's intended ending" risks confusing the model into injecting dramatic story conclusions into the announcer's outro/coda, breaking the strict separation of diegetic and non-diegetic text that this entire fix is trying to enforce.

### 3. Redundant Test Matrix (Section 5)
**Over-engineering:** The plan demands testing the new repair with "Parameter acts1/3/6; three legitimate dramatic speakers vs hint2". 

**Critique:** Testing 1, 3, and 6 acts for this specific reserved-name repair is a waste of test execution time and maintenance surface. The repair logic (removing the announcer and fixing the ending) does not branch on the act count, nor does the validator's detection of the ANNOUNCER string. A single parameterized test proving the repair routing and application on a multi-act draft is sufficient. Multiplying this across act counts tests the Python `len()` function, not the repair logic.

### 4. Ambiguity in `source_rewrite_instruction` kwargs (Section 4)
**Gap:** The plan states: *"Use `source_kwargs['source_rewrite_instruction']=...` inside each phase... **kwargs creates a fresh dict, so this cannot mutate the caller's mapping"*. 

**Critique:** While `**source_kwargs` in the function signature creates a local dictionary, if you pass `source_rewrite_instruction` directly into `_call` via `**source_kwargs`, you must ensure it doesn't collide with `_call`'s explicit signature. `_call` is defined as `def _call(..., source_rewrite_instruction="", **kwargs)`. If a caller passes `source_rewrite_instruction` inside `**source_kwargs`, Python will raise a `TypeError: _call() got multiple values for keyword argument 'source_rewrite_instruction'` because it is defined as a named parameter, not just a sink in `**kwargs`. You must pass it explicitly as a named argument to `_call`, not blindly pack it into `**source_kwargs`.