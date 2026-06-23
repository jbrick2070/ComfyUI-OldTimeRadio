<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: no. Critical state-isolation and data-flow bugs will break the scorer and poison the revision prompts.

MUST-FIX BEFORE BUILD:
1. [Pass isolation] **`meta` dict context loss.** The plan mandates a "FRESH `meta` dict" per pass. `score_outline` (L100) calls `premise_texts(meta)` to extract upstream context (like news briefs) for the grounding palette. If `meta` is fresh/empty, `premise_texts` returns nothing, the grounding palette is empty, and `premise_grounding` scores will silently zero out.
   *Fix:* Do not use a "FRESH" empty dict. Deep-copy the *incoming* `meta` dict for each pass so upstream context is preserved, then merge only the winner's telemetry delta at the end.
2. [Revision overlay wiring] **Mutated intents poisoning the prior_macro.** The plan says to feed the "PRIOR winner's macro shape... (beat intents)" into the revision overlay. `build_sq_data` mutates beat intents in-place to substitute generic crisis nouns (documented in `_otr_story_select.py` L23). If you feed the *composed* (mutated) intents into the next pass, you force the LLM to train on the generic nouns, destroying structural diversity.
   *Fix:* Explicitly construct `prior_macro` using `raw_outline.beats` (the pre-`build_sq_data` intents), never the post-composition mutated outline.
3. [Build chunk 0] **`diversity_hint` misrouting in Path C.** The plan says to wire `diversity_hint` "into the Path C builders". Path C has three builders. `_build_phase_user_prompt` only assigns ALL-CAPS speakers to a skeleton (L872); injecting a structural dramatic hint here will confuse the schema and waste tokens.
   *Fix:* Specify that `diversity_hint` must only be wired into `_build_macro_user_prompt` and `_build_beat_user_prompt`.

SHOULD-FIX:
1. [Grader] **Speaker context in `composed_text`.** The grader evaluates the story based on `composed_text` ("SPOKEN dialogue lines pulled from the ledger rows"). If this extraction only concatenates the raw dialogue text, the LLM cannot track who is speaking or evaluate character consistency.
   *Fix:* Ensure the extraction explicitly prepends the speaker name to each line (e.g., `ALICE: [text]`).
2. [Revision overlay wiring] **`OutlineRequest` dataclass mutation.** The plan adds `prior_critique` and `prior_macro` as "FINAL fields... defaulted". `OutlineRequest` has a strict `__post_init__` (L431) that validates fields.
   *Fix:* Ensure the new fields are explicitly typed as `str = ""` in the dataclass definition in `_otr_outline.py`, and that `__post_init__` trims them safely without raising if they contain unexpected control characters.

OPTIONAL / NICE-TO-HAVE:
- In the never-fail fallback, explicitly log the `grade_error_type` if the grader fails, so telemetry clearly distinguishes between "story composed but graded 0" and "grader JSON parse failed".

CUT THESE (over-engineering):
1. [Pass isolation] **Deep-copying the entire `canon` container.** The canon is typically read-only during outline generation and composition (it holds established facts). Shallow copying or passing by reference is safe and avoids unnecessary memory overhead, unless the composer explicitly mutates it.