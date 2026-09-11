<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview -->

Here is the rigorous critique of the R2 plan.

### Defects and Risks (Must-Fix)

**1. Cross-Pack Regression in Global Source Rewrite (Section 3)**
*   **Defect:** The plan dictates removing `"An act need not repeat every fact"` from the common system text in `nodes/_otr_story_source.py:177`, stating that "Partial-act deferral belongs only to the `_pass_act` instruction above." However, `rewrite_story_source` is a shared global utility used by other packs (e.g., `scifi_news_pro`, `public_domain`). Removing this generic protection globally will cause other packs' partial-artifact rewrites (like single scenes or exchanges) to over-correct and hallucinate the entire source plot into a single slice.
*   **Citation:** Section 3: "Remove generic 'An act need not repeat every fact' from common system text."
*   **Owner:** Root / Source Pipeline Owner.
*   **Must-Fix:** Do not blindly delete this from the global prompt. Either parameterize the partial-artifact protection (e.g., `allow_partial_omissions=True`) or ensure that *all* callers evaluating partial artifacts inject an equivalent rule via the new `instruction` parameter, not just `my_story`'s `_pass_act`.

**2. Semantic Collision in Last-Act Prompting (Section 2)**
*   **Defect:** The plan states `_pass_act` will derive the endpoint from `treatment.ending` when `is_last`, otherwise retaining `plan.ending_state`. In `nodes/_otr_my_story.py:604`, the prompt template expects a granular state: `"- where it should leave the story: %s"`. `treatment.ending` is a global episode summary (one or two sentences), while `plan.ending_state` is a localized act beat. Silently swapping the variable without changing the prompt framing risks the LLM treating the global ending as an abrupt stage direction, or hallucinating epilogue narration outside the spoken dialogue.
*   **Citation:** Section 2: "_pass_act derives the endpoint from existing treatment.ending when is_last... Local endpoint remains available in complete treatment JSON."
*   **Owner:** Root / Prompt Engineering.
*   **Must-Fix:** If substituting the global ending in the final act, the prompt text itself must explicitly frame it as the global conclusion to be realized through dialogue (e.g., `"- episode conclusion to deliver here: %s"`), rather than just dropping a global summary into a local scene-state slot.

**3. Kwargs Leakage Implementation Detail (Section 2)**
*   **Defect:** The plan mandates `_call` explicitly consumes `source_rewrite_instruction=''` and "must not flow through **kwargs to structured_call." 
*   **Citation:** Section 2: "_call explicitly consumes source_rewrite_instruction='', forwards it... It must not flow through **kwargs"
*   **Owner:** Root / Python Implementation.
*   **Must-Fix:** Ensure the implementation explicitly adds `source_rewrite_instruction=""` to the `_call` signature before `**kwargs` in `nodes/_otr_my_story.py:375`. If it is popped from `kwargs` inside the function instead, ensure `kwargs.pop("source_rewrite_instruction", "")` is used so it strictly avoids reaching `structured_call`.