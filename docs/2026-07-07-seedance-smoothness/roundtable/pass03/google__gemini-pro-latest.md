<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

...`
        *   Plan: `model["prompt"] = conditioned_prompt`.
        *   If the existing code uses object attributes (e.g., `model.prompt`), then `model["prompt"]` will crash with `TypeError`.
        *   I must flag this as a MUST-FIX.

    *   *Let's check the Idempotence Rule*:
        *   "Idempotence rule: if the clause, or a stable marker sentence from it, is already present, do not append it again."
        *
