<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: yes-with-fixes. The prompt-conditioning strategy is sound, but the observability plan violates ComfyUI's DAG data flow, and the proposed injection points reference ungrounded methods.

MUST-FIX BEFORE BUILD:

1.  **[Section 1 & Open Questions] Invalid Injection Point:** The plan asks if the conditioner should live in `CloudSeedance2Engine._text_prompt_input()`. The grounding does not show this method; it shows `_partner_inputs()` mapping `model.prompt`.
    *Fix:* Implement the Seedance-only prompt conditioner and verb softener directly inside `CloudSeedance2Engine._partner_inputs()` by mutating `req["text_prompt"]` right before assigning it to the partner node's `model.prompt` field. Do not modify `render_driver.py`, as that will leak Seedance-specific conditioning to Wan and Kling.
2.  **[Section 3] DAG Data Flow Violation:** The plan proposes unified observability linking generation-time data (prompt hash, requested duration) with downstream data (canonical frames, delivered frame status). In ComfyUI, nodes are isolated. You cannot pass generation metadata to `timeline_quality_report` without altering the workflow JSON to wire up a new metadata side-channel.
    *Fix:* Restrict observability to standard Python `logging.info()`. Log the conditioned prompt and requested duration inside `CloudSeedance2Engine`. Log the truncation/loop-fill status independently inside `OTR_SilentComposite`. Do not attempt to build a unified cross-node report object in code.

SHOULD-FIX:

1.  **[Section 2] Brittle String Replacements:** Hardcoded string replacements like `"whip-pans" -> "slowly sweeps"` will fail on case variations or singular forms (e.g., "whip-pan", "Whip-pans").
    *Fix:* Implement the verb softener using case-insensitive regex with word boundaries (e.g., `re.sub(r'(?i)\bwhip-pans?\b', 'slowly sweeps', prompt)`). Alternatively, fork `sci_fi_radio.json` into a Seedance-specific style pack to avoid runtime regex hacking entirely.
2.  **[Open Questions] Prompt Suffix for Truncation:** The plan asks if the prompt should explicitly state "motion begins immediately". Because `OTR_SilentComposite` strictly head-trims clips that exceed the beat (e.g., a 4s minimum provider clip trimmed to a 2s beat), any delayed action will be cut.
    *Fix:* Yes, append "Action and motion begin immediately on the first frame" to the Step 1 conditioner to ensure the usable motion survives the downstream head-trim.

CUT THESE (over-engineering):

1.  **[Section 3] Prompt Hashes:** Hashing the prompt adds no diagnostic value over simply logging the original and conditioned prompt strings in the ComfyUI server log. Cut the hashing logic.
2.  **[Section 5] Reference Video Guidance:** Safe to cut. As noted in the plan, it triggers a different cost envelope and requires new input plumbing. Prompt conditioning is the correct first step.

[ASSUMPTION] The plan assumes `CloudSeedance2Engine` has access to the raw `req["text_prompt"]` before it is sent to the partner node. Based on the grounding of `_partner_inputs()`, this is a safe assumption, provided the mutation happens before the partner node dictionary is constructed.
