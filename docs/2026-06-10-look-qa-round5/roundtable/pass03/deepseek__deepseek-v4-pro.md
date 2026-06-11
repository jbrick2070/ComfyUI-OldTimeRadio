<!-- requested_model: deepseek/deepseek-v4-pro | resolved_model: deepseek/deepseek-v4-pro-20260423 -->

VERDICT: yes-with-fixes. The plan addresses the core defects, but several hidden dependencies and unspecified details must be verified or resolved before building.

MUST-FIX BEFORE BUILD:
1. [F2] The plan assumes `beat_intent` and `arc_phase` fields exist on ledger lines. If absent, the beat clause is silently skipped, and the diversity gate may false-positive (all prompts identical). Verify that the writer emits these fields; otherwise the per-beat variety improvement is lost. [ASSUMPTION]
2. [F4] The plan says "enumerate at build" for speaker-identity fields to update during re-attribution. Without the writer source, the exact set of fields is unknown. Missing one (e.g., `speaker_role`, `voice_id`) would leave stale identity. Verify against `OTR_LedgerScriptWriter.py` and list all fields. [VERIFY]
3. [F5] The announcer char_id resolution depends on a cast table entry with name "ANNOUNCER". Confirm that such an entry exists; otherwise `char_id` remains empty for announcer shots. [ASSUMPTION]
4. [F2] The diversity gate checks "brief-composed LTX sha8s must not all be equal". If there is only one such shot, the condition is trivially true and should not trigger a warning. Add a guard for `n >= 2`.
5. [F2] Trace rows gain `prompt_sha8`, `prompt_source`, `prompt_chars`. The plan states these land in the node-92 /history report. Node-92 (`OTR_VideoRenderBatch`) must be updated to read and include these fields; otherwise the diversity gate cannot access them. [VERIFY]
6. [F5] Manifest rows gain `init_image`. The plan does not specify how `init_image` is resolved. It must be derived from the ledger's image index using `char_id` (similar to `build_request_from_shot`). Without this, the face-acceptance check remains non-mechanical. [ASSUMPTION]

SHOULD-FIX:
1. [F3] The anchor prepends the full `appearance` string, which may be long. Consider truncating to ~100 chars to avoid excessively long prompts that might exceed engine token limits (e.g., HuMo prompt length).
2. [F2] The beat clause mapping table for `beat_intent` is not defined. Provide the fixed table explicitly in the plan or code to avoid ambiguity and ensure consistent mapping.

OPTIONAL / NICE-TO-HAVE:
- None.

CUT THESE (over-engineering):
- None. All parts serve a clear purpose.

[ASSUMPTION] items: F2 beat_intent/arc_phase presence, F5 announcer cast entry, F5 init_image resolution.