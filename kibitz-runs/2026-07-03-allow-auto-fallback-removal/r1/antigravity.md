VERDICT: yes-with-fixes. The code and workflow changes are clean, but missing updates to the central validator's forbidden socket list and outdated architecture documentation create drift.

MUST-FIX BEFORE BUILD:
1. [nodes/_workflow_validation.py] The plan fails to add "allow_auto_fallback" to the central `FORBIDDEN_INPUT_SOCKETS` frozenset. Stale workflows using it on other surfaces (e.g. title or widget) might bypass check 5 [ASSUMPTION]. Fix: Add `"allow_auto_fallback"` to the `FORBIDDEN_INPUT_SOCKETS` frozenset in [nodes/_workflow_validation.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_workflow_validation.py#L135-L146) to tombstone it completely.
2. [docs/2026-06-02-video-engine-architecture__consolidated-final.md] The main architecture documentation still lists `allow_auto_fallback` inside the `policy` sub-model under the Normalized `VideoRequest` definition, leading to drift. Fix: Remove `allow_auto_fallback` from the policy parameters on line 85 of [docs/2026-06-02-video-engine-architecture__consolidated-final.md](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/docs/2026-06-02-video-engine-architecture__consolidated-final.md#L85).

SHOULD-FIX:
1. [tests/debug_prompt.json] The checked-in debug artifact still carries `"allow_auto_fallback": true`, violating the "NO deprecated cruft" directive. Fix: Remove the line on line 288 in [tests/debug_prompt.json](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/tests/debug_prompt.json#L288).

OPTIONAL / NICE-TO-HAVE:
- Update comments in [tests/test_workflow_validator_extended.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/tests/test_workflow_validator_extended.py#L292-L301) to reflect the new addition to `FORBIDDEN_INPUT_SOCKETS`.

CUT THESE (scope / over-engineering):
- None. This is a pure cleanup sprint with no added features or bloat.
