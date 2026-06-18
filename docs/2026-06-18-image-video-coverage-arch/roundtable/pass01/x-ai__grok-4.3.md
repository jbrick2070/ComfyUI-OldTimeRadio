<!-- requested_model: x-ai/grok-4.3 | resolved_model: x-ai/grok-4.3-20260430 -->

VERDICT: no. Open design questions are unresolved and the proposed StillInput/central-approval changes have no grounding in the provided code (required_inputs + role_compat remain the only mechanisms).

MUST-FIX BEFORE BUILD:
1. [Open design questions] No decision on where the still<->video contract lives or its schema. Concrete fix: pick one (add `accepts_still: bool` + `still_input_name: str` to the VideoEngine Protocol in registry.py) and delete the "e.g." example.
2. [What "good" looks like #2 + Current state] "Approval in one place" claim is false: image model selection is already only in OTR_VideoDirector (video_policy_json["image_models"]), but compatibility is still scattered across role_compat.py:engine_fits_role, registry.py:VALIDATED_ENGINES (both), _still_needed_for_role, and assert_usable. Fix: delete the claim or add the single registry surface.
3. [Current state, _still_needed_for_role] The still-skip gate still keys exclusively on `"init_image" in required_inputs` (otr_image_gen_dispatcher.py:155). The plan requires it to key on the new capability instead; no migration or dual-read path is shown. Fix: either update the gate or state that required_inputs stays authoritative.
4. [Back-compat / migration] No concrete steps for engines that already declare required_inputs (ltx_av_talk, humo, ltx_video). Fix: add "read both the new field and required_inputs during transition; required_inputs wins for existing names."
5. [3D engines] "accepts still" with kind=mesh_portrait vs init_image is posed as open but never resolved; three_d_locked_slots + enforce_3d_granularity_lock only look at requires_mesh_portrait. Fix: either drop the kind idea or define how the two capabilities interact.

SHOULD-FIX:
1. [Design goals #4] "Fail-loud, no silent skips" is already violated by the bare except in _still_needed_for_role (otr_image_gen_dispatcher.py:158) that returns True. Change to explicit logging of the unknown engine case.
2. [Constraints] 16 GB / single-resident / no-silent-fallback invariants are stated but the plan adds another capability table without showing it stays under the same cold-import + NVML rules as CAPABILITIES.
3. [otr_image_director.py:three_d_locked_slots] It walks video_policy["video_models"] but the new architecture must also handle image engines that declare a still capability; no cross-check exists.

OPTIONAL / NICE-TO-HAVE:
- Add a one-line `still_input_name` default ("init_image") so most adapters need zero changes.
- Emit a deprecation warning the first time required_inputs is read for still decisions.

CUT THESE (over-engineering):
1. Separate coverage table (open question) -- safe to cut because required_inputs + role_available_inputs already encode exactly the same information and are already read by both directors and the dispatcher.
2. New StillInput capability type -- safe to cut if we simply add the two bool/str fields to the existing VideoEngine Protocol; the extra protocol buys nothing.

[ASSUMPTION] The plan assumes a new capability can be added without touching wrapper_bridge, av_dims, or the LTX-AV graph builders; verify against the real adapter code.
[ASSUMPTION] "Any image model -> any video model" is assumed to be only a still-input question; role_compat still gates on roles, so cross-role coverage may still be empty.