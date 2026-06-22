<!-- requested_model: deepseek/deepseek-v4-pro | resolved_model: deepseek/deepseek-v4-pro-20260423 -->

VERDICT: build-ready as-is? no. The plan has critical gaps: the descriptor roles source is unconfirmed, the FAMILY_REQUIRED_INPUTS update is not explicitly listed, and the auto-selection non-regression mechanism is unspecified.

MUST-FIX BEFORE BUILD:
1. [Changes 4] The plan states FAMILY_REQUIRED_INPUTS for wan's family must become ("text_prompt",) but does not include an explicit code change to update that dict in schemas.py. Without it, the render gate will still require init_image, causing FamilyInputGap errors. Fix: Add a step to change the wan family entry in FAMILY_REQUIRED_INPUTS to ("text_prompt",) and ensure the existing sync assert passes.
2. [Open → R3] The descriptor's roles source for wan_i2v is not resolved. The plan says "Confirm wan_i2v's descriptor roles source" as an open question, yet the entire fix depends on the descriptor's roles being empty. If the director builds the descriptor using something other than the engine's `roles` attribute (e.g., `default_roles` or a hardcoded list), setting `roles = ()` on the engine will not propagate, and wan_i2v will remain excluded. Fix: Verify the exact code in otr_video_director.py that populates the `roles` key. If it does not use `getattr(eng, "roles", ())`, adjust it to do so, or ensure the existing source yields an empty tuple for wan_i2v after the change.
3. [Tests – Auto-SELECTION non-regression] The plan asserts the default engine pick for each slot must not change, but does not explain how this is guaranteed when wan_i2v becomes eligible for announcer_visual. If the selection algorithm simply picks the first eligible engine, wan_i2v could displace the current default. Fix: Either analyze the selection logic to confirm it will not change (e.g., it uses a priority list that still prefers the old default) or add an explicit mechanism (e.g., a preferred engine list) to preserve the old pick. Without this, the invariant is unenforced.

SHOULD-FIX:
1. [Changes 1] The plan adds `optional_inputs = ("init_image",)` to wan_i2v, but no code consumes `optional_inputs`. This is dead code and adds unnecessary surface area. Either wire it into the render gate or engine logic, or remove it to keep the change minimal. If kept, document its intended use.
2. [Changes 3] Adding `roles = ()` default in MotionEngineBase is good, but verify that the descriptor builder uses the `roles` attribute, not `default_roles`. If the builder uses `default_roles`, either change the builder or rename the attribute to avoid confusion.
3. [Tests] The before/after eligibility test should also verify that the render gate's `_assert_family_inputs_satisfiable` continues to pass for requests that previously passed, ensuring no regression in the render-side check after FAMILY_REQUIRED_INPUTS is updated.

OPTIONAL / NICE-TO-HAVE:
- Add a test that the FAMILY_REQUIRED_INPUTS sync assert covers all engines, not just families.
- Document the capability model in the codebase for future engine additions.

CUT THESE (over-engineering):
- The `optional_inputs` attribute on engines can be cut entirely. It is not used by any routing or render logic in the plan, and its presence may mislead future readers. If it is needed later, defer to a separate change.

[ASSUMPTION] The descriptor builder in otr_video_director.py uses `getattr(eng, "roles", ())` to populate the descriptor's `roles` key. If it uses something else, the plan's engine attribute change will not take effect. Verify.
[ASSUMPTION] The `INPUT_TOKENS` constant in role_compat.py is defined and contains all valid input token strings; the plan does not alter it.
[ASSUMPTION] The default engine selection algorithm is deterministic and will not be affected by the eligibility expansion, or there is a separate mechanism (e.g., a hardcoded priority list) that preserves the old default. If not, additional changes are required.