<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. The plan leaves routing-source and family-gate decisions unresolved while also prescribing changes that can silently broaden eligibility and break render-gate behavior.

MUST-FIX BEFORE BUILD:
1. [Changes 1 / Changes 3 / Open -> R3] The descriptor `roles` source is unresolved, but the plan depends on it. Grounding says engines declare `default_roles`, while the plan says `_registry_descriptors` uses `getattr(eng, "roles", ())`; adding `MotionEngineBase.roles = ()` can cause engines that only define `default_roles` to lose their whitelist and become capability-only. Concrete fix: before coding, inspect `_registry_descriptors` and define one normalized source of truth. Preserve legacy `default_roles` restrictions or explicitly migrate every engine to `roles`. Add a descriptor-level test proving `wan_i2v.roles == ()` and a restricted cheap/default-role engine still has its non-empty whitelist.

2. [Changes 4 / Open -> R3] `FAMILY_REQUIRED_INPUTS` assert-equal is mandated before resolving whether multiple engines in a family can have different `required_inputs`. If they can, the proposed assert is wrong and the render gate remains incapable of expressing engine-specific minima. Concrete fix: inventory engine family membership first. If all engines in a family share the same requirements, enforce that with the assert. If not, change the render gate to use engine-level `required_inputs` or split the family identifiers.

3. [Changes 1 / Changes 4] Changing `wan_i2v.required_inputs` from `("init_image",)` to `("text_prompt",)` makes the render gate require an actual non-empty `request["text_prompt"]`. Role capability only proves the role theoretically supplies text; it does not prove every concrete request does. Concrete fix: add an integration test for each Wan-routed role/slot, including `announcer_video_model`, that builds the actual render request and passes `_assert_family_inputs_satisfiable`. If any request lacks `text_prompt`, fix the director/request builder before changing `FAMILY_REQUIRED_INPUTS`.

4. [Changes 1] The plan assumes Wan can render when only `text_prompt` is present and `init_image` is absent because the still is “DERIVED from the prompt per beat.” The grounding excerpts do not show that derivation path or the Wan renderer contract. [ASSUMPTION] If the current Wan implementation actually requires an init image at call time, this plan just moves the failure from routing to render/runtime. Concrete fix: verify Wan’s render path accepts prompt-only requests or add the missing prompt-to-still derivation before invoking Wan; test a request with no `asset_refs.init_image`.

5. [Changes 2] “empty/None -> capability-only” is underspecified and can accidentally treat malformed descriptors as valid. The current function fails closed when `roles is None or required is None`; the plan must not also allow missing `required_inputs`. Concrete fix: implement explicitly: `required_inputs is None => False`; normalize only `roles is None` or empty tuple/list to “no whitelist”; keep unknown required tokens fail-closed.

6. [Tests] The before/after eligibility test is not implementable as written after the code changes, because both the algorithm and engine declarations change. Computing “before” from the modified descriptors will not represent the real old state. Concrete fix: check in a baseline eligibility snapshot from current code before the change, or implement a test helper with the old algorithm and old per-engine required-input values for the affected engines.

7. [Tests / Invariants] “Auto-SELECTION non-regression” is asserted but not protected by the proposed routing changes. Expanding eligibility for all empty-role engines can change picks if selection depends on iteration order or first-fit behavior. Concrete fix: add golden selection tests for each existing slot/default workflow before the change. If any pick changes, either restore priority/order explicitly or add a deliberate override.

8. [Invariants / Changes 4] “NO workflow-JSON change (engine attrs + role_compat only)” contradicts the required `schemas.py::FAMILY_REQUIRED_INPUTS` change and new schema/render-gate tests. Concrete fix: rewrite the invariant to allow `schemas.py`/test changes, or drop Change 4. Do not leave the scope contradictory.

9. [Changes 1] “Apply the same to any other i2v b-roll engine” is not a buildable instruction. Concrete fix: enumerate the exact engine files/classes to change, or add a discovery test that fails every video b-roll/i2v engine whose true minimum is prompt-only but still declares `init_image` required.

SHOULD-FIX:
1. [Changes 2] Normalize `roles` types defensively. Existing code does `tuple(roles)`, which behaves badly if an engine accidentally declares a string instead of a tuple/list. Concrete fix: reject bare strings or convert only tuple/list/set to tuple; otherwise fail closed.

2. [Changes 1 / Changes 3] `optional_inputs` is added to engines/base but `_registry_descriptors` is only described as exporting `roles` and `required_inputs`. If optional inputs are part of the model, they should appear in descriptors/debug output/tests; otherwise this attr is dead metadata. Concrete fix: either add `optional_inputs` to descriptors and a descriptor test, or cut it from this build.

3. [Corrected model] The distinction between `roles` and `default_roles` is a naming hazard. Grounding references `default_roles`; the plan uses `roles`. Concrete fix: migrate to one public attr name with backward-compatible mapping and tests.

4. [Tests] The audio-specials test only names “NOT background_abstract.” It should assert the full matrix against `ROLE_AVAILABLE_INPUTS`, including that `character_video` does not fit audio-required engines because grounding shows it lacks `audio_ref`.

5. [Open -> R3] Aspect routing is explicitly unresolved. If current `roles` were being used to hide aspect incompatibility, making empty roles capability-only can expose engines to wrong-aspect slots. Concrete fix: add a selection/render test covering a wide-only engine on every newly eligible role, or add an explicit aspect capability before broadening eligibility.

OPTIONAL / NICE-TO-HAVE:
- Add debug logging that reports both gates separately: role whitelist result, required-token capability result, and render request present-token result.
- Add a small CLI/test artifact that prints the final engine x role matrix for release review.
- Add comments near `ROLE_AVAILABLE_INPUTS` explaining theoretical supply vs concrete request tokens to avoid future conflation.

CUT THESE (over-engineering):
1. [Changes 1] `optional_inputs = ("init_image",)` can be cut from this build if no code consumes it. The routing fix only needs correct `required_inputs` plus role compatibility; optional metadata can follow after descriptor/render consumers exist.

2. [Tests] Printing additive `False->True` deltas is useful for review but not necessary as an automated requirement. Keep the zero-regression assertion and golden matrix; the printout can be non-blocking.

3. [Invariants] “UTF-8 no BOM; SFW” is unrelated to capability routing and does not reduce the specific build risk here. Keep those in global release checks, not this plan.