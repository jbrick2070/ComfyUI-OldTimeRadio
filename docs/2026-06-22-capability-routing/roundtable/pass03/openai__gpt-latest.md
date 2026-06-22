<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. The plan leaves routing-critical “confirm” items open, and as written can remove the `roles` data while `engine_fits_role` may still fail closed on missing `roles`.

MUST-FIX BEFORE BUILD:
1. [Changes 1/2] `engine_fits_role` change is underspecified and can still reject every descriptor without `roles`. Current grounded code fails if `roles is None or required is None`; merely removing `if role not in tuple(roles)` is not enough, especially if [Changes 2] removes engine `roles` attrs. Concrete fix: remove all `roles` dependency from `engine_fits_role`, including the `roles = descriptor.get("roles")` read and the `roles is None` fail-closed condition. Keep only: descriptor must be dict, `required_inputs` must exist, required tokens must be known, and required tokens must be available for the role. Add a test with a descriptor that has `required_inputs=("init_image",)` and no `roles`, asserting it fits `announcer_visual` and not `background_abstract`.

2. [Wiring opens 2] Auto-selection behavior is not proven. The grounding explicitly says “OPEN: where does the director descriptor's `roles` come from?” and asks whether it is `default_roles`, a separate `roles` attr, or capability-derived. The plan cannot claim “default_roles unchanged => no auto-pick change” until this is verified. Concrete fix: inspect `otr_video_director.py` descriptor construction and auto-pick path; update it so auto defaults use `default_roles` only, not `roles`; add golden tests for each existing slot default before/after the routing change.

3. [Wiring opens 1] Aspect is left as an unresolved runtime compatibility question. Dropping `roles` can make wide engines eligible for portrait roles such as `character_video`; the document only says “CONFIRM aspect is enforced DOWNSTREAM,” which is not a build step. Concrete fix: either add an explicit `supported_aspects` / `render_aspect` compatibility check in routing, or add grounded tests proving that each newly eligible engine can render the role aspect supplied by the director. Do not ship with this as an open confirmation.

4. [Changes 3 / Invariants] The first routing gate uses descriptor `required_inputs`, but the grounded render path has a second gate: `_assert_family_inputs_satisfiable()` compares `FAMILY_REQUIRED_INPUTS` against actual request tokens. The plan says “NO FAMILY_REQUIRED_INPUTS change” but does not prove newly added routes pass the family gate. Concrete fix: for every additive `(engine, role)` delta, assert the family-required tokens are a subset of `ROLE_AVAILABLE_INPUTS[role]` and that a synthesized request for that role passes `_assert_family_inputs_satisfiable()`. If any fail, align descriptor `required_inputs` with family requirements or keep those routes ineligible.

5. [Wiring opens 3 / Changes 2] The before/after eligibility test can become meaningless if implemented after deleting `roles` attrs and using a live “old algorithm helper”: old eligibility would collapse because the old algorithm needs `roles`. Concrete fix: commit a fixture of the pre-change eligibility matrix or pre-change descriptors before deleting/ignoring `roles`; do not compute “before” from post-cleanup descriptors unless the fixture preserves the old `roles` values.

6. [ROOT CAUSE] The root-cause statement depends on `eng_wan_i2v.py:85 roles = (...)`, but the grounding excerpt itself says descriptor `roles` source is still open. [ASSUMPTION] If the real director builds descriptor roles from a source other than the engine class attr, removing class `roles` will not fix the exclusion. Concrete fix: trace the actual descriptor passed to `engine_fits_role` for `wan_i2v` and `announcer_visual`, and verify the missing `announcer_visual` whitelist is the value being checked.

SHOULD-FIX:
1. [Non-regression] “PROVABLY NON-REGRESSIVE” is overstated. The set-theory argument only proves no previously eligible `(engine, role)` pair is removed by the first gate. It does not prove behavior is non-regressive if auto-selection considers the expanded eligible pool, if aspect was implicitly encoded by `roles`, or if the render-side family gate rejects new routes. Concrete fix: reword to “no first-gate removals” and rely on the auto-pick, aspect, and family-gate tests above for non-regression.

2. [Changes 2] Removing dead `roles` attrs in the same patch increases blast radius and can break any unreviewed consumers. Concrete fix: first change `engine_fits_role` to ignore `roles`; leave attrs in place for one release or until a repo-wide search proves there are no consumers other than the old gate.

3. [Invariants] “audio specials gated by capability” needs explicit coverage. Grounding shows `character_video` lacks `audio_ref`, while announcer/music/scene have it. Concrete fix: add parameterized tests that audio-conditioned engines do not fit roles without `audio_ref` and do fit roles with `audio_ref` when their other required tokens are present.

4. [Changes 1] Unknown-role fail-closed behavior is assumed through `role_available_inputs(role)`, but that function is not grounded. [ASSUMPTION] Concrete fix: add a test that an unknown role returns `False` for all engines, or explicitly make `role_available_inputs` return an empty set for unknown roles.

5. [Changes 3] “CUT optional_inputs” is not grounded. [ASSUMPTION] If `optional_inputs` exists in real engine descriptors and is consumed by UI, docs, scoring, or request construction, cutting it may be a behavior change unrelated to the routing bug. Concrete fix: repo-search consumers before removing it; otherwise leave it untouched.

OPTIONAL / NICE-TO-HAVE:
- Add a debug log or test artifact listing additive eligibility deltas, but keep pass/fail assertions focused on expected critical deltas such as `wan_i2v -> announcer_visual`.
- Add one direct regression test for the live wall: selecting `wan_i2v` for slot `announcer_video_model` no longer raises “does not fit any role.”

CUT THESE (over-engineering):
1. [Changes 2] Cut the immediate “declare-once cleanup” of removing all engine `roles` attrs. It is safe to defer because the functional fix is making `engine_fits_role` ignore `roles`; leaving attrs in place reduces risk to unreviewed consumers and preserves old-eligibility fixture data.

2. [Wiring opens 1] Do not add a full `supported_aspects` capability system unless tests prove aspect incompatibility. If downstream truly renders from the role-aspect still and engines accept those dimensions, the explicit aspect matrix is extra routing complexity. If tests fail, then add the minimal check.

3. [Wiring opens 3] Do not require both a committed fixture and an old-algorithm helper. One reliable source of truth is enough. Prefer a committed pre-change eligibility fixture if `roles` attrs may be deleted later.