<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. The direction is sound, but the spec is internally inconsistent and missing the exact migration points needed to avoid breaking routing/render gates.

MUST-FIX BEFORE BUILD:
1. [Proposed direction] False claim: “still+prompt engines (wan_i2v, ltx_video, flux*, ...) -> all roles.” Grounding says `background_abstract` supplies only `{"text_prompt"}` and `wan_i2v.required_inputs = ("init_image",)`, so `wan_i2v` does NOT fit `background_abstract` under the proposed subset rule. Concrete fix: replace with “text-prompt-only engines fit all current roles; init-image engines fit roles whose `ROLE_AVAILABLE_INPUTS` include `init_image`.”

2. [Proposed direction] False claim: audio-required engines “only audio-bearing roles (announcer/music).” Grounding says `scene_broll` also supplies `audio_ref`, and also `init_image`, so `visualizer`, `character_3d`, HuMo-like, and LTX-AV-like engines may become eligible for `scene_broll` by capability. Concrete fix: either update the statement/table to include `scene_broll`, or change `ROLE_AVAILABLE_INPUTS["scene_broll"]` if scene b-roll must not expose audio to routing. Do not claim announcer/music-only unless the role input map proves it.

3. [Invariants] The required “strict SUPERSET” proof cannot be satisfied without a real before/after matrix, and the spec does not provide one. Concrete fix: generate a deterministic table from current descriptors: rows = every engine descriptor, columns = every role in `ROLE_AVAILABLE_INPUTS`, cells = current `engine_fits_role` result vs proposed capability-only result. Assert `before=True => after=True`; list every `False=>True` delta explicitly. Include `wan_i2v -> announcer_visual` as one expected delta.

4. [Current state / grounding_routing.py] The spec treats render-side `_assert_family_inputs_satisfiable` as merely “a SECOND place the capability logic lives,” but it checks actual request token presence, not just theoretical role availability. Removing or weakening it would break loud failure on missing assets. Concrete fix: keep a render-time assertion, but derive the required token set from the same engine/family capability declaration. The role-fit gate answers “can this role theoretically supply the inputs?”; render gate answers “does this concrete request actually contain them?”

5. [Hard questions #2 / grounding_routing.py] The migration point that actually blocks `wan_i2v` is unresolved: grounding explicitly says “OPEN: where does the director descriptor’s `roles` come from?” Concrete fix: before build, verify `otr_video_director.py` descriptor construction and every consumer of `descriptor["roles"]`. Replace eligibility consumers with a shared helper, e.g. `engine_required_inputs(descriptor)` + `engine_fits_role(descriptor, role)`, and preserve/rename role metadata only where it is selection/default policy, not eligibility.

6. [Proposed direction / Invariants] `default_roles` is not redundant with capability eligibility. The doc itself notes it may mean auto-default. Dropping or conflating it with eligibility can change automatic model selection even if eligibility is a superset. Concrete fix: define separate fields:
   - `required_inputs`: hard capability contract.
   - `default_roles` or `auto_default_roles`: selection/defaulting only.
   - no `roles` whitelist for eligibility.
   Update tests to prove default selection for existing working slots is unchanged.

7. [Goal / Hard questions #4] “Apply the same capability-once model to IMAGE engines” is underspecified and not grounded. No image role map, image engine descriptors, input tokens, or consumers are provided. Concrete fix: either split image migration into a separate build item, or add the image equivalents of `ROLE_AVAILABLE_INPUTS`, engine `required_inputs`, descriptor builder, selector, and render/assertion gates. [ASSUMPTION] Image routing has separate consumers that may still depend on role lists; verify before changing.

8. [Invariants] “No-silent-swap safety preserved” is not guaranteed by removing whitelists unless selection behavior is separately constrained. Capability-only eligibility may expand fallback/default pools. Concrete fix: preserve loud behavior for explicitly requested incompatible engines, and add tests that expanded eligibility does not cause an unrequested engine substitution in existing slots. Eligibility expansion and auto-selection must be tested separately.

SHOULD-FIX:
1. [Proposed direction] Token naming is inconsistent: the goal uses `image_in`, `audio_in`, `video/base-clip-in`, while grounding uses `init_image`, `audio_ref`, `base_clip_ref`, `text_prompt`. Concrete fix: standardize on the existing internal token names or provide a one-to-one mapping layer. Do not introduce parallel token names in descriptors.

2. [Current state / grounding_routing.py] `engine_fits_role` currently fail-closes if either `roles` or `required_inputs` is missing. If `roles` is removed from descriptors before all callers are updated, current routing will reject everything. Concrete fix: change `engine_fits_role` first to ignore `roles` and require only valid `required_inputs`; only then remove/populate-away `roles`.

3. [grounding_routing.py] `required_set <= INPUT_TOKENS` validation exists in `engine_fits_role`, but the spec does not say where unknown capability tokens are validated after refactor. Concrete fix: keep the `INPUT_TOKENS` validation in the shared capability accessor and fail closed on unknown tokens.

4. [render_driver.py] `FAMILY_REQUIRED_INPUTS` is keyed by family, while proposed capability declaration is per engine. If multiple engines share a family but require different inputs, deriving render requirements by family may be wrong. Concrete fix: verify whether family uniquely determines required inputs. If not, make render assertion engine-descriptor-based, or key `FAMILY_REQUIRED_INPUTS` by engine name. [ASSUMPTION] Multiple engines per family may exist; verify.

5. [Invariants] The examples “HuMo audio-announcer, ltx_av_music bookends, ltx_video b-roll, flux stills” are not sufficient regression coverage. Concrete fix: test all engines discovered by the descriptor builder, not only named examples.

6. [Current state] `character_video` lacks `audio_ref` in grounding. That means audio-required engines should not fit `character_video`, even if the name suggests character animation. Concrete fix: include this in the before/after table to catch accidental assumptions based on role names.

7. [Proposed direction] `base_clip_ref` is included in role availability but no grounded engine declaration requiring it is shown. Concrete fix: add test coverage for a hypothetical/current base-clip-required engine if one exists; otherwise keep token support but do not claim validation is complete for video/base-clip engines. verify: actual engine declarations requiring `base_clip_ref`.

OPTIONAL / NICE-TO-HAVE:
1. Add a small CLI/test fixture that prints the engine x role eligibility matrix for operator review.
2. Rename `required_inputs` to `consumed_inputs` only if all consumers are migrated at once; otherwise keep existing name to reduce churn.
3. Add comments to `ROLE_AVAILABLE_INPUTS` clarifying that it is theoretical role supply, not proof that a concrete render request contains those assets.

CUT THESE (over-engineering):
1. [Proposed direction] Do not build a new capability schema with parallel names like `image_in/audio_in/video_in` if existing `required_inputs` tokens already express the contract. Safer cut: keep `required_inputs = ("text_prompt", "init_image", "audio_ref", "base_clip_ref")` and make it the single source.

2. [Goal / Hard questions #4] Cut the image-engine migration from this build unless the image routing files and tests are included. It is safe to defer because the live wall is video `wan_i2v -> announcer_visual`, and mixing image migration adds ungrounded blast radius.

3. [Invariants] Do not manually maintain a static before/after table in docs as the primary proof. Generate it in tests from actual descriptors; static tables go stale and can miss new engines.