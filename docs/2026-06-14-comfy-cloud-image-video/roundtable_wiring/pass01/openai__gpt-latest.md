<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. The plan has unresolved scope contradictions, a grounded runtime signature mismatch, and underspecified cost/lease changes that affect correctness and money-spend behavior.

MUST-FIX BEFORE BUILD:
1. [§0, §2b] `cloud_ltx2` is specified as “image-to-video / text-to-video” and “All video roles” in the scope, but `required_inputs=("init_image",)` and `roles=("announcer_visual","music_visual","character_video","scene_broll")` exclude `background_abstract`. Grounding: `role_compat.py` says `background_abstract` supplies only `text_prompt`, so this engine will not fit that role. Concrete fix: choose one:
   - v1 is I2V only: change scope text to “all init-image video roles; not background_abstract/text-to-video”; or
   - support T2V: add a separate `cloud_ltx2_t2v` adapter with `family="text_to_video"`, `required_inputs=("text_prompt",)`, and include `background_abstract`; or update the compatibility model to support optional `init_image` explicitly. Do not claim both with the current required-inputs scheme.

2. [§0, §6] The `other_beats_video_model` dropdown is one slot covering multiple roles, but the plan says `cloud_kling_avatar` is available “on character beats” in that slot. Grounding: `role_compat.py` maps “the three other-beats roles” to the C selector and `cloud_kling_avatar` would fit `character_video` but not `scene_broll` or `background_abstract` because they lack `audio_ref`. A single saved selector cannot be both valid and invalid depending on beat without explicit per-beat handling. Concrete fix: either do not present/recommend `cloud_kling_avatar` for `other_beats_video_model` in v1, or add a separate `character_video_model` selector/policy field and append the widget/link JSON changes, or specify an explicit per-beat fallback rule when `other_beats_video_model=cloud_kling_avatar` encounters scene/background roles.

3. [§1, §2a] The image adapter method signature is wrong for the existing dispatcher. The plan says image adds `render_image(request)->...`, but grounded `otr_image_gen_dispatcher._inprocess_gen_fn` calls `eng.render_image(request, prepared)`. A `cloud_flux_pro.render_image(self, request)` implementation will raise `TypeError`. Concrete fix: require `render_image(self, request, prepared=None)` for `cloud_flux_pro`, matching `FluxGen1ImageEngine.render_image(self, request, prepared=None)`.

4. [§2a, §4.2, §7] Cost enforcement is contradictory and not buildable as written. §2a says `assert_usable` fails when “cost guard would be exceeded”; §4.2 says cost guard is request-context “not `assert_usable`”; §7 again says `assert_usable` fails “over budget.” Grounding: `dispatch_images` calls `_ireg.assert_usable(engine_id, role)` before constructing the full `request` dict with `w/h`, and no episode spend state is passed. Concrete fix: remove cost checks from `assert_usable` tests/spec except “policy system configured”; implement a dispatcher-level `reserve_cloud_cost(engine_id, request, episode_id, request_id)` after the request is assembled and before `render_*`; make cache hits free; make the reserve idempotent by `request_id`; record spent/reserved in a defined ledger/policy location; unknown/stale price raises a fail-closed warning and skips the object/clip.

5. [§4.1, §2a, §2b, §2c] Lease-skip depends on a network marker, but the adapters are not consistently required to define one. §2b mentions `declared_isolation` for `cloud_ltx2`; §2a does not define any marker for `cloud_flux_pro`; §2c also omits `declared_isolation`. Grounding: `dispatch_images` currently unconditionally does `_lease.acquire()` and post-gen `_lease.wait_until_below_mb(...)`. Concrete fix: define one shared marker for all three adapters, e.g. `is_network = True`, and make both image and video lease sites check it before acquire/release/NVML probe. If using `declared_isolation="network"`, verify/update `motion_common` isolation constants/validators first [ASSUMPTION: existing video code may validate isolation values; only `eng_humo.py` shows `_MC.ISOLATION_IN_PROCESS`].

6. [§4.1] The image lease-skip sequencing must be specified against the actual code. Grounding: `dispatch_images` currently does `assert_usable`, checks `gen_fn`, builds `request`, then always acquires lease inside `try`, then `_coerce_pixels(gen_fn(request))`, then always runs the post-generation NVML probe. Concrete fix: resolve the engine object/CAPABILITIES before the `try`, compute `skip_lease = is_network_engine(engine_id)`, only call `_lease.acquire` when false, only release when acquired, and only call `_lease.wait_until_below_mb` when false. Add the same exact branch to the video render lease site after verifying its location.

7. [§4.2, §6, §9.2] The source of the per-episode credit ceiling is undecided but required. The plan says “in the image/video policy,” then §6 says no JSON edit unless a Director widget is added, and §9 leaves the default/behavior open. Concrete fix before build: pick one v1 source. Smallest: environment/config-only ceiling, no workflow JSON change. If it must be in Director policy, append the widget and update `workflows/otr_scifi_16gb_full.json` with validator/link/widget audit; do not keep claiming “no JSON edit.”

8. [§5, §8.S0] The Partner-node invocation seam is correctly marked as a spike, but the rest of the build plan still assumes specific adapter outputs and auth behavior before that spike resolves. Concrete fix: make S0 a hard stop with an artifact: selected invocation mode, exact API-key source, request schema, polling/cancel contract, and output file discovery contract. Do not start S1/S2 adapter implementation until this document is updated with those pinned details.

SHOULD-FIX:
1. [§3, §4.1] CAPABILITIES rows with `vram_class="cpu"` are not enough to distinguish local CPU engines from network engines. If lease-skip reads adapter attrs, it may need to instantiate/inspect adapters; if it reads CAPABILITIES, the row lacks the needed field. Concrete fix: add a registry/capability field such as `"is_network": True` or `"isolation": "network"` and use that for lease-skip/reporting. Verify the capability validator accepts the new key before adding it.

2. [§3] The plan says adding CAPABILITIES rows requires “no `capability_profiles.py` validator change,” but adding a new network/isolation key for lease-skip/reporting would require validator support if unknown keys are rejected. Concrete fix: verify `validate_declaration` behavior; either extend the schema or keep network marking on adapter classes only.

3. [§2b] `target_fps per LTX-2` and “derive duration the same way the local LTX adapter sizes frames” are not concrete enough for a coder. Concrete fix: pin `target_fps`, max duration, min duration, frame-count rounding, and whether duration comes from `timing.target_frame_count / fps`, audio window, or beat budget.

4. [§2c] “duration is the audio’s duration” lacks the actual duration source. Concrete fix: specify whether to read duration from `audio_ref` metadata, probe the WAV header, or use existing request timing. Fail closed if they disagree beyond tolerance.

5. [§4.3] “then Comfy account” is not a buildable auth fallback. No grounded API for a Comfy account lookup is provided. Concrete fix: v1 auth probe should be `OTR_COMFY_API_KEY` only, or §5 must pin the exact account/key discovery API.

6. [§4.4] Retry/cancel behavior is underdefined for paid cloud calls. Concrete fix: specify retryable statuses, max attempts, exponential/backoff timings, idempotency key, and whether retries consume/reserve additional budget.

7. [§6] The statement “six dropdowns are COMBOs built from the registry” is not grounded here for the Director/image director code. Concrete fix: verify the relevant `INPUT_TYPES` implementations before relying on “no JSON edit needed” [ASSUMPTION].

8. [§7] Add a test for the grounded image signature path: `_inprocess_gen_fn` with a fake cloud image engine whose `render_image(request, prepared=None)` returns a path. This catches the §1/§2a signature issue.

9. [§7] Add a test for other-beats partial incompatibility if you keep `cloud_kling_avatar` selectable in the other-beats slot: one character beat passes, scene/background fail closed with LOUD warnings, or fallback as specified.

10. [§3] Guarded imports in package `__init__.py` can hide broken adapters and make `all_engine_names()` miss the cloud rows. Concrete fix: guarded import should log the exception LOUD and tests should assert the rows exist when dependencies are absent but cold-import-safe.

OPTIONAL / NICE-TO-HAVE:
- [§2a, §2b, §2c] Add `provider`, `provider_model`, `pricing_version`, and `terms_url/date_checked` metadata to each adapter for reports and commercial-clean review.
- [§7] Add a dry-run mode that estimates cloud cost for the whole episode without rendering.
- [§5] Record the S0 probe output paths in a small fixture/workflow for future regression tests.

CUT THESE (over-engineering):
1. [§9.3] Cut “visible-but-disabled with help text” for v1. Grounding says the video COMBO is the full static registry and filtering happens at execute time, not dynamic widget mutation. Implement visible static rows plus fail-closed `assert_usable`/report messages; disabled dynamic rows can wait.

2. [§4.3] Cut “Comfy account” auth discovery for v1 unless S0 proves an exact dep-free API. `OTR_COMFY_API_KEY` env-only is sufficient and deterministic.

3. [§4.2] Cut complex “reserve/spent deterministic accounting” beyond one ledger-backed idempotent reserve per `request_id` for v1. Do not build a full billing subsystem before the first paid smoke; just prevent unknown/stale prices and ceiling overrun.

4. [§5] Cut direct provider HTTP fallback from the initial adapter implementation unless both Partner-node and Comfy server workflow submission fail. It adds a third API surface and likely different billing/output semantics.