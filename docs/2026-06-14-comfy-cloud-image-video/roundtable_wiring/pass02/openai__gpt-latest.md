<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. The plan has an internal marker contradiction and, as written, cloud video engines that subclass MotionEngineBase will still take the GPU lease and post-render NVML wait.

MUST-FIX BEFORE BUILD:
1. [§8 S1 vs §4.1/§10.6] Contradiction: §4.1 says the only network marker is `is_network = True` and explicitly says NOT a new `declared_isolation` enum; §8 S1 says to add `declared_isolation="network"`. Concrete fix: delete `declared_isolation="network"` from §8 S1 and any implementation task. Use only `is_network = True`, or, if isolation must be represented, define it consistently everywhere and update `motion_common` constants/tests. Smallest fix: use only `is_network`.

2. [§1/§2b/§2c/§4.2] Cloud video engines are specified to extend `MotionEngineBase`, but grounded `MotionEngineBase.prepare()` unconditionally calls `_GR.acquire(...)`, and `teardown()` waits on `_GR.wait_until_below_mb(...)` when a lease exists. That directly defeats §4.2’s “skip GPU lease and post-gen NVML probe for network engines.” Concrete fix: specify that `cloud_ltx2` and `cloud_kling_avatar` either do not subclass `MotionEngineBase`, or override `prepare()` and `teardown()` as no-lease/no-NVML for `is_network=True`. If the central video render path calls `prepare()` generically, it must branch on `getattr(engine, "is_network", False)` before calling it. Verify: actual video render orchestration site.

3. [§2a/§4.2] Image lease-skip is incomplete if the cloud image adapter exposes `prepare()`. Grounded `_inprocess_gen_fn()` calls `eng.prepare(None, None, None)` whenever callable, before `render_image()`. If `cloud_flux_pro` “mirrors FluxGen1” and inherits/implements a GPU-loading `prepare()`, the dispatcher lease skip will not prevent local residency. Concrete fix: explicitly require `cloud_flux_pro.prepare` to be absent or a no-op returning `None`, with no GPU lease/heavy import; add a test that `_inprocess_gen_fn()` on `cloud_flux_pro` does not acquire the lease or import torch/SDK at prepare time. [ASSUMPTION] Verify whether `FluxGen1ImageEngine` defines `prepare()`.

4. [§7 vs §4.3/§10.3] Test plan contradicts the design: §7 says `assert_usable` fails closed when “over budget,” but §4.3 and §10.3 correctly say cost guard is dispatcher-level and NOT in `assert_usable`. Concrete fix: change §7 to test `assert_usable` only for flag-off / missing API key, and test over-budget via `reserve_cloud_cost(...)` at the dispatcher render path.

5. [§7 vs §0/§2b] Role-compat test for `cloud_ltx2` omits `background_abstract`. §0 and §2b require `cloud_ltx2` to fit all five video roles because `required_inputs=("text_prompt",)` and grounded `ROLE_AVAILABLE_INPUTS["background_abstract"] == {"text_prompt"}`. Concrete fix: update the test bullet to assert `cloud_ltx2` is offered for announcer, character, music, scene, and background_abstract.

6. [§4.3 + grounded `dispatch_images`] Cost-guard ordering is underspecified relative to the actual image dispatcher. Grounded code currently does: cache check -> `assert_usable()` -> `gen_fn is None` check -> build `request` -> acquire lease -> `gen_fn(request)`. §4.3 requires reserve AFTER request assembly and BEFORE render, with cache hits free. Concrete fix: specify exact image order: cache hit returns before auth/cost; cache miss does `assert_usable`; if `gen_fn is None`, skip without reserving; build request including `request_id`; call `reserve_cloud_cost(...)` only for `is_network` engines; then render; then skip/acquire lease based on `is_network`.

7. [§2c] “Output: silent MP4” is not enforceable as written for a cloud provider result. Existing HuMo guarantees silence by encoding frames via `encode_frames_to_silent_mp4`; a cloud avatar provider may return an MP4 with audio. Concrete fix: require the Kling adapter to strip/drop any provider audio before returning/canonicalizing, and assert `has_audio=False` in `canonicalize()`. Add a test with an input MP4 containing audio to verify the returned canonical clip is silent.

SHOULD-FIX:
1. [§4.3] The billing ledger schema is not defined enough to implement idempotency. “Track a running spent total in `ledger["billing"]`” leaves request-id index, currency, price-table version/date, reserved vs actual, and per-engine units unspecified. Concrete fix: define a minimal schema, e.g. `ledger["billing"] = {"currency":"USD","price_table_date":"YYYY-MM-DD","ceiling":float,"reserved_total":float,"requests":{request_id:{engine_id, units, unit_price, estimated_cost, status}}}`.

2. [§4.3] “Unknown/stale price ⇒ fail closed” is not testable because “stale” has no threshold. Concrete fix: define staleness, e.g. price table older than 30 days or missing `valid_from`/`valid_until` fails closed.

3. [§2b] Duration calculation may be wrong if `timing.target_frame_count` is expressed in policy/canvas fps rather than engine `target_fps`. §2b says duration = `target_frame_count / target_fps`. Grounded `OTRVideoDirector` lets the user set `fps` 1–60, while engines have their own `target_fps`. Concrete fix: verify the schema’s frame-count basis. If it is policy fps, compute duration from request timing/canvas fps, then compute provider frames at the engine/provider fps. [ASSUMPTION] Timing schema not shown.

4. [§4.5] Retry/idempotency behavior is underspecified at the provider-call boundary. “429/5xx retry (idempotency key; a retry must not double-reserve)” does not say whether a timeout after submission polls the same job or submits a new one. Concrete fix: require a provider job id/request id to be persisted in the billing/request record before polling; retries after submission must resume/poll, not re-submit unless provider idempotency is confirmed in §5.

5. [§5] The S0 spike needs an explicit failure criterion. It currently lists three invocation options but not when to abandon one. Concrete fix: make S0 output a checked-in note/test fixture proving: key source, minimal request, output discovery, idempotency/retry contract, and whether billing occurs. If none works, stop before S1–S4.

6. [§3] Guarded `__init__.py` imports are required, but the plan does not specify the failure behavior consistently until §10.10. Concrete fix: move “guarded imports must log LOUD and tests assert rows register with flag off” into §3 so the coder implements it with the registry wiring.

7. [§0/§9.3] “Cloud options offered” language is misleading because grounded `OTRVideoDirector.INPUT_TYPES()` builds the COMBO from the full static registry, not role-filtered options. Concrete fix: rename those tables to “compatible at execute time” / “will run for these roles,” not “offered,” to avoid a coder trying to dynamically filter or grey out widgets.

OPTIONAL / NICE-TO-HAVE:
- [§4.3] Add a dry-run/report-only billing mode for tests and operator previews.
- [§7] Add a regression test that selecting `other_beats_video_model=cloud_kling_avatar` accepts the Director slot but ShotLock/render rejects scene/background beats LOUD and allows character beats.
- [§2b/§2c] Record provider job id, provider model version, and normalized duration in the canonical clip provenance for auditability.

CUT THESE (over-engineering):
1. [§8 S1] Cut `declared_isolation="network"` entirely. It conflicts with §4.1 and is unnecessary if `is_network=True` drives lease skipping.

2. [§6] Cut the Director-widget discussion for cloud credit ceiling in v1. Scope says no Director widget and §4.3 already uses env/config `OTR_CLOUD_CREDIT_CEILING`; keeping widget instructions invites positional `widgets_values` risk for no v1 value.

3. [§5 option 1] Do not require in-process Partner-node invocation unless the S0 spike proves billing/key/output behavior. If POSTing a tiny workflow or direct HTTP is the proven seam, carrying an in-process node path adds unnecessary cold-import and billing uncertainty.