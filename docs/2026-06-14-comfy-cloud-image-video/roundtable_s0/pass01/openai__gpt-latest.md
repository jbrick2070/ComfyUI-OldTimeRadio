<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. S0 does not actually prove the `/prompt`/executor seam it claims to gate, and several build steps depend on lease-skip, billing, and role-compat details that are not pinned tightly enough.

MUST-FIX BEFORE BUILD:
1. [§3 + §4.3] The spike is a standalone venv script, but §4.3 says the risk to close is calling `sync_op`/`poll_op` from OTR’s ComfyUI executor-thread `/prompt` path. A standalone script does not prove hidden/headless `/prompt` execution, executor context, or event-loop behavior. Concrete fix: add a temporary in-graph `/prompt` smoke path, e.g. a throwaway node or OTR debug hook that calls the same util-client code from inside a Comfy prompt execution with `OTR_COMFY_API_KEY`, and record that result in `S0_RESULTS.md`. Keep the standalone script only as an import/signature probe.

2. [§2] The “Equivalent alternative” of injecting `api_key_comfy_org` into `/prompt` `extra_data` is not equivalent to the chosen seam. The chosen seam calls `comfy_api_nodes.util.client` directly inside `render_*`; no grounding shows that direct util-client calls read `/prompt` `extra_data`. Concrete fix: remove this alternative from the build path, or make S0 explicitly verify and document the exact installed-code path by which `extra_data["api_key_comfy_org"]` reaches the util-client auth argument.

3. [§5 S1 + `otr_image_gen_dispatcher.py::dispatch_images`] The current grounded dispatcher always acquires `_lease.acquire(...)` before `gen_fn(request)` and always does the post-generation `_lease.wait_until_below_mb(...)` after success. For `cloud_flux_pro`, that would hold/serialize the GPU residency lease across remote network I/O and then run an irrelevant NVML settle probe. Concrete fix: before S2, implement the S1 lease skip concretely in `dispatch_images`: resolve the selected engine once, check `is_network is True`, skip `_lease.acquire/release` and skip post-gen NVML only for that cache-miss render path. Add a test that fails if `_lease.acquire` is called for a network image engine.

4. [§5 S1 + `motion_common.py::MotionEngineBase.prepare`] The grounded `MotionEngineBase.prepare()` always acquires the shared GPU lease and calls `load()`. If `cloud_ltx2` or `cloud_kling_avatar` subclass `MotionEngineBase` without overriding `prepare`, they will hold the GPU lease during upload/poll/download. Concrete fix: define a `CloudMotionEngineBase` or require each cloud video adapter to override `prepare()`/`teardown()` to do no GPU lease, no model load, and no NVML wait. Add a unit test that a cloud video engine’s `prepare()` does not call `_GR.acquire`.

5. [§5 S1/S2/S3/S4] Billing reservation ordering is underspecified. The plan says `reserve_cloud_cost(...)` exists, but not where it brackets uploads, submits, polling, failures, cache hits, or retries. Concrete fix: specify and implement ownership: after cache hit checks and after `assert_usable`, reserve before the first potentially billed util-client call; commit only after a successful billed result is materialized; release/refund on auth failure, upload failure, submit failure, poll timeout, download failure, or canonicalization failure. Do this per image object/per video clip, not once per episode.

6. [§5 S3 + `role_compat.py`] `background_abstract` can supply only `{"text_prompt"}` per grounded `ROLE_AVAILABLE_INPUTS`. §5 S3 says `cloud_ltx2` supports “text_prompt + opportunistic init_image” and “all video roles incl. background_abstract.” If the adapter declares `required_inputs = ("text_prompt", "init_image")`, role filtering will exclude `background_abstract`. Concrete fix: set `cloud_ltx2.required_inputs = ("text_prompt",)` and treat `init_image` as optional at render time.

7. [§3] `S0_RESULTS.md` is checked in, but the spike prints paths, result URL information, and credit data. Provider/download URLs may be signed or otherwise sensitive, and auth mistakes could leak `OTR_COMFY_API_KEY`. Concrete fix: require `S0_RESULTS.md` to redact API keys, auth headers, full URLs, task IDs if reusable, and local user paths; include only module paths, signatures, redacted result-shape/type, and summarized cost source.

SHOULD-FIX:
1. [§3] “Print the credits charged” assumes the util response exposes credit charge data. No grounding excerpt proves that. Concrete fix: make this a best-effort observation, not a seam gate. If the response lacks credit fields, record “not present in response” and source pricing from the dated price table/operator billing page instead.

2. [§5 S1] “`OTR_COMFY_API_KEY` auth probe (dep-free) surfaced via `assert_usable`” is ambiguous. A real key validation probe cannot be dep-free if it calls Comfy API nodes or the network; a dep-free probe can only check env presence/format. Concrete fix: split this into: cold `assert_usable` checks `OTR_ENABLE_CLOUD` and nonempty `OTR_COMFY_API_KEY`; live smoke validates the key; runtime render handles 401/403 fail-closed.

3. [§2 Output handling + `otr_image_gen_dispatcher.py::_coerce_pixels`] `_coerce_pixels` accepts any object with `tobytes`; it does not validate dtype, rank, channel count, or value range. Concrete fix: make `cloud_flux_pro.render_image` assert it returns a `numpy.ndarray` with shape `(H,W,3)`, dtype `uint8`, nonzero dimensions, before handing it to the dispatcher.

4. [§3] The spike has estimated costs but no hard local budget guard. Concrete fix: require an explicit `S0_MAX_SPEND_USD`/confirmation variable or make the script print the planned calls and abort unless `OTR_RUN_LIVE_CLOUD_SPIKE=1`.

5. [§2 Video output] Writing downloaded video bytes to a local `.mp4` is underspecified. Concrete fix: write to a temp file in the episode/output temp area, fsync/close, then atomic rename to final `.mp4`; verify nonzero size before returning the path.

6. [§5 S1] “guarded-import-logs-LOUD + rows present with flag off” needs acceptance criteria. Concrete fix: specify that cloud engines appear in registry/COMBO with `OTR_ENABLE_CLOUD` off, but `assert_usable` fails closed with a LOUD reason and no network import/call occurs.

7. [§3] “Delete after; do NOT commit” removes the exact reproducer for the seam. Concrete fix: either commit a sanitized no-key probe harness behind `OTR_RUN_LIVE_CLOUD_SPIKE=1`, or store the throwaway script content in `S0_RESULTS.md` with secrets redacted.

OPTIONAL / NICE-TO-HAVE:
- [§5] Add a fake util-client test double for `sync_op`, `poll_op`, upload, and download so the cost/lease/auth paths can be tested without live credits.
- [§5] Add timeout/cancel policy for `poll_op` so a provider-side stuck task does not occupy the Comfy executor indefinitely.
- [§2] Record provider result media metadata in the ledger: provider, task id redacted/hash, resolution, duration, and whether audio was stripped.

CUT THESE (over-engineering):
1. [§3] Cut “per-call credit cost observed” as an adapter-coding input. It is useful operator telemetry, but adapters should use the dated price table and fail-closed stale-price logic, not one observed spike charge that may vary by model revision/account/promotion.

2. [§4.5] Cut an independent “confirm `/proxy/`” gate unless S0 shows util-client ambiguity. If the installed util-client signatures and module paths are pinned and the adapters only call that util layer, separately proving internal proxy routing is not necessary for build readiness.

3. [§5 S5] Defer `commercial_clean` dated-table polish until after the three cloud adapters pass live smoke. It does not affect invocation correctness, auth, lease safety, or billing reservation mechanics for the initial feature.