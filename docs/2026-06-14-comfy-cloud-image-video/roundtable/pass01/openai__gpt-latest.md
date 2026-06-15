<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. The plan’s “adapter row / zero dispatcher change” claim is false for the grounded image path, and the cost/fallback/Partner Node invocation seams are not specified enough to build.

MUST-FIX BEFORE BUILD:
1. [4.4, 5] Cloud image adapters cannot “skip the AS-3 GPU-residency lease” with zero dispatcher change. In `dispatch_images`, `_lease.acquire(...)` is unconditional immediately before `gen_fn(request)`, so every cloud image render would still block on the local GPU lease. Concrete fix: add an explicit no-local-GPU/remote execution flag to the selected image engine or capability row, and conditionally skip `_lease.acquire`, `_lease.release`, and the post-gen NVML probe for that engine. Do not leave this as “per-adapter behavior” unless the dispatcher actually asks the adapter before acquiring the lease.

2. [4.3, 4 “Credit cost guard”] The proposed cost guard cannot be implemented solely in adapter `assert_usable` as written for images. The grounded dispatcher calls `_ireg.assert_usable(engine_id, role)` before constructing the full request, so the check has no width/height, duration, resolved provider, slot, budget state, or audio/video length. Concrete fix: define a cost-guard API and call it with the full request context before dispatch. For images, either construct the request before `assert_usable` and pass it through, or add a separate `estimate_cost(request, policy/session_ctx)` step. For video, verify the equivalent dispatcher call path and make the same change if it also lacks request context.

3. [4.1] “Add cloud engines as adapter rows” omits required capability declarations. Both grounded registries contain `CAPABILITIES` tables and comments state they are consumed by `capability_profiles.py` to derive profile enable-sets; “A new engine ships its own row here.” A registered cloud engine without a matching `CAPABILITIES` row risks being excluded or failing validation. Concrete fix: add `CAPABILITIES` entries for every cloud engine in the image/video registries, or explicitly update the capability-profile validator to support a `remote/cloud` execution class. Do not use fake VRAM estimates as the only signal.

4. [4 “Network failure ladder”, 4.3] The plan says cloud should degrade “cloud → local engine → radio floor,” but the grounded image dispatcher does not do that. On `assert_usable` failure or render exception it appends a warning and skips the object; it does not call a fallback resolver or retry with a local engine. Concrete fix: either implement explicit fallback selection/retry in image dispatch and video dispatch, including ledger provenance for the substituted engine, or change the plan to say cloud image failures skip to the existing downstream radio floor. Do not claim local fallback exists for images until it is wired.

5. [1, 4.1, 5] The “Partner Node as adapter” invocation seam is underspecified. The grounded OTR image path calls Python engine methods (`render_image`) through `_inprocess_gen_fn`; it is not composing a Comfy graph node downstream. “Calls an API node under the hood” needs an exact implementation path: whether the adapter imports a Partner Node class, submits a Comfy workflow via internal API, or shells/posts to a running Comfy server. Concrete fix: specify the callable API, required ComfyUI version/extension dependency, auth injection point, polling behavior, and output file discovery contract before build. Preserve the cold-import invariant: no heavy Comfy/torch/network imports at registry import time.

6. [3a, 4.1] `cloud_auto` is not sufficiently specified for per-slot behavior. The grounded image request currently contains `role`, `kind`, object IDs, prompt, seed, and dimensions, but not the resolved policy slot name; `slot` is local to `dispatch_images` and is not placed in `request`. A single `cloud_auto` engine cannot reliably choose “announcer image vs music image vs other_beats image” from slot if slot is not passed. Concrete fix: either create separate engine IDs per slot/namespace, e.g. `cloud_auto_announcer_image`, `cloud_auto_music_image`, etc., or add `slot_name` / `resolved_slot` and available-input metadata to the request passed to the adapter. For video, verify the render request includes slot and `audio_ref` before relying on “Kling Avatar when audio_ref is present.”

7. [5, 6.5] Cache semantics for `cloud_auto` will be wrong unless the resolved provider/model becomes part of the cache key. The grounded image cache key includes `engine_id` and `engine_version`, but not the underlying provider/model selected by `cloud_auto`, API-node version, quality tier, or provider params. If `cloud_auto` changes from Flux Pro to Nano Banana or Luma, old cached results can be reused under the same key. Concrete fix: resolve `cloud_auto` to a concrete provider/model before computing the cache key, and include resolved model ID, provider params, aspect/duration params, and adapter/API-node version in the key or `engine_version`.

8. [4.1, Sources/OTR internals] Registration/import sequencing is missing. The grounded registry exposes `register`, but the excerpt does not show any imports of concrete engine modules. [ASSUMPTION] If current adapters self-register only when their modules are imported elsewhere, new cloud adapters will not appear in dropdowns unless wired into that import path. Concrete fix: verify the existing image/video adapter import mechanism, then add cloud modules to the same package import/bootstrap path and add a cold-import test that `all_engine_names()` includes the cloud rows with `OTR_ENABLE_CLOUD` off.

SHOULD-FIX:
1. [4 “Credit cost guard”] Define budget accounting state, not just estimation. A pre-run estimate is not enough for concurrent/headless runs or partial failures. Concrete fix: add per-episode reserved/spent credit fields, reserve before dispatch, release or mark spent based on success/failure, and make the behavior idempotent across retries.

2. [4 “Auth probe”] Specify the actual API-key source and precedence. “env var or Comfy account” is not buildable. Concrete fix: name the env var/config key, define whether Comfy frontend login state is readable in headless mode, and make missing auth produce an `EngineUnusable`-style error surfaced in the dropdown/report.

3. [4 “Network failure ladder”, 6.4] Add explicit provider timeouts. The grounded image handoff wait only checks a local `.png` path readiness after generation; it is not a network/provider timeout. Concrete fix: each cloud adapter needs connect timeout, total wall-clock timeout, poll interval, retry policy for 429/5xx, and cancellation behavior.

4. [4 “commercial_clean per provider”] Define the truth source and default. “Set the flag conservatively per provider” is underspecified and ToS drift is called out but unresolved. Concrete fix: v1 should default unknown providers to `commercial_clean=False`, pin a dated source URL in the adapter metadata, and expose the reason in usability/wizard output.

5. [2, 3a] Prices are volatile but the proposed guard depends on them. Concrete fix: store a dated price table with provider/model IDs and units (`per_run`, `per_second`, token-based unsupported), and make stale/unknown price entries fail closed rather than run unbounded.

6. [4.2] “Dropdowns auto-populate” may expose disabled cloud rows even when the master flag is off. That may be intended, but the UX needs to distinguish “selectable but unusable” from hidden. Concrete fix: decide whether cloud rows are always visible with disabled/help text, or hidden until `OTR_ENABLE_CLOUD=1`; implement consistently in Director validation.

7. [5] Audio-driven video input mapping is asserted but not grounded. The provided excerpts do not show the video render request includes `audio_ref` or that announcer lines are available at video dispatch time. Concrete fix: verify the video dispatcher/request schema and add a test proving `announcer_video_model=cloud_kling_avatar` receives portrait/init image plus the correct per-line audio path.

8. [4.4, 5] Video output contract is not grounded. The video protocol has `render_clip` and `canonicalize`, but no grounded dispatcher/canonicalizer code is shown. Concrete fix: verify cloud adapters can return whatever the current video canonicalizer accepts, or define a cloud adapter that writes a local MP4 and returns the exact raw type the existing canonicalizer expects.

9. [3] The six-dropdown table says current defaults for video slots are `flux_still`, while the suggested local engines mention `humo / flux_still`, `abstract / ltx / wan`, etc. This is ambiguous for implementation and tests. Concrete fix: separate “current default value” from “compatible existing local engines,” and build tests against actual registry names.

OPTIONAL / NICE-TO-HAVE:
- [6.1] Add a wizard preflight that shows auth status, estimated credits for the whole episode, and which cloud models are selected before dispatch.
- [6.5] Document that cache hits reuse the first generated cloud output and that “same seed” does not imply reproducibility.
- [3a] Add a report line showing `cloud_auto -> resolved_provider/model` for every beat/object.

CUT THESE (over-engineering):
1. [2, 3] Do not surface all named cloud models in v1. Keep only `cloud_auto` plus one explicit image model, one explicit motion model, and one explicit talking-face model. This reduces ToS, pricing, auth, and fallback surface while proving the adapter seam.

2. [4 “commercial_clean per provider”] Do not build dynamic ToS fetching in v1. Use a conservative hardcoded dated table with unknown = not commercial-clean. Fetching legal/commercial policy at runtime adds failure modes and does not improve first-build correctness.

3. [4 “Credit cost guard”] Do not mirror the full OpenRouter spend-guard system unless required by existing reusable code. A minimal per-episode ceiling with deterministic local reservation/spent accounting is enough for v1 cloud media generation.

4. [3a] Avoid one universal `cloud_auto` if it requires complex slot inference. Separate per-namespace/per-slot auto engine IDs are simpler, more testable, and avoid hidden dependence on request fields that are not currently passed.