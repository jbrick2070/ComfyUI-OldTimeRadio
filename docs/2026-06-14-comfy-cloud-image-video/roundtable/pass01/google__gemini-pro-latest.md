<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: no. The plan claims "zero dispatcher change" but requires structural changes to the dispatcher to handle GPU leases, cost guard state, and the proposed fallback ladder.

MUST-FIX BEFORE BUILD:
1. [Section 4] GPU Lease blocking. The plan claims cloud adapters should "skip the AS-3 GPU-residency lease" as a "per-adapter behavior." This is impossible as written: `otr_image_gen_dispatcher.py` unconditionally calls `_lease.acquire()` before calling `gen_fn`. A slow cloud API call will block the local GPU for the entire network wait. Fix: Modify the dispatcher to check the engine's `vram_class` from `_ireg.CAPABILITIES` and skip the lease if it is `"cpu"` or a new `"none"` class.
2. [Section 4] Cost guard placement. The plan states `assert_usable` should fail closed if the episode credit budget is exceeded. However, `assert_usable` is stateless and receives no episode context, ledger, or budget state (the dispatcher calls `_ireg.assert_usable(engine_id, role)`). Fix: Implement the cost guard inside the adapter's `render_image` (raise an exception if the budget is blown) or modify the dispatcher to track and pass budget state.
3. [Section 4] Network failure fallback contradiction. The plan requires network failures to degrade "cloud → local engine → radio floor". This contradicts `otr_image_gen_dispatcher.py`, which explicitly catches render exceptions and skips the object entirely (degrading straight to the radio floor). Fix: Drop the "local engine" fallback requirement and accept the existing behavior (fail straight to radio floor), OR rewrite the dispatcher's exception handler to support in-render engine swapping.
4. [Section 6] Cache key misunderstanding. The plan asks whether to use "hash-the-result vs key-on-request" for cache semantics. Fix: Remove this open question. `otr_image_gen_dispatcher.py` already uses `request_cache_key` (keyed on prompt_hash, seed, engine_id, etc.) for cache hits. The output is only hashed for the filename, not the cache key.

SHOULD-FIX:
1. [Section 1 / 5] Output format assumption. The plan assumes Comfy API nodes "return a file". If the API node returns a standard ComfyUI `IMAGE` tensor, `_coerce_pixels` will fail because it expects a `.png` path or a numpy array. Fix: Ensure the cloud adapter explicitly converts the ComfyUI `IMAGE` tensor to a numpy array (e.g., `(tensor[0] * 255).numpy().astype(np.uint8)`) before returning it to the dispatcher.
2. [Section 3a] `cloud_auto` routing. `cloud_auto` is proposed as a single adapter, but it needs to know which slot it is filling to pick the right model. Fix: Ensure `cloud_auto`'s `render_image` reads `request["role"]` to dynamically route to the correct cloud API node.

OPTIONAL / NICE-TO-HAVE:
- Provide a clear UX in the wizard for users to input their Comfy API key. Relying solely on the README for a newbie audience will cause significant friction.

CUT THESE (over-engineering):
1. [Section 4] In-render fallback to local engines ("cloud → local engine → radio floor"). Why: The dispatcher is already built to fail gracefully to the radio floor on any render exception. Building a stateful retry loop to swap engines mid-render violates the "no architecture rethink" goal and adds unnecessary complexity.

[ASSUMPTION] I am assuming that ComfyUI API/Partner nodes can be instantiated and executed synchronously from within a standard Python function (`_inprocess_gen_fn`), and that they do not strictly require the asynchronous ComfyUI execution graph (PromptExecutor) to manage their polling and credit billing. If they do require the execution graph, the adapter will need to trigger a sub-graph execution.