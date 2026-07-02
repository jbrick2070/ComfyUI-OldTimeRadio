<!-- requested_model: deepseek/deepseek-v4-pro | resolved_model: deepseek/deepseek-v4-pro-20260423 -->

VERDICT: no. The plan lacks a concrete invocation mechanism for partner nodes (Surface A), omits the cost estimation interface, and fails to address how long-running cloud calls stay alive under ComfyUI's watchdog, making the described code unbuildable.

MUST-FIX BEFORE BUILD:
1. [Sec 5, Surface A] The "thin invoke-wrapper around bundled partner node classes" is undefined. Partner nodes (e.g., ElevenLabsTextToSpeech) rely on ComfyUI server internals (hidden inputs, global server object). The plan must specify exactly how credentials are injected and how the node's function is called without a running server. Provide a concrete example for one node, e.g., instantiate the class, set attributes like `self.server` or pass hidden inputs as keyword arguments. [ASSUMPTION: the node class’s `run()` signature accepts a `server` argument or can be patched.]
2. [Sec 2, Budget guard] The pre‑submit budget check requires a per‑request cost estimate, but no interface for `estimate_cost()` is defined, and the pricing table format is missing. Must add: each row’s adapter provides a `cost_estimate(params) -> float` method; the shared backend reads a pricing table (e.g., dict with unit costs) and applies it to request parameters. The budget guard implementation cannot be coded otherwise.
3. [Sec 2, Watchdog / long polling] Cloud generation jobs may block for minutes, causing ComfyUI’s execution watchdog to kill the node. The plan notes the need for heartbeats but does not specify how they will be delivered. Must detail a mechanism, e.g., the cloud adapter must periodically call `comfy.model_management.throw_exception_if_interrupted()` or send a progress update via the node’s `send_sync`, with a concrete frequency and code snippet.

SHOULD-FIX:
4. [Sec 2, Idempotent billing cache] The cache key includes "params" but not whether the random seed is held constant to allow reuse. Clarify how seeds are handled to avoid unintended cache misses and double billing. Also specify the concurrency model (file‑based JSON ledger under episode directory) and locking.
5. [Sec 2, Fallback chain] The "pre‑declared fallback chain" per row is mentioned but not defined. Provide a concrete configuration format (e.g., a list of row IDs in the adapter) so the policy matrix can be implemented.
6. [Sec 3, Canonicalization] For video, the `must_strip_audio` flag requires re‑encoding, which may be slow. State that local ffmpeg will be used and ensure the environment includes it; mention that stripping will not break mux‑later because audio is supplied by the master mixer.
7. [Sec 4a, Voice] The `CastLock` → stock‑voice table mapping is not defined. The adapter must hardcode the mapping; specify the file or data structure now to avoid later drift (e.g., a JSON mapping preset names to ElevenLabs voice IDs). [ASSUMPTION: the curated voice IDs are stable.]

OPTIONAL / NICE-TO-HAVE:
- Cache the pinned node schemas in a versioned file to catch upstream API changes early.
- Provide a mock partner node for CI so adapters can be tested without real API calls.

CUT THESE (over‑engineering):
- None obvious; the plan is already minimal.

ASSUMPTIONS (inferred beyond the document):
- The `comfy_api_nodes` custom nodes are installed and their classes are importable.
- ComfyUI’s hidden inputs (`auth_token_comfy_org` / `api_key_comfy_org`) will be populated on a headless server as advertised.
- The `otr\episodes\<ep>\` directory structure exists and is writable.
- The partner nodes’ `run` methods can be called synchronously from within the OTR node’s execution thread.
- The Comfy Cloud API (Surface B) is either usable or safely quarantined; the plan does not depend on it.