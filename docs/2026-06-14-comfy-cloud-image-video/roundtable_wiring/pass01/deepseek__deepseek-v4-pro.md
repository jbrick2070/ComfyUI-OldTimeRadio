<!-- requested_model: deepseek/deepseek-v4-pro | resolved_model: deepseek/deepseek-v4-pro-20260423 -->

VERDICT: no. The plan has critical gaps around the invocation seam, cost guard, and lease-skip mechanism that would prevent a working build.

MUST-FIX BEFORE BUILD:

1. [§5] The invocation-seam probe is listed as a build-gating spike but the plan defers the entire implementation decision. Without resolving HOW to call the Comfy API nodes (option 1/2/3), the three adapters cannot be coded. The plan must pin this before any adapter work begins.

2. [§2a] `cloud_flux_pro` must return a PNG path, but the plan says "write the returned PNG to ComfyUI/output/..." without specifying HOW the adapter discovers where the Comfy API node wrote the file. The output-file-discovery question is deferred to §5 but is critical for this adapter's output contract.

3. [§4.1] The lease-skip mechanism says "if the resolved engine's `vram_class` is `cpu` AND it is a network engine (new `declared_isolation == "network"` marker, or a `is_network = True` adapter attribute)". The grounding excerpts show `declared_isolation` exists (e.g., `ISOLATION_IN_PROCESS`) but there is NO `"network"` value defined. The plan introduces a new isolation level without specifying where it's defined or how `dispatch_images` checks it. The existing `dispatch_images` code has NO lease-skip logic — the plan must specify the exact code change.

4. [§4.2] The cost guard is described as "a per-episode credit ceiling" with "deterministic reserve/spent accounting" and "dated price table" but provides NO concrete interface — no function signature, no table format, no integration point in the dispatcher. Without this spec, the adapters cannot implement cost enforcement.

5. [§2b, §2c] Both video cloud engines must return MP4 paths through `render_clip` → `canonicalize`, but the grounding excerpts show `MotionEngineBase` and the `render_clip` / `canonicalize` contract is NOT provided. [ASSUMPTION: the cloud adapters will subclass `MotionEngineBase` and implement its interface, but the base class contract is not grounded here — verify the exact return shape expected by the video render pipeline.]

6. [§4.3] The auth probe says "surface a missing key as `EngineUnusable` in `assert_usable`" but the grounding excerpts show `assert_usable` signatures differ between image (`assert_usable(self, host_caps, profile, request_template=None)`) and video (same shape in the Protocol). The plan says "Dep-free helper: is a Comfy API key present (`OTR_COMFY_API_KEY` env, then Comfy account)?" — the question mark indicates uncertainty about the key source. This must be resolved before coding.

7. [§4.4] "Per-adapter network timeouts: connect + total wall-clock + poll interval + 429/5xx retry + cancel" — no concrete values, no integration point in `render_clip` / `render_image`. The plan lists categories but no spec.

SHOULD-FIX:

1. [§2b] `cloud_ltx2` says "derive from the beat's audio/budget window the same way the local LTX adapter sizes frames" but the grounding excerpts don't include the LTX adapter code. [ASSUMPTION: the local LTX adapter's frame-count derivation is the template — verify it's compatible with a cloud engine that has no local audio reference.]

2. [§6] "The six dropdowns... are COMBOs built from the registry at `INPUT_TYPES` time, so the new engine ids appear automatically once registered + imported — no JSON edit needed." The grounding excerpts show `INPUT_TYPES` for `OTRImageGenDispatcher` but NOT for the video director nodes. [ASSUMPTION: the video dropdowns work the same way — verify the video director's `INPUT_TYPES` builds its COMBO from the registry.]

3. [§7] The test plan lists "cold-import: `test_cold_import_no_heavy_libs` still passes" but the cloud adapters will need HTTP client libraries (requests/httpx) for the Comfy API calls. These must be lazy-imported inside `render_*` to maintain cold-import cleanliness. The plan mentions "the SDK/HTTP client is lazy, inside `render_*` only" but doesn't specify which library or its import guard.

4. [§2c] `cloud_kling_avatar` says "Output: silent MP4 (V-1: only `OTR_MasterAudioMux` adds audio)" — but Kling Avatar is audio-driven (it takes `audio_ref`). The output is silent because the audio is added later, but the adapter must confirm the Kling Avatar API accepts audio input and returns video-only output. [ASSUMPTION: the Comfy API node for Kling Avatar works this way — verify.]

OPTIONAL / NICE-TO-HAVE:

1. [§9.2] The per-episode credit ceiling default value is listed as an open decision — recommend a concrete default (e.g., $5.00) with the behavior (skip to radio floor) so the cost guard can be coded without waiting for operator input.

2. [§9.3] Dropdown UX decision (visible-disabled vs hidden) — recommend visible-disabled as stated, which requires no code change to the plan.

CUT THESE (over-engineering):

1. [§7] The test plan lists "Output coercion: cloud image adapter returns a path; `_coerce_pixels` accepts it; a returned IMAGE tensor is rejected LOUD." The `_coerce_pixels` function in the grounding excerpts already handles path strings via the `isinstance(result, str)` branch and rejects unknown types with a `TypeError`. No additional coercion logic is needed — the existing path handles this.

2. [§6] "If the cost guard adds a Director widget (e.g. a per-episode credit ceiling), that is ONE new optional widget APPENDED at the end of the Director's widget list." The cost guard can be an environment variable or policy field — adding a Director widget is premature until the cost guard interface is settled.