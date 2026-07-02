<!-- requested_model: deepseek/deepseek-v4-pro | resolved_model: deepseek/deepseek-v4-pro-20260423 -->

VERDICT: no. The plan assumes a generic Comfy API proxy for partner media nodes that does not exist in the grounding; the actual API surface for cloud image/video/audio generation is undefined, and the audio-reactive claim for the cheap video row is unverified.

MUST-FIX BEFORE BUILD:
1. [§4 Architecture / §2 Cloud catalog audit] The plan relies on a shared transport that submits jobs to partner API nodes via `api.comfy.org`, but the existing Comfy Credits backend (`_otr_comfy_backend.py`) only implements OpenRouter chat completions. There is no evidence of a generic proxy for image/video/audio generation. MUST-FIX: Provide a concrete specification of the HTTP endpoints, request/response formats, and authentication for each selected partner node (Kling, ElevenLabs, Recraft, etc.), or confirm that the existing partner nodes can be used directly as ComfyUI nodes without a custom transport. Without this, the transport module cannot be built.
2. [§3d VIDEO rows] The cheap video row `cloud: wan_i2v` is claimed to be audio-reactive, but `Wan2ImageToVideoApi` only has "optional audio in" and it is not shown that the audio input drives visual generation (as opposed to being muxed). The audio-reactive requirement mandates that the engine consumes `audio_ref` to influence the video. MUST-FIX: Verify that Wan2ImageToVideoApi actually uses the audio to condition the generation, or replace it with a proven audio-reactive engine (e.g., a Kling variant that explicitly accepts audio for motion).
3. [§3a VOICE rows / §4 Transport (B)] The cheap voice row `chatterbox_cc` depends on Comfy Cloud workflow submission (transport B), which is unverified for headless operation. The fallback plan is vague: "fall back to lane (A) providers at slightly higher cost (named in plan, not silent)" but no fallback provider is named. MUST-FIX: Either confirm headless Comfy Cloud submission works, or specify the exact fallback provider (e.g., ElevenLabs Flash tier) and its cost, ensuring the cheap tier remains available.
4. [§4 Architecture] The plan introduces a new capability profile `cloud` that "turns every generative role default to its curated cloud row." The existing capability profile system (grounding: capability profiles derive enable-sets, not defaults) does not support overriding default engine selections. MUST-FIX: Specify how the profile applier will change default engine choices per role without breaking existing workflow JSON or the byte-identical baseline.
5. [§3 Curated cloud rows] The plan does not provide `commercial_clean` status for any cloud provider (ElevenLabs, Kling, Recraft, etc.). The release gate enforces `NONCOMMERCIAL_BLOCKED`, so cloud rows must be audited. MUST-FIX: Perform the ToS audit and set `commercial_clean` + license note for each curated row before promotion, or mark them as research-lane if not cleared.

SHOULD-FIX:
1. [§3a Voice-bank continuity] Voice cloning via `ElevenLabsInstantVoiceClone` adds significant complexity (seeding from CC0 clips, preset mapping). SHOULD-FIX: Defer voice cloning to a later iteration; initially use the cloud TTS with its own voices, and document that voice continuity is not guaranteed.
2. [§6 Open questions] The budget guard behavior (hard abort vs. degrade) is unresolved. SHOULD-FIX: Finalize the policy before S0, as it affects transport design and error handling. The current lean (hard fail-closed) is acceptable but must be decided.
3. [§2 Cloud catalog audit] Pricing data is missing. SHOULD-FIX: Obtain the Comfy partner-node pricing table before S0 to set realistic budget defaults and stamp `approx_cost` on each curated row.

OPTIONAL / NICE-TO-HAVE:
- The 3D seam (S5) is a placeholder; no changes needed.
- The cost ledger per asset is a good addition.

CUT THESE (scope / over-engineering):
- None identified; the plan is focused, but voice cloning could be cut as per SHOULD-FIX.

[ASSUMPTION] The plan assumes that the Comfy account bearer token (`auth_token_comfy_org`) works for all partner API calls, not just OpenRouter. Verify: the auth scope of the token.
[ASSUMPTION] The plan assumes that the partner API nodes' underlying HTTP APIs are stable and can be reverse-engineered. Verify: availability of API documentation.
[ASSUMPTION] The plan assumes that adding new engine names to dropdowns (via registry) does not cause widget index drift because ComfyUI COMBO widgets store string values. Verify: ComfyUI's widget value serialization for engine dropdowns.