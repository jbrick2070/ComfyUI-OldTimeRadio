<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: yes-with-fixes. The plan has converged on a solid, safe architecture, but introduces a few critical contradictions in the latest pass regarding budget pooling and IO invariants that must be resolved before coding begins.

MUST-FIX BEFORE BUILD:
1. **[Sec 2 & Sec 9 #7] Budget Accumulator Contradiction:** Section 9 item #7 mandates "One credit pool chat+media", but Section 2 defines a media state machine tracking "estimated_usd". The existing chat lane (`_otr_comfy_backend.py`) tracks LLM *tokens* (`_run_token_total`), not USD. You cannot pool tokens and USD without building a live conversion engine (massive scope creep).
   *Fix:* Drop "One credit pool chat+media" from Section 9. Explicitly state in Section 2 that the Cloud Media budget is a SEPARATE accumulator tracking USD (`OTR_CLOUD_MAX_USD_PER_RUN`), completely independent of the Comfy Credits LLM token budget.
2. **[Sec 5] IO in Registry `assert_usable`:** Section 5 states "ffmpeg availability is checked in cloud video assert_usable". The grounding explicitly mandates that the registry's `assert_usable` MUST NOT do IO (it enforces the "registry IS the menu" invariant; disk/token checks happen downstream). Checking the system path for ffmpeg is IO.
   *Fix:* Move the ffmpeg check to the adapter's render-lifecycle `assert_usable(host_caps, profile...)` or `prepare()` methods (which are designed for IO/environment checks), NOT the registry's `assert_usable(name, role)`.
3. **[Sec 2] Auth Hidden Inputs on Media Nodes:** Section 2 says the Auth Broker resolves the token into the session, but doesn't explicitly require the media nodes to declare the auth inputs. If they only declare `prompt_id`, the ComfyUI frontend will not send the auth payload to them.
   *Fix:* Explicitly state in Section 2 that cloud-capable media nodes MUST declare the exact same ComfyUI auth hidden inputs (`auth_token_comfy_org`, `api_key_comfy_org`) in their `INPUT_TYPES` as the chat node, so the frontend injects the credentials for the Auth Broker to read.

SHOULD-FIX:
1. **[Sec 5] Protocol Updates:** The plan says `reactivity` and `must_strip_audio` become registered descriptor fields. Ensure the builder explicitly updates the `VideoEngine` Protocol and `role_compat.EngineDescriptor` `TypedDict` to include these new keys, as the grounding shows they currently only expect `required_inputs`.
2. **[Sec 3] Error Normalization:** Add `orphaned_job` to the list of normalized errors in Section 3 to match the cancellation state defined in Section 2.

OPTIONAL / NICE-TO-HAVE:
- In S0, when generating `partner_nodes.yaml`, explicitly capture and log the ComfyUI version or commit hash the pin was generated against to aid in debugging schema drift.

CUT THESE:
1. **[Sec 9 #7] "One credit pool chat+media"** - Safe to cut (and necessary to cut, per Must-Fix #1) because managing two separate env vars (`OTR_COMFY_MAX_TOKENS_PER_RUN` and `OTR_CLOUD_MAX_USD_PER_RUN`) is vastly simpler and less error-prone than unifying currencies.

VERIFY-AT-BUILD:
- **Heartbeat API:** Confirm the exact ComfyUI executor progress/interrupt API name (Item #9) before writing the watchdog loop, otherwise long Kling jobs will trigger the 5-minute stall detector.
- **Wan Audio Capability:** Confirm whether Wan 2.2 I2V actually supports audio conditioning (Item #5); if not, it remains strictly `mute_only` as defined in Section 7.
- **Loudness Reference:** Confirm the exact file/constant location of the existing loudness reference (Item #11) to ensure the canonicalizer matches the local pipeline perfectly.