<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. The plan is not build-ready because its central “zero-local-GPU via Comfy Credits” story mixes unverified cloud surfaces, has a gating contradiction against the grounded audio registry, and does not define the media normalization/schema layer needed for remote assets to enter the existing episode pipeline.

MUST-FIX BEFORE BUILD:
1. [0, 3a, 3b, 4] The plan conflates two different cloud surfaces: “billed via the Comfy account (Comfy Credits partner API nodes)” in [0] versus `provider="comfy_cloud_gpu"` template lanes for Chatterbox and ACE-Step in [3a]/[3b], with [4] admitting Comfy Cloud workflow submission headless auth is unverified. This breaks the architecture arc: the cheapest voice/music picks depend on the least-proven transport. Concrete fix: split the catalog into two explicit surfaces: `comfy_credits_partner_node` and `comfy_cloud_workflow`. Do not promote any `comfy_cloud_workflow` row until S0 verifies headless auth, submission, polling, download, pricing, and cancellation. If that cannot be verified, replace the CHEAP voice/music rows with partner-node-backed rows before build starts.

2. [3d, 5-S3] The video lane says “AUDIO-REACTIVE REQUIRED,” but the CHEAP row `cloud: wan_i2v` is later accepted as “wan_i2v mute b-roll” in [5-S3]. That row does not satisfy the [3d] definition unless it actually declares and consumes `audio_ref`; “optional audio in” is not enough. Concrete fix: create a per-role video default matrix. Example: `announcer_video` default must be `kling_avatar` or another required-`audio_ref`/`lipsync_overlay` engine; `music_video` must use an audio-ref engine or be explicitly classified as non-reactive; `other_beats_video` may use muted I2V only if the requirement is relaxed for b-roll. Do not label the whole curated video set “audio-reactive” while one shipped row is mute.

3. [1, 4] The global/default-off gating story is contradicted by the grounded audio registry. [1] says adapters carry `requires_flag` and cloud rows are gated; [4] relies on `requires_flag` and `OTR_ENABLE_COMFY_CLOUD_MEDIA=1`. But the grounded `nodes/_otr_audio_engines/registry.py` says `assert_usable` has “NO GATED_BY_FLAG case,” “the registry IS the menu,” and a registered role-compatible engine is always usable. If cloud audio adapters are imported, they will be selectable regardless of `requires_flag` unless another layer filters them. Concrete fix: choose one gate mechanism and specify it. Either do not import/register cloud adapters unless the global flag is set, or add a dropdown/profile filtering layer that hides cloud rows when disabled, and add queue-time rejection for disabled cloud rows. Verify the same for video/image registries; do not rely on `requires_flag` unless the actual registry path enforces it.

4. [2, 3, 5-S0] The “CHEAPEST-WORKABLE” selections are not supportable because [2] says per-call pricing is not in the node dump and must be verified later. This directly undermines [3]’s selection rule and [6.1]. Concrete fix: demote every CHEAP label to “candidate” until S0 pulls the pricing table, records `approx_cost`, and runs at least one quality smoke per candidate. Build acceptance for S0 must include a table: row id, provider surface, class_type/template id, price basis, estimated episode cost, and “workable” sample verdict.

5. [4, 5-S0] “Thin adapters” are not credible without a pinned partner-node schema contract. [2] lists class_type names, but no required input fields, output field names, upload semantics, job status shape, file download shape, or error taxonomy for media nodes. A submit/poll/download backend cannot be built from class_type names alone. Concrete fix: add an S0 deliverable that captures/pins `object_info` or equivalent schema for each curated node/template: class_type, required/optional inputs, media upload fields, output artifact fields, job lifecycle states, retryable failures, and cost metadata. Adapters should be generated or validated against that schema.

6. [4] The plan lacks a media canonicalization layer, which is concept-level required for the existing episode invariants. Remote providers will return inconsistent sample rates, channel layouts, loudness, durations, image sizes, video FPS/codecs, container formats, and possibly embedded audio tracks. [4] only says “download” and “assets land straight in `otr\episodes\<ep>\`.” Concrete fix: define a canonical output contract before adapter work:
   - voice/music: WAV format, sample rate, channels, loudness/normalization, exact duration/trim rules;
   - stills: dimensions/aspect, color mode, file type, portrait-hash/reference preservation handling;
   - video: FPS, resolution/aspect, codec/container, duration policy, no embedded audio unless explicitly stripped, mux-LAST compliance;
   - all: content hash, ledger path, provider/job metadata, retry/fallback restamp behavior.

7. [1, 4] Auth reuse is underspecified. Grounding for `_otr_comfy_backend.py` shows the existing Comfy Credits auth is captured by the writer node at run time via hidden inputs. [4] says the media backend will reuse that auth, while [1] says existing media dropdown seams avoid node surgery. There is no concept-level statement for how `OTR_ImageGenDispatcher`, audio nodes, or `OTR_VideoDirector` receive `auth_token_comfy_org` / `api_key_comfy_org`. Concrete fix: add an explicit auth-wiring design: either every cloud-capable media node declares the hidden Comfy auth inputs, or a central run credential broker is introduced. verify: current media nodes have access to Comfy hidden auth inputs before assuming this is “reuse.”

8. [4, 6.5] Fallback/budget behavior is internally unresolved. [4] says every cloud asset restamps loudly on fallback; [6.5] asks whether budget should hard abort, degrade-to-local, or degrade-to-cheap-row and currently leans hard fail. These are different user-facing behaviors and affect ledger correctness. Concrete fix: define a policy matrix before build:
   - budget exceeded before request: hard abort or alternate row?
   - provider failure before asset: hard abort or fallback?
   - provider failure after partial asset: discard/retry/fallback?
   - fallback allowed only to local or only to cheaper cloud?
   Each path must state ledger wording and whether episode assembly continues.

SHOULD-FIX:
1. [4, 5-S4] The `cloud` capability profile is underspecified: “turns every generative role default to its curated cloud row” does not say whether the default is CHEAP, BEST 1, or role-specific. This matters especially for video because the cheap row may be non-reactive. Concrete fix: define exact cloud-profile defaults per role and tier-selection policy.

2. [3a] Voice-bank continuity via `ElevenLabsInstantVoiceClone` creates a new persistent remote identity/mapping subsystem that is not necessary for the first cloud lane and carries ToS/commercial risk. Concrete fix: defer voice cloning until basic TTS rows ship; first build should map CastLock presets to provider-native voices or a small audited preset table.

3. [4] Commercial-clean metadata is asserted but not specified for media rows. The grounded LLM catalog has explicit license fields, while the shown audio/video protocols only expose `commercial_clean` and no provider ToS/license-note schema. Concrete fix: define a media row metadata schema with provider ToS status, license note, audit date, allowed use, and whether the row can pass release gating.

4. [2, 3d] Native-audio video models are excluded from “audio-reactive,” but the plan does not define whether their generated audio must be stripped at canonicalization time or whether selection is blocked for OTR roles. Concrete fix: add a `generates_audio` / `must_strip_audio` capability flag and enforce mux-LAST by canonicalization.

5. [5-S2, 5-S3] The sprint order delays video audio-reactive validation until S3, but video is the modality with the hardest requirement in the operator brief. Concrete fix: move one minimal audio-ref/lipsync smoke into S0/S1 as a transport feasibility test before stills consume build time.

6. [1] The grounding claim “one shared engine-registry pattern, three namespaces ... all on dep-free `engine_registry_base.py`” is not fully true for the grounded audio registry: `nodes/_otr_audio_engines/registry.py` is its own implementation and explicitly says there is no flag-gate case. Concrete fix: revise [1] to distinguish “parallel pattern” from “same base implementation,” or migrate audio to the shared base before relying on shared behavior.

OPTIONAL / NICE-TO-HAVE:
- [6] Add a small “1940s radio aesthetic” evaluation rubric before arguing BEST rows: voice noise/age appropriateness, poster-style stills, restrained motion, black-and-white/period color controls, and lip-sync tolerance.
- [4] Add provider-level rate-limit/concurrency controls; otherwise full-episode parallel generation may trip partner throttles. [ASSUMPTION]
- [4] Add cancellation/resume semantics for long video jobs so an aborted local orchestration run does not orphan billable cloud work. [ASSUMPTION]

CUT THESE (scope / over-engineering):
1. [3e, 5-S5] Cut the 3D seam from this build. The plan says “nothing downstream consumes meshes yet,” so registry tokens and future candidate rows do not help the zero-local-GPU episode goal and risk calcifying unused abstractions.

2. [3a] Cut `ElevenLabsInstantVoiceClone` / cloned voice-bank mapping from the first implementation. It is not required to provide 1 cheap + 2 best voice rows, and it adds ToS audit, remote persistent IDs, preset mapping, and failure modes before the basic cloud TTS lane exists.

3. [3b] Cut `SoniloVideoToMusic optional later` from this campaign. The stated role is `theme_music`; video-to-music does not serve the current episode pipeline and can be reintroduced when there is an actual video-conditioned music requirement.

4. [3a, 3b, 4] Cut or quarantine `comfy_cloud_gpu` template rows until transport B is proven. It is safe because partner API-node alternatives exist in the catalog, and the current goal is specifically framed around Comfy Credits partner API nodes.