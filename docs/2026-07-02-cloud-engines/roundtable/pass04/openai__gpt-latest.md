<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. Still not build-ready as-is: row counts violate the stated brief, and cache-keying, auth, partner-output mapping, reactivity descriptors, interrupt semantics, and verify steps remain build-blocking ambiguous.

MUST-FIX BEFORE BUILD:
1. [Brief / §7] Defect: The stated goal is “1 CHEAP + 2 BEST cloud rows per modality,” but §7 lists only 2 rows for VOICE and MUSIC: VOICE has `elevenlabs_flash` + `elevenlabs_tts`; MUSIC has `stability_audio` + `sonilo_music`. Concrete fix: either add a second BEST row for VOICE and MUSIC, or explicitly amend the goal to “1 CHEAP + 1–2 BEST, depending on modality” and update acceptance tests accordingly. As written, a builder cannot satisfy both §7 and the brief.

2. [§4] Defect: Cache lookup is impossible/ambiguous if `sha256` is part of `CloudAssetKey`, because §2 says dispatch reserves budget “after cache lookup, immediately before submit,” while §4 says `sha256` is computed only after canonicalization mutations, i.e. after provider output exists. Concrete fix: split the concepts:
   - `RequestCacheKey`: deterministic pre-submit key from provider/row/schema/version/prompt/inputs/seed-if-supported/canonicalizer-version/etc.
   - `content_sha256`: post-canonicalization manifest field for integrity and ledger proof.
   Do not require `content_sha256` to participate in the pre-submit cache lookup key.

3. [§2] Defect: Auth precedence is under-specified and conflicts with the grounded chat lane shape. Grounding shows hidden names `auth_token_comfy_org` and `api_key_comfy_org`, and chat `_bearer()` prefers configured API key over bearer. §2 says `OTR_COMFY_API_KEY env > server config > logged-in hidden-input token` but does not define “server config,” does not name whether `api_key_comfy_org` is treated as server config or hidden input, and does not name the exact hidden constants. Concrete fix: make the media broker order explicit:
   `OTR_COMFY_API_KEY` env > verified Comfy server/API-key source [name exact source] > hidden `api_key_comfy_org` > hidden `auth_token_comfy_org`.
   If “server config” is not a concrete API available to nodes, remove it or make it a verify-at-build item before implementation.

4. [§1 / §3] Defect: `partner_nodes.yaml` pinning does not specify output selection. S0 reads `RETURN_TYPES` and `FUNCTION`, but the plan never says how an adapter chooses the returned media object/path when a partner node has multiple outputs or non-obvious return shape. Different implementors could pick first output, first path-like output, or modality-specific output, producing incompatible behavior. Concrete fix: require `partner_nodes.yaml` to include, per row:
   - expected `RETURN_TYPES`;
   - selected output index/name;
   - expected media kind/path/blob shape;
   - a fixture assertion from S0 proving the selected output can be canonicalized.
   Runtime drift must compare this pinned output contract before invoking.

5. [§1 / §3] Defect: Error taxonomy is inconsistent. §1 says missing/drifted rows fail with named `unsupported_schema` / `MALFORMED_CONFIG`; §3 normalized errors list includes `unsupported_schema` but not `malformed_config`. Grounding registries use `MALFORMED_CONFIG` as a usability reason. Concrete fix: define one canonical public error-code set and exact mapping:
   - unregistered/malformed saved engine id -> `malformed_config`;
   - pinned partner class missing or schema/return drift -> `unsupported_schema`;
   - role/profile mismatch -> `incompatible_profile`;
   then use the same spelling/case in resolver, invoke, ledger, and tests.

6. [§5 / §7] Defect: `reactivity` and `must_strip_audio` are required descriptor fields, but §7 assigns only `wan_i2v` as `mute_only`; it does not assign concrete values for `kling_avatar` or `seedance_2`, and it does not assign `must_strip_audio` for any row. Concrete fix: add a video-row descriptor table in §7 for every shipped video row, e.g. row id, `reactivity`, `must_strip_audio`, required inputs, fallback chain, license gate. Also add a test that `descriptor_for_engine()` exposes these fields. Do not leave adapters to infer them.

7. [Brief / §5 / §7] Defect: The “audio-reactive video” goal is not enforced for `music_visual`. §5 only gates talking/announcer beats, while grounding shows all surviving video roles, including `music_visual`, supply `audio_ref`. §7 also ships `wan_i2v` as `mute_only`, which could be selected for music and still pass the current policy. Concrete fix: either:
   - enforce the reactivity gate for `music_visual` too, requiring `required_audio_ref`, `lipsync_overlay`, or a clearly defined audio-reactive category; or
   - explicitly narrow the goal to “audio-reactive talking/announcer video only; music visuals may be mute.”
   Add matrix tests for all three grounded roles: `announcer_visual`, `music_visual`, `character_video`.

8. [§3 / §2] Defect: Cancellation is said to be cut, but the watchdog is required to poll the interrupt flag every tick. The plan does not say what happens when the interrupt flag is set after a provider job has been submitted. Concrete fix: define interrupt semantics without provider cancellation:
   - before submit: mark `ABORTED`/`RELEASED`, no billing reservation remains;
   - after submit: stop local wait, emit `ORPHANED_JOB {provider, job_id, submitted_at, estimated_usd}`, release semaphore, classify as `interrupted` or map explicitly to `timeout`;
   - add that error code to §3 taxonomy if not mapped to an existing one.

9. [§1 / §8] Defect: The invariant says shipped rows register unconditionally and the registry is the menu, but §8 only requires a registration test “when flag on.” That misses the saved-COMBO validity invariant when `OTR_ENABLE_COMFY_CLOUD_MEDIA` is off. Concrete fix: add tests for both flag states:
   - flag off: all shipped cloud row ids are registered and dropdown values remain valid, but resolver raises `GATED_BY_FLAG`;
   - flag on: all shipped rows resolve to yaml-backed descriptors or fail closed on schema drift.

10. [§1 / §2 / §7 / §8] Defect: Cloud rows must participate in the existing `CAPABILITIES` invariant, fallback cpu checks, profile enable-set derivation, and no-GPU cloud profile, but the plan never states the required CAPABILITIES rows for the new cloud adapter names. Concrete fix: for every row in §7, add a CAPABILITIES declaration keyed exactly to adapter `engine.name`, with cloud-safe values such as `vram_class: "cpu"`, `vram_estimate_mb: 0`, `required_toolchain: None`, `requires_sidecar: False`, `cpu_ok: True`, and empty/local model requirements unless a row has a real local dependency. Add per-namespace consistency tests: registered cloud row ids == CAPABILITIES keys subset, no orphan CAPABILITIES rows.

11. [§9] Defect: Verify item #7 “One credit pool chat+media” has no concrete verify step in §8 or §9. Concrete fix: add an S0 live-smoke or operator verification step that runs one chat credit call and one media partner-node call against the same logged-in/API-key account and confirms both debit the same Comfy credit pool, or documents that media uses a separate pool and updates budget/ledger wording.

12. [§9] Defect: Verify item #12 “comfy_api_base override honored by media partner nodes” has no concrete test. Grounding shows chat supports `OTR_COMFY_API_BASE` / `OTR_COMFY_CHAT_PATH`; media partner nodes may instead use Comfy install defaults. Concrete fix: add a pre-smoke test that sets the Comfy API base override and proves the media partner node uses it, or explicitly document “media partner nodes are install-base-only; `OTR_COMFY_API_BASE` applies only to chat.”

SHOULD-FIX:
1. [§2] “No module globals” conflicts rhetorically with a backend-owned keyed session table, which will likely be a process singleton. Concrete fix: clarify “no credential/budget/session state in adapter module globals; the backend session table is the allowed process singleton and is lock-guarded.”

2. [§2] The session key is “prompt_id (+ episode_id once parsed)” but teardown is “assembler done signal or prompt completion.” Concrete fix: specify collision behavior and teardown order for multiple episodes under one prompt and for failed prompts where episode_id was never parsed.

3. [§2] Rate-limit provider ID normalization is specified, but the source field is not. Concrete fix: require a pinned yaml field `provider_id`, normalized to `[A-Z0-9_]+`, and fail S0 if two rows normalize to the same env var name.

4. [§6] `actual_duration_s` is named, but no migration behavior is specified for old script/ledger artifacts. Concrete fix: validators should fail with a named missing-field error for cloud runs, while local legacy artifacts either remain supported or are explicitly migrated.

5. [§8] S0 includes both control plane and live Kling smoke. That is fine, but acceptance should separate no-network CI from live-smoke acceptance so ordinary CI cannot accidentally require credits. Concrete fix: label each S0 acceptance item as no-network, fixture-only, or `OTR_RUN_CLOUD_SMOKE=1`.

OPTIONAL / NICE-TO-HAVE:
- Add an example `partner_nodes.yaml` row showing `node_key`, hidden constants, selected output index, pricing version, provider_id, seed support, sync/job mode, and descriptors.
- Add ledger examples for cache hit, provider-rejected release, timeout orphan, fallback hop, and schema drift.

CUT THESE:
1. [§5] Cut `optional_audio_ref` unless a shipped S0-pinned video row actually needs it. It weakens the policy model and creates ambiguity about whether “audio-reactive” is guaranteed. Safe to cut because current §7 only names `wan_i2v` as `mute_only` and does not assign any row `optional_audio_ref`.

2. [§2] Cut “sync-vs-job” from the high-level S0 bullet unless it is made a concrete yaml field used by invocation. As written it is jargon without an enforceable build output. Safe to cut if replaced by explicit yaml fields: `execution_mode: sync|job`, `job_id_field`, `poll_status_field`, `terminal_states`.

3. [§8] Cut the subjective “SA3-vs-Stability listening test” as a build gate, or move it to optional QA, unless it has pass/fail criteria. Safe to cut from build lock because it does not prove wiring correctness; keep cached rerun and structural audio acceptance as required gates.

VERIFY-AT-BUILD checklist:
1. [§8 / §9.1] Headless auth injection: S0 smoke #1 must prove hidden/API token injection through `/prompt` works without node surgery.
2. [§8 / §9.2] Kling audio conditioning: S0 smoke #2 must prove a WAV fixture actually conditions the generated clip, not merely uploads alongside it.
3. [§8 / §9.3] Pricing table version: S0 must stamp a versioned pricing table into yaml/ledger and fail if row pricing is absent.
4. [§8 / §9.4] ToS per provider: early S0 audit must mark every row commercial allowed/blocked/pending; pending remains blocked for commercial runs.
5. [§9.5] Wan audio conditioning: verify whether Wan supports audio conditioning; until proven, `wan_i2v` remains `mute_only`.
6. [§8 / §9.6] SA3-vs-Stability: if kept as a gate, define pass/fail criteria; otherwise move to optional QA.
7. [§9.7] One credit pool chat+media: missing concrete step; add S0 operator verification or document separate pools. See MUST-FIX #11.
8. [§5 / §9.8] ffmpeg in all run modes: verify in local, headless, and cloud-profile runs; `assert_usable` must fail before paid submit if unavailable.
9. [§3 / §9.9] Executor progress/interrupt API: resolve exact Comfy executor progress and interrupt APIs before live smokes; heartbeat tick must be <=30s.
10. [§1 / §9.10] Hidden-input constants: S0 yaml must record exact constants and drift CI must compare them against live `INPUT_TYPES()`.
11. [§6 / §9.11] Loudness reference: locate the existing lane’s source-of-truth loudness constant and reuse it for cloud/local canonicalization.
12. [§9.12] `comfy_api_base` override: missing concrete step; add a media partner-node override test or document media as install-base-only.