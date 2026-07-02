# OTR Cloud Engine Lanes -- pass02 (R2-synthesized)

DOCS ONLY campaign; build gated on operator go + free coder baton.
Brief: per modality 1 CHEAPEST-WORKABLE + 2 BEST-OF-CLASS cloud rows;
video serves the audio-reactive pipeline; zero-local-GPU episodes.

## 0. Goal (unchanged from pass01)

Cloud lane per generative modality; local CPU orchestration; opt-in,
fail-closed, cost-guarded; local byte-identical defaults untouched; no
sfx-role resurrection; OTR custom nodes never run on Comfy Cloud.

## 1. Provider surface (SINGLE, this build)

SURFACE A `comfy_credits_partner_node`: the 214 hosted partner API nodes
on the running install, executed by invoking bundled `comfy_api_nodes`
classes in-process under a defined contract (sec 5). Billing: Comfy
account credits. SURFACE B (Comfy Cloud workflow submission) is CUT from
this build -- unproven headless, and its polling would stall the local
worker; it lives in Appendix B as a research flag. Consequences:
- MUSIC rows: CHEAP-candidate `stability_audio`, BEST `sonilo_music`
  (both A). ACE-Step moves to Appendix B.
- VOICE rows: ElevenLabs is the only A-surface TTS provider; the lane
  ships two tiers (flash CHEAP-candidate, premium BEST). Chatterbox
  voice-continuity row moves to Appendix B. OPERATOR NOTE: the "2
  best-of-class" ask for voice/music is provider-bounded on surface A
  today; Appendix B rows restore the third option when proven.

## 2. Registration, gating, enforcement (resolves R1/R2 conflict)

- REGISTER UNCONDITIONALLY. The registry IS the menu (audio C6 invariant,
  registry.py ~line 151) and COMBO widgets validate saved values against
  the options list -- unregistered rows would break saved workflows.
  Cloud adapters always register; dropdown labels carry a `cloud$` prefix
  so opt-in is visible.
- ENFORCE AT THE PROFILE RESOLVER (the layer that already owns
  disk/token/commercial checks): if `OTR_ENABLE_COMFY_CLOUD_MEDIA` != 1,
  resolving any cloud row raises `EngineUnusable(GATED_BY_FLAG)` as a
  queue-time named error. The single-sourced reason taxonomy already
  contains GATED_BY_FLAG. Registry code untouched; no import-order
  dependency (kills the profile-timing bug).
- CAPABILITIES CONSISTENCY: every cloud row adds a CAPABILITIES entry
  (invariant: no registered engine without a row and vice versa):
  `{"vram_class": "cloud", "vram_estimate_mb": 0, "required_toolchain":
  None, "requires_sidecar": False, "cpu_ok": True,
  "model_requirements": ["comfy_account"]}`.
- PROFILE DEFAULTS: `default_engine_for_role(role)` is NOT touched
  (byte-identical baseline). New helper
  `resolve_default_engine_for_role(role, profile)` consults the profile's
  DEFAULT-OVERRIDE map first, falls back to the registry default;
  profile-aware callers (headless applier, dropdown pre-select under
  `--profile cloud`) use the new helper only.
- LICENSING: rows carry `license_audit_status:
  Literal["pending","commercial_clean","noncommercial_blocked"]`
  (mirrors the LLM catalog field); `pending` is BLOCKED for commercial
  runs by the release gate; boolean `commercial_clean` derives from it.

## 3. Control plane (CloudMediaSession -- no module globals)

`CloudMediaSession` object threaded through dispatch (concurrency-safe;
the module-global `set_auth` pattern of the chat lane is NOT reused):
run_id, episode_id, auth (token/api-key), budget ceiling (env
`OTR_CLOUD_MEDIA_BUDGET_USD` read ONCE as a static ceiling) + per-run
spent accumulator (mirrors `_run_token_total` reset pattern; no os.environ
mutation), ledger handle, cache root, cancel token, per-provider
semaphores.

- AUTH: hidden inputs `auth_token_comfy_org`/`api_key_comfy_org` declared
  on cloud-capable dispatch nodes (3a/3b/3c, ImageGenDispatcher,
  VideoRenderBatch) using the real constant names verified from the
  install at build time. HEADLESS reality (S0 smoke #1): hidden inputs
  are populated by the web app; a headless /prompt submission must
  INJECT them -- the smoke + the headless runner inject from server
  config / `OTR_COMFY_API_KEY`, precedence: explicit env > server config
  > logged-in token. Missing auth = fail-closed named error.
- BUDGET: each adapter implements `estimate_cost(request) -> CostQuote
  {provider, row_id, unit, unit_price_usd, quantity, estimated_usd,
  max_usd, pricing_source_version}`. Guard RESERVES estimated spend
  pre-submit, reconciles ACTUAL post-completion (ledger labels
  ESTIMATED vs ACTUAL when providers omit cost metadata). The
  per-episode cost gate runs MID-RUN: after script/ledger parse, before
  first media dispatch (beat counts are writer-dynamic; a pre-run gate
  is impossible). Exceed -> hard abort leg, LOUD, `BUDGET_ABORT`.
- FALLBACK POLICY: retry x2 w/ backoff -> pre-declared per-row fallback
  chain (list of row ids in the adapter) -> abort. The fallback RESOLVER
  consults the active profile enable-set + CAPABILITIES `cpu_ok` before
  selecting ANY local row: on a no-GPU host under `--profile cloud`, the
  chain is cloud-only then abort. Every hop restamps the ledger LOUDLY.
- CANCELLATION: CUT for v1; local abort logs `ORPHANED_JOB {provider,
  job_id, submitted_at, estimated_usd}`. No invented cancel hooks.
- RATE LIMITS: `OTR_CLOUD_MAX_CONCURRENCY_<PROVIDER>` (default 2, Kling
  1); semaphore keyed by provider; retries re-acquire (do not hold).

## 4. Billing cache (global, idempotent)

Location: `otr\cache\cloud_media\` (GLOBAL -- cross-episode dedup;
per-episode caching would re-bill identical assets). Canonical assets are
copied/hardlinked into `otr\episodes\<ep>\` so the deliverable invariant
holds. Key = `CloudAssetKey`: row id + resolved provider/model slug +
normalized request params + pinned seed (C7 env-seed pattern: cloud
requests always carry an explicit seed) + ALL input-asset content hashes
(init/reference image, base clip, audio slice) + output-contract version
+ adapter version + canonicalizer version + schema version. Atomic write
(temp + rename) with manifest; entries validated before marking
CACHED/BILLED. Cache dependency DAG: voice/music resolve first ->
slice hashes -> video keys; S2 acceptance includes a full re-run
producing 100% CACHED audio; S3 likewise for video.

## 5. Invocation contract (Surface A, the build's keystone)

- S0 produces `partner_nodes.yaml` pinned FROM THE LIVE INSTALL by
  importing each candidate class in-process and reading
  `INPUT_TYPES()` / `RETURN_TYPES` / `FUNCTION` (authoritative; hidden
  inputs visible -- /object_info is NOT the capture point). Per node:
  import path, class name, required/optional/hidden inputs, return
  slots, sync-vs-job-backed, upload semantics. All class names in sec 6
  are `node_class_name_candidate` until pinned; a missing class DROPS
  the row at registration with a LOUD log, never a crash.
- `invoke_partner_node(class_name, inputs, hidden_auth, *, timeout_s,
  cancel_token, session) -> PartnerResult` -- owned by the shared
  backend: class lookup from the pinned yaml, instantiation, FUNCTION
  dispatch, async bridge (backend-owned event-loop thread; adapters
  block with timeout), upload/download streaming (streaming SHA-256,
  never whole-media-in-memory), normalized error classes {auth, budget,
  retryable_transport, provider_rejected, timeout, corrupt_output,
  unsupported_schema}.
- WATCHDOG: while polling job-backed nodes the wrapper checks the
  interrupt/cancel token and emits a progress heartbeat every poll tick
  (<=30s) so the 5-min stall detector never false-kills a healthy cloud
  leg; verify the concrete progress API at S0 (verify-register #9).

## 6. Media canonicalization contract

Per-modality canonicalizers, fail-closed, never write partial assets
into episode paths: `canonicalize_<modality>(raw: PartnerResult,
request, session) -> CanonicalAsset {path, sha256, media_type,
duration_s, width, height, fps, container, provider_job_id, cost_quote,
actual_cost, validation_warnings}`.
- voice/music: WAV 44.1kHz; channel policy per stereo_policy widget;
  loudness normalized to the SAME reference the local lane produces at
  that stage (verify where loudness lives today -- do NOT invent a new
  LUFS convention); per-line duration tolerance +/-250ms w/ head/tail
  silence padding; ACTUAL duration emitted into line metadata (captions
  + delivery vectors validate against it). Per-line granularity only.
- stills: role canvas exact (e.g. 1472x832), sRGB PNG; portrait-hash /
  in-character checks re-run on cloud output.
- video: role fps/res/container; rows carry `reactivity:
  Literal["required_audio_ref","lipsync_overlay","mute_only",
  "optional_audio_ref"]` + `must_strip_audio: bool`; canonicalizer
  strips embedded audio via ffmpeg when flagged (ffmpeg presence =
  verify item; master audio frozen upstream, mux LAST).

## 7. Curated rows (all class names = candidates until S0 pins)

VOICE (per-line only): CHEAP-cand `cloud$ elevenlabs_flash`
(ElevenLabsTextToSpeech, flash/turbo tier) | BEST `cloud$ elevenlabs_tts`
(premium voices). Preset mapping: curated table in the adapter file
`{castlock_preset_id: {provider_voice_id, voice_name,
license_audit_status, fallback_voice_id}}`, audited like the pinned LLM
catalog; unmapped preset = fail-closed. Voice CLONING deferred;
TextToDialogue = experiment flag (breaks per-line captions/ledger).

MUSIC: CHEAP-cand `cloud$ stability_audio` (StabilityTextToAudio) |
BEST `cloud$ sonilo_music` (SoniloTextToMusic -- VERIFIED present in the
live install dump lines 1774-88; a panel claim that it is a hallucination
was itself the misread). SA3-continuity = listening test at S2.

STILLS: CHEAP-cand `cloud$ recraft` | BEST `cloud$ flux_pro` (BFL) |
BEST `cloud$ nano_banana_2` (reference-image edit -> character
consistency).

VIDEO (reactivity matrix): announcer/talking beats REQUIRE
required_audio_ref or lipsync_overlay -> default `cloud$ kling_avatar`
(KlingAvatarNode / KlingLipSyncAudioToVideoNode), alt `cloud$ seedance_2`
(ByteDance2ReferenceNode, audio-ref + identity). music_video +
other_beats b-roll: mute I2V allowed -> CHEAP-cand `cloud$ wan_i2v`
(Wan2ImageToVideoApi, mute_only until audio CONDITIONING is proven).
Native-audio models (Veo/Sora) only ever mute_only + must_strip_audio.
Every cloud video adapter declares exact `required_inputs`; matrix tests
via `descriptor_for_engine`.

## 8. Sprints

- S0 CONTROL PLANE: partner_nodes.yaml pinning; CloudMediaSession; budget
  guard + CostQuote + pricing table (source: Comfy partner pricing page,
  versioned); billing cache; ledger (append-only JSONL: run_id,
  episode_id, request_id, row_id, provider, provider_job_id, cache_key,
  status, estimated_usd, actual_usd, fallback_from/to, error_class,
  timestamps, sha256); rate limits; canonicalizer skeleton; ToS audit ->
  license_audit_status per row. SMOKES (live class, see tests): #1
  headless auth injection end-to-end on one cheap image node; #2 Kling
  audio-driven clip (the hard requirement, proven before stills work).
  Nice-to-have: `otr-cloud-doctor` (flags, visible rows, auth, budget,
  schema versions) + dry-run manifest mode (keys + estimates, no submit).
- S1 STILLS. RISK NOTE: "lowest risk" assumes ImageGenDispatcher resolves
  through the image registry -- CONFIRMED the registry + CAPABILITIES
  exist (`_otr_image_engines/registry.py`); dispatcher wiring check is
  R3's first job. Acceptance: 3-beat image set on-model via cloud.
- S2 VOICE + MUSIC. Acceptance: full audio episode, zero local GPU,
  STRUCTURAL bar (duration tolerances, loudness lint, pre-mux master ==
  muxed track); re-run = 100% CACHED; SA3-vs-Stability listening test.
- S3 VIDEO matrix. Acceptance: talking beat driven by episode audio;
  b-roll wan_i2v; re-run CACHED; mux-LAST intact.
- S4 `cloud` PROFILE: DEFAULT-OVERRIDE map + resolve_default_engine_for_
  role + no-GPU end-to-end acceptance + cost report. Workflow JSON
  default changes (if any) happen ONLY here, operator-gated, same-change
  rule + validator + widget audit.

TESTS (both classes): (a) no-network CI suite -- mock partner node
fixture + canned responses keyed to pinned schema versions; unit tests:
budget matrix, cache-key stability, canonicalization golden files,
auth-broker fail-closed, GATED_BY_FLAG resolver rejection when flag off,
schema-drift (pinned yaml vs live INPUT_TYPES import mismatch fails
LOUDLY); (b) live smokes behind `OTR_RUN_CLOUD_SMOKE=1` + credentials +
budget; S0 promotion requires smokes #1 and #2 green.

## 9. Appendix A: 3D (docs-only). Appendix B: Surface B (research flag)

3D unchanged (Tripo P1 / Rodin Gen2.5 / Meshy candidates; no sprint).
Surface B: Comfy Cloud workflow submission for ACE-Step music +
Chatterbox voice-continuity; requires its own flag
`OTR_ENABLE_COMFY_CLOUD_WORKFLOWS=1` + a recorded headless lifecycle
smoke artifact before any row registers; revisit post-S4.

## 10. Verify-at-build register

1. Headless auth injection (S0 smoke #1). 2. Kling audio conditioning
(S0 smoke #2). 3. Partner pricing table -> approx_cost per row + version.
4. ToS -> license_audit_status per provider. 5. Wan audio CONDITIONING
(else mute_only stands). 6. SA3-vs-Stability continuity listening test.
7. Credit pool shared across chat + media partner nodes. 8. ffmpeg
available to canonicalizer in all run modes. 9. Concrete progress/
interrupt API for watchdog heartbeats. 10. Real hidden-input constant
names on the running install. 11. Where loudness normalization lives in
the local lane today. 12. ImageGenDispatcher resolves via image registry
(R3 first job).
