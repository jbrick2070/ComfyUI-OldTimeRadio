# OTR Cloud Engine Lanes -- pass03 (R3-synthesized: wiring locked)

DOCS ONLY campaign; build gated on operator go + free coder baton.
Brief: 1 CHEAP + 2 BEST cloud rows per modality; audio-reactive video;
zero-local-GPU episodes. Goal/scope per pass02 sec 0-1 (Surface A only;
Surface B + 3D in appendices).

## 1. Rows ship from a CHECKED-IN pin, never live imports

- S0 produces `partner_nodes.yaml` by importing candidate partner classes
  in-process on the live install and reading INPUT_TYPES()/RETURN_TYPES/
  FUNCTION (+ hidden-input constant names, seed_supported, sync-vs-job).
  The yaml is CHECKED IN. Only rows whose class pinned successfully SHIP.
  (Sonilo re-verifies here like every row; the install dump says it
  exists, the pin proves it -- absent = row dropped LOUDLY at build time.)
- RUNTIME NEVER DROPS ROWS. Shipped rows register unconditionally
  (registry IS the menu; saved COMBO values stay valid; CAPABILITIES
  invariant keyed to exact engine.name in EACH modality's own registry
  module, cross-registry consistency test per namespace). On a target
  install where a class is missing/drifted, the row stays registered and
  fails CLOSED at resolve/invoke with a named `unsupported_schema` /
  MALFORMED_CONFIG error. Schema drift fails LOUDLY -- no best-effort
  remapping (silent-swap class of bug; fail-closed is the invariant).
- Dropdown construction reads registry + yaml only -- never imports
  partner classes. No-network CI uses the checked-in yaml + mock
  fixtures.
- Flag gate unchanged (pass02): resolver raises GATED_BY_FLAG when
  `OTR_ENABLE_COMFY_CLOUD_MEDIA` != 1.

## 2. Session: backend-owned keyed table (no protocol changes)

- `CloudMediaSession` lives in a lock-guarded backend table keyed by
  prompt_id (+ episode_id once parsed). Cloud-capable NODES obtain
  prompt_id via additive hidden inputs (PROMPT/UNIQUE_ID pattern already
  used by otr_save_to_episode_workspace; MERGE into existing hidden
  dicts, never replace). ADAPTERS -- including audio, whose FROZEN
  AudioEngine protocol signatures (generate_voice/generate_clip) cannot
  take a session arg -- FETCH the session from the table instead of
  receiving it. No AudioEngine signature change, no module globals.
- Lifecycle: lazy-create at first cloud call; teardown on assembler done
  signal or prompt completion; leak check in tests.
- AUTH BROKER precedence: `OTR_COMFY_API_KEY` env > server config >
  logged-in hidden-input token -- resolved INTO the session by the
  broker (the chat lane's `set_auth()` module globals are not reused;
  its `_bearer()` reads no env). Headless runners inject tokens into the
  /prompt payload (S0 smoke #1 proves it, no node surgery needed).
- BUDGET: ceiling re-read PER RUN via the existing `_int_env` pattern
  (hot-adjust between runs; never mutated mid-run). Spent accumulator
  guarded by a threading.Lock. State machine per request: RESERVED ->
  SUBMITTED -> BILLED_ESTIMATE | BILLED_ACTUAL, with RELEASED (no-submit
  / provider-rejected) and ABORTED; orphaned reservations only for
  submitted-unknown-billing. Reservation happens INCREMENTALLY per
  dispatch: after cache lookup, immediately before submit (the single
  mid-run episode gate was impossible -- video cost/keys depend on
  generated audio). The per-episode ESTIMATE (rows x parsed request
  list) prints after script/ledger parse as a REPORT, not a gate.
- ORDERING: cloud lanes ride the EXISTING gate/done chain (audio legs
  complete -> audio_done -> slice manifest -> video legs consume the
  manifest as an INPUT, never filesystem timing). Budget preflight nodes
  are not added; the existing serialization is the ordering edge.
- RATE LIMITS: provider semaphore held from submit through terminal
  state per attempt (released for backoff, reacquired on retry); IDs
  normalized `[A-Z0-9_]+` (`OTR_CLOUD_MAX_CONCURRENCY_<ID>`); resolved
  env-var name echoed in ledger/debug. Submit-rate-only providers
  declare that separately in the yaml.
- FALLBACK: pre-declared per-row chains EXTEND the shipped
  `_otr_shared/fallback.py` machinery (humo->latentsync->still_kenburns
  precedent) -- ONE fallback system. Every hop re-checks: profile
  enable-set, CAPABILITIES cpu_ok, license/commercial release gate, and
  budget. LOUD ledger restamp per hop. No-GPU + `--profile cloud` =
  cloud-only chain then abort.
- CANCELLATION stays cut: timeout + `ORPHANED_JOB {provider, job_id,
  submitted_at, estimated_usd}`; cancel_token REMOVED from the invoke
  signature (v1).

## 3. Invocation contract (updated)

`invoke_partner_node(node_key, inputs, session, *, timeout_s) ->
PartnerResult` where node_key = `(import_path, class_name)` from the
yaml (class_name alone is under-keyed). Async bridge: backend-owned
event-loop thread; adapters block with timeout. Isolation reality: we
ALWAYS run inside a full ComfyUI process -- S0 verifies each candidate
node's FUNCTION executes correctly when invoked from another node's
execute context (never from a bare script; comfy.* globals exist).
WATCHDOG: heartbeat implementation is an S0 PROMOTION REQUIREMENT --
poll loop checks the interrupt flag and emits progress via the executor
progress API every tick (<=30s); the concrete API name is verify item
#9 and must be resolved BEFORE live smokes (5-min stall detector
otherwise kills long Kling jobs). Errors normalize to {auth, budget,
retryable_transport, provider_rejected, timeout, corrupt_output,
unsupported_schema}. Streaming download + streaming SHA-256 only.

## 4. Cache + ledger (updated)

Global `otr\cache\cloud_media\` (same output-base resolution helper as
episodes; .gitignore; excluded from obs_publish). Canonical assets are
COPIED into `otr\episodes\<ep>\` (hardlink optimization cut -- Windows
edge cases). CloudAssetKey as pass02 PLUS: sha256 computed AFTER
canonicalization mutations (a stripped video hashes as its delivered
bytes); `seed` participates in the key only when the pinned schema says
seed_supported (else recorded as request metadata); adapter/
canonicalizer versions are simple integers bumped on output-contract
change. Concurrency: per-cache-key lock + double-checked validation;
single writer lock/queue for the billing JSONL. Cache manifest carries
must_strip_audio proof + canonicalizer version; stale entries (manifest
lacks proof) re-canonicalize before reuse. Billing JSONL and the
production ledger stay SEPARATE with request_id linking both (financial
vs artistic sources of truth; no dual-write of one fact).

## 5. Reactivity is a descriptor field + a ShotLock policy gate

Capability fit alone CANNOT enforce the matrix: surviving video roles
supply every token (audio_ref included), so a text/image-only mute
engine capability-fits announcer beats. Fix shipped in this plan:
`reactivity` (`required_audio_ref|lipsync_overlay|mute_only|
optional_audio_ref`) + `must_strip_audio` become REGISTERED descriptor
fields (descriptor_for_engine extended), and OTR_ShotLock adds a
MANDATORY policy check: talking/announcer beats reject engines whose
reactivity is not required_audio_ref or lipsync_overlay -- named
queue-time error BEFORE money is spent. Matrix tests via the extended
descriptors. ffmpeg availability is checked in cloud video
assert_usable (pay-then-crash prevention; canonicalizer strip must
never discover a missing ffmpeg after credits are spent).

## 6. Canonicalization (additions)

Loudness: ONE source-of-truth constant (defined S0, consumed by local
and cloud paths) -- match the existing lane's reference (verify item
#11), do not invent a new LUFS convention. Line metadata field is NAMED
at build: `actual_duration_s` on each script line; caption + delivery-
vector validators updated to consume it. Everything else per pass02
sec 6.

## 7. Rows (unchanged picks; all class refs = yaml candidates)

VOICE: `cloud$ elevenlabs_flash` CHEAP-cand | `cloud$ elevenlabs_tts`
BEST. MUSIC: `cloud$ stability_audio` CHEAP-cand | `cloud$ sonilo_music`
BEST. STILLS: `cloud$ recraft` CHEAP-cand | `cloud$ flux_pro` |
`cloud$ nano_banana_2`. VIDEO: `cloud$ kling_avatar` (talking default) |
`cloud$ seedance_2` (alt) | `cloud$ wan_i2v` CHEAP-cand mute_only.
Voice-preset table, licensing tri-state, per-line-only constraints: per
pass02. ToS audit runs EARLY in S0 (all rows default pending = blocked
for commercial runs until it lands).

## 8. Sprints (wiring-scoped)

- S0 CONTROL PLANE (no node surgery): checked-in partner_nodes.yaml;
  session table + auth broker + budget state machine + cache + ledger +
  rate limits + canonicalizer skeleton + loudness constant + heartbeat
  impl; ToS audit early; pricing table versioned; checked-in short WAV
  fixture; SMOKE #1 headless auth injection on one cheap image node;
  SMOKE #2 Kling audio-driven clip using the WAV fixture (independent
  of S2). Live smokes behind OTR_RUN_CLOUD_SMOKE=1 + operator setup
  step (logged-in Comfy OR OTR_COMFY_API_KEY) named in acceptance.
- S1 STILLS: ImageGenDispatcher hidden inputs (MERGE) + image adapters
  + registration import lines + registration test (every shipped row id
  registered when flag on). Dispatcher is registry-driven (CONFIRMED:
  otr_image_gen_dispatcher.py:38 imports _otr_image_engines.registry).
- S2 VOICE + MUSIC: audio node hidden inputs; adapters fetch session
  from the table (frozen protocol untouched); structural acceptance +
  100% CACHED re-run + SA3-vs-Stability listening test.
- S3 VIDEO: descriptor extension + ShotLock reactivity gate + video
  adapters + slice-manifest input wiring; talking-beat acceptance;
  cached-asset strip-proof validation.
- S4 `cloud` PROFILE: DEFAULT-OVERRIDE map beside enable-set derivation
  (one source of truth) + resolve_default_engine_for_role + no-GPU
  end-to-end + cost report; workflow JSON changes ONLY here,
  operator-gated, same-change rule + validator + widget audit.
- Each lane sprint ends: regression suite + Bug Bible + commit AND push.

## 9. Verify-at-build register (rolled forward)

1. Headless auth injection (S0#1). 2. Kling audio conditioning (S0#2).
3. Pricing table version. 4. ToS per provider (early S0). 5. Wan audio
CONDITIONING else mute_only stands. 6. SA3-vs-Stability listening test.
7. One credit pool chat+media. 8. ffmpeg in all run modes (also gated
in assert_usable). 9. Executor progress/interrupt API for heartbeats
(pre-smoke). 10. Hidden-input constants recorded in yaml + drift CI.
11. Existing loudness reference location. 12. comfy_api_base override
honored by media partner nodes (or document install-base-only).

## Appendices: A (3D docs-only), B (Surface B research flag) -- pass02.
