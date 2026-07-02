# OTR Cloud Engine Lanes -- FINAL (pass04, R4-converged)

DOCS ONLY campaign output; build gated on operator go + free coder baton.
GOAL (amended, explicit): per modality 1 CHEAP + 1-2 BEST cloud rows
(voice + music ship 1+1 on Surface A today -- ElevenLabs is the only
partner TTS provider and Sonilo/Stability the music pair; third rows
return via Appendix B when the Comfy-Cloud-workflow surface is proven).
AUDIO-REACTIVE VIDEO (OPERATOR AMENDMENT 2026-07-02, post-convergence):
audio reactivity is DEFAULT-ON for EVERY video role -- talking/announcer,
music_visual, AND b-roll all default to audio-reactive rows. Nothing
reactive sits behind an off switch. `mute_only` rows (wan_i2v) remain
REGISTERED as an explicit per-role OPT-DOWN the operator selects
deliberately (cost control), never a silent default. Zero-local-GPU
episodes via `--profile cloud`.

## 1. Rows ship from a CHECKED-IN pin, never live imports

Per pass03 sec 1, plus R4 tightening:
- `partner_nodes.yaml` per-row fields (concrete): `node_key
  {import_path, class_name}`, `provider_id` (normalized `[A-Z0-9_]+`;
  S0 fails on collisions), hidden-input constant names, `seed_supported`,
  `execution_mode: sync|job` + `job_id_field` + `poll_status_field` +
  `terminal_states`, expected RETURN_TYPES + SELECTED OUTPUT index/name
  + expected media shape (multi-output partner nodes are ambiguous
  otherwise) + an S0 fixture assertion proving the selected output
  canonicalizes; pricing row + pricing_source_version; ComfyUI
  version/commit the pin was generated against. An example yaml row and
  example ledger lines (cache hit, provider-rejected release, timeout
  orphan, fallback hop, schema drift) ship as S0 doc deliverables.
- Runtime never drops rows; missing/drifted class = registered row that
  fails CLOSED at resolve/invoke. CANONICAL ERROR CODES (one spelling
  everywhere -- resolver, invoke, ledger, tests): `malformed_config`
  (unregistered/bad saved engine id), `unsupported_schema` (pinned class
  missing or schema/return drift), `incompatible_profile`,
  `gated_by_flag`, `auth`, `budget`, `retryable_transport`,
  `provider_rejected`, `timeout`, `interrupted`, `corrupt_output`,
  `orphaned_job`.
- CAPABILITIES: every row gets an entry in ITS modality registry keyed
  to exact engine.name -- `cpu_ok: True`, `vram_estimate_mb: 0`,
  `required_toolchain: None`, `requires_sidecar: False`; the exact
  `vram_class` label for cloud rows is chosen at build against what
  capability_profiles.py consumers accept (verify #13) -- the hard
  requirements are the zero-VRAM/cpu-ok semantics. Per-namespace
  consistency tests: registered cloud ids == CAPABILITIES keys, no
  orphans.

## 2. Session, auth, money

Per pass03 sec 2, plus R4 tightening:
- SESSION TABLE: the backend session table is the ONE allowed
  lock-guarded process singleton ("no module globals" means no
  credential/budget state in adapter modules). Key = prompt_id ALONE;
  episode_id is attached metadata once parsed (never part of the key;
  no collision ambiguity). Teardown: assembler done or prompt
  completion; ABORTED/crashed runs are covered by a sweep at next
  session-create that evicts entries older than N hours and logs
  LEAKED_SESSION with any unreleased reservations.
- AUTH BROKER precedence (concrete, vague "server config" dropped):
  `OTR_COMFY_API_KEY` env > hidden `api_key_comfy_org` > hidden
  `auth_token_comfy_org`. Cloud-capable media nodes MUST declare BOTH
  auth hidden inputs (exact constants, same as the writer node) PLUS
  the PROMPT/UNIQUE_ID hidden inputs -- the frontend only injects what
  is declared. Headless runners inject into /prompt (S0 smoke #1).
- BUDGET: SEPARATE USD accumulator (`OTR_CLOUD_MEDIA_BUDGET_USD`),
  fully independent of the chat lane's token ceiling -- no currency
  unification. The SESSION owns the state machine and exposes
  `reserve(estimate_usd)->reservation_id / submit(rid, job_id) /
  bill(rid, actual_usd) / release(rid) / abort(rid)`; adapters only
  call these. States: RESERVED -> SUBMITTED -> BILLED_ESTIMATE |
  BILLED_ACTUAL, RELEASED, ABORTED. Incremental reserve per dispatch
  (post-cache-lookup, pre-submit); episode estimate = report not gate.
- INTERRUPT SEMANTICS (cancellation stays cut): interrupt before submit
  -> ABORTED/RELEASED, no reservation remains; after submit -> stop
  local wait, emit ORPHANED_JOB {provider, job_id, submitted_at,
  estimated_usd}, release the provider semaphore, classify
  `interrupted`.
- Rate limits, fallback (single system extending _otr_shared/
  fallback.py, license + enable-set + cpu_ok re-checked per hop),
  ordering on the existing gate/done chain + slice-manifest input:
  unchanged from pass03.

## 3. Invocation contract

`invoke_partner_node(node_key, inputs, *, timeout_s) -> PartnerResult`.
SESSION IS RESOLVED INTERNALLY from the backend table via the current
prompt context -- it is NOT a parameter (adapters never receive or pass
a session object; the frozen AudioEngine protocol stays untouched).
`PartnerResult` (TypedDict): `{path: str, content_type: str,
duration_s: float | None, provider_job_id: str | None, raw_meta: dict}`
-- downloads stream to a temp path (never whole-media-in-memory); the
canonicalizer consumes exactly this shape. Async bridge, in-full-
ComfyUI execution-context verification, heartbeat-as-S0-promotion-
requirement, normalized error codes (sec 1 list): per pass03 sec 3.

## 4. Cache + ledger

- SPLIT CONCEPTS (R4): `RequestCacheKey` = deterministic PRE-SUBMIT key
  from row id + resolved slug + normalized request params + seed (only
  when seed_supported) + INPUT-asset content hashes + output-contract /
  adapter / canonicalizer / schema versions. `content_sha256` =
  POST-canonicalization output hash recorded in the manifest + ledger
  (integrity/proof), NEVER part of the lookup key.
- Manifest carries must_strip_audio proof + canonicalizer version;
  stale entries re-canonicalize before reuse. Per-key lock +
  double-checked validation; single ledger writer. Global cache dir via
  the same output-base path helper; COPIED into episodes; excluded from
  obs_publish by an explicit path allowlist IN obs_publish (named
  mechanism, not a comment).
- Billing JSONL and production ledger linked by request_id (financial
  vs artistic truth), per pass03.

## 5. Reactivity policy (simplified) + descriptor table

`reactivity` values: `required_audio_ref | lipsync_overlay | mute_only`
(`optional_audio_ref` CUT -- no shipped row used it; policy stays
crisp). DEFAULT POLICY (operator amendment): ALL video roles require a
reactive engine (`required_audio_ref` or `lipsync_overlay`) unless the
role carries an EXPLICIT opt-down (`OTR_VIDEO_MUTE_OK_ROLES` env /
profile field listing role names, default EMPTY). Gate LOCATION
(concrete): `OTR_ShotLock.validate()` immediately after resolving the
engine pick, raising `EngineUnusable(reason=INCOMPATIBLE_PROFILE)` with
a message naming the role's required reactivity AND the opt-down knob
-- BEFORE any reservation (money) and before dispatch. A `mute_only`
engine resolves ONLY for roles named in the opt-down list; the ledger
stamps `MUTE_OPT_DOWN <role>` so the choice is auditable. Matrix tests
cover all grounded video roles (announcer_visual, music_visual,
character_video) x all shipped rows x opt-down empty/populated via the
EXTENDED descriptors (`descriptor_for_engine` + the role_compat
EngineDescriptor TypedDict + VideoEngine Protocol gain `reactivity` +
`must_strip_audio` -- explicit protocol/TypedDict updates, tested).

VIDEO ROW DESCRIPTOR TABLE (nothing left for adapters to infer):
| row | reactivity | must_strip_audio | required_inputs | fallback chain |
|-----|-----------|------------------|-----------------|----------------|
| kling_avatar (KlingAvatarNode) | required_audio_ref | True | (init_image, audio_ref) | -> seedance_2 -> local talking chain (fallback.py) -> abort |
| kling_lipsync (KlingLipSyncAudioToVideoNode) | lipsync_overlay | True | (base_clip_ref, audio_ref) | -> kling_avatar -> local -> abort |
| seedance_2 (ByteDance2ReferenceNode) | required_audio_ref | True | (init_image, audio_ref) | -> kling_avatar -> local -> abort |
| wan_i2v (Wan2ImageToVideoApi) | mute_only | True | (init_image, text_prompt) | -> local still/parallax -> abort |

ALL provider audio is stripped unconditionally (master audio frozen,
mux LAST); must_strip_audio=True across the board. ffmpeg availability
is checked in the ADAPTER's render-lifecycle `assert_usable(host_caps,
profile, ...)` / `prepare()` -- NOT the registry's assert_usable, which
does no IO (registry invariant; R4 correction).

## 6. Canonicalization

Per pass03 sec 6, plus: `actual_duration_s` validators fail with a
named missing-field error on CLOUD runs; legacy local artifacts remain
supported unmigrated. PartnerResult (sec 3) is the canonicalizer input
shape.

## 7. Rows (final)

VOICE: `cloud$ elevenlabs_flash` CHEAP-cand | `cloud$ elevenlabs_tts`
BEST. MUSIC: `cloud$ stability_audio` CHEAP-cand | `cloud$ sonilo_music`
BEST. STILLS: `cloud$ recraft` CHEAP-cand | `cloud$ flux_pro` BEST |
`cloud$ nano_banana_2` BEST (present in the live template catalog;
S0 pin gates it like every row). VIDEO: per sec 5 table + role
defaults (ALL REACTIVE, operator amendment): talking/announcer ->
kling_avatar; music_visual -> seedance_2 (audio-ref, driven by the
theme/beat audio slice); b-roll/character beats -> seedance_2.
wan_i2v ships as the mute OPT-DOWN row only (sec 5 knob). CHEAP-slot
consequence: the cheapest REACTIVE row is what the cheap tier means
for video now -- wan_i2v claims it ONLY if S0 proves Wan audio
CONDITIONING (verify #5); otherwise the cheapest reactive row is
whichever of seedance_2 / kling std-tier prices lower in the S0
pricing table, and wan_i2v remains opt-down-only.
Voice-preset table, licensing tri-state (pending=blocked), per-line
granularity, ToS audit early-S0: per pass02/03. An operator-facing
summary table (row, tier, surface, approx_cost, license status) is
GENERATED from registry+yaml into docs at S0 end (tables cannot drift
from shipped truth).

## 8. Sprints + tests

Per pass03 sec 8, plus R4 tightening:
- Every S0 acceptance item is labeled `no-network | fixture-only |
  OTR_RUN_CLOUD_SMOKE=1`; ordinary CI can never require credits.
- BOTH flag states tested with process isolation (env toggling leaks;
  use subprocess or explicit reload): flag OFF -> all shipped rows
  REGISTERED (COMBO validity) and resolver raises gated_by_flag; flag
  ON -> rows resolve to yaml-backed descriptors or fail closed on
  drift.
- S0 smoke #2 names its row id (kling_avatar) + the checked-in WAV
  fixture, and must prove the audio CONDITIONS the clip (not merely
  uploads alongside it).
- NEW concrete verify steps: credit-pool check = one chat call + one
  media call on the same account, confirm one pool debits (else
  document separate pools and update wording); api-base check = set
  the override pre-smoke and prove the media node honors it (else
  document media as install-base-only).
- SA3-vs-Stability listening test MOVED to optional QA (subjective; the
  build gates are the structural audio acceptance + cached re-run).
- Session-leak test (LEAKED_SESSION sweep) in the offline suite.
- Repo process (unchanged, operator directive): each lane sprint ends
  with regression suite + Bug Bible + commit AND push to v2.0-alpha.

## 9. Verify-at-build register (final)

1. Headless auth injection via /prompt (S0#1, no node surgery).
2. Kling audio CONDITIONING proof (S0#2, fixture WAV, row kling_avatar).
3. Versioned pricing table; S0 fails on missing row pricing.
4. ToS -> license_audit_status per provider (early S0; pending=blocked).
5. Wan audio conditioning (else mute_only stands). 6. [moved to
optional QA] SA3-vs-Stability. 7. Credit pool: concrete S0 check (sec
8). 8. ffmpeg in local/headless/cloud-profile runs (adapter-level gate).
9. Exact executor progress/interrupt API for <=30s heartbeats (pre-
smoke). 10. Hidden-input constants in yaml + drift CI. 11. Existing
loudness constant location; cloud canonicalizer reuses it. 12.
comfy_api_base: concrete override test or document install-base-only.
13. Exact vram_class label for cloud CAPABILITIES rows accepted by
capability_profiles consumers.

## Appendices

A: 3D docs-only (Tripo P1 / Rodin Gen2.5 / Meshy candidates; no
sprint). B: Surface B (Comfy Cloud workflow) research flag
(`OTR_ENABLE_COMFY_CLOUD_WORKFLOWS=1` + recorded headless lifecycle
smoke artifact required before any row registers; restores ace_step +
chatterbox voice-continuity rows -- the third-row-per-lane path).
