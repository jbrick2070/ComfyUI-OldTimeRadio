# OTR Cloud Engine Lanes -- voice / stills / video (3D appendix) -- pass01

R1-synthesized. DOCS ONLY campaign; build gated on operator go + free coder
baton. Operator brief: per modality 1 CHEAPEST-WORKABLE + 2 BEST-OF-CLASS
cloud rows; video lane must serve the audio-reactive pipeline.

## 0. Goal

Cloud provider lane for every generative modality so a full episode renders
with ZERO local GPU: local ComfyUI orchestrates on CPU; heavy generation
executes remotely on the Comfy account (partner API nodes / Comfy Cloud).
Local byte-identical defaults untouched; cloud is opt-in, fail-closed,
cost-guarded. Non-goals: running OTR custom nodes on Comfy Cloud; replacing
local engines; reintroducing the deleted sfx role.

## 1. Two provider surfaces (explicit -- do not conflate)

- SURFACE A `comfy_credits_partner_node` (PRIMARY): the 214 hosted partner
  API nodes present on the running install (Kling, ElevenLabs, Stability,
  Sonilo, Recraft, BFL, Seedance, Wan-API...). Executed by INVOKING the
  bundled `comfy_api_nodes` node classes IN-PROCESS -- they own their
  endpoints, job polling, retries. We do NOT hand-roll per-provider HTTP.
  Billing: Comfy account credits. Auth: ComfyUI hidden inputs
  (`auth_token_comfy_org` / `api_key_comfy_org`) -- see auth broker (S0).
- SURFACE B `comfy_cloud_workflow` (QUARANTINED): submitting open-source
  template workflows (ACE-Step music, Chatterbox TTS) to Comfy Cloud GPUs.
  Cheapest raw compute, but headless API-key auth + submit/poll/download +
  cancellation are UNPROVEN for unattended batch. NO row on this surface is
  load-bearing until S0 smoke passes; every B row has a named A fallback.
  If B fails S0, B rows drop to a research flag and the plan stands.

Verified constraints (2026-07-02 grounding):
- `_otr_comfy_backend.py` proves surface-A auth capture + fail-closed
  pattern for CHAT only; media generality is a build-time verification.
- Audio registry `assert_usable` has NO flag-gate: "the registry IS the
  menu" (registry.py line ~151; the class docstring above it is stale).
  Video/image registries share `engine_registry_base` + role_compat
  capability tokens (`text_prompt|init_image|audio_ref|base_clip_ref`);
  audio is a parallel FROZEN implementation, not the shared base.
- COMBO widgets serialize the SELECTED STRING: adding dropdown options
  causes no widgets_values positional drift. New WIDGETS would; none are
  added before S4.

## 2. Gating, auth, money (the control plane)

- IMPORT-GATED REGISTRATION: cloud adapters register only when
  `OTR_ENABLE_COMFY_CLOUD_MEDIA=1` (default OFF). Flag off = rows do not
  exist anywhere (dropdowns, resolver, tests). This respects "registry IS
  the menu" instead of trusting `requires_flag` the audio registry never
  enforces. Per-row experiment flags gate registration the same way.
- AUTH BROKER (S0, shared backend): one capture point for the Comfy
  auth hidden inputs per run, shared via session context to all cloud
  adapters. Cloud-capable dispatch nodes (voice 3a/3b, music 3c,
  ImageGenDispatcher, VideoRenderBatch) DECLARE the hidden inputs at
  build time -- acknowledged node surgery, one-time, additive-only.
  S0 smoke #1: hidden inputs populate on a HEADLESS server (the whole
  no-GPU story rides on this; if they do not, fall back to
  `api_key_comfy_org` from server config, verify before S1).
- BUDGET + POLICY MATRIX (decided, not open):
  - Pre-submit budget check fails -> hard abort the leg, LOUD named error,
    ledger `BUDGET_ABORT`. No silent degradation.
  - Provider failure pre-asset -> retry x2 w/ backoff, then pre-declared
    fallback chain (cloud row -> named cheaper cloud row -> local row if
    profile permits -> abort). EVERY hop restamps the ledger LOUDLY.
  - Partial/corrupt asset -> discard, one re-submit, then chain as above.
  - Budget env `OTR_CLOUD_MEDIA_BUDGET_USD`, reset per run (BUG-296).
- IDEMPOTENT BILLING CACHE: cloud asset cache keyed by content hash
  (row id + prompt + params + audio-slice hash) resolving into
  `otr\episodes\<ep>\`; transport checks cache before submit; ledger marks
  CACHED vs BILLED. Re-runs never re-bill completed assets.
- Pre-run COST ESTIMATE printed per episode (rows x beat counts) before
  first submit; post-run ledger totals actuals.
- Rate limiting: per-provider concurrency knob + serialization in the
  shared backend (defaults conservative); cancellation: local abort
  cancels in-flight cloud jobs where the node class supports it, else
  logs ORPHANED_JOB with job id.

## 3. Media canonicalization contract (S0 deliverable, per modality)

Remote outputs are normalized BEFORE entering the episode tree:
- voice/music: WAV, 44.1kHz, channel policy per stereo_policy widget,
  loudness normalized to the lane's LUFS target, duration trim/pad rules
  per line; per-line granularity PRESERVED (captions/delivery vectors
  assume line-granular audio).
- stills: exact role canvas (e.g. 1472x832 landscape), sRGB, PNG;
  portrait-hash + in-character invariants re-verified on cloud output.
- video: role fps/resolution/container per workflow config; ANY embedded
  provider audio STRIPPED at canonicalization (`must_strip_audio` flag on
  rows whose models generate native audio) -- master audio is frozen
  upstream, mux is LAST, unconditionally.
- all: content hash, provider/job id/cost metadata into ledger; LOUD
  restamp on any fallback.

## 4. Curated rows (CHEAP = candidate until S0 prices + smokes it)

Every row: surface tag, `commercial_clean` ToS audit REQUIRED in S0 before
promotion (release gate enforces NONCOMMERCIAL_BLOCKED), approx_cost
stamped from the Comfy partner pricing table (S0).

### 4a. VOICE (char_voice, announcer_voice) -- per-line granularity only
| Tier | Row | Surface | Backing |
|------|-----|---------|---------|
| CHEAP-candidate | `cloud: elevenlabs_flash` | A | ElevenLabs cheap/flash tier via `ElevenLabsTextToSpeech` |
| BEST 1 | `cloud: elevenlabs_tts` | A | `ElevenLabsTextToSpeech` premium voices |
| BEST 2 (quarantined) | `cloud: chatterbox_cc` | B | Comfy Cloud `audio-chatterbox_tts` -- voice continuity w/ local sidecar; fallback = elevenlabs_flash |

CastLock keeps assigning presets; cloud adapter maps preset -> curated
stock-voice table (audited). Voice CLONING (`ElevenLabsInstantVoiceClone`
from the CC0 bank) DEFERRED post-S2 (ToS audit + per-voice cost + new
persistent-identity subsystem). `ElevenLabsTextToDialogue` demoted to
experiment flag: whole-conversation blobs break per-line captions/ledger.

### 4b. MUSIC (theme_music)
| Tier | Row | Surface | Backing |
|------|-----|---------|---------|
| CHEAP-candidate (quarantined) | `cloud: ace_step_1_5` | B | ACE-Step 1.5 templates; fallback = stability_audio |
| BEST 1 | `cloud: sonilo_music` | A | `SoniloTextToMusic` |
| BEST 2 | `cloud: stability_audio` | A | `StabilityTextToAudio` (style continuity w/ local SA3 is a HYPOTHESIS -- listening test at S2) |

### 4c. STILLS (announcer/music/other_beats image)
| Tier | Row | Surface | Backing |
|------|-----|---------|---------|
| CHEAP-candidate | `cloud: recraft` | A | Recraft image nodes |
| BEST 1 | `cloud: flux_pro` | A | BFL Flux pro tier (prompt continuity w/ local flux_gen1) |
| BEST 2 | `cloud: nano_banana_2` | A | Gemini/Nano Banana 2 (reference-image edit -> character consistency) |

### 4d. VIDEO -- per-role REACTIVITY MATRIX (replaces blanket requirement)
| Role / beat class | Reactivity requirement | Default row | Alt rows |
|-------------------|------------------------|-------------|----------|
| announcer_video / talking beats | REQUIRED: `audio_ref` consumed OR `lipsync_overlay` on base clip | `cloud: kling_avatar` (`KlingAvatarNode`; lipsync variant `KlingLipSyncAudioToVideoNode`) | `cloud: seedance_2` (`ByteDance2ReferenceNode`, audio-ref + identity preservation) |
| music_video beats | OPTIONAL (classified non-reactive; OTR music replaces any track) | `cloud: wan_i2v` mute | kling std, luma_ray |
| other_beats b-roll | OPTIONAL, mute I2V allowed | `cloud: wan_i2v` (`Wan2ImageToVideoApi`) CHEAP-candidate | seedance_2 |

VERIFY-AT-BUILD (S0 smoke #2): one Kling audio-driven clip end-to-end
(upload audio slice -> avatar/lipsync -> download) -- proves the hard
requirement BEFORE stills consume build time. Wan "optional audio in" must
be shown to CONDITION generation before wan_i2v may ever claim reactive;
until then it is mute-only. Native-audio models (Veo/`GeminiVideoOmni`,
`OpenAIVideoSora2`) enter only as mute-I2V rows with `must_strip_audio`.

## 5. Adapters (thin, per existing pattern)

One adapter file per row in the matching registry namespace; honest
`required_inputs` + family; zero local VRAM; import-gated (sec 2). The
shared backend (`nodes/_otr_shared/cloud_media_backend.py`) provides: auth
broker, budget guard + policy matrix, billing cache, cost ledger, rate
limits, canonicalization, and a thin invoke-wrapper around bundled partner
node classes (surface A) / cloud workflow client (surface B, post-smoke).
S0 pins per-node schemas from `/object_info` (inputs, outputs, upload
semantics, job lifecycle, error taxonomy) -- adapters are validated
against pinned schemas, not guessed from class names.

## 6. Cloud capability profile (S4, gated)

New profile `cloud` beside 16gb/8gb/cpu_floor. Mechanism (answers the
enable-set-vs-default gap): the profile applier gains a per-role
DEFAULT-OVERRIDE map (role -> row id) consumed by
`default_engine_for_role`; enable-set continues to gate usability. The
shipped workflow JSON keeps LOCAL defaults; a `cloud` run overrides at
apply time (headless `--profile cloud`), so the byte-identical local
baseline never moves. Only S4 may touch workflow JSON (same-change rule +
validator + widget audit) if any default widget changes are wanted for a
cloud-first operator, and that change is operator-gated.
Acceptance: full episode on `CUDA_VISIBLE_DEVICES=''` host, assets in
`otr\episodes\<ep>\`, final in `otr\obs\`, cost report printed.

## 7. Sprints

- S0 CONTROL PLANE: shared backend (auth broker, budget+policy, cache,
  ledger, rate limits, canonicalization skeleton); pricing table pulled +
  approx_cost stamped; ToS audit per row; schema pinning via /object_info;
  smoke #1 headless auth; smoke #2 Kling audio-driven clip; transport-B
  feasibility verdict (promote or quarantine-to-research). All cloud tests
  mocked in the no-network suite.
- S1 STILLS lane (lowest risk). Acceptance: 3-beat image set on-model.
- S2 VOICE + MUSIC (no cloning). Acceptance: full audio episode zero local
  GPU; STRUCTURAL bar (per-line duration tolerance, loudness lint,
  pre-mux master == muxed track), NOT byte-stability (cloud is
  nondeterministic; byte-identical applies to the local baseline only).
- S3 VIDEO matrix incl. reactive paths. Acceptance: talking-radio beat
  driven by episode audio; mux-LAST intact; b-roll via wan_i2v.
- S4 `cloud` profile + no-GPU end-to-end + per-episode cost report.

## 8. Appendix: 3D (docs-only, no sprint)

Operator said "maybe 3d". Recorded candidates for a future build: Tripo P1
(cheap, game-ready), Rodin Gen2.5 (quality modes), Meshy (7 nodes).
Cloud removes the local cu128/ninja blocker that parked 3D, but nothing
downstream consumes meshes; no registry tokens, no adapters, no sprint.

## 9. Verify-at-build register

1. Headless hidden-input auth (S0 smoke #1). 2. Kling audio conditioning
(S0 smoke #2). 3. Transport B headless lifecycle (submit/poll/download/
cancel/price). 4. Partner pricing table -> approx_cost per row. 5. ToS /
commercial_clean per provider. 6. Wan audio conditioning (else mute-only).
7. SA3-vs-Stability style continuity listening test. 8. Credit balance is
one pool across chat + media partner nodes. 9. Cloud job long-poll vs
watchdog heartbeats (transport must heartbeat while polling).
