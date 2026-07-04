# R1 Judgment — model-selection round (GPT-5.5 + Gemini 3.1 Pro + DeepSeek v4-pro)

Panel spend R1: ~$0.122 (0.045 + 0.077). Claude anchor + judge.

## ACCEPTED (grounded CONFIRMED against real files)
- **A-1 ELEVENLABS_VOICE typed-input flow (all 3 models).** TTS node requires a
  typed `voice: ELEVENLABS_VOICE`; the only producer is the AUX
  `cloud_elevenlabs_voice_selector` (COMBO -> ELEVENLABS_VOICE). A curated
  voice_id string must be turned INTO that typed input. Design the resolver
  path explicitly. CONFIRMED in partner_nodes.yaml.
- **A-2 voice-bank schema blocks the naive ref_path plan (GPT#3/DeepSeek#3).**
  voice_bank_entry_schema.json REQUIRES ref_path (minLength 1, VALIDATE_INPUTS
  checks local-disk presence) AND ref_sha256 (minLength 1). A voice_id in
  ref_path fails the disk check. BUT additionalProperties:true -> add a
  `provider_voice_id` field + make the disk/sha check engine-conditional
  (skip for runtime=cloud). CONFIRMED via schema read.
- **A-3 fail-loud needs a NEW preflight gate (GPT#4) — highest-value catch.**
  cast_lock.py:386 `_resolve_character_voices_fail_soft` "NEVER raises", repairs
  missing voices (PD1 audio-is-king), and leaves orphans "for the node-81
  engine fallback" (:510), UNCONDITIONALLY (:186). This CONTRADICTS the cloud
  "missing key/voice/quota = LOUD stop / no fallback" directive. The build MUST
  add a cloud-specific admission gate AFTER CastLock, BEFORE invoke, that fails
  hard on: missing ElevenLabs auth/quota, and any cloud-selected char/announcer
  line with no resolvable provider voice (no silent node-81 local fallback).
  CONFIRMED via code.
- **A-4 `cloud` runtime undefined (GPT#5/DeepSeek#2).** audio_engine_profiles.yaml
  only has runtime in_graph|oop_venv. Define `runtime: cloud` + the profile
  fields that map a profile to a partner row (partner_row, provider_id, required
  param defaults, auth, billing category, canonicalizer, error policy, valid
  roles) + dispatch routing to invoke_partner_node. CONFIRMED.
- **A-5 combo_options_excluded (GPT#2).** partner_nodes.yaml pin_meta
  `combo_options_excluded: true`; `model` is COMFY_DYNAMICCOMBO_V3 -> real tier
  options hidden. V3 expansion/pin of the ElevenLabs `model` (and the required
  output_format / apply_text_normalization / language_code defaults) is a build
  PREREQUISITE. Do not name a specific tier until expanded. CONFIRMED.
- **A-6 music role is singular (GPT#6).** audio_engine_profiles.yaml uses
  `role: music`, `meta.music_engine` singular; the scoping doc's
  music_open/close/inter are SPEAKER/cue roles, not engine-profile roles.
  Decide: one `music` engine profile mapped to all cue types, stamp per-episode
  `meta.music_engine` (+ optionally per-cue seed/duration). CONFIRMED.
- **A-7 reframe determinism (GPT#8).** seed_supported proves a seed socket, not
  provider byte-identical output over time. Contract = deterministic REQUEST
  construction + durable logging (row, model string, voice_id, seed, duration,
  text/prompt hash); keep test_audio_byte_identical scoped to local/mock paths.
- **A-8 drop "flash = cheap soak" (GPT#7).** Price is flat across tiers; one
  `elevenlabs` engine, tier via `model` param. flash vs premium = quality only.

## MODEL-SELECTION RESOLUTIONS (the point of R1)
- ONE `elevenlabs` cloud engine; tier via `model` default_param (post V3 expand).
- Library voices only for v1; InstantVoiceClone deferred (all 3 agree cut).
- Pinned signature announcer voice v1; seed-shuffle deferred (GPT/anchor).
- Music v1 default = `cloud_sonilo_music` (cheaper $0.15/60s, native duration,
  "BEST"); `cloud_stability_audio` = documented next candidate, not fully wired.
- Local stable_audio_3 stays the engine default until operator promotes cloud.

## VERIFY-AT-BUILD (UNVERIFIABLE now)
- Whether the ElevenLabs `model` DYNAMICCOMBO expands to real quality tiers vs
  voice-model variants (live V3 expansion).
- Whether cloud_elevenlabs_voice_selector accepts stable voice IDs or only
  mutable display labels (node internals on the live install).
- Sonilo short-duration (<10s) fidelity / clamp behavior (first live cue).
- ToS/license of specific ElevenLabs premade voices (operator review).

## REJECTED / DEFERRED
- Pre-render cost-estimate path (GPT SF#5): good idea, but it's a build feature,
  not a model-selection decision -> fold as an optional S-late deliverable.
- "Move test_audio_byte_identical to wiring round" (Gemini/DeepSeek): accepted
  as sequencing, not a cut of the invariant.

## Convergence
R1 CONVERGED on model selection. The coding-contract gaps it surfaced (schema
variant, cloud runtime, preflight gate, ELEVENLABS_VOICE flow, music role) are
handed to the kibitz local-repo panel (codex + Claude, brief agy) for the
coding/wiring hardening, per the operator's stated sequence for this session.
