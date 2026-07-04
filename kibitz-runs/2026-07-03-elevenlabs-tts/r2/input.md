# BUILD PLAN — Cloud AUDIO on Comfy Cloud: ElevenLabs TTS + cloud MUSIC

Branch `v2.0-alpha`. Operator directives win. Fail-loud, no fallbacks, no hidden
promotion — the dropdown pick IS the enable. UTF-8 no BOM, SFW, determinism
seed-keyed. Every node/wiring change lands in `workflows/otr_scifi_16gb_full.json`
in the SAME change. Suite + Bug Bible + B7 green + push per green chunk.

Status: model selection CONVERGED via roundtable R1 (GPT-5.5 + Gemini 3.1 Pro +
DeepSeek v4-pro, ~$0.12, Claude anchor+judge; see roundtable/pass01_judgment.md).
This plan folds the grounded, code-CONFIRMED findings into build sprints. Next
gate: kibitz (codex + Claude, brief agy) on coding/wiring, then a Fable final
grounded gate before merge (CLAUDE.md §9).

## Model decisions (settled)
- ONE `elevenlabs` cloud voice engine (char + announcer); flash/tts are the same
  `ElevenLabsTextToSpeech` class at a flat $0.24/1K chars — tier rides the
  `model` DYNAMICCOMBO, NOT a second engine. ~$1.10/episode dialogue.
- Library voices only for v1; `InstantVoiceClone` deferred (unanimous cut).
- Pinned signature announcer voice for v1; seed-shuffle deferred.
- Cloud music v1 default = `cloud_sonilo_music` ($0.15/60s, native `duration`,
  "BEST music"); `cloud_stability_audio` = documented next candidate, not wired.
- Local `stable_audio_3` stays the engine default until operator promotes cloud.

## Grounded contracts the build MUST honor (all CONFIRMED against real files)

C1. **`runtime: cloud` profile contract (new).** `config/audio_engine_profiles.yaml`
today only allows `in_graph | oop_venv`. Add `runtime: cloud` plus the fields
that bind a profile to a partner row: `partner_row` (e.g. `cloud_elevenlabs`),
`provider_id`, `required_param_defaults` (model/output_format/language_code/
apply_text_normalization/stability), `auth_required`, `billing_category`,
`canonicalizer` target, `error_policy: fail_loud`, and valid roles
(char_voice / announcer_voice / music). The engine resolver must route a
`runtime: cloud` profile to `cloud_media_invoke.invoke_partner_node`.

C2. **Voice-bank cloud identity (schema change).** `voice_bank_entry_schema.json`
REQUIRES `ref_path` (VALIDATE_INPUTS checks local-disk presence) AND `ref_sha256`
(both minLength 1). A voice_id cannot ride in `ref_path`. Use
`additionalProperties:true` to add `provider_voice_id` (string) and make the
disk/sha requirement ENGINE-CONDITIONAL: skip disk presence + sha for
`engine=elevenlabs` (runtime cloud); populate `ref_path`/`ref_sha256` with a
schema-valid sentinel or relax minLength via a cloud-entry branch. CastLock's
scorer (`assign_voice_for_slot`) then casts cloud entries unchanged (it scores
gender/timbre/role/age, not the ref file).

C3. **ELEVENLABS_VOICE typed-input resolver.** The TTS node needs a typed
`voice: ELEVENLABS_VOICE`. Design the path from a cast-stamped
`provider_voice_id` to that typed input: preferred = the cloud adapter constructs
the ELEVENLABS_VOICE payload directly from the voice_id (verify the payload shape
on the live install); fallback = drive `cloud_elevenlabs_voice_selector` with the
id. VERIFY-AT-BUILD: does the selector accept stable voice IDs or only mutable
display labels? Choose voice once at CAST time, reuse per line (like local
`voice_ref_id`).

C4. **Cloud fail-loud ADMISSION GATE (new node/step).** `cast_lock.py:386`
`_resolve_character_voices_fail_soft` NEVER raises and leaves orphan lines "for
the node-81 engine fallback" (PD1 audio-is-king), running unconditionally. It
CANNOT enforce the cloud no-fallback contract. Add a cloud admission gate AFTER
CastLock and BEFORE any cloud invoke that fails LOUD on: missing ElevenLabs
auth/quota; any cloud-selected character or announcer line with no resolvable
`provider_voice_id`; budget exhausted. No silent swap to a local engine.
OPERATOR DECISION: for a cloud char line with no voice, fail-loud OVERRIDES PD1's
repair — confirm this is wanted (directive says yes).

C5. **V3 DYNAMICCOMBO expansion (build prerequisite).** `partner_nodes.yaml`
pin_meta `combo_options_excluded: true`; `model` is COMFY_DYNAMICCOMBO_V3 with
options hidden. Before naming a default tier, expand+pin the ElevenLabs `model`
options and the required combos `output_format` / `apply_text_normalization`
(and `language_code` policy). Ship a profile->schema conformance test (same
GOTCHA flagged for seedance).

C6. **Music role reconciliation.** `audio_engine_profiles.yaml` uses singular
`role: music`; `meta.music_engine` is singular. The cloud music engine is ONE
`music` profile mapped to all cue types (open/close/inter); stamp per-episode
`meta.music_engine` (the credits MUSIC line already reads it) plus per-cue
seed/duration in the ledger. Node-83 `done` output stays the pattern but the
durable `stamp_durable` write is the reliable path (node-83 `done` is unlinked
in the JSON today).

C7. **Determinism contract (reframed).** `seed_supported` proves a seed socket,
not byte-identical provider output over time. Contract = deterministic REQUEST
construction + durable logging (partner_row, resolved model string,
provider_voice_id, seed, duration, text/prompt hash). `test_audio_byte_identical`
stays scoped to local/mock paths; it compares muxed output vs the master INPUT,
so a cloud cue is fine as long as it flows through the FROZEN assembler unchanged
after mint.

## Sprint slices
- **S0 — pins & schema (no render):** V3-expand + re-pin ElevenLabs `model` +
  required combos (C5); add `runtime: cloud` + `provider_voice_id` schema changes
  (C1, C2); profile->schema conformance test. Gate workflow-JSON edits until here
  is green.
- **S1 — cloud voice adapter:** the `elevenlabs` char/announcer adapter (cloud
  variant of the registry pattern) on the reduced audio-engine protocol via the
  S0 backend bridge; canonicalize_audio (SR/format/loudness owned exactly like
  the still/video canonicalizers); per-line WAV out consumed by the SAME
  assembler. ELEVENLABS_VOICE resolver (C3). Delivery vector -> `stability` +
  `seed` ONLY (no similarity_boost/style/speed — not on the node).
- **S2 — curated voice pool + casting:** checked-in ElevenLabs voice manifest
  (voice_id + gender/age/accent/timbre, ToS-clean premade voices), mapped into
  `voice_reference_bank.json` cloud entries; pinned announcer voice; CastLock
  casts characters + announcer deterministically (OTR_CAST_SEED). Pool coverage
  acceptance: enough by gender/age/timbre to avoid forced reuse.
- **S3 — fail-loud admission gate (C4)** + budget/no-fallback enforcement
  (reuse the built budget state machine; $10 default cap).
- **S4 — durable stamps + credits:** stamp `cast[].voice_ref_id`/`voice_engine`,
  fold the `meta.cast_voice_slots` durable-stamp gap, `meta.music_engine`; verify
  they surface in `OTR_CreditsRoll`.
- **S5 — cloud music adapter:** `cloud_sonilo_music` engine (cloud variant of
  `stable_audio_theme`), prompt via the Meta brief protocol
  (`_otr_music_prompt.py`), native `duration`; trim inside the FROZEN assembler
  only if the provider overshoots (never re-open the ripped credits-music loop).
- **S6 — workflow JSON wiring:** land the selector/TTS + cloud-music nodes and
  links in `otr_scifi_16gb_full.json` (append-only positional widgets);
  re-run `OTR_WorkflowValidator` + link/widget audit.
- **S7 — acceptance:** live 30-word episode with ElevenLabs voices for a
  character AND the announcer + a cloud Sonilo cue; delivered voice IDs +
  `meta.music_engine` durably stamped and in the credits roll; audio
  byte-identical through the mux; no-key/no-quota fails LOUD; casting
  seed-reproducible.

## Open operator decisions (surfaced, not guessed)
1. Announcer voice: pinned signature (recommended v1) vs seed-shuffled.
2. Voice-pool size + license/ToS review of the chosen ElevenLabs premade voices.
3. v1 = library-only (recommended) vs include cloning.
4. Cloud music v1 default: `cloud_sonilo_music` (recommended) vs
   `cloud_stability_audio`; all 3 cue roles vs open/close only.
5. Confirm cloud char line with no voice = fail-loud (overrides PD1 repair) —
   directive says yes; confirm.
6. Music length: provider-native `duration` (recommended) vs OTR post-trim/loop.
