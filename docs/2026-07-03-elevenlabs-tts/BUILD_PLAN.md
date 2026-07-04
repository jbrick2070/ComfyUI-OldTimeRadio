# BUILD PLAN — Cloud AUDIO on Comfy Cloud: ElevenLabs TTS + cloud MUSIC

Branch `v2.0-alpha`. Operator directives win. Fail-loud, no fallbacks, no hidden
promotion — the dropdown pick IS the enable. UTF-8 no BOM, SFW, determinism
seed-keyed. Every node/wiring change lands in `workflows/otr_scifi_16gb_full.json`
in the SAME change. Suite + Bug Bible + B7 green + push per green chunk.

Status: model selection CONVERGED via roundtable R1 (GPT-5.5 + Gemini 3.1 Pro +
DeepSeek v4-pro, ~$0.12, Claude anchor+judge; roundtable/pass01_judgment.md).
Coding contracts HARDENED via kibitz r2 (codex read-only sandbox + Claude
anchor/judge; antigravity benched; kibitz-runs/2026-07-03-elevenlabs-tts/r2/
final.md). Every K-finding below was spot-verified against the real files.
Remaining gate: a Fable final grounded pass (CLAUDE.md §9) — this touches casting
+ the render path.

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

C1. **`runtime: cloud` = new adapters + model changes, NOT just YAML (kibitz K1).**
`_otr_engine_profiles.py:35` `_VALID_RUNTIMES={"in_graph","oop_venv"}`; `:67`
`EngineProfile` is `extra="forbid"`; `:95` the validator RAISES on an unknown
runtime. Dispatch also IGNORES `profile.runtime` — voice runs through
`adapter.generate_voice()` (`_otr_voice_node_common.py:354/:559`), music through
`stable_audio_theme.py:203/:224`. So the correct shape is: register real cloud
ADAPTERS (`elevenlabs`, `cloud_sonilo_music`) whose `generate_voice` /
`generate_clip` call `cloud_media_invoke.invoke_partner_node`; ADD `"cloud"` to
`_VALID_RUNTIMES` and DECLARE the new profile fields on `EngineProfile`
(`partner_row`, `provider_id`, `required_param_defaults`, `auth_required`,
`billing_category`, `canonicalizer`, `error_policy: fail_loud`, valid roles).

C2. **`partner_row` must be a REAL pinned key (kibitz K2).** `cloud_elevenlabs`
is not a row. Use `cloud_elevenlabs_tts` (voice) / `cloud_sonilo_music` (music);
`invoke_partner_node` rejects unknown keys (`cloud_media_invoke.py:603/:620`).

C3. **Voice-bank cloud identity — SENTINEL, not schema conditionals (kibitz
K3+K4).** `voice_bank_entry_schema.json` REQUIRES `ref_path` (disk-checked) AND
`ref_sha256` (both minLength 1); the custom validator `_validate_entry()`
(`_otr_voice_bank.py:121`) supports NO oneOf/if/then, and `_entry_from_dict()`
indexes both keys unconditionally (`:166/:167`). So: add a `provider_voice_id`
property (schema allows extras) and use NON-EMPTY cloud SENTINELS
(`ref_path="cloud:elevenlabs/<voice_id>"`, `ref_sha256="cloud"`) + skip the
disk/sha check when `engine=elevenlabs`. Thread `provider_voice_id` through the
WHOLE chain — it is silently dropped today: add it to the schema, the
`VoiceBankEntry` dataclass (`:77`), `_entry_from_dict()` (`:158`), CastLock
`_stamp()` (which persists only voice_ref_id/voice_engine/commercial_clean,
`cast_lock.py:650-654`), the durable cast stamp, and the admission gate.
CastLock's scorer (`assign_voice_for_slot`) casts cloud entries unchanged (it
scores gender/timbre/role/age, not the ref file).

C4. **ELEVENLABS_VOICE typed-input resolver (kibitz K7 verify).** The TTS node
needs a typed `voice: ELEVENLABS_VOICE`; the only producer is the AUX
`cloud_elevenlabs_voice_selector` (COMBO->ELEVENLABS_VOICE). S0/S1 MUST include a
live-object capture test of `ElevenLabsVoiceSelector.EXECUTE_NORMALIZED` to
codify EITHER a direct typed-payload constructor from `provider_voice_id` OR
`selector(provider_voice_id)`. VERIFY-AT-BUILD: does the selector accept stable
voice IDs or only mutable display labels? Choose the voice once at CAST time,
reuse per line.

C5. **Cloud fail-loud ADMISSION GATE (kibitz K5 / roundtable A-3).**
`cast_lock.py:187` calls `_resolve_character_voices_fail_soft` UNCONDITIONALLY;
that routine (`:386`) NEVER raises and leaves orphans "for the node-81 engine
fallback" (`:510`) — PD1 audio-is-king. It CANNOT enforce no-fallback. Add a
cloud admission gate AFTER CastLock, BEFORE any cloud invoke, that fails LOUD on:
missing ElevenLabs auth/quota; any cloud-selected character/announcer line with
no resolvable `provider_voice_id`; budget exhausted. No silent swap to local.
(Alternative: make the repair cloud-aware and raise instead of appending the
fallback note when the selected profile runtime is `cloud`.) A dedicated node is
more auditable. OPERATOR DECISION: fail-loud OVERRIDES PD1's repair for cloud
char lines — confirm (directive says yes).

C6. **Budget cap is INERT unless adapters pass estimates (kibitz K6).**
`invoke_partner_node` defaults `estimated_usd=0.0` and reserves the passed
estimate (`cloud_media_invoke.py:603/:605/:623`); the backend cap exists
(`cloud_media_backend.py:110/:287/:292`) but does nothing on a 0 estimate. Build
cost estimators — ElevenLabs = chars*($0.24/1K), Sonilo = duration*($0.15/60s) —
and pass a NONZERO `estimated_usd` per line/cue. Hidden auth injects from
`session.auth` (`:363/:379`); confirm these audio adapters receive Comfy hidden
auth or require `OTR_COMFY_API_KEY`.

C7. **V3 DYNAMICCOMBO expansion (build prerequisite).** `partner_nodes.yaml`
pin_meta `combo_options_excluded: true`; `model` is COMFY_DYNAMICCOMBO_V3, tiers
hidden. Expand+pin the ElevenLabs `model` options + the required combos
`output_format` / `apply_text_normalization` (+ `language_code` policy) before
naming a default. One ElevenLabs adapter/profile only — tier via the resolved
`model` combo default (NO separate `_flash` engine).

C8. **Conformance tests currently XFAIL these exact rows (kibitz K7).**
`tests/test_cloud_partner_conformance.py:28-33` `KNOWN_UNADAPTERED` xfails
`cloud_elevenlabs_tts/_flash` + `cloud_sonilo_music`; `_engine_by_node_key()`
scans only image/video (`:24/:53`). Extend it to the audio registry + remove the
xfails in the SAME sprint that registers the adapters.

C9. **Music role reconciliation.** `audio_engine_profiles.yaml` uses singular
`role: music`; `meta.music_engine` singular. ONE `music` cloud profile mapped to
all cue types (open/close/inter); stamp per-episode `meta.music_engine` (credits
MUSIC line already reads it) + per-cue seed/duration in the ledger. The durable
`stamp_durable` write is the reliable path (node-83 `done` is unlinked in the
JSON today). Include `stable_audio_3` + the cloud engine in the
`StableAudioTheme` hard fallback tuple (`stable_audio_theme.py:36`, kibitz K8).

C10. **Determinism contract (reframed).** `seed_supported` proves a seed socket,
not byte-identical provider output. Contract = deterministic REQUEST construction
+ a ledger request-hash (text/prompt hash, resolved model, provider_voice_id,
seed, duration, partner_row). `test_audio_byte_identical` stays scoped to
local/mock; it compares muxed output vs the master INPUT, so a cloud cue is fine
as long as it flows through the FROZEN assembler unchanged after mint.

## Sprint slices (order tightened by kibitz)
- **S0 — pure code, no render (the gate for everything):** implement
  `canonicalize_audio` (`cloud_media_canonical.py:127` is a NotImplementedError
  stub today; `LOUDNESS_REFERENCE_SOURCE` is `"UNRESOLVED"` `:68` — resolve it to
  the LOCAL lane's real loudness reference); add `"cloud"` to `_VALID_RUNTIMES` +
  the new `EngineProfile` fields (C1); add `provider_voice_id` through schema +
  dataclass + loader + `_stamp` + durable stamp (C3); V3-expand + re-pin the
  ElevenLabs `model`/combos (C7); extend the conformance test + drop the xfails
  (C8). Gate workflow-JSON edits until S0 is green.
- **S1 — cloud voice adapter:** register the `elevenlabs` char/announcer adapter
  whose `generate_voice` calls `invoke_partner_node` (C1) via `cloud_elevenlabs_tts`
  (C2); ELEVENLABS_VOICE resolver + capture test (C4); per-line WAV consumed by
  the SAME assembler through `canonicalize_audio`; delivery vector -> `stability`
  + `seed` ONLY; cost estimator passed as nonzero `estimated_usd` (C6).
- **S2 — curated voice pool + casting:** checked-in ElevenLabs voice manifest
  (voice_id + gender/age/accent/timbre, ToS-clean premade voices) mapped into
  `voice_reference_bank.json` cloud entries (sentinel ref fields, C3); pinned
  announcer voice; CastLock casts characters + announcer deterministically
  (OTR_CAST_SEED). Pool coverage: enough by gender/age/timbre to avoid forced
  reuse.
- **S3 — fail-loud admission gate (C5)** + budget/no-fallback enforcement (C6).
- **S4 — durable stamps + credits:** `cast[].voice_ref_id`/`voice_engine` +
  `provider_voice_id`, fold the `meta.cast_voice_slots` durable-stamp gap,
  `meta.music_engine`; verify they surface in `OTR_CreditsRoll`.
- **S5 — cloud music adapter:** `cloud_sonilo_music` engine (cloud variant of
  `stable_audio_theme`) whose `generate_clip` calls `invoke_partner_node`; prompt
  via the Meta brief protocol (`_otr_music_prompt.py`), native `duration`, cost
  estimator (C6); trim inside the FROZEN assembler only on overshoot (never
  re-open the ripped credits-music loop).
- **S6 — workflow JSON wiring:** land the selector/TTS + cloud-music nodes and
  links in `otr_scifi_16gb_full.json` (append-only positional widgets); re-run
  `OTR_WorkflowValidator` + link/widget audit.
- **S7 — acceptance:** live 30-word episode with ElevenLabs voices for a character
  AND the announcer + a cloud Sonilo cue; delivered voice IDs + `meta.music_engine`
  durably stamped and in the credits roll; audio byte-identical through the mux;
  no-key/no-quota fails LOUD; casting seed-reproducible.

## Fable final-gate corrections — FOLD BEFORE BUILD (CLAUDE.md §9)
Grounded fan-out (model Fable, read-only) found seven cross-cutting misses the
roundtable + codex did not; each verified against the real files. These OVERRIDE
the sprint text above where they conflict.

- **F-MF1 — VRAM-tier profiles will silently REVERT the cloud pick (dormant-wire
  class).** `config/profiles/16gb_full.json:17-22` `slot_overrides` pin
  `char_voice_engine=indextts2 / announcer_voice_engine=kokoro /
  music_engine=stable_audio_3` and `widget_mapping.json` stamps them onto the
  live widgets on every headless `--profile` run. S7's acceptance render through
  the standard harness would stamp the widgets BACK to local and quietly test the
  local stack. FIX: S7 cloud acceptance runs profile-less (or via a dedicated
  cloud profile variant); naming a cloud engine in any profile requires F-MF2
  first (`capability_profiles.py:331` rejects engines not in CAPABILITIES).
- **F-MF2 — registering adapters breaks an unlisted test.**
  `tests/test_capability_profiles.py:213` asserts
  `set(CAPABILITIES)==set(_REGISTRY)`. Adding `elevenlabs` (S1) + `cloud_sonilo_music`
  (S5) requires matching rows in the audio `CAPABILITIES` table
  (`nodes/_otr_audio_engines/registry.py:184-207`; suggest
  `cpu_ok:True, requires_sidecar:False, model_requirements:[]`), same sprint.
- **F-MF3 — three hardcoded tuples or the engine can NEVER be selected.**
  Append `elevenlabs` to `_LEGACY_FIRST_ENGINES` char_voice+announcer_voice and
  the cloud music engine to music (`_otr_engine_profiles.py:42-51`, APPEND ONLY —
  index 0 stays the byte-identical default). Add a new cloud bank id to
  `_VOICE_BANKS` (`cast_lock.py:39`) allowed ONLY on the elevenlabs profile
  (`_resolve_char_engine` walks the tuple AND checks `voice_bank in
  allowed_voice_banks`, `cast_lock.py:657-676`). Miss any → `auto_registry`
  returns `target_engine=None` → silent zero cloud casting.
- **F-MF4 — announcer comes from the ADAPTER, not CastLock.**
  `cast_lock.py:41` `_DEFAULT_ANNOUNCER_ENGINE="kokoro"` is hardcoded; CastLock
  has no announcer-engine widget. The elevenlabs adapter pins its announcer via
  `announcer_voice_ref("elevenlabs")` in `begin_episode` (kokoro's own pattern,
  `_otr_voice_node_common.py:383-385`), which RAISES unless the S2 manifest ships
  an elevenlabs entry with `"announcer_voice"` in `roles` (single-entry pool OK).
- **F-MF5 — the C3 sentinel as written triggers a SILENT BARK fallback.** A
  `cloud:elevenlabs/...` ref fails `os.path.exists` in `_resolve_ref_to_disk`
  (`_otr_voice_node_common.py:472-485`), and `:514-552` then renders the line on
  BARK, local + invisible to budget/ledger (a no-fallback violation). CORRECT
  shape (replaces C3's disk-skip wording): the elevenlabs adapter sets
  `requires_voice_ref=False`, NO `missing_ref_fallback`, and
  `voice_ref_field="provider_voice_id"` so `:402-405` feeds the voice_id straight
  to `generate_voice`. This is why threading `provider_voice_id` through `_stamp`
  (C3) is load-bearing. No scattered `engine=="elevenlabs"` disk conditionals.
- **F-MF6 — do NOT drop the xfails in S0.** `KNOWN_UNADAPTERED`
  (`test_cloud_partner_conformance.py:28-35`) removal must move into the sprint
  that registers each adapter (elevenlabs in S1, sonilo in S5) or S0→S1 is red
  (billed row, no adapter, no xfail). `cloud_elevenlabs_flash` +
  `cloud_stability_audio` get NO adapter here — their xfails STAY (reworded) or
  their rows drop.
- **F-MF7 — S6 wires DROPDOWN VALUES, not graph nodes.** The workflow JSON has
  ZERO cloud partner graph nodes (cloud image rides existing dropdown values like
  `flux_gen1`). Graph-wiring the real ElevenLabs/Sonilo partner nodes would
  BYPASS `invoke_partner_node` (no budget/canonicalize/ledger/gate). S6 =
  select the new engine via widget VALUES (append-only positional) + validator/
  audit. The selector→ELEVENLABS_VOICE payload is built ADAPTER-INTERNAL
  in-process (C4), never in the graph.

Fable SHOULD-FIX: (1) if `elevenlabs` joins `APPROVED_VOICE_ENGINES`
(`_otr_voice_bank.py:216`), `test_voice_bank_coverage.py:11-18` requires >=5
adult-male AND >=5 adult-female elevenlabs entries — size the S2 manifest to it;
(2) K9/ledger test must assert the PER-LINE estimate scale (an episode-total
chars-per-line estimator would trip the $10 cap ~line 9); (3) pin the elevenlabs
YAML row `sample_rate: 44100` to match the canonical WAV; (4) the S0 re-pin
regenerates the WHOLE partner_nodes.yaml — run the image/video conformance suites
in the same chunk; (5) put `stable_audio_3` at index 0 of the `StableAudioTheme`
fallback tuple (`stable_audio_theme.py:36`), not merely "include" it.

Fable CONFIRMED-SOUND (no change needed): byte-identical spine is safe (cloud is
dropdown-opt-in, defaults unchanged); sentinel bank entries never leak into local
pools (every consumer is engine-filtered); the casting ladder raises a caught
`VoiceCastingError` on a too-small pool → S3 gate fails loud (so S2 pool coverage
is operational, not cosmetic); the durable chain carries `provider_voice_id`
(once `_stamp` adds it) + `meta.music_engine` and `OTR_CreditsRoll` reads both.

## Open operator decisions (surfaced, not guessed)
1. Announcer voice: pinned signature (recommended v1) vs seed-shuffled.
2. Voice-pool size + license/ToS review of the chosen ElevenLabs premade voices.
3. v1 = library-only (recommended) vs include cloning.
4. Cloud music v1 default: `cloud_sonilo_music` (recommended) vs
   `cloud_stability_audio`; all 3 cue roles vs open/close only.
5. Confirm cloud char line with no voice = fail-loud (overrides PD1 repair) —
   directive says yes; confirm.
6. Music length: provider-native `duration` (recommended) vs OTR post-trim/loop.
