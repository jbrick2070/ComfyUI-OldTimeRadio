# BUILD PLAN — Cloud AUDIO on Comfy Cloud: ElevenLabs TTS + cloud Sonilo music

Branch `v2.0-alpha`. Operator directives win. Fail-loud, no fallbacks, no hidden
promotion — the dropdown pick IS the enable. UTF-8 no BOM, SFW, determinism
seed-keyed. Every node/widget/wiring change lands in
`workflows/otr_scifi_16gb_full.json` in the SAME change. Suite + Bug Bible + B7
green + push per green chunk.

Two coupled lanes, one campaign, both reusing the built cloud backend
(`invoke_partner_node`): (A) ElevenLabs cloud TTS as a casting-integrated
character + announcer voice engine; (B) a truly-cloud Sonilo music engine.

Provenance: model selection converged via roundtable R1 (GPT-5.5 + Gemini 3.1 Pro
+ DeepSeek v4-pro, ~$0.12); coding contracts hardened via kibitz r2 (codex +
Claude anchor/judge; antigravity benched) and a Fable §9 grounded gate. Every
file:line below was verified against the real Windows files. This is the CLEAN
folded plan — all seven Fable cross-cutting fixes are integrated inline.

---

## 1. Model decisions (settled)
- ONE `elevenlabs` cloud voice engine (char + announcer) on partner_row
  **`cloud_elevenlabs_tts`**. `_flash` and `_tts` are the same
  `ElevenLabsTextToSpeech` class at a flat $0.24/1K chars (~$1.10/episode); tier
  rides the `model` combo, NOT a second engine. No separate `_flash` engine.
- Library voices only for v1; `InstantVoiceClone` deferred.
- Pinned signature announcer voice for v1; seed-shuffle deferred.
- Cloud music v1 default = **`cloud_sonilo_music`** ($0.15/60s, native
  `duration`, "BEST"); `cloud_stability_audio` documented as the next candidate,
  NOT wired. Local `stable_audio_3` stays the engine default until you promote
  cloud.

## 2. Invariants (non-negotiable)
- **Audio SPINE frozen.** Cloud produces per-line WAVs / cue AUDIO consumed by
  the SAME assembler; master mix + mux-LAST unchanged. `test_audio_byte_identical`
  stays green (it compares a default-widget baseline; cloud is dropdown-opt-in,
  so defaults never change).
- **No fallback / fail-loud.** Missing key/voice/quota = LOUD stop; never a
  silent swap to a local engine.
- **Determinism.** Casting from `OTR_CAST_SEED`; a curated checked-in voice pool
  reproduces. `seed_supported` is only a seed socket — the real contract is a
  deterministic REQUEST + a ledger request-hash, not byte-identical provider
  audio.
- **Append-only positional widgets** in the workflow JSON.

---

## 3. Grounded engineering contracts (all file:line verified)

### C1 — `runtime: cloud` = new ADAPTERS + registry rows, not just YAML
`_otr_engine_profiles.py:35` `_VALID_RUNTIMES={"in_graph","oop_venv"}`; `:67`
`EngineProfile` is `extra="forbid"`; `:95` the validator RAISES on an unknown
runtime. Dispatch IGNORES `profile.runtime` — voice runs through
`adapter.generate_voice()` (`_otr_voice_node_common.py:354/:559`), music through
`stable_audio_theme.py:203/:224`. Therefore:
- Add `"cloud"` to `_VALID_RUNTIMES` and DECLARE the new fields on
  `EngineProfile` (else `extra="forbid"` rejects them): `partner_row`,
  `provider_id`, `required_param_defaults`, `auth_required`, `billing_category`,
  `canonicalizer`, `error_policy: fail_loud`, valid roles.
- Register real cloud ADAPTERS — `elevenlabs` (voice) + `cloud_sonilo_music`
  (music) — whose `generate_voice` / `generate_clip` call
  `invoke_partner_node` with `partner_row="cloud_elevenlabs_tts"` /
  `"cloud_sonilo_music"` (real pinned keys; `invoke_partner_node` rejects
  unknown keys, `cloud_media_invoke.py:603/:620`).
- **CAPABILITIES parity (Fable F-MF2):** `tests/test_capability_profiles.py:213`
  asserts `set(CAPABILITIES)==set(_REGISTRY)`. Add matching rows to the audio
  `CAPABILITIES` table (`nodes/_otr_audio_engines/registry.py:184-207`;
  `cpu_ok:True, requires_sidecar:False, model_requirements:[]`) in the SAME
  sprint each adapter registers.

### C2 — engine must be SELECTABLE: three hardcoded tuples (Fable F-MF3)
Miss any one and `auto_registry` returns `target_engine=None` → "character
voices preserved" (silent zero cloud casting, `cast_lock.py:555-559`).
- APPEND `elevenlabs` to `_LEGACY_FIRST_ENGINES` char_voice + announcer_voice,
  and the cloud music engine to music (`_otr_engine_profiles.py:42-51`) — APPEND
  ONLY; index 0 stays the byte-identical default.
- Add a new cloud bank id to `_VOICE_BANKS` (`cast_lock.py:39`) allowed ONLY on
  the elevenlabs profile's `allowed_voice_banks` (`_resolve_char_engine` walks
  the tuple AND checks `voice_bank in allowed_voice_banks`,
  `cast_lock.py:657-676`; adding it to existing profiles would let indextts2 win
  the walk).

### C3 — voice identity = adapter metadata, NOT a disk sentinel (Fable F-MF5)
The naive `cloud:elevenlabs/...` ref_path fails `os.path.exists` in
`_resolve_ref_to_disk` (`_otr_voice_node_common.py:472-485`) and then
`:514-552` renders the line on BARK — silent, local, invisible to budget/ledger
(a no-fallback violation). Correct shape:
- The elevenlabs adapter sets `requires_voice_ref=False`, NO
  `missing_ref_fallback`, and `voice_ref_field="provider_voice_id"` — then
  `:402-405` feeds `cast.get("provider_voice_id")` straight to `generate_voice`.
- Thread **`provider_voice_id`** end-to-end (it is dropped today): add it to
  `voice_bank_entry_schema.json` (schema allows extras), the `VoiceBankEntry`
  dataclass (`_otr_voice_bank.py:77`), `_entry_from_dict()` (`:158`), CastLock
  `_stamp()` (persists only voice_ref_id/voice_engine/commercial_clean today,
  `cast_lock.py:650-654`), the durable cast stamp, and the admission gate.
- Bank entries still need schema-valid `ref_path`/`ref_sha256` (both required,
  minLength 1; the custom validator `_validate_entry()` at `:121` has no
  if/then). Use non-empty sentinels (`ref_path="cloud:elevenlabs/<voice_id>"`,
  `ref_sha256="cloud"`) purely to satisfy the schema — the disk check is never
  hit because `requires_voice_ref=False`. Every local pool consumer is
  engine-filtered, so `cloud:` refs never leak into local casting.

### C4 — ELEVENLABS_VOICE resolver is ADAPTER-INTERNAL
The TTS row needs a typed `voice: ELEVENLABS_VOICE`; the only producer is the AUX
`cloud_elevenlabs_voice_selector` (COMBO→ELEVENLABS_VOICE). Build the typed
payload IN-PROCESS inside the adapter from `provider_voice_id` (not in the
graph). S0/S1 include a live-object capture test of
`ElevenLabsVoiceSelector.EXECUTE_NORMALIZED` to codify the payload shape.
VERIFY-AT-BUILD: does the selector accept stable voice IDs or only mutable
display labels? Choose voice once at CAST time, reuse per line.

### C5 — announcer is pinned by the ADAPTER, not CastLock (Fable F-MF4)
`cast_lock.py:41` `_DEFAULT_ANNOUNCER_ENGINE="kokoro"` is hardcoded and CastLock
has no announcer-engine widget — it always casts the announcer from kokoro's
bank. The elevenlabs adapter pins its announcer via
`announcer_voice_ref("elevenlabs")` in `begin_episode` (kokoro's own pattern,
`_otr_voice_node_common.py:383-385`), which RAISES unless the S2 manifest ships
an elevenlabs entry with `"announcer_voice"` in `roles` (single-entry pool is
fine — lowest id, no ladder). A char line arriving with no `provider_voice_id`
must fail LOUD at the S3 gate — never inherit the announcer pin.

### C6 — fail-loud ADMISSION GATE (roundtable A-3 / kibitz K5)
`cast_lock.py:187` calls `_resolve_character_voices_fail_soft` UNCONDITIONALLY;
that routine (`:386`) NEVER raises and leaves orphans "for the node-81 engine
fallback" (`:510`). It cannot enforce no-fallback. Add a cloud admission gate
(a dedicated node — more auditable) AFTER CastLock, BEFORE any cloud invoke, that
fails LOUD on: missing ElevenLabs auth/quota; any cloud-selected
character/announcer line with no resolvable `provider_voice_id`; budget
exhausted. OPERATOR DECISION: fail-loud OVERRIDES PD1's "audio is king" repair
for cloud char lines (directive says yes; confirm).

### C7 — budget cap is INERT unless adapters pass estimates (kibitz K6)
`invoke_partner_node` defaults `estimated_usd=0.0` and reserves the passed
estimate (`cloud_media_invoke.py:603/:605/:623`); the backend cap exists
(`cloud_media_backend.py:110/:287-301`, $10 default) but does nothing on a 0
estimate. Build PER-LINE / PER-CUE cost estimators — ElevenLabs =
`chars*($0.24/1K)`, Sonilo = `duration*($0.15/60s)` — passed as nonzero
`estimated_usd`. The estimator MUST be per-line scale (an episode-total-chars
estimate per line would trip the $10 cap ~line 9). Hidden auth injects from
`session.auth` (`cloud_media_invoke.py:363/:379`) — confirm the audio adapters
receive Comfy hidden auth or require `OTR_COMFY_API_KEY`.

### C8 — canonicalize_audio is a STUB this campaign must build (kibitz K5)
`cloud_media_canonical.py:127` `canonicalize_audio` raises via `_not_built_yet`;
`LOUDNESS_REFERENCE_SOURCE="UNRESOLVED"` (`:68`). Build it FIRST: WAV 44.1 kHz,
stereo policy, loudness matched to the LOCAL lane's real reference (resolve that
constant/module — do not invent a fresh LUFS convention), ±250 ms tolerance with
head/tail silence padding, `actual_duration_s` to line metadata. Pin the
elevenlabs YAML row `sample_rate: 44100` to match.

### C9 — music role reconciliation + durable stamp
`audio_engine_profiles.yaml` uses singular `role: music`; `meta.music_engine`
singular. ONE `music` cloud profile mapped to all cue types (open/close/inter);
stamp per-episode `meta.music_engine` (`stable_audio_theme.py:179-183` already
ships this; `otr_credits_roll.py:161-205` reads it) + per-cue seed/duration in
the ledger. Put `stable_audio_3` at index 0 of the `StableAudioTheme`
profiles-import-failure fallback tuple (`stable_audio_theme.py:36`, currently
`("musicgen","stable_audio_music")`); include the cloud engine once registered.

### C10 — durable chain (confirmed sound, Fable)
`stamp_durable` copies the whole cast section
(`production_ledger.py:336-337`) so `provider_voice_id` survives once `_stamp`
adds it (C3); `meta.music_engine` + cast voices are read by `OTR_CreditsRoll`.
Fold the known `meta.cast_voice_slots` durable-stamp gap here (S4).

---

## 4. Sprint slices (sequenced so every chunk is green)

- **S0 — pure code, no render (the gate for everything):**
  build `canonicalize_audio` + resolve `LOUDNESS_REFERENCE_SOURCE` (C8); add
  `"cloud"` + the new `EngineProfile` fields (C1); thread `provider_voice_id`
  through schema/dataclass/loader/`_stamp`/durable stamp (C3); V3-expand + re-pin
  the ElevenLabs `model`/`output_format`/`apply_text_normalization` combos
  (running the image/video conformance suites in the SAME chunk — the re-pin
  regenerates the whole `partner_nodes.yaml`). Do NOT touch the conformance
  xfails yet. Gate workflow-JSON edits until S0 is green.
- **S1 — cloud voice adapter:** register the `elevenlabs` adapter
  (`generate_voice`→`invoke_partner_node` via `cloud_elevenlabs_tts`) with
  `requires_voice_ref=False` + `voice_ref_field="provider_voice_id"` (C3); add
  its `CAPABILITIES` row + append to `_LEGACY_FIRST_ENGINES` + `_VOICE_BANKS`
  (C1/C2); ELEVENLABS_VOICE payload + capture test (C4); per-line WAV through
  `canonicalize_audio`; delivery vector → `stability`+`seed` ONLY; per-line cost
  estimator (C7); REMOVE the `cloud_elevenlabs_tts` xfail
  (`test_cloud_partner_conformance.py:28-35`) in THIS sprint. `cloud_elevenlabs_flash`
  + `cloud_stability_audio` xfails STAY (reworded — no adapter).
- **S2 — curated voice pool + casting:** checked-in ElevenLabs voice manifest
  (voice_id + gender/age/accent/timbre, ToS-clean premade voices) → cloud bank
  entries with sentinel ref fields (C3) incl. an `"announcer_voice"`-role entry
  (C5); pinned announcer via the adapter's `begin_episode` (C5); CastLock casts
  characters deterministically (OTR_CAST_SEED). Pool coverage is OPERATIONAL: a
  too-small same-gender pool raises a caught `VoiceCastingError` → S3 fails loud.
  If `elevenlabs` joins `APPROVED_VOICE_ENGINES` (`_otr_voice_bank.py:216`),
  `test_voice_bank_coverage.py:11-18` requires ≥5 adult-male AND ≥5 adult-female
  entries — size the manifest to it.
- **S3 — fail-loud admission gate (C6)** + budget/no-fallback enforcement (C7).
- **S4 — durable stamps + credits:** `cast[].voice_ref_id`/`voice_engine` +
  `provider_voice_id`, fold the `meta.cast_voice_slots` gap, `meta.music_engine`;
  verify all surface in `OTR_CreditsRoll`.
- **S5 — cloud music adapter:** `cloud_sonilo_music` engine
  (`generate_clip`→`invoke_partner_node`) on the reduced protocol; prompt via the
  Meta brief (`_otr_music_prompt.py`), native `duration`, cost estimator (C7),
  `CAPABILITIES` row + `_LEGACY_FIRST_ENGINES` music append; REMOVE the
  `cloud_sonilo_music` xfail in THIS sprint; trim inside the FROZEN assembler
  only on overshoot (never re-open the ripped credits-music loop).
- **S6 — workflow JSON: DROPDOWN VALUES, not graph nodes (Fable F-MF7):** the JSON
  has ZERO cloud partner graph nodes; cloud rides existing dropdown VALUES (like
  cloud image with `flux_gen1`). Select the new engines via widget values
  (append-only positional), then `OTR_WorkflowValidator` + link/widget audit.
  Graph-wiring the real partner nodes would bypass `invoke_partner_node`
  (no budget/canonicalize/ledger/gate) — do NOT.
- **S7 — acceptance (PROFILE-LESS, Fable F-MF1):** the standard `--profile`
  harness stamps `char_voice_engine=indextts2 / music_engine=stable_audio_3`
  from `config/profiles/16gb_full.json:17-22` via `widget_mapping.json`, which
  would silently revert the cloud pick. Run the cloud acceptance profile-less
  (or via a dedicated cloud profile variant, which itself needs the CAPABILITIES
  rows from C1 or `capability_profiles.py:331` rejects it). Live 30-word episode
  with ElevenLabs voices for a character AND the announcer + a cloud Sonilo cue;
  delivered voice IDs + `meta.music_engine` durably stamped and in the credits
  roll; audio byte-identical through the mux; no-key/no-quota fails LOUD; casting
  seed-reproducible.

## 5. Test additions
- profile→schema conformance for the expanded ElevenLabs combos (S0).
- ELEVENLABS_VOICE live-object capture test (S1).
- ledger request-hash test asserting PER-LINE estimate scale + the hash fields
  (text/prompt hash, resolved model, provider_voice_id, seed, duration,
  partner_row) (S3).
- conformance xfail removals land in the adapter's own sprint (S1/S5), never S0.

## 6. Open operator decisions (surfaced, not guessed)
1. Announcer voice: pinned signature (recommended v1) vs seed-shuffled.
2. Voice-pool size + license/ToS review of the chosen ElevenLabs premade voices
   (min ≥5 M / ≥5 F adult if `elevenlabs` is an APPROVED_VOICE_ENGINE).
3. v1 = library-only (recommended) vs include cloning.
4. Cloud music v1 default: `cloud_sonilo_music` (recommended) vs
   `cloud_stability_audio`; all 3 cue roles vs open/close only.
5. Confirm cloud char line with no voice = fail-loud (overrides PD1 repair).
6. Music length: provider-native `duration` (recommended) vs OTR post-trim/loop.
