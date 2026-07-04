# Cloud AUDIO on Comfy Cloud — Model-Selection Plan (pass00)

Two coupled lanes, one campaign, both on Comfy Cloud partner nodes, both reusing
the already-built cloud backend (`invoke_partner_node`): (A) ElevenLabs TTS as a
casting-integrated character + announcer voice engine, and (B) a truly-cloud
MUSIC engine. This pass hardens the MODEL-SELECTION decisions first (which rows,
which tiers, which defaults); coding/wiring come in later rounds.

## Grounded facts (verified against the real Windows files, 2026-07-03)

Roster pins (`nodes/_otr_shared/partner_nodes.yaml`, generated 2026-07-03):
- `cloud_elevenlabs_flash` and `cloud_elevenlabs_tts` are the **SAME class**
  `ElevenLabsTextToSpeech` (import `comfy_api_nodes.nodes_elevenlabs`). Inputs
  (required): `text` STRING, `voice` ELEVENLABS_VOICE, `model`
  COMFY_DYNAMICCOMBO_V3, `stability` FLOAT, `seed` INT, `language_code` STRING,
  `output_format` COMBO, `apply_text_normalization` COMBO. Returns AUDIO.
  `seed_supported: true`. The only expressive knob exposed is **`stability`** —
  NOT similarity_boost / style / speed. Tier (flash/turbo vs premium) rides on
  the `model` DYNAMICCOMBO, whose real options are hidden by the V3 dynamic
  schema (shallow static pin — needs V3 expansion at pin time).
- `cloud_elevenlabs_voice_selector` = AUX, `api_node: false`, no billing,
  input `voice` COMBO -> output `ELEVENLABS_VOICE`.
- `cloud_sonilo_music` = `SoniloTextToMusic`; inputs `prompt` STRING,
  `duration` INT, `seed` INT; returns AUDIO; `seed_supported: true`;
  note "BEST music".
- `cloud_stability_audio` = `StabilityTextToAudio`; required `model` COMBO,
  `prompt` STRING; optional `duration` INT, `seed` INT, `steps` INT; returns
  AUDIO; note "CHEAP-cand music".

Pricing (`docs/2026-07-02-cloud-engines/PRICING.md`, 211 cr = $1):
- ElevenLabs TTS: **FLAT** 50.64 cr / 1K chars = $0.24 / 1K chars regardless of
  tier; ~790-word script ~= 4.6K chars ~= $1.10 for ALL dialogue. Tier split is
  QUALITY-only, not price.
- Sonilo music: 0.5275 cr/sec -> 60s ~= $0.15. Stability audio: 42.2 cr/run =
  $0.20/run.
- InstantVoiceClone (deferred lane): 31.65 cr = $0.15/voice.

Integration surface (verified):
- Engines register as rows in `config/audio_engine_profiles.yaml`, one per
  (role, engine), with `runtime: in_graph | oop_venv`. **There is no `cloud`
  runtime today** — cloud engines need a new runtime value.
- Voices live in `config/voice_reference_bank.json`, each entry coding to
  `config/voice_bank_entry_schema.json`: `voice_ref_id`, `engine`, `gender`,
  `timbre[]`, `roles[]`, `age_band`, `ref_path`, `ref_sha256`,
  `commercial_clean`. Local engines resolve `ref_path` to a disk clip; an
  ElevenLabs entry would carry the ElevenLabs **voice_id** where `ref_path`
  sits and has no disk clip / no sha.
- Casting: `nodes/cast_lock.py` casts characters via
  `assign_voice_for_slot(role, engine, char_id, gender, timbre, age_band,
  episode_seed, ...)` (deterministic scorer keyed on `episode_seed` /
  `OTR_CAST_SEED`, gender100/timbre40/role20/age10 ladder, honors a hybrid-LLM
  proposal then falls closed). Announcer is pinned per engine via
  `announcer_voice_ref(engine)` (raises if the active announcer engine has no
  reference). So a NEW engine slots in cleanly IF it supplies bank entries + an
  announcer reference.
- Music default today = `stable_audio_3` (rank 1, `is_default: true`) in the
  same profiles file; `caps: {duration_control: true}`.

## Section A — ElevenLabs voice model decisions

A1. **Which pinned row is the OTR voice engine?** flash vs tts are the same
class at the same price; recommend picking ONE row (the `model` DYNAMICCOMBO
carries the tier) and exposing tier as an engine param, not two engines.
Proposal: use one `cloud_elevenlabs` engine; default `model` = a
multilingual/quality tier; a flash/turbo option for cheap soak lanes.

A2. **Curated deterministic voice pool vs live library.** ElevenLabs' library
varies over time; determinism requires a checked-in manifest of a fixed OTR
voice pool (voice_id + gender/age/accent/timbre metadata) so `OTR_CAST_SEED`
reproduces. Proposal: curate a small licensed pool (premade/ToS-clean voices),
map each into a `voice_reference_bank.json` entry (engine=`elevenlabs`,
`ref_path`=voice_id), so CastLock's existing scorer casts them unchanged.

A3. **Announcer voice.** Pinned signature vs seed-shuffled (open operator
decision). Proposal: pinned signature announcer voice_id by default (a stable
show identity), operator env to shuffle; separate 1-voice announcer sub-pool.

A4. **Per-line delivery.** OTR's per-line delivery vector must map onto the
ONLY exposed knob, `stability` (+ seed). Higher stability = flatter/byte-stable;
lower = expressive. similarity_boost/style/speed are NOT on the pinned node —
do not scope a mapping that assumes them. Confirm at V3 expansion whether the
DYNAMICCOMBO hides more knobs.

A5. **Clone lane.** Library-only for v1; `InstantVoiceClone` ($0.15/voice) is a
later opt-in. Recommend defer.

## Section B — Cloud music model decision

B1. **v1 default: sonilo vs stability-audio.** Sonilo is cheaper ($0.15/60s vs
$0.20/run), marked "BEST music", and has a native `duration` INT for
open/close/inter cue lengths. Stability-audio is the "CHEAP-cand" with optional
duration + steps + a `model` COMBO. Proposal: `cloud_sonilo_music` = v1 cloud
default; `cloud_stability_audio` = selectable alt. Local `stable_audio_3` stays
the engine default until operator promotes cloud.

B2. **Which music roles go cloud.** `music_open` / `music_close` /
`music_inter`. Proposal: all three route to the chosen cloud engine when the
music dropdown selects it; no partial routing (simpler stamp + budget).

B3. **Length handling.** Prefer provider-native duration (both expose a
`duration`); if a provider ignores a short request, OTR trims inside the FROZEN
assembler — must NOT re-open the just-ripped credits-music loop.

## Shared contract (both lanes)
- No fallbacks / no hidden promotion: the dropdown pick IS the enable; missing
  key / voice / quota = LOUD stop (no silent swap to a local engine).
- Reuse the built cloud backend: `cloud_media_invoke.invoke_partner_node` +
  hidden auth + budget state machine + billing JSONL + a `canonicalize_audio`
  analog to the still/video canonicalizers (SR/format so the FROZEN assembler
  consumes cloud cues/lines identically).
- Durable stamps for the credits roll: `cast[].voice_ref_id` / `voice_engine`
  (+ fold the `meta.cast_voice_slots` durable-stamp gap) and `meta.music_engine`.
- Determinism: casting from `OTR_CAST_SEED`; music seed-keyed where supported,
  else log LOUD that the cue is non-reproducible.
- Every node/widget/wiring change lands in `workflows/otr_scifi_16gb_full.json`
  in the same change (append-only positional widgets); `test_audio_byte_identical`
  stays green through the mux.

## Open operator decisions (do not guess)
- Announcer: pinned vs shuffled.
- Voice-pool size + license/ToS review of the chosen ElevenLabs voices.
- v1 includes cloning or library-only.
- v1 cloud music default: sonilo vs stability-audio; all 3 roles or open/close only.
- Default when the ElevenLabs key is unset (directive = fail-loud; confirm no
  silent local fallback wanted).
