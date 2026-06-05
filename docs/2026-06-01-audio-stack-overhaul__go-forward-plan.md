# Go-Forward Plan -- Audio Engine Overhaul (model-agnostic registry)

**Date:** 2026-06-01 | **Branch:** `v2.0-alpha` | **Canonical plan:** this doc + `otr_audio_overhaul__unified_plan_v2.md`

## Scope (locked 2026-06-01)
**In:** a model-agnostic audio-engine registry for **music**, **announcer voice**, and **character voice** -- engines selectable per role, new models drop in as one adapter. Good narration is paramount.
**Out:** **SFX.** Not part of this work. The existing SFX subsystem (AudioGen / ProcSFX / `speaker_role="sfx"`) and the parked SFX/clean-ledger ROADMAP track stay as their own independent thing -- decoupled, not deleted.
**Target:** **stereo end to end.** Mono is transitional. Native-stereo engines (Stable Audio, Chatterbox) keep their channels; the mono assembly chain is the one thing standing in the way (see Stereo below).

## Spine -- the registry (BUILT)
`nodes/_otr_audio_engines/` -- a thin registry (`AudioEngine` Protocol, `register` / `engines_for_role` / `assert_usable`). The default-for-role engine sorts first so the ComfyUI dropdown defaults to the byte-identical choice; a non-default engine runs only when its `OTR_ENABLE_*` flag is on, else it resolves to the role default. So an un-flagged lane never changes the rendered audio. Adding a model = one adapter file + one import line.

Engine matrix:

| role | default (byte-identical) | opt-in | notes |
|---|---|---|---|
| char_voice | `bark` | `chatterbox` (MIT, commercial-clean), `indextts2` (Bilibili NON-commercial) | Chatterbox is the intended new default once enabled; IndexTTS2 stays flag-gated/non-commercial |
| announcer_voice | `kokoro` (Apache-2.0) | `chatterbox` | separate slot from characters (your LLM-slot model) |
| music | `musicgen` | `stable_audio_music` (Stability Community license, stereo-native) | |

## Determinism / PD1
Everything ships default-off; with defaults selected the assembled-WAV SHA-256 stays identical to `baseline_v1.5.wav`. Legacy engines (`interface="batch"`) delegate to the existing node -> byte-identical by construction. The per-line **delivery vector** (`_otr_delivery_vector.py`, 8-dim, pure-Python, no LLM) is stamped additively, ignored by Bark, so it does not move the baseline. Re-baseline is an operator-only act at promotion.

## Stereo (the "mono is the past" workstream)
`canonical_audio` already keeps `[B, C, T]` (channel-preserving), so the engines are stereo-ready today. The blocker is downstream: `OTR_SceneSequencer._extract_clips_from_audio` / `_resample_audio` assume mono 1-D, and `EpisodeAssembler` mixes mono. Going stereo end to end means upgrading those two nodes -- and because it changes the assembled bytes for **every** engine (including the legacy defaults), it is a **re-baseline-level change under PD1**, not a free win. Plan: keep `mono_safe` as the transitional bridge; do the SceneSequencer/EpisodeAssembler stereo upgrade as its own gated sprint with an operator re-baseline. Until then new stereo engines are downmixed at the node boundary so nothing breaks.

## Build status
- **Sprint A** `9b76d78` -- registry + `[B,C,T]` audio utils (`canonical_audio`/`mono_safe`/`audio_sha16`). 14 tests.
- **Sprint B** `1b5a39b` -- deterministic 8-dim delivery vector. 7 tests.
- **Sprint C** `c79cc51` -- six engine adapters (3 legacy defaults + Chatterbox/IndexTTS2/Stable-Audio-music opt-ins), SFX excluded. 11 tests.
- **ROADMAP** `2161439` -- SFX track decoupled from this work.
- Full `tests/` green at every step (3432 passed / 12 skipped); default workflow untouched.

## Remaining
- **Sprint D (nodes):** `OTR_BatchCharacterVoices` (default `bark`), `OTR_AnnouncerVoice` (default `kokoro`), `OTR_StableAudioTheme` (default `musicgen`). Each reads its engine from the registry; `interface="batch"` engines delegate to the existing node (byte-identical), `interface="per_line"`/`"clip"` engines call the adapter. `stereo_policy` widget defaults to preserve-stereo (with the mono bridge until the assembly upgrade).
- **Sprint E (wiring):** opt-in workflow copy `workflows/otr_scifi_16gb_audio_v2_optin.json` (patch the voice/theme nodes, defaults=legacy; default workflow frozen) + the license-clean reference-voice bank (`config/voice_reference_bank.json` + validator) bridging Bark presets -> reference clips.
- **Operator (GPU):** isolated dependency pilot for Chatterbox/IndexTTS2 (no xformers on cu130) + Stable Audio (SageAttention-off); wire each opt-in's inference; capture `baseline_v2_*` per engine; flip defaults to promote.

## Rejected / not doing
Hard cutover (no fallback, breaks PD1); LLM-generated emotion in v1 (deterministic vector covers it; no PD6); SFX of any kind in this track; stereo assembly without a deliberate re-baseline.
