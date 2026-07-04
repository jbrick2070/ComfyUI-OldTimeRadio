# SCOPING PROMPT — Cloud AUDIO on Comfy Cloud: ElevenLabs TTS (casting + characters + announcer) AND a true cloud MUSIC engine

> Two coupled lanes, one campaign, both on Comfy Cloud partner nodes: (A) ElevenLabs TTS as a first-class casting-integrated voice engine, and (B) a truly-cloud MUSIC engine (Comfy Cloud, not local Stable-Audio). Scope them together — they share the cloud backend, the durable stamp, and the credits provenance. Section A is below; Section B (cloud music) is at the end.

---

## SECTION A — ElevenLabs TTS as a first-class OTR voice engine (casting + characters + announcer)

**Paste this to the coding window (or a kibitz/roundtable) to SCOPE the work. This is a scoping brief, not a build plan — the first deliverable is the build plan.**

Branch `v2.0-alpha`. Operator directives win. Ground every claim against the REAL Windows files via Desktop Commander + the Windows venv (never the Linux mount). No fallbacks, no hidden auto-defaults — the dropdown pick IS the enable; missing creds/voice FAIL LOUD at invoke. UTF-8 no BOM, SFW, determinism seed-keyed.

## Objective
Make **ElevenLabs cloud TTS a selectable, first-class character AND announcer voice engine** in OTR — fully integrated with CastLock casting (deterministic per-episode voice assignment), the durable-ledger provenance stamp (so it shows in the new credits roll), and the workflow JSON — with per-line delivery control and a hard budget/no-fallback contract.

Today ElevenLabs exists only as the AUX `cloud_elevenlabs_voice_selector` row (it emits an `ELEVENLABS_VOICE` output and is NOT wired as a `char_voice_engine`). The `cloud_elevenlabs_tts` / `_flash` TextToSpeech row is priced (~$0.24/1K chars, ~$1.10/episode flat) but not a selectable voice. This sprint turns it into a real casting-integrated voice lane.

## Ground these anchors FIRST (report exact files/lines before proposing anything)
- **Audio engine registry + adapters:** `nodes/_otr_audio_engines/` — how `indextts2` / `bark` / `chatterbox` / `dia` / `kokoro` self-register, expose the `char_voice_engine` dropdown, resolve a `voice_ref_id` to disk, and honor the per-line delivery vector. ElevenLabs must slot into this SAME pattern (cloud variant).
- **Casting:** `nodes/cast_lock.py` — the hybrid voice-fit (`~565-596`), the delivered `_stamp` (`~628-632`: `voice_ref_id` / `voice_engine`), the fall-closed deterministic scorer, and the seeded picker. Determine how a NEW engine's voice inventory (ElevenLabs voice IDs) is offered to the caster and assigned per character + the announcer.
- **Announcer path:** the `announcer` speaker role — how its voice is chosen today vs the character path; decide whether the announcer gets a PINNED signature voice or a seed-shuffled one (operator call — surface it).
- **Cloud backend (already built):** `nodes/_otr_shared/cloud_media_backend.py`, `cloud_media_invoke.py` (`invoke_partner_node`), `cloud_media_canonical.py`, and the pinned rows in `nodes/_otr_shared/partner_nodes.yaml` — how the ElevenLabs partner node(s) are pinned, their inputs/outputs (esp. the `voice_selector -> ELEVENLABS_VOICE -> TextToSpeech` chain), auth (hidden `api_key_comfy_org` via V3 `PREPARE_CLASS_CLONE`), budget state machine, and billing JSONL.
- **Durable stamp (credits provenance):** `nodes/production_ledger.py::stamp_durable` + how CastLock/the dispatcher copy the local wire ledger into the singleton. The delivered ElevenLabs voice MUST be durably stamped (`cast[].voice_ref_id`/`voice_engine`) AND ideally `meta.cast_voice_slots[<char>].speech_signature`, because the NEW `OTR_CreditsRoll` (`nodes/otr_credits_roll.py`) credits Cast & Voices from those stamps. (Known gap: `cast_voice_slots` is not durably stamped today — fold that fix in here.)
- **Workflow source of truth:** `workflows/otr_scifi_16gb_full.json` — any new voice-selector/TTS node or wiring lands IN this file in the SAME change; `widgets_values` is POSITIONAL (append-only); re-run `OTR_WorkflowValidator` + link/widget audit.

## Hard design questions the scope MUST resolve
1. **Voice inventory → casting.** ElevenLabs exposes a voice LIBRARY (voice IDs, gender/age/accent metadata). How does CastLock deterministically pick a voice per character (seed-keyed via `OTR_CAST_SEED`) so a re-run reproduces? Curate a fixed OTR voice pool (a checked-in manifest of ElevenLabs voice IDs + metadata) rather than a live-varying library, so casting is deterministic and offline-describable.
2. **Announcer voice.** Pinned signature announcer voice vs seed-shuffled? Configurable? Same or separate pool from characters?
3. **The selector→TTS chain.** How does `cloud_elevenlabs_voice_selector`'s `ELEVENLABS_VOICE` output feed the TextToSpeech call per line? Is voice chosen ONCE per character (cast time) and reused per line, or per line? (Should be cast-time, stable per character — like the local engines' `voice_ref_id`.)
4. **Per-line delivery.** Map OTR's delivery vector to ElevenLabs voice settings (stability / similarity_boost / style / speed). Which lines get expressive vs byte-stable?
5. **Billing / budget / no-fallback.** ~$1.10/episode flat — enforce the budget state machine; fail LOUD (no silent swap to a local engine) if the key/voice/quota is missing. Decide the DEFAULT budget behavior (the $10 cap already exists).
6. **Cloning lane (defer or include?).** `InstantVoiceClone` (deferred lane, ~$0.15/voice) vs library voices — recommend library voices for v1, clone as a later opt-in.
7. **Cost of the AUX selector row.** Confirm the selector is free/cheap and the TextToSpeech row is where the char billing lands.

## Constraints / invariants (non-negotiable)
- No fallbacks, no hidden promotion; the char_voice_engine dropdown pick IS the enable; missing key/voice = LOUD stop.
- Determinism: casting draws from `OTR_CAST_SEED` (OS entropy on normal runs); a curated voice pool makes it reproducible + offline-describable.
- Audio SPINE stays FROZEN: the master mix + mux-LAST + `test_audio_byte_identical` remain green; ElevenLabs produces per-line WAVs consumed by the SAME assembler path the local engines use.
- Every node/widget/wiring change lands in `otr_scifi_16gb_full.json` in the same change. Suite + Bug Bible + B7 green + push per green chunk. Single resident heavy ≤ 14.5 GB (cloud TTS holds no local VRAM). UTF-8 no BOM, SFW.

## Deliverables the scope should produce
1. A grounded BUILD PLAN (exact files/anchors, sprint-sliced) covering: the ElevenLabs char-voice adapter (cloud variant of the registry pattern); the curated voice-pool manifest (checked-in, licensed/ToS-clean); CastLock integration for characters + announcer; the selector→TTS wiring in the workflow JSON; the durable voice stamp (+ the `cast_voice_slots` gap); per-line delivery mapping; budget/no-fallback enforcement; tests.
2. The panel record (kibitz codex+antigravity, Cowork Claude anchor+judge) + a Fable final grounded gate before merge (CLAUDE.md §9), since this touches casting + the render path.
3. Acceptance: a live 30-word episode renders with ElevenLabs voices for BOTH a character and the announcer, the delivered voice IDs are durably stamped and appear in the credits roll, audio stays byte-identical through the mux, budget/no-key paths fail LOUD, and casting is seed-reproducible.

## Open operator decisions to surface (do not guess)
- Announcer voice: pinned vs shuffled.
- Voice pool size/curation + license/ToS review of the chosen ElevenLabs voices.
- Whether v1 includes voice cloning or library-only.
- Default when the ElevenLabs key is unset (fail-loud is the directive; confirm no silent local fallback is wanted).

---

## SECTION B — Truly cloud MUSIC engine (Comfy Cloud)

**Objective:** make the OTR music engine **truly Comfy Cloud** (a cloud partner audio model) — for the opening / closing / inter-scene cues — as a selectable engine alongside the local Stable-Audio default, with the SAME no-fallback / durable-stamp / budget contract as the voice lane.

Candidate Comfy Cloud partner music rows (from `docs/2026-07-02-cloud-engines/PRICING.md`): **`cloud_sonilo_music`** (~$0.15 / 60s) and **`cloud_stability_audio`** (stable-audio-2.5, ~$0.20 / run). Confirm which are pinned/available in `partner_nodes.yaml` and their real inputs/outputs before choosing the v1 default.

### Ground these anchors FIRST
- **Current music path:** `nodes/stable_audio_theme.py` (node 83 `OTR_StableAudioTheme`) — how it takes the music PROMPT (via the Meta brief protocol, `nodes/_otr_music_prompt.py`), produces the cue AUDIO, and emits `music:done:engine=...`. This is the pattern the cloud engine must match (prompt in → cue AUDIO out).
- **Music prompt protocol:** `nodes/_otr_music_prompt.py` — the closing/opening cue prompts route through the Meta brief; the cloud engine consumes the SAME prompt.
- **Assembler / spine:** how the theme cues feed `OTR_EpisodeAssembler` into the master mix. The AUDIO SPINE is FROZEN — the cloud cue must flow through the SAME assembler path; `test_audio_byte_identical` compares the muxed output vs the master INPUT it received (not a fixed baseline), so a different (cloud) cue is fine as long as it goes through the assembler unchanged after mint.
- **Music roles:** `music_open` / `music_close` / `music_inter` speaker roles — which cues the engine must produce, and whether all route to the same cloud engine.
- **Durable stamp / credits:** `meta.music_engine` is what `OTR_CreditsRoll` credits (MUSIC line). The cloud music engine MUST durably stamp `meta.music_engine` (the credits work already reads it) — confirm the node-83 `done` output vs a direct `stamp_durable` write (today node 83's `done` output is UNLINKED in the JSON; the stamp is the reliable path).
- **Cloud backend reuse:** same `cloud_media_invoke.invoke_partner_node` + auth + budget + billing + canonicalize path as the voice/still/video lanes.

### Hard design questions
1. **Prompt → cloud model.** Map the OTR music brief (mood/era/tempo) to the cloud model's prompt/params (sonilo vs stability-audio). Duration control (open ~10s, close ~8s, inter short) — does the cloud model honor a requested length, or does OTR trim/loop post-mint (loop must stay inside the frozen assembler, not re-open the credits-music loop that was just ripped)?
2. **Canonicalization.** Cloud returns an audio asset — canonicalize to OTR's SR/format so the assembler consumes it identically to local cues (`canonicalize_audio` analog to the still/video canonicalizers).
3. **No-fallback + budget.** Missing key/quota = LOUD stop (no silent Stable-Audio swap). Music budget line (~$0.15-0.20/cue) folds into the episode budget guard.
4. **Selectability.** The music engine dropdown pick IS the enable; local `stable_audio` stays the default until the operator promotes the cloud engine. No hidden auto-default.
5. **Determinism.** Seed-keyed where the provider supports it; otherwise note the cue is non-reproducible and log it LOUD (like the cast-RNG entropy log).

### Deliverables (fold into the same campaign as Section A)
- Cloud music adapter (cloud variant of the `stable_audio_theme` pattern, on the reduced audio-engine protocol), the pinned partner row confirmation, canonicalize_audio, `meta.music_engine` durable stamp, the workflow JSON wiring (node + links in `otr_scifi_16gb_full.json`, same change), budget/no-fallback, tests, and the kibitz + Fable gate.
- Acceptance: a live 30-word episode renders with the CLOUD music cue in the master mix, `meta.music_engine` is durably stamped and appears in the credits MUSIC line, audio stays byte-identical through the mux, and no-key fails LOUD.

### Open operator decisions (do not guess)
- v1 default cloud music model: `cloud_sonilo_music` vs `cloud_stability_audio` (cost vs quality vs licensing).
- Whether all three music roles use the cloud engine or only open/close.
- Length handling: provider-native duration vs OTR post-trim/loop.
