# OTR Audio-Engine Overhaul -- Sprint Plan (round-robin hardening draft)

**Date:** 2026-06-01 | **Branch:** `v2.0-alpha` | **Status:** A/B/C/C.1 SHIPPED; D onward open
**Owner:** Jeffrey A. Brick | **Drafted by:** Claude (grounded against live code)
**Purpose:** a self-contained plan to harden via round-robin (ChatGPT -> Gemini -> synthesis). Decision-forks and open questions are called out in S13.

---

## 0. Goal & scope

Build a **model-agnostic audio-engine registry** so each audio *role* picks its engine from a shared, extensible pool -- the same shape as the writer's two-slot LLM design. New models slot in as a one-file adapter, not node surgery.

**In scope:** `music`, `announcer_voice`, `character_voice`. **Good narration is the priority.**
**Out of scope:** **SFX.** The existing SFX subsystem stays untouched here and its removal is a separate staged cleanbreak (S12). Do not entangle the two.
**Stereo:** the target is stereo end to end; mono is transitional (S10).

---

## 1. Locked decisions (do not reopen in the round-robin)

1. **Default-OFF opt-in lanes.** Bark / Kokoro / MusicGen stay the byte-identical default; new engines run only behind `OTR_ENABLE_*`. Mirrors the shipped OpenRouter / Comfy-Credits pattern (`nodes/_otr_openrouter_backend.py:143`, `nodes/_otr_comfy_backend.py:184`).
2. **Registry superstructure**, not the dead `nodes/_voice_backends/` stub (that was never wired). Fresh thin registry under `nodes/_otr_audio_engines/`.
3. **IndexTTS2 is NOT commercial-clean** (Bilibili license). Characters default to **Chatterbox (MIT)** once the lane is enabled; IndexTTS2 is flag-gated and non-commercial-flagged forever.
4. **Announcer and character voices are independent slots** sharing one engine pool.
5. **Deterministic 8-dim delivery vector**, stamped additively post-freeze. No LLM (no PD6), no RNG (C7-safe).
6. **Downstream per-engine `prepare_text`**, NOT an engine-aware upstream writer -- the canonical script stays engine-neutral so engines stay swappable.
7. **`[B, C, T]` canonical audio**; never assume `waveform.shape[0] == 2`.
8. **SFX deletion is its own cleanbreak** (S12); its operator re-baseline is paired with the stereo re-baseline (S10) so we re-baseline once.

---

## 2. Prime-directive constraints

- **PD1 (audio is king).** The assembled-WAV SHA-256 must stay identical to `tests/fixtures/baseline_v1.5.wav` on the default path. Legacy engines reach byte-identity by **delegating to the existing node**, not reimplementing. Any new-engine audio or stereo assembly change is a deliberate, operator-run **re-baseline**, never silent. Runtime gate: `tests/test_audio_byte_identical.py` (sha256 vs fixture; `FIXED_SEEDS` pin every audio node to 42; gated behind `OTR_REGRESSION_RUNTIME=1`).
- **PD2 (14.5 GB VRAM).** New voice engines load after the LLM writer is unloaded, so the GPU is mostly free. Use `_flush_vram_keep_llm()` (never `force_vram_offload()` between LLM phases), a 1-token warmup, and a per-node vram sentinel. IndexTTS2 ~8 GB, Chatterbox modest, Stable Audio small -- all fit.
- **PD3 (wire the workflow JSON).** A node change is not done until the workflow JSON matches. New nodes land only in the **opt-in copy** `workflows/otr_scifi_16gb_audio_v2_optin.json`; the default `workflows/otr_scifi_16gb_full.json` stays frozen. `FIXED_SEEDS` keys + the seed-target test update when node identities change.
- **PD6 (LLM-slot tagging).** The engines are not LLM calls. The deterministic delivery vector + `prepare_text` add no LLM call. An *optional* engine-specific LLM doctor (S13 Q3) would be `# LLM slot: creative`, model id wired from the writer's `creative_writing_model` output socket -- no new widget.
- **Guardrails.** The `engine` dropdown key is not in `_MODEL_WIDGET_KEYS`, so it clears `tests/test_b6_wiring_guardrails.py` without needing the `NON_LLM_MODEL_WIDGET_OK` exemption. New code must avoid the `docs/_s28_forbidden_sweep.py` patterns (`alias`, `shim`, `legacy fallback`, `back-compat`, hardcoded `C:/Users/jeffr`, ...) in runtime lines.

---

## 3. Architecture (as-built spine)

```
nodes/_otr_audio_engines/
  registry.py        AudioEngine Protocol + register / get_engine /
                     engines_for_role (default-first) / assert_usable
  __init__.py        re-exports + imports every adapter (self-register)
  eng_bark.py        char_voice  (default; interface="batch")
  eng_kokoro.py      announcer_voice (default; interface="batch")
  eng_musicgen.py    music (default; interface="batch")
  eng_chatterbox.py  char_voice + announcer_voice (opt-in MIT; per_line; prepare_text)
  eng_indextts2.py   char_voice (opt-in Bilibili; per_line; prepare_text + emo_list)
  eng_stable_audio.py stable_audio_music (opt-in; interface="clip"; stereo-native)
nodes/_otr_audio_utils.py     canonical_audio([B,C,T]) / mono_safe / audio_sha16
nodes/_otr_delivery_vector.py deterministic_delivery_vector / stamp_delivery_vectors
nodes/_otr_script_prep.py     clean_spoken_text (engine-neutral base)
```

**Adapter contract.** `name: str`, `roles: tuple`, `default_roles: tuple`, `commercial_clean: bool`, `requires_flag: Optional[str]`, `interface: "batch"|"per_line"|"clip"`, `load()/unload()`. By interface:
- `batch` (legacy): `make_batch_node()` lazy-returns the existing node -> the generic node delegates the whole ledger -> **byte-identical**.
- `per_line` (voice opt-ins): `prepare_text(text, delivery_vector) -> str`, `generate_voice(text, ref_clip_path, delivery_vector, seed) -> AUDIO`.
- `clip` (music): `generate_clip(prompt, duration_s, seed) -> AUDIO`.

**Engine matrix:**

| role | default (byte-identical) | opt-in | flag |
|---|---|---|---|
| character_voice | `bark` | `chatterbox` (MIT), `indextts2` (Bilibili, non-commercial) | `OTR_ENABLE_CHATTERBOX` / `OTR_ENABLE_INDEXTTS2` |
| announcer_voice | `kokoro` (Apache-2.0) | `chatterbox` | `OTR_ENABLE_CHATTERBOX` |
| music | `musicgen` | `stable_audio_music` (Stability Community, stereo) | `OTR_ENABLE_STABLE_AUDIO` |

---

## 4. Determinism & re-baseline

The default path stays byte-identical because (a) the default workflow JSON is frozen, (b) the flags are off, (c) legacy engines delegate to the existing node, (d) the delivery vector is additive and Bark ignores it. **Re-baseline events** (operator, GPU host, seeded): (1) promoting any new voice/music engine to default; (2) the stereo assembly switch (S10). Pair them so there is a single `baseline_v2.wav` capture. The promotion PR records host, driver, torch/cu130 build, and SageAttention on/off (Blackwell determinism is sensitive to all).

---

## 5. Sprint A -- registry + audio utils  [SHIPPED `9b76d78`]
`registry.py` + `_otr_audio_utils.py`. 14 tests. No node, no workflow, no model -> byte-identical. **Exit met.**

## 6. Sprint B -- deterministic delivery vector  [SHIPPED `1b5a39b`]
`_otr_delivery_vector.py`: 8-dim `{happy/angry/sad/afraid/disgusted/melancholic/surprised/calm}` from keyword + punctuation + tension; `stamp_delivery_vectors(ledger)` adds `line['delivery']={emotion_vector, version="v1"}`. 7 tests. Additive. **Exit met.**

## 7. Sprint C -- engine adapters  [SHIPPED `c79cc51`]
Six adapters (3 legacy defaults + Chatterbox/IndexTTS2/Stable-Audio-music opt-ins). SFX excluded. Lazy imports; opt-in inference pilot-gated. 11 tests. **Exit met.**

## 8. Sprint C.1 -- prepare_text hook  [SHIPPED `f49d4f9`]
`_otr_script_prep.clean_spoken_text` (strip speaker label + parens + bracket tags); `prepare_text` on Chatterbox + IndexTTS2. 8 tests. **Exit met.**

## 9. Sprint D -- generic voice + theme nodes  [NEXT]
**Objective:** registry-driven nodes that default to the legacy engine (byte-identical) and expose the engine dropdown.
**Files:** `nodes/batch_character_voices.py` (`OTR_BatchCharacterVoices`, default `bark`), `nodes/announcer_voice.py` (`OTR_AnnouncerVoice`, default `kokoro`), `nodes/stable_audio_theme.py` (`OTR_StableAudioTheme`, default `musicgen`). Register in `__init__.py`.
**INPUT_TYPES:** `script_json` (forceInput), `engine` (`engines_for_role(<role>)` -> legacy first = default), role-specific params (temperature / durations), `seed`, `stereo_policy` (`["preserve_stereo","mono_safe"]`).
**Dispatch:** `engine = assert_usable(engine, role)`; if the resolved engine's `interface=="batch"`, delegate the whole ledger to `make_batch_node()` (byte-identical); else iterate dialogue lines (`_otr_ledger_consumers.iter_lines(roles=...)`), call `prepare_text` then `generate_voice` / `generate_clip`, pack via the same AUDIO contract Bark emits.
**Output:** char voice `("AUDIO","STRING")` = `(tts_audio_clips, batch_log)`; announcer `(announcer_audio_clips, ...)`; theme `("AUDIO","AUDIO","AUDIO","STRING")` (open/close/inter/log) -- match the existing sockets so the link graph is unchanged.
**PD:** Gate-3 voice-preset check (`batch_bark_generator.py:287-295`) generalizes to "valid for the active engine" so `idx:<ref>` can coexist with `bark:v2/...`. No `model_id` widget (the `engine` key is clean). VRAM sentinel + `_flush_vram_keep_llm()`.
**Tests:** node INPUT_TYPES structure; default engine == legacy; `assert_usable` fallback with flags off; delegation path returns the same AUDIO shape; stereo_policy default.
**Exit:** full `tests/` green; default workflow still names the legacy nodes; `OTR_BatchCharacterVoices @ engine=bark` delegates to `BatchBarkGenerator` unchanged (byte-identity is operator-verified at re-baseline, structurally verified headless).

## 10. Sprint E -- opt-in workflow copy + reference-voice bank
**Workflow:** generate `workflows/otr_scifi_16gb_audio_v2_optin.json` from the canonical, patching the voice/theme node types to the registry nodes (widget defaults = legacy engines). **Do not touch the link graph** -- the AUDIO sockets already match. Re-run `OTR_WorkflowValidator`; update the seed-target test.
**Reference-voice bank:** `config/voice_reference_bank.json` (maps each Bark `v2/en_speaker_*` -> a license-clean 3-10 s reference WAV, tagged gender/timbre/age/role) + `nodes/_otr_voice_reference_bank.py` (validator: consent/license/duration). New `python_assign_voice_ref` mirrors `_otr_casting.py:807 python_assign_voice_preset` (same gender->timbre->age ranking, one `rng.choice` over `open_voice_ref_pool` modeled on `config/cast_pools.py:477`) -> C7 preserved. Clips are self-recorded or open-license -- never scraped.
**Tests:** default-workflow-unchanged; opt-in-workflow has the v2 node types; bank validation; ref-assignment determinism + gender/age parity with the Bark pool.
**Exit:** opt-in workflow validates; bank validator green; default workflow byte-identical.

## 11. Sprint F -- operator dependency pilots (GPU, your machine)
Isolated venv import test per opt-in lib BEFORE it touches the main env. Must prove: no `xformers` install, no torch downgrade off cu130 nightly, no cu121/124 pin, no transformers conflict, no Linux-only path, no startup crash. Diff `pip freeze` before/after. Chatterbox + IndexTTS2 + Stable Audio (Stable Audio also: confirm output with SageAttention OFF). Confirm IndexTTS2 license terms before any non-test use.
**Exit:** each lib imports clean on the Blackwell stack; licenses recorded.

## 12. Sprint G -- wire opt-in inference + per-engine prepare_text tuning
After F passes: fill in `generate_voice` / `generate_clip` real calls (Chatterbox API is known; IndexTTS2 + Stable Audio confirmed in F), tune each engine's `prepare_text`, free-run duration first (do not pin token-count -- HuMo lip-syncs to the line). Per-engine output cache keyed on `(engine, ref_id|prompt, text, emo_vec, seed)` like MusicGen/AudioGen already do.
**Exit:** a flagged opt-in episode renders end to end on the GPU host.

## 13. Sprint H -- stereo end to end  [re-baseline]
Upgrade `OTR_SceneSequencer._extract_clips_from_audio` (`scene_sequencer.py:508`) and `_resample_audio` (`:109`) and `EpisodeAssembler` (`:949`) from mono 1-D to channel-aware `[B,C,T]`. Because this changes the assembled bytes for **every** engine, it is a re-baseline. Keep `mono_safe` as the bridge until this lands.
**Exit:** stereo through the chain; new `baseline_v2.wav` captured; PD1 re-anchored.

## 14. Sprint I -- promotion + re-baseline runbook
Operator: set flags, `OTR_REGRESSION_RUNTIME=1`, render once per engine, save `baseline_v2_<key>.wav` + `.sha256`; flip defaults; tag `v2.0-alpha-stable`. Rollback = inverse flag flip. Legacy engines remain registered as permanent fallback.

## (separate) SFX cleanbreak
Tracked independently: delete `speaker_role="sfx"` (woven through `scene_sequencer` ~86, `_otr_ledger_freeze` ~34, `_otr_ledger_consumers` ~21, `_otr_outline` ~14, `video_engine` ~14, `_otr_speaker_role` ~11), `OTR_BatchAudioGenGenerator` (node 15), ~57 test files, and the canonical-workflow node; pair its operator re-baseline with S13. Staged: node -> role -> ledger/freeze/assembly/video -> tests -> workflow re-wire, full regression per stage. Authorized 2026-06-01.

---

## 15. Risks
- **xformers contamination** on Blackwell silently downgrades torch off cu130 -> S11 isolated pilot is mandatory.
- **IndexTTS2 license** non-commercial -> stays flag-gated, never default; Chatterbox (MIT) is the character default.
- **Stable Audio x SageAttention** reported bad output unless SageAttention is off -> pin the working state and record it in every baseline.
- **Stereo re-baseline blast radius** -- S13 changes the default-path bytes; must be a deliberate operator capture, not a toggle.
- **Reference-clip licensing** -- self-recorded / open-license only; document provenance.
- **Determinism** -- diffusion (Stable Audio) + AR-with-temperature (Chatterbox) are deterministic only with pinned seed + steps + temperature + stable kernels; any kernel/driver change is a re-baseline trigger.

## 16. Open questions for the round-robin
1. **Reference-voice bank source** -- self-recorded vs curated open-license packs vs synthesized anchors (Kokoro/Chatterbox-generated). Legal-clean is mandatory; which gives the best 5-7 voice ensemble coherence?
2. **Stereo timing** -- do S13 now (one re-baseline, paired with SFX removal) or keep stereo behind the opt-in lane and re-baseline later? Trade-off: sooner stereo vs a larger single re-baseline.
3. **Per-engine LLM doctor** -- is the deterministic `prepare_text` enough, or is an opt-in engine-specific LLM rewrite pass (PD6) worth it for IndexTTS2/Chatterbox expressiveness? Cost vs quality.
4. **Duration control** -- free-run TTS + audio-driven shot length, vs pinning IndexTTS2 token count to a planned shot. Free-run avoids clipped VO under HuMo; does it hurt pacing?
5. **Chatterbox vs IndexTTS2 as the character workhorse** -- MIT/commercial-clean vs more expressive but non-commercial. Default to Chatterbox; keep IndexTTS2 for personal/non-commercial renders?
6. **Announcer engine** -- keep Kokoro (Apache, in-stack, cheap) as default, or move the announcer to Chatterbox for a consistent voice model across roles?
7. **Cache key** -- is `(engine, ref_id|prompt, text, emo_vec, seed)` sufficient, or do per-engine config knobs (steps, temperature, exaggeration) need to enter the key to avoid stale-cache drift?

## 17. Test / file map (as-built)
| concern | path |
|---|---|
| registry | `nodes/_otr_audio_engines/registry.py` + `tests/test_audio_engine_registry.py` |
| adapters | `nodes/_otr_audio_engines/eng_*.py` + `tests/test_audio_engine_adapters.py` |
| audio utils | `nodes/_otr_audio_utils.py` + `tests/test_otr_audio_utils.py` |
| delivery vector | `nodes/_otr_delivery_vector.py` + `tests/test_delivery_vector.py` |
| script prep | `nodes/_otr_script_prep.py` + `tests/test_script_prep.py` |
| PD1 gate | `tests/test_audio_byte_identical.py` + `tests/fixtures/baseline_v1.5.*` |
| assembly (mono->stereo target) | `nodes/scene_sequencer.py:445,508,602,949,1049,1088` |
| cast voice assignment | `nodes/_otr_casting.py:645,807` + `config/cast_pools.py:253,477` |
| ledger line / voice_preset | `nodes/_otr_ledger_consumers.py:51,71,138` |
| legacy engine classes | `BatchBarkGenerator` / `KokoroAnnouncer` / `MusicGenTheme` (`__init__.py:163-168`) |

**Commits this track:** A `9b76d78`, B `1b5a39b`, C `c79cc51`, ROADMAP-decouple `2161439`, C.1 `f49d4f9`. Full `tests/` green at each (3440 passed / 12 skipped).
