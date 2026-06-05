# OTR Audio-Engine + Voice-Casting Overhaul -- CONSOLIDATED FINAL PLAN (code-verified)

**Date:** 2026-06-02 | **Branch:** `v2.0-alpha` | **Status:** A/B/C/C.1 SHIPPED; D onward open
**Supersedes** every prior doc on this track (FINAL sprint plan, FULL-FINAL plan, the 3-review consolidations, the voice-casting subsystem spec, the round-3 hardening). This is the single source of truth.
**Verification:** every code claim in §0 is verified against the live repo (five subagent passes). §0 is ground truth; nothing below overrides it.

---

## 0. Verification corrections (ground truth, V1-V22)

**Audio path (prior rounds):** V1 `default_tts="bark"` inert. V2 master already 48k stereo; HuMo gets stereo today; HuMo-16k is present-tense. V3 five drama fields discarded pre-freeze; keyword delivery v1 stays; structured source keys on **`arc_phase`** (+ `speaker_role`), not `beat_intent` (prose). V4 no Stable Audio "Qwen reprompt". V5 SR ladder 48k/24k. V12 native-stereo theme already survives the assembler -> **defer S13**. V13 AudioEnhance widen mis-processes true-stereo voice (only relevant if a native-stereo voice engine lands). V14 legacy `script_json` round-trip is byte-safe (json.loads only). V15 HuMo 16k-mono must be node-51-internal, Whisper-encode-only. V16 HuMo frame-count is duration-based (resample-safe). V17 VRAM flush must use `comfy.model_management`, not raw torch.

**Casting (prior round):** V6 `char_id` stable + rename-proof (no derive-id needed). V7 legacy age is a no-op on the default pool path (`age_band=None`). V8 additive cast schema already in production (`tts_model`/`voice_params` already on rows; no strict validator; no `CastCharacter` class). V9 voice assignment computed once at `lock_cast`, stamped on the frozen cast. V10 `beat_intent` is prose; `arc_phase` is the enumerable persisted field. V11 `line_id` deterministic (positional).

**This round (decisive):**
- **V18 -- engine-complete cast (H1) is BYTE-SAFE on the default path. [decisive PASS]** The legacy caster draws from a single dedicated `cast_rng = rng or random.Random()` (`_otr_casting.py:1289`), threaded through `precompute_ensemble_slots`/`cast_one_character`/`python_assign_voice_preset` (`:1342/:1373/:890`); grep found **zero** bare global `random.*` draws; the two seeded helpers build their own isolated `random.Random(f"{cast_seed}:{char_id}")`. **A new caster that uses its own fresh `random.Random(per_slot_seed)` (never `cast_rng`, never global `random`) is a disjoint PRNG stream and cannot change the legacy `voice_preset` outcomes.** => stamp BOTH `voice_preset` + `voice_ref_id` at `lock_cast` on every row; downstream widgets select only. No upstream-widgets fallback needed.
- **V19 -- cast iteration order is already stable** (`enumerate(open_slots)`, positional; voice pick `sorted(pick_from, key=ps[0])` `:889`). H5 per-slot stable seeds are feasible and won't fight existing order.
- **V20 -- HuMo encode tensor is SHARED with mux + duration. [FAIL -> hard rule]** `chunk["audio"]` feeds the Whisper encode (`batch_humo_render.py:2350`) AND `_save_clip_via_ffmpeg` mux (`:2733`) AND duration math (`:2244`) -- one object. An in-place 16k-mono downmix would corrupt the muxed video audio (PD1) + duration. **S0.1 MUST `.clone()` (or build a separate dict) before downmixing.** The existing mux is read-only, so the risk is strictly a new in-place op.
- **V21 -- the video branch already gates on EVICTION, not data.** Nodes 51 + 72 wait on `OTR_UnloadAll` (node 24) via forceInput STRING links (83/217); `OTR_UnloadAll` runs `mm.unload_all_models()` + `gc.collect()` + `soft_empty_cache(force=True)` + explicit `_unload_bark()` **before** returning (`visual/unload_all.py:185-294/348-350`). So M1's window is structurally mitigated for the existing boundary; **the new opt-in engines must join that eviction chain** (unload before the video gate fires).
- **V22 -- comfy mm frees patchers, NOT out-of-band caches. [NUANCE -> FM2 is real]** HuMo models are locals returned as model-patchers that `mm.unload_all_models()` tracks (`_otr_humo_tier_loader.py:42/496-504`) -- no `del` needed. But Bark/LLM/MusicGen live in **module-global caches** (`_BARK_CACHE`, `_otr_bark_lib.py:106`) that mm cannot see; they require an explicit `del` + `gc.collect()` + `empty_cache()` (the `_unload_bark()` pattern, `:251-260`, called by `OTR_UnloadAll`). **A new HF-pipeline engine (Chatterbox/IndexTTS2/Stable Audio) parked on an instance attr / global MUST mirror that explicit teardown** before defrag; `unload_all_models()` alone leaks it.

---

## 1. Goal & scope
Model-agnostic per-role audio-engine registry **and** its upstream voice-casting subsystem: **music + announcer_voice + character_voice**, casting resolved Python-side from a versioned registry, engine-complete at `lock_cast`. Good narration is the priority. **SFX out** (separate S12). Stereo deferred (V12).

## 2. Locked decisions (unchanged)
1 default-OFF lanes; legacy = byte-identical default. 2 fresh registry. 3 IndexTTS2 non-commercial -> Chatterbox (MIT) char default, IndexTTS2 flag-gated. 4 announcer + character independent slots, shared pool. 5 deterministic keyword delivery vector v2 (structured = B.2 on `arc_phase`+`speaker_role`; LLM-emitted rejected). 6 downstream `prepare_text`. 7 `[B,C,T]` canonical audio. 8 SFX is its own cleanbreak; re-baselines SEQUENCED, profile-named.

## 3. Prime-directive constraints
- **PD1 [H1] raw delegation:** generic node hands the legacy node the unmodified `script_json` string (round-trip verified safe, V14) + its exact widget tuple from a frozen, test-asserted `config/legacy_invocation_manifest.json`; zero transform; output verbatim; never `canonical_audio(48000)` the 24k output (V5).
- **PD2 VRAM (16 GB / 14.5 ceiling):**
  - `AUDIO_ENGINE_LOCK` (process-global) + single-residency `AudioEngineSession`; `defrag_cuda()` (`gc.collect()`+`empty_cache()`+`ipc_collect()`) **before load AND after unload**.
  - **[V22/FM2] explicit teardown:** an engine on an instance attr / module-global cache MUST `del self.model` + `gc.collect()` + `empty_cache()` on unload (mirror `_unload_bark`); `mm.unload_all_models()` does not see it. Use Comfy's manager for patcher models (V17).
  - **[V21/M1] join the eviction chain:** the new opt-in engines must be unloaded before the `OTR_UnloadAll`-gated video load fires -- the existing pipeline already gates video on eviction, so the new engines hook into the same discipline (their session `__exit__` completes before the video gate).
  - **[FM3] pin autocast dtype** for the audio block (`torch.autocast(device_type="cuda", dtype=...)` inside `deterministic_inference`); don't rely on engine defaults (sm_120 bf16 drift).
  - Measured peak in S11: TF32-off, Stable Audio full T5 pipeline, **post-video-load** (FLUX/HuMo touched).
- **PD3:** new nodes only in the opt-in workflow copy; default frozen.
- **PD6:** engines + delivery vector + delivery profiles + `prepare_text` add no LLM call.
- **[H9] determinism:** `CUBLAS_WORKSPACE_CONFIG=:4096:8` set **before CUDA init** (startup bootstrap; `assert_determinism_env_ready()` raises at import); baseline `warn_only=False`; S11 pilots strict determinism against the real forward passes + a documented reduced-determinism fallback if sm_120 kernels hard-crash; `flash_attn`/`flash-attn` on the S11 import hard-fail set (M2). Bit-exact opt-in baselines not assumed until S11 (D4).
- **Guardrails:** `engine` key clears b6; avoid `_s28` patterns.

## 4. Architecture
`[H3]` flag-independent stable `engines_for_role` + `DEFAULT_ENGINE_BY_ROLE`. `[H2]` `assert_usable` fails closed. `[H7]` typed `VoiceRequest`/`MusicRequest`, interface-blind node, `prepare_text->str` text-only, `VoiceRequest` carries `stage_directions`. **[H9-profiles]** the dropdown identity is an **engine profile** (`config/audio_engine_profiles.yaml`: `profile_id/role/engine/commercial_clean/model_path/model_sha256/default_params/allowed_voice_banks`), not a bare engine name -- a swap is a profile addition. (Its own sprint between D and G, D5.)

---

## 5. Voice-casting subsystem (engine-complete, byte-safe per V18)

**Two casters at `lock_cast`, both stamped (V18 byte-safe):**
```
LEGACY (FROZEN): python_assign_voice_preset, dedicated cast_rng, age no-op on default (V7)
  -> voice_preset = "v2/en_speaker_*"   (default path: unchanged bytes)
NEW (own per-slot random.Random(stable_cast_seed), disjoint stream):
  commercial-filter -> gender(100)/timbre(40)/role(20)/age(10, light) -> stable sort -> one rng.choice
  -> voice_ref_id   (+ engine profile id)
```
Both run at `lock_cast`; **both ids stamped on every `char_id` row** so the frozen cast is valid for any engine. **[H1]** Downstream `OTR_BatchCharacterVoices` widgets SELECT over pre-stamped data, never re-cast (`auto_registry` = use `voice_ref_id`; `preserve_ledger` = use `tts_model`); fail-closed assert that the resolved engine has a populated id on every row.

**[H5] new-caster determinism:** per-`char_id` `stable_cast_seed(episode_seed, casting_policy_version, char_id, gender, timbre, role, age_band)` (sorted-keys SHA), stable slot order (V19) -- kills the add/remove-character cascade re-roll. One seeded `rng.choice` over the top tier.

**[H6] insufficient-voice ladder:** gender+timbre+role+age -> drop age -> drop role -> gender-only -> **fail** unless `allow_voice_reuse` is explicit + stamped. Never silent reuse.

**Cast contract (additive, V8):** rows already carry `char_id`(V6), `name`, `gender`, `casting_traits`, `tts_model`, `voice_preset`, `voice_params`. **Add** `voice_ref_id` + `commercial_clean` (nullable; byte-neutral default-path). Voice keys on `char_id`, never `name`.

**[H3-validator] char_id integrity:** in `OTR_LedgerScriptWriter`, BEFORE freeze, assert `set(line.char_ids) <= set(cast.char_ids)` and fail the node immediately (no unmapped char_id reaches audio); stamp `cast_lock_revision` and assert the line index was built from it. Compatible with V6 (it's an upstream guard, not a post-index rename).

**[H7] manual overrides:** resolved at freeze, char_id-only keys (name rejected), schema-checked against the engine profile, unknown char_id/voice_id/param fails, non-commercial stamps `commercial_clean:false`, override-JSON SHA stamped. Render node consumes the frozen assignment, never re-interprets the JSON.

**[H4] compatibility resolver (fail closed, no auto-repair):** `bark` requires `preserve_ledger`+`bark_legacy` bank; ref-clip engines reject `bark_legacy`; `preserve_ledger` with a missing `voice_ref_id` RAISES ("run auto_registry"), never self-heals; non-commercial engine without `allow_noncommercial_models` raises.

**[H2] commercial-clean gate -- all leak paths closed:**
1 role-agnostic scan (character + announcer + music profiles + overrides + bank entries + cache sidecars + ledger `audio_meta`); 2 fail CLOSED on None/missing (never `get(...,True)`); 3 read RENDER TRUTH (adapter stamps `commercial_clean` from its own registry entry into `audio_meta` at render; cast-row stamp is advisory); 4 manual override fails closed at ASSIGNMENT; 5 `allow_noncommercial_models` enables a personal render but NEVER bypasses `assert_release_clean`; 6 **defense-in-depth filename mangling** -- any non-commercial render is renamed `[name]_NON_COMMERCIAL_RESEARCH_ONLY.wav` at EpisodeAssembler/SaveToEpisodeWorkspace (metadata can be stripped by FFmpeg).

**[H10] announcer** is pinned like a cast row (`char_id:"announcer"`, kokoro, `bm_george`, `speed:0.95`, `commercial_clean:true`, `announcer_policy_v1`); no `random`; delivery profiles do NOT touch the announcer.

**[H12] reference-bank provenance:** each entry carries `license_id/license_url_or_file/provenance/voice_clone_allowed/commercial_use_allowed/proof_sha256`; missing `voice_clone_allowed` -> release gate fails. "Open license" != clone-clean. (Bank source: do NOT reopen LibriTTS-R as automatically clone-clean -- prefer self-recorded / explicitly clone-licensed; provenance is mandatory.)

**Delivery profiles (Locked-5-safe):** named deterministic transform of the keyword vector before per-engine projection; ship the **plumbing in v2 with only `neutral` (identity no-op) populated**; populated profiles (`radio_drama`...) defer to v2.1; `delivery_profile`+`version` in the cache key + baseline; character-dialogue only.

**[H11] ledger-root audit metadata:** one block records `audio_engine_profile_set`, `voice_bank_id`+`sha`, `casting_policy_version`, `delivery_profile`+`version`, `commercial_clean`. Re-baseline triggers now include `casting_policy_version`, `delivery_profile_version`, `delivery_version`, kernel/driver.

---

## 6. Sprints

**A/B/C/C.1 [SHIPPED]** -- retrofits: DEFAULT_ENGINE_BY_ROLE+stable-order, fail-closed assert_usable, import-audit (+`flash_attn`), typed requests, PreparedLine. B stays keyword v2.

**D -- generic nodes [NEXT].** Registry-driven, default=legacy; raw-delegation batch branch (manifest widgets, verbatim, no canonical_audio); per-line branch builds typed requests. No SceneSequencer decoupling (V1). Tests: byte-identity passthrough (value, manifest widgets) + exact-string no-mutation, fail-closed, stable dropdown, default==legacy.

**D5 -- engine-profile sprint** (between D and the casting work): profile YAML, compatibility matrix (H4), profile hashes (H9).

**E.0-E.5 -- voice-casting.** E.0 engine-complete cast: stamp both ids at `lock_cast` (V18 byte-safe) + add `voice_ref_id`/`commercial_clean` (V8). E.1 voice registry + reference bank (provenance H12). E.2 the new caster (per-slot seeds H5, degradation ladder H6). E.3 delivery profiles (neutral-only). E.4 widget surface (select-only; reject `voice_engine_mode` + `deterministic_inference`-as-widget). E.5 commercial gate (H2) + char_id validator (H3) + override validator (H7) + compatibility resolver (H4).

**S0.1 -- HuMo 16k-mono [PRESENT-TENSE].** Node-51-internal, Whisper-encode-only, **`.clone()` before downmix (V20, hard rule)**; preserve duration (V16); must NOT mutate `episode_audio`/mux/duration. Tests: shape/sr/`episode_audio` hash unchanged/token-alignment.

**F -- operator pilots (GPU).** Isolated venv import (xformers + **flash_attn** + torch-change HARD-fail; offline env enforced); Stable Audio SageAttention OFF (+ no enhancer V4; record T5 size); determinism pilot vs real forward passes (strict vs documented fallback, + autocast-dtype pin FM3); render-twice **cross-process** (engine); measured peak (post-video-load); watermark `disabled`; **Chatterbox param names pinned** (`exaggeration`/`cfg_weight` vs `cfg`); IndexTTS2 license confirmed.

**G -- wire inference + cache.** Real `generate_voice`/`generate_music` inside `deterministic_inference`; free-run duration + guardrails; `stable_line_seed` (line_id deterministic, V11); **[H8] cache key** on POST-override resolution + license namespace (`commercial_clean/license_id/provenance_sha/allowed_for_release/profile_id+sha/bank_id+sha/override_sha/casting_policy_version/delivery_profile+version`) + **float-fidelity** (the engine receives the *same reconstructed quantized float* the cache string used -- never raw widget value); release mode refuses any record not `allowed_for_release`.

**H -- native stereo [DEFERRED, V12/V13].** Only if a native-stereo VOICE engine is enabled; then the work is the AudioEnhance widen guard (V13), mono-path byte-exact branch, episode-level cross-process render-twice, new capture.

**I -- promotion.** Profile-named baselines, sequenced; `assert_release_clean` pre-flight (H2); legacy permanent fallback.

**S12 (separate) -- SFX cleanbreak.** Unchanged; delete the inert `default_tts` widget here.

---

## 7. Failure modes (verified)
- **M1 [V21 PASS, refine]:** existing pipeline already gates video on `OTR_UnloadAll` eviction (not `audio_done`) -- new opt-in engines must complete their session `__exit__`/unload before that gate.
- **FM2 [V22 confirmed]:** HF-pipeline engines on an instance attr/global need explicit `del`+`gc`+`empty_cache` (mirror `_unload_bark`); `mm.unload_all_models()` won't free them.
- **FM3:** pin autocast dtype in `deterministic_inference`.
- **M2:** `flash_attn` on the S11 import hard-fail set; pilot notes any non-deterministic attention fallback (feeds D4).
- **S0.1 [V20]:** clone before downmix (mandatory).

## 8. Decision calls (my recommendations -- your confirm)
- **D1** keyword v2; B.2 on **`arc_phase` + `speaker_role`** (R2; arc_phase alone too coarse); LLM-emitted rejected.
- **D2** profile-named sequenced re-baselines.
- **D3** defer S13 (V12).
- **D4** pinned-config bit-exact first; **define the epsilon metric now** (`max|delta_sample| < eps`, pinned per engine, tolerance harness) as the documented fallback; legacy stays SHA-exact unconditionally.
- **D5** engine-profile sprint between D and casting.
- **D6** delivery profiles: ship **neutral-only plumbing in v2**; populated profiles to v2.1.
- **D7** keep age light (10 pts) in the new bank; legacy untouched (inert on default, V7).
- **H1 placement** confirmed byte-safe (V18) -> engine-complete cast with select-only downstream widgets; no upstream-widgets fallback.

## 9. Verify-in-pilot (not repo-resolvable)
Chatterbox exact param names; Stable Audio T5 size + no enhancer; strict-determinism survival + flash/SDPA non-determinism on sm_120/cu130 (drives D4); the cache float-fidelity wiring is a Sprint-G implementation rule (build it so the engine gets the reconstructed quantized float). Everything else from all source docs' verify lists is RESOLVED in §0 (V1-V22).

## 10. Tests (carry the round-3 set)
Engine-complete cast (both ids stamped; missing-id raises; policy select-not-recast); commercial gate (missing fails closed; scans all roles+cache; render-truth; override fails at assignment; allow-noncommercial != release bypass; filename mangled); char_id (subset-or-fail; name-key rejected; revision match); compatibility matrix + preserve_ledger-no-repair; new-caster per-slot-seed stable under unrelated edit + insufficient-pool ladder; cache (post-override; quantized-float==engine-input; profile metadata stamped); VRAM (video-load-waits-for-unload; session `del`s custom model before defrag; flash_attn hard-fail; autocast pinned); delivery-profile neutral no-op + cache-key.

## 11. Execution order
1 Sprint D (sealed legacy delegation). 2 D5 engine profiles (YAML + H4 matrix + H9 hashes). 3 E.0-E.5 (engine-complete H1 + H5 seeds + H6 ladder + H7 overrides + H2 gate + H3 validator + H4 resolver + H12 provenance). 4 S0.1 (clone-before-downmix). 5 F (deps + determinism + VRAM pilots: M1/FM2/FM3/M2). 6 G (inference + cache H8 + line seeds). 7 H only if a native-stereo voice exists. 8 I (profile baseline + commercial pre-flight).

## 12. Evidence appendix (this round)
caster rng `_otr_casting.py:1289/1342/1373/890`; no global random (grep 0); isolated helpers `:632/:1210`; iteration `:683/:1347`, sorted pick `:889`; age no-op `:1350-1351/:873-884`. HuMo shared tensor `batch_humo_render.py:2350/2733/2244/2257-2259`, slice `:808-822`, mux read-only `:1018/1028`. video eviction gate `visual/unload_all.py:185-294/348-350`, links 83/217, nodes 24/51/72. comfy frees patchers `_otr_humo_tier_loader.py:42/496-504`; Bark global cache `_otr_bark_lib.py:106/251-260`.

**Commits this track:** A `9b76d78`, B `1b5a39b`, C `c79cc51`, ROADMAP `2161439`, C.1 `f49d4f9`. Full `tests/` green (3440 / 12 skipped).
