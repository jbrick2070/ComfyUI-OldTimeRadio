# OTR Audio + Voice-Casting Overhaul -- FULL FINAL Plan (code-verified)

**Date:** 2026-06-01 | **Branch:** `v2.0-alpha` | **Status:** A/B/C/C.1 SHIPPED; D onward open
**Supersedes:** `audio_engine_overhaul__FINAL-sprint-plan.md` + `audio_engine_FINAL__consolidated_3review.md` + `voice_selection_and_casting__subsystem_spec.md`.
**This pass folds in the upstream voice-casting subsystem** and re-verifies every new code claim against the live repo (two subagents). §0 corrections are ground truth and override any conflicting reviewer text. Hardening tags `[H#]`/`[R#]` carried for traceability.

---

## 0. Verification corrections (ground truth)

Prior round (kept): **V1** `default_tts="bark"` inert. **V2** master already 48k stereo; HuMo gets stereo today; HuMo-16k is present-tense. **V3** five drama fields discarded pre-freeze; keyword delivery v1 stays. **V4** no Stable Audio "Qwen reprompt". **V5** SR ladder 48k/24k confirmed.

This round (casting + assembly, all file:line-verified):
- **V6 -- `char_id` is already stable + rename-proof.** Assigned at cast-build (`_otr_casting.py:1063/1082/1096`, `c01..`), never from name; the LLM rename pass keys by char_id and rewrites only `name`, inside `lock_cast` (`:1204-1224`, `:1430`). Voice identity may key on `char_id` directly -- **no "derive stable id" prerequisite.** Caveat: the *ledger line* char_id is a name->id lookup built post-rename with a no-duplicate-name assert (`OTR_LedgerScriptWriter.py:754/2526-2553`); do not add a rename pass after that index is built.
- **V7 -- legacy age is already a no-op on the default path.** `python_assign_voice_preset` (`_otr_casting.py:807-891`) ranks gender->timbre->age-band->one `rng.choice`, but pool mode passes `age_band=None` so the age filter is inert (C7 byte-identical) on the default path (`:870-872`); age only changes the draw on the CastPlanner/`llm_slot_fill` path. Keep age (do not delete); the PD1 risk is planner-path-only.
- **V8 -- additive cast schema already proven in production; no strict validator.** The ledger is plain-dict-backed (no pydantic/jsonschema, no `extra=forbid`; `save()` merges unknown fields, `production_ledger.py:980-1001`). `tts_model` + `voice_params` ALREADY exist as nullable fields on every cast row (`cast_pools.py:432/438/457/463`, `_otr_casting.py:1405/1411`). **There is no `CastCharacter` class** -- cast rows are plain dicts; the pydantic models are `DescriptionResponse`/`CastingResponse` (`extra="ignore"`). Adding `voice_ref_id` + `commercial_clean` is byte-safe and trivial.
- **V9 -- voice assignment is computed once at `lock_cast` and stamped on the frozen cast** (`_otr_casting.py:1396-1417`, read by Bark `:1627-1631`); never re-derived at render. The new caster mirrors this.
- **V10 -- `beat_intent` is freeform prose (4-200 chars), NOT enumerable** (`_otr_outline.py:102-107`). The enumerable, persisted fields are **`arc_phase`** (setup/complication/resolution/climax, `_otr_outline.py:125`) and `speaker_role`. Any structured-delivery hybrid (B.2) keys on `arc_phase`, not `beat_intent`.
- **V11 -- `line_id` is deterministic** (`= beat_id = b{NNN}`, pinned `^b\d{3}$`, uniqueness-tested). Usable in a stable per-line seed; positional (an inserted beat shifts later ids), not a content hash.
- **V12 -- native-stereo theme already survives the assembler.** EpisodeAssembler only pads channels UP; a stereo segment passes through unmodified (`scene_sequencer.py:1049-1054`, `_extract_waveform:1547` never downmixes). **=> S13 is not needed for theme stereo; defer it (no theme re-baseline).**
- **V13 -- AudioEnhance widen mis-processes a true-stereo VOICE input.** `_mono_to_stereo` guards (already-stereo passes, `audio_enhance.py:60`), but `_haas_delay`/`_stereo_decorrelate` gate on "is stereo," not "was input stereo" (`:67/:86/:390/:400`) -> a native-stereo source gets re-widened. Only matters if a native-stereo *voice* engine is enabled; then add an "input-was-stereo" guard. Theme bypasses AudioEnhance, so theme is unaffected.
- **V14 -- the legacy `script_json` round-trip is verified SAFE.** All three legacy nodes consume via `_OTRLC.load_ledger` = `json.loads` only; the raw string is never hashed/cached/byte-compared (`batch_bark_generator.py:428`, `kokoro_announcer.py:176`, `musicgen_theme.py:544`, `_otr_ledger_consumers.py:51-68`). `[H1-string]` is belt-and-suspenders, not a correctness gate. (Still forward the original string -- zero-risk.)
- **V15 -- HuMo 16k-mono must be INTERNAL to node 51, Whisper-encode-only.** The assembled `audio` is used at three sites: Whisper encode (`batch_humo_render.py:2350-2353`), **mp4 mux** (`_save_clip_via_ffmpeg`, `:935/:1013`), and duration/slice math (`:1986/:2244`). A 16k-mono node between 7->51 would degrade the muxed video audio (PD1). S0.1 downmixes+resamples ONLY the chunk feeding `AudioEncoderEncode`, leaving mux/duration audio full-SR.
- **V16 -- HuMo frame-count is duration-based, resample-safe** (`humo_length_for_dur(dur_s)`, `:513/:2125`). A 48->16k resample at the encode cannot shift frame boundaries.
- **V17 -- VRAM flush must cooperate with `comfy.model_management`.** HuMo uses `mm.unload_all_models`/`soft_empty_cache`/`load_models_gpu` (`batch_humo_render.py:2469/231/171`). `_flush_vram_keep_llm()` (`story_orchestrator.py:63-83`) is the light keep-LLM flush; `force_vram_offload` is `_vram_log.py`. A new engine's flush must use Comfy's manager, not raw `torch.cuda.empty_cache()` only.

---

## 1. Goal & scope
Model-agnostic per-role audio-engine registry **plus** its upstream **voice-casting subsystem**: **music + announcer_voice + character_voice**, casting resolved Python-side per a versioned registry. Good narration is the priority. **SFX out** (separate S12). Stereo deferred per V12.

## 2. Locked decisions (carried; #5 refined)
1-8 unchanged from the consolidated plan. **#5 refinement:** keyword delivery vector for v2; the deferred structured source (B.2) keys on **`arc_phase`** (enumerable, persisted), not `beat_intent` (prose, per V10); LLM-emitted rejected. **#8:** re-baselines SEQUENCED, profile-named.

## 3. Prime-directive constraints (hardened + verified)
- **PD1 [H1] raw delegation** + **[H1-manifest]** legacy widget tuple sourced from `config/legacy_invocation_manifest.json` (recorded from the canonical JSON, test-asserted to match) + **[H1-string]** forward the original `script_json` string (verified safe per V14, kept zero-risk) + **[H1+]** never `canonical_audio(48000)` the 24k legacy output.
- **PD2 [H8] VRAM:** process-global `AUDIO_ENGINE_LOCK` + single-residency session; `defrag_cuda()` before-load AND after-unload; **flush via `comfy.model_management` (V17)** + `_flush_vram_keep_llm()`, not raw torch; measured peak (TF32-off, Stable Audio full T5 pipeline, post-video-load).
- **PD3:** new nodes only in the opt-in workflow copy; default frozen.
- **PD6:** engines + delivery vector + delivery profiles + `prepare_text` add no LLM call.
- **[H9] determinism:** `CUBLAS_WORKSPACE_CONFIG=:4096:8` set **before CUDA init** via a startup bootstrap (`assert_determinism_env_ready()` raises at import if unset); baseline uses `warn_only=False`; **S11 pilots strict determinism against the real forward passes** and records a documented reduced-determinism fallback if sm_120/cu130 kernels hard-crash. Bit-exact opt-in baselines are NOT assumed until S11 proves them (§D4).
- **Guardrails:** `engine` key clears b6; avoid `_s28` patterns.

## 4. Architecture (hardened)
`[H3]` flag-independent stable `engines_for_role` + `DEFAULT_ENGINE_BY_ROLE`. `[H2]` `assert_usable` fails closed. `[H7]` typed `VoiceRequest`/`MusicRequest`, interface-blind node; `prepare_text -> str` text-only; `VoiceRequest` carries `stage_directions`. `[R2#9] engine profiles` (`config/audio_engine_profiles.yaml`: model_path/sha/default_params) -- the dropdown shows names, the request resolves a profile, so a swap is config+adapter. (Adopt as its own small sprint between D and G -- §D5.)

---

## 5. Voice-casting subsystem (the upstream work -- NEW, folded in)

**Two casters, side by side (the PD1-safe core):**
```
LEGACY (FROZEN, default path): python_assign_voice_preset
  gender -> timbre -> age-band(no-op in pool mode, V7) -> rng.choice  over the Bark pool
  -> voice_preset="v2/en_speaker_*", stamped once at lock_cast (V9). UNTOUCHED.
NEW (opt-in path): assign_voice_for_slot
  commercial-filtered -> gender -> timbre -> role -> age(light) -> stable sort -> seeded rng.choice
  over the voice registry; char_id-keyed; -> voice_ref_id, stamped at lock_cast like the legacy one.
```
The default workflow keeps the legacy caster verbatim; only the opt-in workflow routes character casting through the new one. This is what keeps the whole change byte-safe.

**Cast contract (mostly already exists, V8):** plain-dict cast rows already carry `char_id` (stable, V6), `name`, `gender`, `casting_traits`, `tts_model`, `voice_preset`, `voice_params`. **Add two nullable fields:** `voice_ref_id` (ref-clip engines only) and `commercial_clean`. Additive, byte-neutral on the default path (Bark rows: `tts_model="bark"`, `voice_ref_id=null`, `commercial_clean` per engine). Voice keys on `char_id`, never `name`.

**Voice registry:** `voice_id, tts_model, gender/timbre/role/age tags, ref_audio_path, default_voice_params, commercial_clean, model_id/model_sha256/library_version/adapter_version`. `get_all_registered_voices()` returns a **stable `voice_id`-sorted** list (reproducible tie-break). `voice_reference_bank_sha256` (from .wav bytes at load) enters the cache key.

**New caster** = the spec's `assign_voice_for_slot`: commercial gate at source (`allow_noncommercial` bypass), collision set, gender(100)/timbre(40)/role(20)/age(10) scoring as a frozen `casting_policy_v1`, stable candidate order + one seeded `rng.choice` (seed from episode-seed + fixed casting salt, never `hash()`), result stamped per `char_id` on the frozen cast.

**Delivery profiles (Locked-5-safe lever):** a named episode-level deterministic transform of the keyword v1 vector before per-engine projection (`neutral / radio_drama / heightened_noir / subtle_documentary / emergency_broadcast`), a frozen versioned table; `delivery_profile`+`delivery_profile_version` enter the cache key. Independent of B.2.

**Commercial-clean release gate:** stamp `commercial_clean` on the cast row + `audio_meta` + WAV metadata; promotion pre-flight `assert_release_clean(ledger)` **fails a release build** if any voice is non-clean (an IndexTTS2 render can't ship by accident).

**Widget surface (character-voice node):** `voice_bank`, `cast_voice_policy` (preserve_ledger|auto_registry|manual_overrides), `manual_voice_assignments_json` (per char_id, respects the commercial gate), `delivery_profile`, `allow_noncommercial_models`, `cache_enabled`. **Reject** `voice_engine_mode` enum (anti-model-agnostic -- keep the per-role registry dropdown) and `deterministic_inference`-as-widget (it's a startup global).

**Engine routing:** Bark = legacy characters + PD1 baseline + fallback; Kokoro = announcer (single pinned clip, [H13]); Chatterbox = commercial-clean character workhorse; IndexTTS2 = research-only behind `allow_noncommercial_models`/`OTR_ENABLE_INDEXTTS2`, non-commercial-stamped; Stable Audio/MusicGen = music/theme only, never a voice.

---

## 6. Sprints

**A/B/C/C.1 [SHIPPED]** -- carry the retrofits (DEFAULT_ENGINE_BY_ROLE+stable-order, fail-closed assert_usable, import-audit, typed requests, PreparedLine). **B stays keyword for v2** (do not fold arc_phase now; B.2 later).

**D -- generic voice + theme nodes [NEXT].** Registry-driven nodes, default=legacy. `[H1]` raw-delegation batch branch (manifest widgets, output verbatim, no canonical_audio); per-line branch builds typed requests. Per V1 no SceneSequencer decoupling; per V14 round-trip is safe but forward the original string. Tests: byte-identity passthrough (value, manifest widgets) + no-mutation, fail-closed, stable dropdown, default==legacy.

**E.0-E.5 -- voice-casting subsystem.** E.0 add `voice_ref_id`+`commercial_clean` (additive, V8) + affirm char_id keying (V6). E.1 voice registry + reference bank (LibriTTS-R primary; `ref_sha256` from .wav bytes; stable-sorted pools). E.2 the NEW caster beside the frozen legacy one (V9 pattern). E.3 delivery profiles. E.4 widget surface. E.5 commercial release gate. Opt-in workflow copy; link graph untouched; `[H14]` removed (V1).

**S0.1 -- HuMo 16k-mono [PRESENT-TENSE].** Per V15: INTERNAL to node 51, Whisper-encode-only; pin downmix method, preserve duration (V16 frame-safe); must NOT touch `episode_audio`/mux/duration. Tests: shape/sr/episode-audio-hash-unchanged/token-alignment. Do before enabling a non-Bark voice that changes the master.

**F -- operator dependency pilots (GPU).** Isolated venv import (xformers/torch-change HARD-fail; offline env enforced); SageAttention OFF for Stable Audio (+ V4 no enhancer; record T5 size for VRAM honesty); **determinism pilot** against real forward passes (strict vs documented fallback); render-twice **cross-process** (engine); measured peak; watermark policy=disabled unless licensing requires; **Chatterbox param names (`exaggeration`/`cfg_weight` vs `cfg`) pinned here** (FC-3, unverifiable from repo).

**G -- wire opt-in inference.** Real `generate_voice`/`generate_music` inside `deterministic_inference`; free-run duration + guardrails; `[R2#5]` `stable_line_seed(base, role, speaker_id, line_id, start_s_3dp, text_sha)` (line_id deterministic per V11); `[H5b]` full canonical cache key (sorted/quantized floats; adapter source-file hash; lib version; model_sha; ref_sha + bank_sha; prepared text; delivery+profile+versions; seed; params; sr; stereo_policy; watermark; driver/torch in metadata not key).

**H -- native stereo [DEFERRED per V12].** Theme already survives the assembler -> reclassify S13 to optional polish; only needed if a native-stereo VOICE engine is enabled, and then the work is the AudioEnhance widen guard (V13), not SceneSequencer. Keep `mono_safe` bridge. If built: mono-path byte-exact branch + episode-level cross-process render-twice + new v2 capture.

**I -- promotion + re-baseline.** Profile-named baselines; sequenced (stereo then SFX); commercial-clean pre-flight gate; legacy permanent fallback.

**S12 (separate) -- SFX cleanbreak.** Unchanged; delete the inert `default_tts` widget here.

---

## 7. Decisions to confirm (my calls)
- **D1 delivery source:** keyword v2; **B.2 structured keys on `arc_phase`** (V10), not beat_intent; LLM-emitted rejected. Do not fold anything into the v2 vector.
- **D2 re-baseline:** sequence, profile-named.
- **D3 stereo:** **DEFER S13** (theme already survives, V12); reclassify to optional polish; revisit only with a native-stereo voice engine (then AudioEnhance widen guard, V13). *Removes a sprint + a re-baseline.*
- **D4 PD1 bit-exact for opt-in:** pursue pinned-config bit-exact; per-op documented tolerance fallback only if S11 proves a kernel can't; legacy stays SHA-exact.
- **D5 engine profiles:** adopt as a small sprint between D and G.
- **D6 delivery profiles:** in for v2, or defer to v2.1? (small, deterministic, free of B.2, but a new baseline dimension.)
- **D7 age in the new bank:** keep light (10 pts); legacy untouched (and inert on the default path per V7).

## 8. Verify-in-pilot (only items not resolvable from the repo)
Chatterbox exact param names (`exaggeration`/`cfg_weight` vs `cfg`) -- S11. Stable Audio T5 conditioner size + no prompt-enhancer -- S11 VRAM honesty. Strict-determinism survival on sm_120/cu130 -- S11 (drives D4). Everything else in both source docs' "verify against repo" lists is RESOLVED in §0.

## 9. Evidence appendix (this round)
char_id stable `_otr_casting.py:1063/1082/1096/1204-1224/1430`; line char_id name-lookup `OTR_LedgerScriptWriter.py:754/2526-2553`. age no-op pool mode `_otr_casting.py:870-872`. assignment stamped `_otr_casting.py:1396-1417/1627-1631`. plain-dict ledger + merge `production_ledger.py:980-1001`; tts_model/voice_params already present `cast_pools.py:432/438/457/463`. line_id `production_ledger.py:764`+`_otr_outline.py:1457-1461`. beat_intent prose `_otr_outline.py:102-107`; arc_phase enumerable `:125`. assembler pad-up `scene_sequencer.py:1049-1054/1547`. AudioEnhance widen `audio_enhance.py:60/67/86/390/400`. legacy json.loads `batch_bark_generator.py:428`/`kokoro_announcer.py:176`/`musicgen_theme.py:544`. HuMo audio 3 sites `batch_humo_render.py:2350/935/1986`; frame math `:513/2125`; comfy mm `:2469/231/171`; `_flush_vram_keep_llm` `story_orchestrator.py:63-83`.

**Commits this track:** A `9b76d78`, B `1b5a39b`, C `c79cc51`, ROADMAP `2161439`, C.1 `f49d4f9`. Full `tests/` green at each (3440 / 12 skipped).
