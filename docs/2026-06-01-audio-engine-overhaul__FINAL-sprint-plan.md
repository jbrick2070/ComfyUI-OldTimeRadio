# OTR Audio-Engine Overhaul -- FINAL Sprint Plan (code-verified)

**Date:** 2026-06-01 | **Branch:** `v2.0-alpha` | **Status:** A/B/C/C.1 SHIPPED; D onward open
**Supersedes:** `audio_engine_sprint_plan__HARDENED_consolidated.md` (the 3-review merge).
**This pass:** every reviewer claim about the real code was verified against the live files + the on-disk workflow JSON (subagent, Desktop Commander). Corrections in §0 override the hardened plan where they conflict. The hardened plan's [H#] tags are preserved for traceability.

---

## 0. Verification corrections (these override the hardened plan)

**V1 -- [H14] SceneSequencer `'bark'` is NOT a blocker. DROPPED.**
`OTR_SceneSequencer` declares a `default_tts` combo defaulting to `"bark"` (`nodes/scene_sequencer.py:490-493`) but **never reads it** -- the method comment at `:549` states per-line routing reads `voice_assignments` / `cast.voice_preset`, not the widget. Character clips arrive on the `tts_audio_clips` AUDIO bus and are sequenced by ledger `start_s` regardless of which engine produced them, so the bus is **already engine-agnostic**. No SceneSequencer "decoupling" is required before a non-Bark voice. *Optional tidy:* delete the dead `default_tts` widget (its own micro-task; not gating).

**V2 -- the master is ALREADY stereo; reframe the stereo sprint and pull HuMo-16k forward.**
`OTR_AudioEnhance` (node 4) unconditionally widens mono->stereo (`nodes/audio_enhance.py:374` `_mono_to_stereo`, `:391` Haas, `:401` mid-side; node wired `spatial_width=0.3`) BEFORE `EpisodeAssembler` (which only pads channels up, `scene_sequencer.py:1049-1054`). So the assembled/episode master is **48 kHz stereo today**, and **HuMo already consumes a stereo master** (node 51 `audio` <- link 78 <- node 7 `episode_audio`; encoder <- node 72 `whisper_large_v3_fp16`). Consequences:
- "Stereo end to end" is not a greenfield mono->stereo migration. The real goal (S13): make SceneSequencer's **per-clip path channel-aware** so *native* stereo from Stable Audio / Chatterbox survives to AudioEnhance instead of being mono-downmixed and re-synthesized as fake width.
- **HuMo 16 kHz-mono extraction is PRESENT-TENSE, not future.** ComfyUI's `AudioEncoderEncode` is either silently downmixing the stereo master for Whisper today (working by luck) or it is a latent correctness risk. Add an explicit dedicated 16 kHz-mono derivation feeding node 51's Whisper encode, and verify token alignment -- **now**, as a standalone hardening item (S0.1 below), independent of S13.
- Re-baseline blast radius is **smaller** than the hardened plan assumed: the legacy/default path (mono engines -> AudioEnhance widen) is unchanged by S13, so default byte-identity holds without a re-baseline; only the new native-stereo engine outputs need the v2 capture.

**V3 -- [H10] structured-field delivery vector needs a persistence prerequisite; keep keyword v1.**
The drama fields the hardened plan wants to source from (`beat_objective`, `beat_turn`, `next_turn`, `beat_tension`, `dramatic_question`) live only on the prompt-time `LineRequest` (`nodes/_otr_line_composer.py:695-701`), are consumed only to build the prompt (`:1134-1154`), and are **discarded** -- the composer returns the text string and nothing else (`:1747`). The frozen ledger line schema (`nodes/production_ledger.py:763-788`) persists `beat_intent` + `target_words` but **none of those five fields**; the freeze module has zero references to them. So "derive the vector from structured fields, additively, post-freeze" is **not available off the shelf**.
- **Decision:** the shipped keyword-based delivery vector (Sprint B, `1b5a39b`) stays for v1 -- it *is* truly additive, post-freeze, deterministic, and works today. The reviewers' quality critique (keyword mis-handles negation / homographs / sarcasm) is valid but its fix has a prerequisite they could not see.
- **Available-now partial:** `beat_intent` IS persisted on the line -- a `beat_intent`-informed + keyword-fallback hybrid is additively available with no new persistence. **Full** structured sourcing = a later sprint (B.2) that first stamps the five drama fields onto each line row at write-time (schema-additive, survives freeze); that is a writer-side change with its own blast radius, NOT a Sprint B retrofit.

**V4 -- [H8] Stable Audio "Qwen reprompt" is not a local concern.** No such enhancer/reprompt model is referenced anywhere in the repo; the local engine takes the raw prompt directly (`nodes/_otr_audio_engines/eng_stable_audio.py:47-55`). Downgrade to a **pilot-time note** (confirm the chosen Stable Audio ComfyUI integration doesn't pull a prompt-enhancer model), not a residency requirement.

**V5 -- SR ladder confirmed.** 48 kHz standardize at `scene_sequencer.py:602`; Bark 24 kHz (`batch_bark_generator.py:713`); the 24->48 upcap happens once via AudioEnhance (`audio_enhance.py:369`). The raw-delegation "do not double-resample the legacy path" discipline [H1+] is correct and stands.

*Method note:* the canonical workflow JSON is NOT truncated on disk (46,913 bytes, 29 nodes, `last_node_id:79`) -- the earlier "truncation" was only the Read-tool display cap. A sandbox Python probe reads the full file; use that for future JSON verification.

---

## 1. Goal & scope
Model-agnostic per-role audio-engine registry: **music + announcer_voice + character_voice**. Good narration is the priority. **SFX out** (separate S12 cleanbreak). Stereo per V2.

## 2. Locked decisions (carried; only #5 source corrected)
1. Default-OFF opt-in lanes; legacy stays the byte-identical default.
2. Fresh registry under `nodes/_otr_audio_engines/` (not the dead stub).
3. IndexTTS2 non-commercial -> Chatterbox (MIT) is the character default; IndexTTS2 flag-gated forever.
4. Announcer and character are independent slots, shared pool.
5. Deterministic 8-dim delivery vector, additive post-freeze, no-LLM, no-RNG. **v1 source = keyword (shipped); structured-field source deferred to B.2 (needs persistence, per V3).**
6. Downstream per-engine `prepare_text`; canonical script stays engine-neutral.
7. `[B,C,T]` canonical audio; never assume `shape[0]==2`.
8. SFX deletion is its own cleanbreak (S12). **Re-baseline captures SEQUENCED, not paired (see §18.2).**

## 3. Prime-directive constraints (carried, hardened)
- **PD1 [H1] raw delegation:** the generic node hands the legacy node the *unmodified* `script_json` + its *exact* legacy widget tuple, does **zero** transform (no delivery stamp, no `canonical_audio`, no cache, no normalization), returns output verbatim -- a sealed evidence bag. **[H1+]** never route the 24 kHz legacy output through `canonical_audio(48000)` (double-resample breaks the baseline; SceneSequencer/AudioEnhance already do 24->48).
- **PD2 [H8] VRAM:** single-engine residency (`load()` unloads any other resident engine), hard per-engine budget checked at runtime, explicit `gc.collect()` + `torch.cuda.empty_cache()` + `torch.cuda.ipc_collect()` after each engine batch. Peak for the worst-case 3-role opt-in graph is **measured in S11**, not assumed. (Qwen-reprompt item -> pilot note per V4.)
- **PD3:** new nodes only in the opt-in copy `workflows/otr_scifi_16gb_audio_v2_optin.json`; default frozen.
- **PD6:** engines + delivery vector + `prepare_text` add no LLM call.
- **[H9] determinism context (NEW, critical for cu130/Blackwell):** opt-in inference runs inside `deterministic_inference(seed)` -- `CUBLAS_WORKSPACE_CONFIG=:4096:8`, TF32 off (matmul+cuDNN), `cudnn.deterministic=True/benchmark=False`, `torch.use_deterministic_algorithms(True, warn_only=True)`, CPU+CUDA seeds, explicit `torch.Generator("cuda")` for diffusion. Without it `baseline_v2.wav` is unreproducible run-to-run.
- **Guardrails:** `engine` key clears `test_b6_wiring_guardrails`; avoid `_s28_forbidden_sweep` patterns in runtime lines.

## 4. Architecture (carried, hardened)
- **[H3] `engines_for_role` is flag-independent + stable (legacy first)**; default pinned by an explicit `DEFAULT_ENGINE_BY_ROLE = {"character_voice":"bark","announcer_voice":"kokoro","music":"musicgen"}` constant, not an emergent sort. (My shipped `engines_for_role` is already flag-independent + stable-sorted; this adds the explicit constant + a stable-order test -- a clarity/safety retrofit, not a bug fix.)
- **[H2] `assert_usable` FAILS CLOSED:** explicit opt-in selection with its flag off **raises** (no silent fallback to Bark); fallback only behind `OTR_AUDIO_ENGINE_ALLOW_FALLBACK=1` with a loud log. *(Changes my shipped silent-fallback `assert_usable` + `test_assert_usable_optin_off_resolves_to_default` -- a deliberate, good retrofit.)*
- **[H7] typed `VoiceRequest` / `MusicRequest`, interface-blind node:** the node builds a typed request and calls one method per family; interface variance stays behind the adapter, so the next engine wanting a negative prompt / BPM / loopability / different ref shape is one adapter, not a node rewrite. *(Touches the shipped Sprint C adapter stubs -- cheap now, no real inference yet.)*

## 5-8. Shipped sprints (carry the retrofits)
- **A `9b76d78`** registry + utils. Retrofit: `DEFAULT_ENGINE_BY_ROLE` + stable-order test [H3]; fail-closed `assert_usable` [H2].
- **B `1b5a39b`** delivery vector. **Per V3: keep keyword source for v1.** Add B.2 later for structured-field source (needs write-time persistence of the drama fields).
- **C `c79cc51`** adapters. Retrofit: import-audit test [H5] (`import nodes._otr_audio_engines` must not import chatterbox/indextts/stable_audio_tools/xformers); migrate stubs to `VoiceRequest`/`MusicRequest` [H7].
- **C.1 `f49d4f9`** prepare_text. Retrofit [H11]: emit `PreparedLine(spoken_text, stage_directions, parens, delivery_vector)` -- preserve expressive cues for engines to opt into, don't globally delete them.

## 9. Sprint D -- generic voice + theme nodes  [NEXT]
Files: `batch_character_voices.py` (default `bark`), `announcer_voice.py` (default `kokoro`), `stable_audio_theme.py` (default `musicgen`).
INPUT_TYPES: `script_json` (forceInput), `engine` (stable list, legacy first), role params, `seed`, `stereo_policy` (`["mono_safe","preserve_stereo"]`, default `mono_safe`).
**[H1] dispatch:** if resolved engine `interface=="batch"` -> `make_batch_node().generate(script_json, *legacy_exact_widgets)` raw, no transform, output verbatim (NO `canonical_audio`). Else per-line: `stamp_delivery_vectors(json.loads(script_json))` (per-line path only), build `VoiceRequest`/`MusicRequest`, `prepare_text` -> `generate_voice`/`generate_music`, pack to the Bark AUDIO contract.
**Per V1: no SceneSequencer decoupling needed** -- the clip bus is engine-agnostic. (Optionally delete the dead `default_tts` widget as a separate micro-task.)
Tests [H1/H2/H3]: byte-identity passthrough (sha256 value, all 3 legacy engines) + no-mutation of `script_json`; opt-in-without-flag raises; dropdown order stable across flag toggles; default==legacy.
Exit: full `tests/` green; default workflow still names the legacy nodes; passthrough + no-mutation green.

## 9.1 Sprint S0.1 -- HuMo 16 kHz-mono extraction  [PRESENT-TENSE hardening, per V2]
Independent of stereo. Add an explicit 48 kHz-stereo -> 16 kHz-mono derivation feeding node 51's Whisper `AudioEncoderEncode`; verify Whisper token alignment + no cu130 tensor-layout panic. Confirm whether ComfyUI's encoder already downmixes (then this is defensive/explicit) or not (then this fixes a latent bug). Do before any stereo work and before enabling a non-Bark voice that changes the master.

## 10. Sprint E -- opt-in workflow copy + reference-voice bank
Opt-in workflow copy (patch node types only, link graph untouched); re-run `OTR_WorkflowValidator`; update seed-target test. **[H14] removed -- not needed (V1).**
Reference-voice bank: `config/voice_reference_bank.json` (Bark preset -> license-clean 3-10 s WAV + gender/timbre/age/role + **ref SHA-256**) + validator. `python_assign_voice_ref` mirrors `python_assign_voice_preset` (one `rng.choice`, C7). **[H13]** announcer ref = single pinned clip by fixed id (one consistent voice), not a pool draw.

## 11. Sprint F -- operator dependency pilots (GPU)
Isolated venv import test per opt-in lib (no xformers / no torch downgrade off cu130 / no cu121-124 pin / no transformers conflict / no startup crash; diff `pip freeze`). Stable Audio with **SageAttention OFF**; **V4: confirm the SA integration pulls no prompt-enhancer model.** **[H9] render-twice determinism gate** (byte-identical twice per engine). **[H8] VRAM gate** (measured peak, 3-role worst case). **[H11b] Chatterbox watermark** (`resemble-perth` on Py3.12): pin-and-require or disable -- byte-identity must not depend on an optional package; record in baseline metadata. IndexTTS2 license confirmed before non-test use.

## 12. Sprint G -- wire opt-in inference + prepare_text tuning
Real `generate_voice`/`generate_music` inside `deterministic_inference`; tune `prepare_text`; free-run duration (no token-pin -- HuMo lip-syncs to the line; guardrails: min/max line dur, max episode dur, line timeout, compression only for extreme outliers). **[H5b] full cache key** on **prepared** text + adapter_version + model_id + model_sha256 + ref_sha256 + delivery + delivery_version + seed + params + sr + stereo_policy + normalization; driver/torch/SageAttention in **baseline metadata, not the key** [H6]. Exit: a flagged opt-in episode renders end to end.

## 13. Sprint H -- native stereo through the per-clip path  [re-baseline, reframed per V2]
Make `SceneSequencer._extract_clips_from_audio` (`:508`) + `_resample_audio` (`:109`) channel-aware so native stereo from Stable Audio / Chatterbox survives to AudioEnhance (the master is already stereo; this stops the mono-downmix-then-fake-widen). `mono_safe` stays the bridge until this lands. Default/mono path is unchanged -> only the native-stereo engine outputs need the v2 capture. Exit: native stereo preserved; HuMo still gets clean 16 kHz mono (S0.1); new `baseline_v2_stereo.wav`.

## 14. Sprint I -- promotion + re-baseline runbook
Operator: set flags, `OTR_REGRESSION_RUNTIME=1`, render once per engine, save `baseline_v2_<key>.wav` + `.sha256`; flip defaults; tag. Rollback = inverse flag flip. Legacy engines = permanent fallback.

## (separate) S12 -- SFX cleanbreak
Unchanged. Re-baseline **sequenced** after stereo (§18.2).

## 15. Risks (corrected)
xformers contamination (S11 pilot) | IndexTTS2 non-commercial (`allowed_for_release=False` metadata stamp) | Stable Audio x SageAttention (off; record) | nondeterministic CUDA ops (`deterministic_inference` + render-twice) | VRAM frag (single-residency + defrag + measured) | **HuMo stereo-master TODAY (V2 -> S0.1 16k-mono extraction, present-tense)** | reference-clip licensing (open-license/self-recorded + SHA) | determinism = re-baseline trigger on kernel/driver change.

## 16. Decisions resolved (from the review, kept)
Q1 voice-bank source: **curated open-license REAL human speech (LibriTTS-R) primary, self-recorded control, synthesized rejected** (cloning-from-synthetic compounds artifacts); ~12 anchors, ship 5-7. Q2 stereo: **isolate (S13), after opt-in inference, before promotion** (V2 makes it native-stereo-preservation, smaller). Q3 per-engine LLM doctor: **No** (breaks C7/PD2/PD6). Q4 duration: **free-run + audio-driven shot length** + guardrails. Q5 **Chatterbox is the workhorse** (legally shippable); IndexTTS2 personal/non-commercial. Q6 **Kokoro stays announcer default**. Q7 **full cache key** [H5b].

## 17. Test / file map -- carried from hardened §17 plus: **`test_humo_16k_mono_extraction` (S0.1)**, and **`test_delivery_vector` stays keyword-sourced for v1** (structured-source test lands with B.2).

## 18. Decisions to confirm (your call)
1. **Delivery-vector source.** v1 = keyword (shipped, truly additive). I am **more conservative than the hardened plan**: per V3 even the "structured fields" version needs a write-time persistence prerequisite (B.2), and the LLM-emitted version (reviewer C) changes Locked-5 + forces a writer re-baseline + is writer-model-dependent -> **reject for v2**. Recommendation: ship keyword for v2; do B.2 (persist beat fields -> structured source) as a deliberate post-v2 sprint; LLM-emitted only as a separately-baselined experiment.
2. **Re-baseline pairing.** **SEQUENCE, do not pair:** `baseline_v2_stereo.wav` (verify) then `baseline_v3_no_sfx.wav`. Merged, a determinism regression on the operator-only runtime gate is unbisectable. Cost: one extra capture; benefit: bisectability. (Agrees with hardened §18.2; V2 further shrinks the stereo capture to opt-in engines only.)
3. **Dead `default_tts` widget (new, minor).** Per V1 it controls nothing -- leave it (zero-risk) or delete it in the S12/cleanbreak. Your call; non-gating.

---

## Appendix -- verification evidence (file:line)
- default_tts inert: `scene_sequencer.py:490-493` (widget), `:549` (comment), routing `:654-744`; JSON node 3 `widgets_values[...,"bark",...]` unused.
- master already stereo: `audio_enhance.py:374/391/401`; assembler pad-up `scene_sequencer.py:1049-1054`; HuMo audio `batch_humo_render.py:1319-1323`, encode `:2345-2354`; JSON node 51 audio<-link78<-node7, encoder<-link214<-node72 (`whisper_large_v3_fp16`); 16k-mono contract `scripts/render_humo_batch.py:338,348,356,364`.
- drama fields not persisted: `_otr_line_composer.py:695-701` (LineRequest), `:1134-1154` (prompt-only use), `:1747` (returns text); ledger schema `production_ledger.py:763-788` (has beat_intent/target_words, not the five); writer writeback `OTR_LedgerScriptWriter.py:3555-3562`; freeze module zero refs.
- SR ladder: `scene_sequencer.py:602` (48k), `batch_bark_generator.py:713` (24k), `audio_enhance.py:369` (24->48 upcap).
- Stable Audio no-enhancer: `_otr_audio_engines/eng_stable_audio.py:47-55`; "Qwen" hits are all unrelated (LLM writer / license docs / comfy slug).

**Commits this track:** A `9b76d78`, B `1b5a39b`, C `c79cc51`, ROADMAP-decouple `2161439`, C.1 `f49d4f9`. Full `tests/` green at each (3440 passed / 12 skipped).
