# Problem Statement -- Audio Generation Stack Overhaul (Bark -> IndexTTS2, MusicGen/AudioGen -> Stable Audio 3)

**Date:** 2026-06-01
**Branch:** `v2.0-alpha`
**Author:** Jeffrey A. Brick (problem framed); code review + research by Claude
**Status:** DRAFT for expert round-robin (Jeffrey runs the synthesis)

---

## 1. Goal

Replace OTR's three generated-audio engines with newer, more expressive, commercially-cleaner models, **without breaking the "audio is king" pipeline**:

1. **TTS (dialogue):** Bark -> **IndexTTS2** (zero-shot voice cloning + emotion control + duration control).
2. **Music (theme):** MusicGen (`facebook/musicgen-medium`) -> **Stable Audio 3** (Small-Music / Medium).
3. **SFX (foley):** AudioGen (`facebook/audiogen-medium`) -> **Stable Audio 3** (Small-SFX) -- *only if it fits.*

This is a feature-track change, not a hotfix. It must land under the Prime Directives (PD1 audio integrity, PD2 14.5 GB VRAM ceiling, PD3 workflow-JSON wiring, PD6 LLM-slot tagging) and survive the Blackwell / sm_120 / cu130-nightly / Windows stack.

---

## 2. Current state (grounded in code)

### 2.1 Pipeline order
`LedgerScriptWriter -> FreezeCascade -> BatchBark -> SceneSequencer -> AudioEnhance -> EpisodeAssembler` (`__init__.py:13`). Visual (HuMo/LTX/FLUX) consumes the assembled audio downstream.

### 2.2 The three engines today

| Engine | Node (class / id) | Model | File | Output | SR |
|---|---|---|---|---|---|
| Dialogue TTS | `OTR_BatchBarkGenerator` (node 11) | `suno/bark` | `nodes/batch_bark_generator.py` | `("AUDIO","STRING")` = `tts_audio_clips` | 24000 |
| Announcer TTS | `OTR_KokoroAnnouncer` | `hexgrad/Kokoro-82M` | `nodes/kokoro_announcer.py` | `announcer_audio_clips` | 24000 |
| Theme music | `OTR_MusicGenTheme` (node 14) | `facebook/musicgen-medium` | `nodes/musicgen_theme.py` | `("AUDIO","AUDIO","AUDIO","STRING")` open/close/+log | MUSICGEN_SAMPLE_RATE |
| SFX/foley | `OTR_BatchAudioGenGenerator` (node 15) | `facebook/audiogen-medium` | `nodes/batch_audiogen_generator.py` | `("AUDIO","STRING")` = `sfx_audio_clips` | 32000 |

All clips converge in `OTR_SceneSequencer` (`nodes/scene_sequencer.py:445`), which **resamples everything to 48000** (line 602), adds room tone, applies offsets. `OTR_EpisodeAssembler` (`scene_sequencer.py:949`) prepends/appends theme with crossfades and peak-normalizes to **-1.0 dBFS**. No `.wav` is persisted by a node in the normal flow; the final mux is done by the video node (FFmpeg). The PD1 baseline WAV is captured by the test harness `tests/_run_baseline.py`.

### 2.3 Backend abstraction = STUB only (important)
There is a `nodes/_voice_backends/` package (`_protocol.py` `VoiceBackend` Protocol: `load/generate/unload`, returns WAV bytes; registry in `__init__.py`; `KNOWN_ENGINES={bark,kokoro,cosyvoice,xtts,piper}`) and a parser `nodes/_otr_voice_resolver.py` (`parse_voice_spec("engine:preset")`). **None of it is wired into the render path** -- `bark.py` / `kokoro.py` backends raise `*NotMigrated`. Production dispatch is hardcoded per node. So there is **no live backend-selection seam** to plug IndexTTS2 into; the real template is the Kokoro node pattern (dedicated node, hardcoded engine, emits batched `AUDIO`).

### 2.4 Voice selection (the thing to "re-jigger")
- Authoritative pool: `config/cast_pools.py:253-269` `VOICE_PROFILES` -- **10 fixed Bark presets** `v2/en_speaker_*`, each a tuple `(preset, gender, lang, {traits})` where traits encode vocal quality + **age band** (20s..60s) + role hints. (`en_speaker_7` is the FIX-3 female reclass.)
- Assignment is **pure Python**, byte-deterministic (C7): `nodes/_otr_casting.py` `precompute_ensemble_slots` (645) -> `python_assign_voice_preset` (807-891) ranks by gender -> timbre -> age-band (`_otr_castplanner.AGE_BAND_VOICE_TAGS`) -> one `rng.choice`. `open_voice_pool` (`cast_pools.py:477`) drops taken presets so casts never collide.
- The cast-contract pydantic model carries `voice_preset` as a `v2/en_speaker_*` string; the Bark node's Gate 3 (`batch_bark_generator.py:287-295`) hard-raises if any character line lacks a `v2/`-prefixed preset.

### 2.5 Per-line expressiveness data = DOES NOT EXIST
- `DramaticState` (`nodes/_otr_dramatic_state.py:48-149`) is **episode-level plot architecture only** -- docstring explicitly says it carries no tone/mood/style.
- Line-composer hints (`nodes/_otr_line_composer.py:553` `LineRequest`: `beat_objective`, `beat_turn`, ...) are **prompt-time only and discarded**; they do not persist onto the dialogue line as a delivery instruction.
- Per-line ledger audio fields (`stamp_per_line_audio_meta`) are **forensic/timing only** (`tts_engine`, `voice_preset`, `render_ms`, `dur_s`, `audio_sample_hash`). **Zero affect.**
- => An expressive TTS has nothing to read today. A new per-line emotion/delivery field must be produced and threaded onto the ledger.

### 2.6 PD1 byte-identity gate
`tests/test_audio_byte_identical.py` compares **SHA-256 of the full assembled episode WAV** against `tests/fixtures/baseline_v1.5.wav` (+ `.sha256`), with `FIXED_SEEDS` pinning Bark/Kokoro/AudioGen/MusicGen to 42. The runtime comparison is gated behind `OTR_REGRESSION_RUNTIME=1` (operator-run); always-on tests are structural (fixture integrity, seed-target node presence, workflow validity).

---

## 3. Target engines (research, 2026-06-01)

### 3.1 IndexTTS2 (dialogue TTS)
- **What:** Bilibili IndexTeam, autoregressive zero-shot TTS. HF `IndexTeam/IndexTTS-2` (arXiv 2506.21619). EN + ZH.
- **Voice:** zero-shot clone from a **single 3-10 s reference clip** -- *no preset IDs.*
- **Expressiveness:** 3 emotion-input methods -- (a) **Emo-Text** (infer emotion from text), (b) **Emo-Vector** (8 dims: happy/angry/sad/afraid/disgusted/melancholic/surprised/calm, each 0-1), (c) **Emo-Audio** (mimic an emotion-reference clip). Emotion is disentangled from speaker identity.
- **Duration control:** can pin token count for exact duration, or free-run while preserving prosody. (Useful: HuMo lip-sync needs the clip length to match the line.)
- **VRAM:** >= 8 GB stated for inference; near-real-time on a 4090. Fits the 14.5 GB ceiling with headroom (TTS runs after the LLM writer is unloaded, so the GPU is largely free).
- **ComfyUI:** community nodes exist (e.g. `kana112233/ComfyUI-kaola-IndexTTS2`) -- **NOT native**; Blackwell/cu130/no-xformers compat must be verified.
- **License:** confirm -- IndexTTS-1.5 is Apache-2.0, but IndexTTS-2's repo license is not clearly Apache and may be custom/non-commercial. **Flag for legal-clean check** before shipping.

### 3.2 Stable Audio 3 (theme music; maybe SFX)
- **What:** Stability AI text-to-audio, **native day-0 ComfyUI support** (comfy.org). Stereo. Trained on **fully-licensed data, licensed for commercial use** (good for OTR's legal posture).
- **Variants:** **Small-SFX** (SFX up to 2:00, CPU-capable), **Small-Music** (music loops up to 2:00), **Medium** (structured tracks up to ~6:20). HF `stabilityai/stable-audio-open-1.0` (1.0) + `stable-audio-open-small`; "3" is the current ComfyUI-supported line.
- **VRAM:** Small runs on CPU / tiny VRAM; Medium needs a modest GPU. All well under 16 GB.
- **Fit for OTR:** theme is short (opening/closing) -> Small-Music or Medium; foley is short one-shots -> Small-SFX. One model family can serve **both** music and SFX. **Native ComfyUI = lowest Blackwell-dep risk of the three swaps.**
- **Caveat:** Stable Audio is **stereo**; MusicGen/AudioGen are mono. SceneSequencer/EpisodeAssembler currently assume mono buffers -> a real integration point (stereo handling or downmix).

---

## 4. Cross-cutting constraints and gaps

### 4.1 PD1 is THE central tension
All three swaps change the waveform. The `baseline_v1.5.wav` SHA-256 **will break by design**. PD1 says audio stays byte-identical to baseline at every gate; a change that alters audio "normally triggers revert." => This feature cannot be byte-identical to v1.5. The honest options:
- **(A) Re-baseline:** capture `baseline_v2.wav` after the swap; retire v1.5; PD1 then means "byte-identical to the new IndexTTS2/StableAudio baseline." Re-capture is operator-run on the GPU host.
- **(B) Default-off lane** (mirrors the OpenRouter/Comfy-Credits pattern): land all three new engines behind `OTR_ENABLE_INDEXTTS2` / `OTR_ENABLE_STABLE_AUDIO`, default OFF; Bark/MusicGen/AudioGen stay the byte-identical default; re-baseline only when/if promoted to default. **Recommended starting posture** -- keeps PD1 green on the default path while the new engines mature.

### 4.2 VRAM (PD2 14.5 GB)
IndexTTS2 (~8 GB) and Stable Audio (small) both fit; TTS/music/SFX run after the LLM writer is unloaded. New nodes must use `_flush_vram_keep_llm()` (never `force_vram_offload()` between LLM phases), a 1-token/warmup pass, and their own vram-sentinel budget.

### 4.3 Blackwell / Windows deps
- Stable Audio 3 = native ComfyUI -> low risk.
- IndexTTS2 = community node -> **must verify** it does not pull `xformers` (silently downgrades torch off the cu130 nightly and breaks the env), and that its `transformers`/`torch` pins tolerate the nightly. Pilot in an isolated env first.

### 4.4 PD3 wiring + PD6 model-pick rules
- Each node swap is "not done" until `workflows/otr_scifi_16gb_full.json` is re-wired (class/`type`, AUDIO socket into SceneSequencer / EpisodeAssembler, `FIXED_SEEDS` keys in the PD1 test, seed-target-node test).
- A Stable Audio node may expose a checkpoint **combo** the way AudioGen/MusicGen already do -- set `NON_LLM_MODEL_WIDGET_OK = True` (`batch_audiogen_generator.py:206`, `musicgen_theme.py:482`); a bare `model_id` widget is otherwise rejected by the forbidden-sweep + `_MODEL_WIDGET_KEYS` allowlist.
- The TTS/music/SFX engines are not LLM calls, so the swaps themselves do not trigger PD6. **BUT** generating the new per-line emotion tags (4.5) with an LLM does: tag the call `# LLM slot: creative`, wire the model id from the writer's `creative_writing_model` output socket (no new widget/slot), update the routing table + wiring pin.

### 4.5 Two genuinely new subsystems (the real work, beyond a model swap)
1. **Reference-voice bank** (replaces the preset table): IndexTTS2 needs a 3-10 s reference clip per voice, tagged (gender, timbre, age, role). `VOICE_PROFILES` / `open_voice_pool` / cast-contract `voice_preset` semantics change from `v2/en_speaker_*` -> a reference-clip id/path. Source of the clips must be **license-clean** (curated open-license voice packs, or self-recorded) -- not scraped voices.
2. **Per-line emotion signal** (4.5 §2.5 gap): the writer/line-composer must emit a per-line delivery tag (an Emo-Vector, an Emo-Text phrase, or a per-character Emo-Audio ref) that survives onto the ledger line for the TTS node to read.

---

## 5. Work breakdown (per swap)

**A. Stable Audio 3 for music (node 14) -- lowest risk, do first as proof:**
- New node `OTR_StableAudioTheme` (or re-skin MusicGen node) using the native ComfyUI Stable Audio nodes; `NON_LLM_MODEL_WIDGET_OK` checkpoint combo; keep the per-episode SHA cache keyed on (prompt, duration, seed, model_id); handle stereo; feed EpisodeAssembler. Re-wire JSON + PD1 seed keys.

**B. Stable Audio 3 for SFX (node 15) -- same pattern, "if it fits":** Small-SFX one-shots; same cache + stereo handling; feed SceneSequencer `sfx_audio_clips`.

**C. IndexTTS2 for dialogue (node 11) -- the heavy one:** new `OTR_BatchIndexTTSGenerator` (Kokoro-node template); reference-voice bank + new selection logic; per-line emotion signal + writer/line-composer change (PD6 if LLM-tagged); duration-control to match HuMo lip-sync line length; VRAM sentinel; ledger writeback (reuse `stamp_per_line_audio_meta`); keep Bark text-cleaning lessons (parenthetical stripping, ASCII fold) or re-evaluate since IndexTTS2 handles expressive text differently.

**D. Shared:** byte-identity strategy (4.1), stereo path through SceneSequencer/EpisodeAssembler, default-off flags, full `tests/` + new node tests + PD3 wiring + re-baseline plan.

---

## 6. Decision forks (for the experts)

1. **Migration posture:** default-off lane (Bark/MusicGen/AudioGen stay default; new engines opt-in) vs. hard cutover + re-baseline. (Recommendation: default-off lane first.)
2. **One model for both music AND SFX** (Stable Audio Medium + Small-SFX) vs. keep AudioGen for SFX and only swap music. Does Stable Audio's SFX quality beat AudioGen for short foley?
3. **Reference-voice bank source:** self-recorded vs. curated open-license voice packs vs. a fixed set of synthesized anchor voices. Legal-clean is mandatory.
4. **Emotion signal design:** Emo-Vector (8 numeric dims, deterministic, no LLM) vs. Emo-Text (LLM infers, richer, PD6 applies) vs. per-character Emo-Audio refs. Where is it computed and how does it stay C7-deterministic?
5. **Duration control vs. lip-sync:** pin IndexTTS2 token-count to the planned shot length, or free-run and let the shot-duration calculator follow the audio?
6. **Byte-identity redefinition:** is PD1 "byte-identical to a re-captured v2 baseline," and do we keep Bark as a permanent fallback engine?

---

## 7. Open questions to verify before build

- IndexTTS2 **license** (commercial OK?) and the **ComfyUI node's** Blackwell/cu130/no-xformers behavior (pilot import test).
- Stable Audio 3 **stereo handling** through the mono-assuming SceneSequencer/EpisodeAssembler.
- IndexTTS2 English-only quality + consistency across a 5-7 voice ensemble (it is EN/ZH; OTR is EN).
- Whether IndexTTS2 per-line cloning is **fast enough** for a full episode's dialogue line count within the render budget.
- Re-baseline mechanics (who/when captures `baseline_v2.wav`).

---

## 8. Paste-ready question block for the round-robin

> Context: a local ComfyUI "old-time radio" sci-fi pipeline on Windows + RTX 5080 (16 GB, Blackwell sm_120, torch cu130 nightly). Prime directives: audio output must be deterministic and currently byte-identical to a fixed baseline; 14.5 GB VRAM ceiling; everything local/offline; commercial-clean licensing.
>
> We want to (1) replace Bark TTS with IndexTTS2 (zero-shot voice cloning + emotion control), (2) replace MusicGen theme music and possibly AudioGen SFX with Stable Audio 3 (native ComfyUI).
>
> Questions: (a) Best migration posture -- default-off opt-in lane keeping Bark/MusicGen/AudioGen as the byte-identical default, or hard cutover + re-baseline? (b) For IndexTTS2 expressiveness with a 5-7 voice ensemble, is a deterministic 8-dim Emo-Vector per line preferable to an LLM-inferred Emo-Text tag, given a strict reproducibility requirement? (c) Does Stable Audio 3 SFX quality justify retiring AudioGen, or keep AudioGen for foley and only swap music? (d) How to build a license-clean reference-voice bank for zero-shot cloning of fictional characters? (e) Pitfalls swapping a mono (MusicGen/AudioGen) assembly chain to stereo (Stable Audio)? (f) IndexTTS2 on Blackwell/cu130 -- known dependency traps (xformers/transformers pins)?

---

## Sources
- IndexTTS2 model: https://hf.co/IndexTeam/IndexTTS-2 (paper https://arxiv.org/pdf/2506.21619)
- IndexTTS2 features overview: https://index-tts2.org/ , https://indextts.ai/index-tts2
- IndexTTS2 ComfyUI node (community): https://github.com/kana112233/ComfyUI-kaola-IndexTTS2
- Stable Audio 3 day-0 ComfyUI support: https://blog.comfy.org/p/stable-audio-3-day-0-support
- Stable Audio 3 ComfyUI tutorial: https://docs.comfy.org/tutorials/audio/stable-audio/stable-audio-3
- Stable Audio Open models: https://hf.co/stabilityai/stable-audio-open-1.0 , https://hf.co/stabilityai/stable-audio-open-small
- Blackwell sm_120 ComfyUI (cu130 nightly, no-xformers): https://github.com/Comfy-Org/ComfyUI/discussions/6643
