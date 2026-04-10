# ComfyUI-OldTimeRadio — ROADMAP

---

## NEW CONVERSATION HANDOFF — READ THIS FIRST

### 1) Current Shipped State
- **Last shipped:** `v1.4` (tagged 2026-03-15)
- **Active branch:** `main`
- **Post-v1.4 hotfixes merged:** Parser v3/v4 hardening, QA Peer Review Guide, workflow JSON widget fixes (6 commits post-tag on `main`)

### 2) v1.4 Final Highlights
- **Mistral Nemo 12B Flagship**: Default LLM upgraded to `mistralai/Mistral-Nemo-Instruct-2407`. 18.1 tok/s on RTX 5080 with 4-bit NF4 quantization. 12.4B parameters in a 7 GB VRAM envelope.
- **Zero-Prime Cache Hardening**: All `from_pretrained()` calls (LLM, Bark, MusicGen) now explicitly pass `cache_dir` derived from `HF_HOME`. Eliminates redundant Hub fetches, `.incomplete` deadlocks, and Windows symlink issues. Models load instantly from NVMe after first download.
- **16GB Flagship Hardening**: Zero-Prime loading logic, 2GB Sovereignty Buffer, and 4-bit NF4 "Wing Ding Protection" verified on RTX 5080/Blackwell (18.1 tok/s peak).
- **Kokoro Announcer**: `repo_id` explicitly locked to `hexgrad/Kokoro-82M` to suppress migration warnings.
- **208 GB Global Cache Cleanup**: Identified and purged duplicate model weights from `~/.cache/huggingface/hub` that were wasting disk space.
- **First-Name Shield**: Roster generation hardened to prevent character name collisions/hallucinations.
- **VoiceHealth & Telemetry Fixes**: Forced CUDA for health checks and fixed case-sensitive tracking for 9B/12B/14B models.
- **True Single Switch Architecture**: Restructured the `LLMScriptWriter` to be the sole repository for `model_id` state.
- **Subtle Pacing Overhaul**: 50% duration reduction for all dramatic beats and pauses for tighter narrative flow.
- **Total VRAM Cleanup**: Aggressive explicit `model.cpu()`, inline Python `del` sweeping, and ComfyUI `soft_empty_cache()` inside the GC block.
- **AudioGen SFX Integration**: `OTR_BatchAudioGenGenerator` shipped with SHA-256 caching, per-prompt generation via `facebook/audiogen-medium`, and explicit VRAM cleanup.
- **Workflow verification**: 100% regression pass (89/89 tests).

### 3) Post-v1.4 Hotfixes (already on main)
- **Parser v3/v4 Rewrite**: `_parse_script()` now handles next-line dialogue (Mistral Nemo style) and shorthand `[CHAR, traits]` tags. Blocklist uses first-word check to reject stage directions (`ACT 1`, `SCENE 3`, `CONTINUED`).
- **QA Peer Review Guide**: `QA_PEER_REVIEW_GUIDE.md` added with VRAM validation, parser fault tolerance, workflow JSON integrity, and regression test protocols.
- **Workflow JSON Fixes**: Added missing `widget` mapping blocks and `guidance_scale` values across all 3 shipped workflow JSONs.
- **MusicGen Load Fix**: `device` and `dtype` defined before model load to prevent initialization errors.

### 4) Next Priority Feature (v1.5)
**AudioGen ↔ SceneSequencer Integration (priority 1):** Wire the `BatchAudioGenGenerator` SFX output into the `SceneSequencer` audio assembly timeline so generated Foley actually plays in the episode.

### 5) First Moves for v1.5 (Phase 1)
- **Branch `v1.5-audiogen-dsp`** off `main`.
- **OTR_VRAMGuardian node**: Wrap `force_vram_offload()` as a wirable ComfyUI node (1 day).
- **Parser v5**: Unify SFX cue extraction into canonical `_parse_script()` (2-3 days).
- **Regression sweep**: Run full 89-test suite + new SFX parser tests (1 day).
- Pass **Phase 1 Test Gate** → advance to Phase 2 (Audio Pipeline).

---

## Current State (as of 2026-04-10)

- **Last shipped tag:** `v1.4` — Mistral Nemo 12B flagship, Zero-Prime cache hardening, AudioGen SFX
- **Active branch:** `main` (6 hotfix commits post-tag)
- **Repo:** https://github.com/jbrick2070/ComfyUI-OldTimeRadio
- **Sister repo (bug bible):** https://github.com/jbrick2070/comfyui-custom-node-survival-guide

### What v1.4 ships
- Arc Enhancer full pipeline, timeout sentinel path, plot-spine injection
- Kokoro Announcer + MusicGen theme end-to-end
- True Single Switch LLM architecture in place
- Per-node VRAM snapshot logging (`_vram_log.py` → `otr_runtime.log`)
- All v1.3 timeout / corruption fixes verified
- AudioGen generative Foley (`batch_audiogen_generator.py`) with SHA-256 caching
- Parser v3/v4 multi-format support (hotfixed post-tag)
- QA Peer Review Guide (hotfixed post-tag)

### Hardware Reality (do not violate)
- RTX 5080 Laptop, 16 GB VRAM, Blackwell sm_120
- torch 2.10.0 + CUDA 13.0, Windows, Python 3.12
- SageAttention + SDPA. Flash Attention 2 is NOT available on this platform.
- 100% local, open source, offline-first. No cloud, no API keys.
- Real-world VRAM peak target: **≤ 14.5 GB**

---

## v1.5 — Phased Build Plan

v1.5 is delivered in **3 phases**. Each phase ends with a test gate — no advancing until the gate passes. Dependencies flow downward: Phase 2 depends on Phase 1 plumbing, Phase 3 depends on Phase 2 audio.

### Phase Overview

| Phase | Name | Features | Effort | Target |
|-------|------|----------|--------|--------|
| **1** | **Plumbing & Infrastructure** | VRAMGuardian node, Parser v5 (unified SFX), Regression tests | ~4-5 days | 2026-04-18 |
| **2** | **Audio Pipeline** | AudioGen ↔ SceneSequencer SFX bus, Tape Emulation DSP | ~6-9 days | 2026-05-02 |
| **3** | **Video Polish & Ship** | Adaptive Brightness Gating, TA_Offset Pins, Final regression + ship sign-off | ~3-4 days | 2026-05-10 |

---

### Phase 1 — Plumbing & Infrastructure
*Branch: `v1.5-audiogen-dsp` off `main`*

This phase builds the foundation that Phase 2 depends on. The VRAMGuardian node is standalone, but Parser v5 **must** land first because the SFX bus wiring in Phase 2 needs canonical `{"type": "sfx"}` items in the script JSON — not the duplicate regex currently in BatchAudioGen.

| Item | VRAM | Files | Effort |
|------|------|-------|--------|
| **OTR_VRAMGuardian node** | 0 GB | New `nodes/vram_guardian.py`, update `__init__.py` | 1 day |
| **Parser v5 — unified SFX cue extraction** | 0 GB | `nodes/story_orchestrator.py`, `nodes/batch_audiogen_generator.py` | 2-3 days |
| **Regression + new parser test cases** | 0 GB | `tests/test_core.py` | 1 day |

**OTR_VRAMGuardian node:** Wrap `force_vram_offload()` from `_vram_log.py` into a first-class ComfyUI node. Users wire it into workflow graphs for explicit VRAM flush points between heavy nodes. Passthrough design — accepts any input, flushes VRAM, forwards the input unchanged.

**Parser v5 — Unified SFX Cue Extraction:** Currently `_parse_script()` in `story_orchestrator.py` handles dialogue but ignores `[SFX:]` tags. `BatchAudioGenGenerator` has its own duplicate regex extraction (lines 130-136). Unify: the canonical parser emits `{"type": "sfx", "description": "...", "index": N}` items inline with dialogue. BatchAudioGen then consumes them directly from `script_json` instead of re-parsing raw text.

**Regression Tests:** Extend `tests/test_core.py` with test cases for the v3/v4 parser patterns that landed post-v1.4, plus new SFX extraction tests. All 89+ existing tests must stay green.

#### Phase 1 Test Gate ✓
- [ ] `OTR_VRAMGuardian` visible in ComfyUI node picker
- [ ] VRAMGuardian triggers `force_vram_offload()` and logs to `otr_runtime.log`
- [ ] Script JSON output contains `{"type": "sfx"}` items for all `[SFX:]` tags
- [ ] BatchAudioGen reads SFX from script JSON (no internal regex)
- [ ] 89+ regression tests green
- [ ] VRAM ≤ 14.5 GB
- [ ] UTF-8 no BOM on all new files

---

### Phase 2 — Audio Pipeline
*Depends on: Phase 1 (Parser v5 provides clean SFX items)*

This is the big payload. After this phase, episodes **sound different** — AudioGen SFX is mixed into the timeline and tape emulation adds analog warmth. These two features are independent of each other but grouped here because they both touch the audio assembly path.

| Item | VRAM | Files | Effort |
|------|------|-------|--------|
| **AudioGen ↔ SceneSequencer SFX bus** | 0 GB | `nodes/scene_sequencer.py`, `nodes/batch_audiogen_generator.py` | 3-5 days |
| **Tape Emulation DSP** | 0 GB (CPU-only) | `nodes/audio_enhance.py` | 3-4 days |

**AudioGen ↔ SceneSequencer SFX Bus:** The `BatchAudioGenGenerator` outputs a batched waveform tensor, but `SceneSequencer` (45 KB) does not consume it. Wire an `sfx_audio` input into the sequencer that overlays SFX clips at the timecodes matching their `{"type": "sfx"}` positions in the script. The sequencer already processes `dialogue`, `pause`, `environment`, and `scene_break` items — SFX becomes the 5th item type.

**Tape Emulation DSP:** Add analog tape emulation to `audio_enhance.py`: tape saturation (soft-clipping waveshaper), wow/flutter (low-frequency pitch modulation), tape hiss (band-limited noise injection), and high-frequency rolloff (gentle low-pass). All CPU-only numpy/scipy. Exposed as a node with intensity controls (`off`, `subtle`, `medium`, `heavy`). This is the signature aesthetic feature.

#### Phase 2 Test Gate ✓
- [ ] SFX clips audible at correct timecodes in final episode MP4
- [ ] SFX volume balanced against dialogue (no drowning out voices)
- [ ] Tape emulation audibly warm on A/B comparison
- [ ] No peak clipping on tape-processed audio
- [ ] No VRAM usage from DSP (CPU-only verified)
- [ ] Full regression green (89+ tests)
- [ ] VRAM ≤ 14.5 GB on full episode run

---

### Phase 3 — Video Polish & Ship
*Depends on: Phase 2 (audio pipeline stable)*

Small targeted improvements to the video engine and audio sync, followed by the final ship verification sweep. Low risk, low effort — this is the "tighten the bolts" phase.

| Item | VRAM | Files | Effort |
|------|------|-------|--------|
| **Adaptive Brightness Gating** | 0 GB | `nodes/video_engine.py` | 1-2 days |
| **TA_Offset Pins** | 0 GB | `nodes/scene_sequencer.py` | 1 day |
| **Final regression + ship criteria** | 0 GB | All | 1 day |

**Adaptive Brightness Gating:** Extend the CRT renderer to auto-adjust scene brightness based on audio energy. Quiet scenes dim toward dark navy; loud scenes brighten toward full phosphor green. The per-frame `volume_curve` from `_analyze_audio()` already provides the data — this adds a global brightness multiplier to the render pass. Subtle by default; dramatic on high-energy scenes.

**TA_Offset Pins:** Expose `sfx_offset_ms` and `dialogue_offset_ms` float inputs on SceneSequencer. These shift audio clip placement on the assembly timeline for fine-grained sync tuning. Default 0. Tested with ±500ms offsets.

**Final Regression & Ship:** Full 89+ test sweep, `vram_profile_test.py` green, end-to-end episode generation with all new features enabled, UTF-8 no BOM scan. Jeffrey confirms ship → merge to `main` → tag `v1.5`.

#### Phase 3 Test Gate ✓ (Ship Gate)
- [ ] Brightness variation visible between quiet/loud scenes
- [ ] TA_Offset pins shift audio placement correctly (±0.5s test)
- [ ] Full regression sweep: 89+ tests green
- [ ] `tests/vram_profile_test.py` green
- [ ] VRAM peak ≤ 14.5 GB on full episode run
- [ ] UTF-8 no BOM on all new/modified files
- [ ] End-to-end episode: script → audio → SFX → video → MP4 (all features on)
- [ ] **Jeffrey personally confirms ship.** Merge to main, tag `v1.5`.

---

### Already Shipped (not in scope — do not re-implement)
These features were proposed by external analysis as v1.5 work but **already exist**:
- ✅ Per-node VRAM snapshot logging (`_vram_log.py`)
- ✅ VRAM monitoring & ceiling alerts (`vram_snapshot()`, `VRAM_CEILING_EXCEEDED`)
- ✅ `force_vram_offload()` infra (exists, just needs node wrapper)
- ✅ Audio-reactive video visualizer (CRT ring, FFT bars, particles, waveform)
- ✅ FFmpeg video encoding with NVENC auto-detection
- ✅ Telemetry HUD post-roll credits
- ✅ Multi-format script parser (v3/v4 patterns)
- ✅ QA Peer Review Guide

---

## Punted to v1.6+

Formally deferred out of v1.5 scope. Not started, not blocked, not forgotten.

| Feature | Reason for Deferral |
|---|---|
| **RVC Voice Locking** | 2nd-pass GPU inference, VRAM headroom uncertain on 16 GB. Needs narrator vs character contract. |
| **MusicGen Interstitial Wiring** | SceneSequencer needs a scene-transition bus. Architectural work beyond v1.5. |
| **Expanded TTS Stack (Fish-Speech, XTTS)** | Voice identity risk. Blocked by narrator/character voice contract. |
| **Bark Alt Voice UI Selector** | Voice pool exists but full dropdown UI is scope creep. |
| **RDKit Phoneme Assignment** | Research-level. No clear integration path. |
| **Regional Accent Injection** | Depends on voice stack expansion (TTS/RVC). |
| **Spatial Ambient Reverb** | Nice-to-have but lower priority than tape emulation. |
| **One-off Audio FX Library (bonk/slap/whoosh)** | Small but deferred until SFX bus integration proves out. |
| **Shared Latent SFX Grafting** | Research only. |
| **Asynchronous Expert Scheduling** | Policy: sequential execution only. |
| **High-rank LoRA Fine-tuning UI** | Exceeds 16 GB VRAM. |
| **AI-driven Automated Ducking** | Full mixing stack rewrite. |
| **Visual-audio Lip-sync Bridges** | Out of scope — this is an audio drama project. |

*(Tentative dates: Q3–Q4 2026.)*

---

## v1.5 Ship Criteria

Ship criteria are defined per-phase above. All three test gates must pass:
- **Phase 1 Gate** — VRAMGuardian functional, SFX in canonical parser, regression green.
- **Phase 2 Gate** — SFX audible in MP4, tape emulation warm, no clipping.
- **Phase 3 Gate (Ship)** — Brightness gating visible, TA_Offset working, full end-to-end episode, Jeffrey confirms ship → merge to `main` → tag `v1.5`.

---

## Hard Rules (permanent)

- 16 GB VRAM ceiling, 14.5 GB real-world target.
- 100% local, open source, offline-first.
- Sequential execution only.
- Flash Attention 2 is not available on this platform. Stop asking.
- Git pushes via PowerShell blocks handed to the user with `cd` baked in.
- Lockstep verify local HEAD vs GitHub HEAD after every push.
- UTF-8 no BOM on Windows. Verify before declaring done.
- Every change ships with a regression pass.

---

## Credits
Created by Jeffrey Brick. Built on top of Bark (Suno AI), Kokoro TTS, MusicGen (Meta), AudioGen (Meta), and Mistral Nemo (Mistral AI). Patterns from the ComfyUI Custom Node Survival Guide.
