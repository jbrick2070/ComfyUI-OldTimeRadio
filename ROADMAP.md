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

### 5) First Moves for v1.5
- **Branch `v1.5-audiogen-dsp`**: Initialize the new feature branch.
- **SceneSequencer SFX bus**: Implement the audio insertion logic that consumes the batched AudioGen waveforms.
- **Tape emulation DSP**: Add analog tape saturation filters to `audio_enhance.py`.
- **OTR_VRAMGuardian node**: Promote `force_vram_offload()` to a wirable ComfyUI node.
- Run full 89-test regression sweep against the parser v3/v4 changes.

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

## v1.5 — Feature Plan

### Priority Order (April–May 2026)

| Priority | Feature | VRAM Impact | Status | Effort | Target |
|----------|---------|-------------|--------|--------|--------|
| **1** | **AudioGen ↔ SceneSequencer SFX bus** | 0 GB | Not started | Medium (3-5 days) | 2026-04-25 |
| **2** | **OTR_VRAMGuardian node** | 0 GB | Not started (infra exists in `_vram_log.py`) | Small (1 day) | 2026-04-18 |
| **3** | **Tape Emulation DSP** | 0 GB (CPU-only) | Not started | Medium (3-4 days) | 2026-05-01 |
| **4** | **Parser v5 — unified SFX cue extraction** | 0 GB | Not started | Medium (2-3 days) | 2026-05-05 |
| **5** | **Video Engine — Adaptive Brightness Gating** | 0 GB | Not started | Small (1-2 days) | 2026-05-08 |
| **6** | **TA_Offset Pins** (timecode offsets on SceneSequencer) | 0 GB | Not started | Small (1 day) | 2026-05-10 |
| **7** | **Full Regression + New Parser Test Cases** | 0 GB | Not started | Small (1 day) | Ship gate |

### Feature Details

#### Priority 1 — AudioGen ↔ SceneSequencer SFX Bus
The `BatchAudioGenGenerator` node already generates high-quality SFX via `facebook/audiogen-medium` and outputs a batched waveform tensor. But `SceneSequencer` (`scene_sequencer.py`, 45 KB) does not consume it. This wiring is the single most valuable v1.5 deliverable — it makes the AudioGen investment audible in the final episode.

**Files:** `scene_sequencer.py`, `batch_audiogen_generator.py`
**Success Criteria:** SFX clips appear at correct timecodes in the final MP4 audio track.

#### Priority 2 — OTR_VRAMGuardian Node
Promote the existing `force_vram_offload()` function from `_vram_log.py` into a standalone ComfyUI node (`OTR_VRAMGuardian`). Users can then wire explicit VRAM flush points into their workflow graphs — useful for long multi-episode batch runs.

**Files:** New `nodes/vram_guardian.py`, update `__init__.py`
**Success Criteria:** Node appears in ComfyUI node picker, executes a full VRAM flush when triggered, passes VRAM profile test.

#### Priority 3 — Tape Emulation DSP
Add analog tape emulation filters to `audio_enhance.py`: tape saturation (soft clipping), wow/flutter (pitch modulation), tape hiss (filtered noise injection), and optional high-frequency rolloff. All CPU-only numpy/scipy, zero VRAM. Core aesthetic feature for the "old time radio" brand.

**Files:** `nodes/audio_enhance.py`
**Success Criteria:** Audible analog warmth on A/B comparison. No peak clipping. No VRAM usage.

#### Priority 4 — Parser v5 — Unified SFX Cue Extraction
Currently `_parse_script()` in `story_orchestrator.py` handles dialogue parsing, but `BatchAudioGenGenerator` has its own separate `[SFX:]` regex extraction. Unify SFX cue extraction into the canonical parser so both dialogue and SFX flow through one pipeline.

**Files:** `nodes/story_orchestrator.py`, `nodes/batch_audiogen_generator.py`
**Success Criteria:** SFX cues appear in the canonical script JSON as `{"type": "sfx", ...}` items. BatchAudioGen consumes them instead of re-parsing.

#### Priority 5 — Adaptive Brightness Gating
Extend `video_engine.py`'s CRT renderer to auto-adjust scene brightness based on audio energy phases (quiet scenes dim, loud scenes brighten). The per-frame volume curve already exists in `_analyze_audio()` — this is a small extension to the render pass.

**Files:** `nodes/video_engine.py`
**Success Criteria:** Visible brightness variation between quiet/loud scenes in output video.

#### Priority 6 — TA_Offset Pins
Expose time-alignment offset inputs on SceneSequencer nodes for fine-grained audio-video sync tuning. Simple float inputs that shift audio clip placement on the assembly timeline.

**Files:** `nodes/scene_sequencer.py`
**Success Criteria:** Offset values shift audio placement. Tested with ±0.5s offsets.

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

- [ ] AudioGen SFX clips audible in final episode MP4.
- [ ] `OTR_VRAMGuardian` node visible in ComfyUI picker and functional.
- [ ] Tape emulation audibly warm on A/B test.
- [ ] SFX cues flow through canonical parser (no duplicate regex).
- [ ] Adaptive brightness gating visible in video output.
- [ ] TA_Offset pins shift audio placement correctly.
- [ ] Full regression sweep: 89+ tests green.
- [ ] VRAM peak ≤ 14.5 GB on full episode run.
- [ ] `tests/vram_profile_test.py` green.
- [ ] UTF-8 no BOM on all new files.
- [ ] **Jeffrey personally confirms ship.** Only then: merge to main and tag `v1.5`.

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
