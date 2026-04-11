# ComfyUI-OldTimeRadio — ROADMAP

---

## NEW CONVERSATION HANDOFF — READ THIS FIRST

### 1) Current Shipped State
- **Last shipped:** `v1.4` (tagged 2026-03-15)
- **Active branch:** `main`
- **Post-v1.4 hotfixes merged:** Parser v3/v4 hardening, workflow JSON widget fixes (6 commits post-tag on `main`)

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

### v1.5 CLEAN — Narrative Pipeline Hardening (2026-04-10)
Architectural overhaul of the script generation pipeline based on hierarchical narrative decomposition research.

| Change | What | Why |
|:---|:---|:---|
| **7-Line Micro-Spine Protocol** | Open-Close generates 3× 7-line structural spines (~100 tokens) instead of 3× full outlines (~450 tokens) | Cuts Open-Close from ~12 min to ~2 min. Eliminates KV cache exhaustion and VRAM_CEILING_EXCEEDED warnings. Forces LLM to focus on narrative structure, not prose. |
| **Story Editor (Critique-Guided Writing)** | After outline generation, a "Story Editor" pass critiques the outline and generates per-act briefs. Each act prompt receives its brief + overall critique. | Critique drives writing BEFORE dialogue is generated, not after. Produces stronger, more purposeful acts from first draft. |
| **Self-Critique Gating** | Global revision pass auto-skipped for scripts with >3 acts. Critique runs upstream via Story Editor instead. | Prevents "Summarization Collapse" where the LLM condensed 5 acts into ~30 lines during revision. |
| **Arc Enhancer v2 — Critique + Act Summaries** | Arc Enhancer now receives critique findings + act-by-act narrative summaries from chunked generation | Opening/closing polish uses the complete story picture for start-to-end coherence, not just an extracted spine. |
| **Dialogue Inflation (1.5x)** | `words_per_act` target increased from 1.2× to 1.5× | Pushes each act toward ~500-600 words, guaranteeing 8-10 min of actual dialogue. |
| **Dynamic Token Budgets** | `act_budget = max(1024, min(2048, words_per_act × 2.5))` instead of hardcoded 1536 | Scales with episode length. Longer episodes get proportionally more generation headroom. |
| **force_vram_offload() Between Acts** | Replaced raw `gc.collect()`/`torch.cuda.empty_cache()` with the proper 3-step teardown | Ensures dangling model references are cleaned up, not just cached memory. |
| **Prompt Hardening** | Strict "Do NOT summarize" directives in act generation prompts | Forces the LLM to write every beat as full dialogue instead of narrative summary. |
| **v1.5.1 Prompt Guard** | Truncates input prompts to `context_cap` | **Saves 110s stall and 15GB VRAM spike.** Prevents pre-fill explosion on 10k+ prompts. |
| **v1.5.1 CUDA Warmup** | 1-token dummy generate on load | Eliminates ~60s cold-start stall on first token by front-loading JIT kernel compilation. |
| **v1.5.1 Flush Etiquette** | `_flush_vram_keep_llm()` helper | Stops redundant 13s model reloads (saves ~2 min total) by keeping weights on GPU between phases. |


### What v1.4 ships
- Arc Enhancer full pipeline, timeout sentinel path, plot-spine injection
- Kokoro Announcer + MusicGen theme end-to-end
- True Single Switch LLM architecture in place
- Per-node VRAM snapshot logging (`_vram_log.py` → `otr_runtime.log`)
- All v1.3 timeout / corruption fixes verified
- AudioGen generative Foley (`batch_audiogen_generator.py`) with SHA-256 caching
- Parser v3/v4 multi-format support (hotfixed post-tag)

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
- [x] `OTR_VRAMGuardian` visible in ComfyUI node picker
- [x] VRAMGuardian triggers `force_vram_offload()` and logs to `otr_runtime.log`
- [x] Script JSON output contains `{"type": "sfx"}` items for all `[SFX:]` tags
- [x] BatchAudioGen reads SFX from script JSON (no internal regex)
- [x] 89+ regression tests green (113 passing)
- [x] VRAM ≤ 14.5 GB
- [x] UTF-8 no BOM on all new files

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
- [x] SFX clips audible at correct timecodes in final episode MP4
- [x] SFX volume balanced against dialogue (no drowning out voices)
- [x] Tape emulation audibly warm on A/B comparison
- [x] No peak clipping on tape-processed audio
- [x] No VRAM usage from DSP (CPU-only verified)
- [x] Full regression green (113 tests passing)
- [x] VRAM ≤ 14.5 GB on full episode run (peak 9.6 GB)

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
- [x] Brightness variation visible between quiet/loud scenes
- [x] TA_Offset pins shift audio placement correctly (±500ms range)
- [x] Full regression sweep: 113 tests green
- [ ] `tests/vram_profile_test.py` green
- [x] VRAM peak ≤ 14.5 GB on full episode run (peak 9.6 GB)
- [x] UTF-8 no BOM on all new/modified files
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

---

## Punted to v1.6

Formally deferred out of v1.5 scope. Not started, not blocked, not forgotten.
v1.6 is the **audio polish** release — incremental improvements to voice, music, and SFX.

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
| **AI-driven Automated Ducking** | Full mixing stack rewrite. |

*(Tentative: Q3 2026)*

---

## v2.0 — Visual Drama Engine (Research Phase)

**Vision:** Evolve OldTimeRadio from audio-only drama to **complementary visual storytelling**. The existing script pipeline becomes the scene graph that drives both audio AND visual generation. The CRT retro visualization remains as the aesthetic fallback/overlay.

> **Full research report:** `docs/v2_research_report.md`

### Two Forks, One Pipeline

| Fork | Technology | Strengths | Weaknesses |
|---|---|---|---|
| **A: Deterministic 3D Sim-Renderer** | Blender headless, Rhubarb NG lip-sync, TripoSR/Meshy for asset gen | Absolute character consistency (rigged meshes can't hallucinate). Infinite clip length (geometric state preservation). Deterministic lip-sync via viseme→blendshape mapping. **Fits 8-12 GB VRAM.** | Requires upfront asset library. Stylized/animated aesthetic unless heavily art-directed. |
| **B: LTX 2.3 Latent Diffusion (22B DiT)** | FP8-quantized DiT, IP-Adapter + CLIP Vision anchoring, FeatureExtractorV2 audio/video separation, spatio-temporal tiled VAE | Cinematic photorealism from base weights. Zero asset prep. FeatureExtractorV2 enables latent lip-sync from 48kHz audio. | 22B params needs FP8/GGUF to fit 16GB. ~20s context window (recursive chaining). "Silent Lip Bug" from aspect ratio tensor misalignment. Multi-character needs timestamped localized masking. |

### Decision Matrix (from Research Report)

| Feature | Path A (3D) | Path B (LTX 2.3) |
|---|---|---|
| Visual Consistency | **Absolute** | Fragile (IP-Adapter/LoRA dependent) |
| VRAM Footprint | **8-12 GB** | 14.5+ GB with FP8 |
| Clip Length | **Infinite** (geometric state) | ~20s (recursive chaining) |
| Lip Sync | **Rhubarb NG** (offline, deterministic) | FeatureExtractorV2 (latent, aspect-ratio sensitive) |
| Aesthetic | Stylized/animated | **Cinematic/photorealistic** |
| Setup Friction | Heavy upfront (assets) | Heavy runtime (debugging) |

### Hybrid Architecture (Proposed)

Neither fork alone is optimal. The hybrid composes both:

1. **Script pipeline** extracts scene descriptions, character profiles, and SFX cues (already built)
2. **Fork A** generates character meshes (TripoSR from reference images) → Blender headless renders consistent character poses with Rhubarb-driven lip-sync mapped to 48kHz audio
3. **Fork B** generates establishing shots and scene backgrounds via LTX 2.3 I2V (IP-Adapter anchored to key frames generated by Flux/SDXL)
4. **Compositor** layers 3D character renders over diffusion backgrounds, synced to the audio timeline
5. **CRT overlay** adds the retro aesthetic as a post-processing pass

**Why hybrid wins:** Character consistency via Path A (meshes don't drift), cinematic environments via Path B (no assets to build for backgrounds), all within 16GB via sequential VRAM handoff.

### Audio-First Architecture (Already Built)

The research confirms that **both paths mandate audio-first** — the exact architecture OTR v1.5 already implements:
- 48kHz normalized audio timeline locked before visual rendering
- JSON production manifests drive all downstream nodes
- VRAM Power Wash protocol between model phases

### VRAM Strategy

Sequential handoff (same pattern as LLM → Bark → MusicGen):
```
LLM script → unload → Bark TTS → unload → Rhubarb visemes (CPU) →
TripoSR 3D mesh → unload → Blender render (CPU/low VRAM) →
LTX 2.3 backgrounds (FP8) → unload → FFmpeg composite
```

### Lip-Sync Architecture

| Path | Pipeline | Tool | Input | Output |
|---|---|---|---|---|
| **A (3D)** | Audio → phoneme detection → viseme JSON → blendshape keys | Rhubarb NG CLI | 48kHz WAV | Timestamped viseme JSON |
| **B (Diffusion)** | Audio → FeatureExtractorV2 → audio_aggregate_embed → DiT cross-attention | LTX 2.3 node | 48kHz WAV | Latent lip motion |

### Temporal Extension

| Path | Method | Limit |
|---|---|---|
| **A (3D)** | Save ending geometric + camera state → load as frame 0 of next scene | **None** |
| **B (Diffusion)** | Extract final frame → feed back as I2V anchor for next block (recursive chaining) | ~20s per block, latent outpainting |

### Quick Proof-of-Concept (v2.0-alpha spike)

| Step | Tool | Output |
|---|---|---|
| Extract scene prompt from OTR episode | Existing script pipeline | Text prompt + `visual_plan` JSON |
| Generate character reference image | Flux/SDXL | Anchor frame for IP-Adapter |
| Fork A: Image → 3D mesh → render | TripoSR + Blender headless | Character turntable + posed frames |
| Fork B: Reference → 5s background clip | LTX 2.3 (FP8, I2V mode) | Cinematic environment clip |
| Lip-sync (A) | Rhubarb NG → Blender blendshapes | Synced character animation |
| Composite | FFmpeg (existing `video_engine.py`) | 3D characters over diffusion backgrounds |

### Production Plan Extension

The Director's `production_plan_json` gains a `visual_plan` section:
```json
{
  "visual_plan": [
    {
      "scene_id": "scene_1",
      "description": "Monitoring station command bridge, emergency lighting",
      "characters_present": ["KAEL", "HENRY"],
      "camera": "medium two-shot, slight dutch angle",
      "duration_sec": 45,
      "method": "hybrid"
    }
  ]
}
```

### Research Inputs

- ✅ Deep research report completed: `docs/v2_research_report.md` (authored by Jeffrey)
- ✅ Bug Bible patterns for VRAM handoff and model sequencing apply directly
- ✅ Existing ComfyUI node infrastructure supports TripoSR, LTX, and Wan 2.1
- ⬜ LTX 2.3 FP8 quantized weights availability check
- ⬜ Rhubarb NG Windows binary validation

### Open Questions

- Multi-character compositing: Z-depth sorting for overlapping characters?
- Blender render time budget: target <30s per frame for real-time viability?
- LTX 2.3 vs Wan 2.1 for backgrounds: which fits 16GB better after FP8?
- Timeline: v2.0-alpha spike after v1.5 ships, full v2.0 is Q4 2026 / Q1 2027

*(This is a research-phase document. No code committed until the spike proves feasibility.)*


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
- **NEVER use `force_vram_offload()` between LLM passes** within a single run (use `_flush_vram_keep_llm()` instead).
- **ALWAYS truncate LLM input prompts** to `context_cap - max_new_tokens`. Prevents pre-fill stalls and 25GB VRAM spikes (BUG-12.33).
- **ALWAYS perform a 1-token warmup pass** on LLM load to front-load CUDA kernel JIT.
- Flash Attention 2 is not available on this platform. Stop asking.
- Git pushes via PowerShell blocks handed to the user with `cd` baked in.
- Lockstep verify local HEAD vs GitHub HEAD after every push.
- UTF-8 no BOM on Windows. Verify before declaring done.
- Every change ships with a regression pass.

---

## Credits
Created by Jeffrey Brick. Built on top of Bark (Suno AI), Kokoro TTS, MusicGen (Meta), AudioGen (Meta), and Mistral Nemo (Mistral AI). Patterns from the ComfyUI Custom Node Survival Guide.
