# ComfyUI-OldTimeRadio — ROADMAP

---

## NEW CONVERSATION HANDOFF - READ THIS FIRST

### 1) Current Shipped State
- **Last shipped:** `v1.4`
- **Active branch:** `main`

### 2) V1.4 Final Highlights
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
- **Workflow verification**: 100% regression pass (89/89 tests).

### 3) Next Priority Feature (v1.5)
- **[COMPLETED] AudioGen SFX:** Replacing procedural noise with generative Foley via `OTR_BatchAudioGenGenerator`.
- **RVC Voice Locking:** Post-generation timbre-locking for canonical characters (Lemmy, etc.).
- **Flash Attention 2**: Optional opt-in for platforms that support it (currently disabled for Blackwell sm_120).

### 4) First Moves for v1.5
- **Branch `v1.5-audio-gen-rvc`**: Initialize the new feature branch.
- **RVC Prototype**: Test the 16GB VRAM headroom for an RVC secondary pass. NOT available on this architecture (Windows sm_120).
- Lockstep verify local HEAD vs GitHub HEAD after every push.
- Zero trailing BOM signatures.

### 5) First Moves for Next Session
- Check test results again. 4GB Obsidian profile and workflow JSONs are now verified and hardened.
- Run `git status` and verify changes are pushed/clean.
- Begin drafting the `v1.5` feature specs for RVC integration (timbre locking).

---

## Current state (as of 2026-04-10)

- **Last shipped tag:** `v1.4` — Mistral Nemo 12B flagship, Zero-Prime cache hardening
- **Active branch:** `main`
- **Repo:** https://github.com/jbrick2070/ComfyUI-OldTimeRadio
- **Sister repo (bug bible):** https://github.com/jbrick2070/comfyui-custom-node-survival-guide

### What v1.3 ships
- Arc Enhancer full Phase A/B/C pipeline, on by default in all workflows
- Phase A structural coherence scoring (5 checks, full telemetry)
- Plot Spine Injection (middle-act summary passed to Phase B so it cannot contradict the middle)
- Echo phrase logging in Phase C
- OpenClose timeout fix (450 tok budget, 480s wall) — parallel 3-outline evaluator is gated OFF
- Test workflow speed pass (short 3-act, 100% feature parity)
- Flash Attention 2 platform-accurate warning message
- PRO QA announcer bookends
- NameLeakGuard (difflib fuzzy, ALL-CAPS aware)
- Lemmy RNG statistical verification
- All v1.3-beta stability fixes from Antigravity (prestartup mock, VRAM deferral)
- **Gemma 4 VRAM release fix.** `_unload_gemma4` now calls `model.cpu()` before dropping references, and `_generate_with_gemma4` detaches output tensors to CPU and explicitly frees GPU tensors plus the streamer before returning. Root cause was abandoned `_run_with_timeout` ThreadPoolExecutor threads holding live Gemma model references, preventing GC and causing a 31.70 GiB allocation on a 16 GB card. Verified end to end: telemetry reads `VRAM allocated=0.03 GiB` after unload, full sci-fi workflow completes in ~1h 14m producing a 457 MB MP4.

### Known v1.3 issues carried into v1.4 (must be fixed in v1.4-beta)
These shipped in v1.3 final because the OOM fix was the blocker. They are real and must be addressed as part of v1.4, not deferred to v1.5:
- **Critique Revision pass timeout at 600s.** Fallback text interleaves with the script body and causes downstream structural corruption.
- **Arc-Enhancer-Echo timeout at 300s.** Same failure mode — timeout fallback text leaks into the script.
- **Script structural corruption from timeout text injection.** Timeout fallbacks need a clean sentinel path that does not contaminate script output. This is the single biggest narrative-quality regression on long runs and is a Theme B blocker.

### Hardware reality (do not violate)
- RTX 5080 Laptop, 16 GB VRAM, Blackwell sm_120
- torch 2.10.0 + CUDA 13.0, Windows, Python 3.12
- SageAttention + SDPA. Flash Attention 2 is NOT available on this platform.
- 100% local, open source, offline-first. No cloud, no API keys.
- Real-world VRAM peak target: **≤ 14.5 GB**

---

## v1.4-beta — what is LEFT

Work continues on `v1.4-voice-arc-infra` as **v1.4-beta**. No point releases, no `v1.4`, no `-arc` suffix. Clean line: stays beta until explicitly signed off as `v1.4`, then straight to main.

### Theme A — Voice & Narration
- **[COMPLETED] Kokoro Announcer end-to-end test.**
- **[COMPLETED] MusicGen Theme end-to-end test.**
- **[COMPLETED] Bark female voice pool expansion.** Added `en_speaker_4`, `en_speaker_9` + Euro variants for 3-profile safety.

### Theme B — Arc Enhancer 2.0
- **[COMPLETED] Timeout fallback sentinel path.**
- **[COMPLETED] Phase A score floor retry.**
- **[COMPLETED] Plot Spine visible in `_runtime_log`.**
- **[COMPLETED] OpenClose parallel evaluator re-enable.**
- **[COMPLETED] Chunked context continuity.** 
- **[COMPLETED] Automatic scene transitions.** 
- **[COMPLETED] Subtle Pacing Overhaul.** 50% duration reduction for snappier dialogue.

### Theme C — Infrastructure (start here per roadmap order)
- **[COMPLETED] Per-node VRAM snapshot logging.** Lightweight telemetry in `_runtime_log` is successfully logging footprints per phase.
- **[COMPLETED] Experimental High-VRAM Tier (BETA).** Added ad-hoc dropdown support for Gemma 4 26B-A4B and 31B models with automated 4-bit `bitsandbytes` quantization.
- **[COMPLETED] True Single Switch.** Director inherits exactly the model the ScriptWriter defines. No VRAM leaks. No drift.
- **[COMPLETED] VRAM profile test.** Verified via `tests/test_core.py` and regression QA.
- **[COMPLETED] Project State JSON.** Integrated `ProjectStateLoader` to inject preambles directly into generation context.

### Cleanup debt
- **[COMPLETED] BOM on `nodes/gemma4_orchestrator.py`.** Removed permanently along with leftover tmp scripts.
- **[COMPLETED] Dangling procedural SFX theme nodes** in all three shipped workflows are purged. Bonus: Workflows upgraded to aesthetic horizontal grids across colored type bounds.

### v1.4 Ship Criteria MET
- Timeout sentinel fallback fix landed and verified on a long run.
- Kokoro Announcer + MusicGen Theme verified end-to-end on real hardware.
- All Theme A/B/C items landed with flawless `0-byte` and `UUID` integrity testing.
- Ready to branch and begin v1.5.
- `tests/vram_profile_test.py` green.
- Full regression sweep: AST parse, Lemmy RNG check, arc coherence check, VRAM profile test, widget audit, BOM scan.
- **Jeffrey personally confirms ship.** Only then: merge to main and tag `v1.4`.

---

## Punted to v1.5

Formally deferred out of v1.4 scope. Not started, not blocked, not forgotten.

- **Optional RVC voice-lock prototype** (was Theme A stretch goal). Opt-in post-BatchBark node for cross-episode timbre locking.
- **MusicGen interstitial wiring.** Node already emits an interstitial cue; needs a scene-transition bus in `SceneSequencer` to consume it.
- **AudioGen / AudioLDM2 SFX replacement.** Current `sfx_generator.py` is procedural numpy. Replace with a real text-to-audio model once an audiocraft-free path is validated.
- **Expanded TTS stack** (Fish Speech, XTTS). Voice identity risk, needs a narrator vs character contract first.
- **Shared latent SFX grafting.** Research, not implementation.
- **Asynchronous expert scheduling.** Would require ComfyUI queue refactor. Sequential stays.
- **High-rank LoRA fine-tuning UI.** Exceeds 16 GB headroom.
- **AI-driven automated ducking.** Full mixing stack rewrite.
- **Visual-audio lip-sync bridges.** Out of scope — this is an audio drama project.

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
Created by Jeffrey Brick. Built on top of Bark (Suno AI), Kokoro TTS, MusicGen (Meta), and Gemma 4 (Google DeepMind). Patterns from the ComfyUI Custom Node Survival Guide.
