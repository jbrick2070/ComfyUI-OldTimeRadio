# ComfyUI-OldTimeRadio — ROADMAP

---

## Current state (as of 2026-04-08)

- **Last shipped tag:** `v1.3` → `ddbed87` (v1.3 final, includes Gemma 4 VRAM release fix)
- **Active branch:** `v1.4-voice-arc-infra` (working as v1.4-beta until explicit ship signoff)
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

Work continues on `v1.4-voice-arc-infra` as **v1.4-beta**. No point releases, no `v1.4.1`, no `-arc` suffix. Clean line: stays beta until explicitly signed off as `v1.4`, then straight to main.

### Theme A — Voice & Narration
- **Kokoro Announcer end-to-end test.** Node shipped. Needs full-episode hardware run with measured VRAM delta.
- **MusicGen Theme end-to-end test.** Node shipped. Needs full-episode hardware run with measured VRAM delta during Gemma → MusicGen → Bark handoff.
- **Bark female voice pool expansion** (Bug Bible `10.06`). Not started. Add at least one additional distinct female preset, keep the 3-female safety rule.

### Theme B — Arc Enhancer 2.0
- **Timeout fallback sentinel path (BLOCKER).** Critique Revision (600s) and Arc-Enhancer-Echo (300s) currently inject fallback text directly into the script body on timeout, corrupting downstream structure. Fix: route timeout fallbacks through a sentinel that is detected and skipped by the assembler, not concatenated. This must land before any other Theme B work.
- **Phase A score floor retry.** If arc score < 3/5, auto-retry Phase B once. Log both attempts.
- **Plot Spine visible in `_runtime_log`.** Currently fed to Phase B silently.
- **OpenClose parallel evaluator re-enable.** Currently gated (`ENABLE_3_OUTLINE_EVALUATOR = False`). Re-activate with v1.3 timeout/budget fixes as per-outline contract.
- **Chunked context continuity.** Replace `acts[-1][:3000]` with sentence-boundary-aware truncation.
- **Automatic scene transitions.** Programmatic `[TRANSITION: ...]` injection on weak handoffs.

### Theme C — Infrastructure (start here per roadmap order)
- **Per-node VRAM snapshot logging.** Lightweight telemetry in `_runtime_log`. Prerequisite for the VRAM profile test.
- **VRAM profile test.** `tests/vram_profile_test.py` snapshotting VRAM between major nodes, assert ≤ 14.5 GB peak.
- **Project State JSON.** Per-series "bible" file. Read-only during generation, writable between episodes. Character voice locks, forbidden patterns, tone contracts.

### Cleanup debt
- **BOM on `nodes/gemma4_orchestrator.py`.** Pre-existing UTF-8 BOM violates the CLAUDE.md no-BOM rule. Strip it.
- **Dangling procedural SFX theme nodes** in all three shipped workflows. `OTR_SFXGenerator` stubs that used to feed opening/closing theme audio are still present but disconnected. Remove them for a clean v1.4 JSON.

### v1.4-beta → v1.4 ship criteria
- Timeout sentinel fallback fix landed and verified on a long run.
- Kokoro Announcer + MusicGen Theme verified end-to-end on real hardware with VRAM telemetry ≤ 14.5 GB peak.
- All Theme A/B/C items landed OR explicitly punted to v1.5 with VRAM numbers.
- OpenClose parallel evaluator stable across a 10-batch run.
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
