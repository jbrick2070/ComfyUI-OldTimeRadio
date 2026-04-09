# ComfyUI-OldTimeRadio — ROADMAP

---

## Current state (as of 2026-04-08)

- **Last shipped tag:** v1.3 (main)
- **Active branch:** `v1.4-voice-arc-infra`
- **Status:** v1.4 partial. Two Theme A nodes landed in code but **NOT battle-tested**. A v1.3 full-workflow OOM regression was observed on the RTX 5080 Laptop and must be investigated before any v1.4 hardware pass.
- **Repo:** https://github.com/jbrick2070/ComfyUI-OldTimeRadio

### Hardware reality (do not violate)
- RTX 5080 Laptop, 16 GB VRAM, Blackwell sm_120
- torch 2.10.0 + CUDA 13.0, Windows, Python 3.12
- SageAttention + SDPA. Flash Attention 2 is NOT available on this platform.
- 100% local, open source, offline-first. No cloud, no API keys.
- Real-world VRAM peak target: **≤ 14.5 GB**

---

## v1.3 full-workflow OOM — BLOCKER

The v1.3 full workflow OOM'd on the 5080 after v1.4 branch work. This must be root-caused and fixed before v1.4 hardware regression can run. Until this is resolved, all v1.4 end-to-end testing is blocked.

Investigation checklist:
- Repro on clean `main` (tag `v1.3`) to confirm this is a v1.3 regression and not a v1.4 branch side-effect.
- VRAM snapshot between Gemma unload → BatchBark load (suspect: Gemma not fully releasing).
- Check `torch.cuda.empty_cache()` call sites in `gemma4_orchestrator.py` and `batch_bark_generator.py`.
- Confirm MusicGen / Kokoro code paths are NOT active in the failing run (v1.3 full should not touch them).
- Bisect: `v1.3` tag → `v1.4-voice-arc-infra` HEAD.

---

## v1.4 — what is LEFT

Two nodes landed in code but are unverified on hardware. Everything else in the original v1.4 plan is still open.

### Theme A — Voice & Narration
- **Kokoro Announcer end-to-end test.** Node shipped. Needs full-episode hardware run with measured VRAM delta.
- **MusicGen Theme end-to-end test.** Node shipped. Needs full-episode hardware run with measured VRAM delta during Gemma → MusicGen → Bark handoff.
- **Bark female voice pool expansion** (Bug Bible `10.06`). Not started. Add at least one additional distinct female preset, keep the 3-female safety rule.

### Theme B — Arc Enhancer 2.0
- **Phase A score floor retry.** If arc score < 3/5, auto-retry Phase B once. Log both attempts.
- **Plot Spine visible in `_runtime_log`.** Currently fed to Phase B silently.
- **OpenClose parallel evaluator re-enable.** Currently gated (`ENABLE_3_OUTLINE_EVALUATOR = False`). Re-activate with v1.3 timeout/budget fixes as per-outline contract.
- **Chunked context continuity.** Replace `acts[-1][:3000]` with sentence-boundary-aware truncation.
- **Automatic scene transitions.** Programmatic `[TRANSITION: ...]` injection on weak handoffs.

### Theme C — Infrastructure
- **Project State JSON.** Per-series "bible" file. Read-only during generation, writable between episodes. Character voice locks, forbidden patterns, tone contracts.
- **VRAM profile test.** `tests/vram_profile_test.py` snapshotting VRAM between major nodes, assert ≤ 14.5 GB peak.
- **Per-node VRAM snapshot logging** in `_runtime_log`.

### Cleanup debt
- **BOM on `nodes/gemma4_orchestrator.py`.** Pre-existing UTF-8 BOM violates the CLAUDE.md no-BOM rule. Strip it.
- **Dangling procedural SFX theme nodes** in all three shipped workflows. `OTR_SFXGenerator` stubs that used to feed opening/closing theme audio are still present but disconnected. Remove them for a clean v1.4 JSON.

### v1.4 Ship Criteria
- v1.3 OOM root-caused and fixed.
- All Theme A/B/C items landed OR explicitly punted to v1.5 with VRAM numbers.
- Kokoro Announcer + MusicGen Theme verified end-to-end on real hardware.
- OpenClose parallel evaluator stable across a 10-batch run.
- `tests/vram_profile_test.py` green (≤ 14.5 GB peak).
- Full regression sweep: AST parse, Lemmy RNG check, arc coherence check, VRAM profile test.

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
