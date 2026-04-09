# ComfyUI-OldTimeRadio — ROADMAP

---

## NEW CONVERSATION HANDOFF — READ THIS FIRST

If you are a fresh AI session picking up this project, read this section top to bottom before touching anything. Do not skip it. Do not jump to the code.

### Current shipped state
- **Version:** v1.3 (tag pushed, locked)
- **Branch:** main
- **Last commit:** `3656508` (docs: update README to v1.3)
- **Tag:** `v1.3` → `a0b9ccd`
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

### Hardware reality (do not violate)
- **RTX 5080 Laptop, 16 GB VRAM, Blackwell sm_120**
- **torch 2.10.0 + CUDA 13.0 on Windows**
- **SageAttention + SDPA active. Flash Attention 2 is NOT available on this platform — no prebuilt wheel exists. Do not attempt install.**
- **100% local, open source, offline-first. No cloud. No API keys.**
- Real-world VRAM peak target: **≤ 14.5 GB** (leaves room for fragmentation)

### Standing rules (DO NOT VIOLATE)
1. Every story has a start, middle, and end. Arcs matter.
2. Safe for work, non-violent.
3. No curse words.
4. Clean code, clean logs, meaningful names. The reader and end user matter.
5. Never skip regression testing for widget errors and bugs.
6. Always follow the ComfyUI Custom Node Survival Guide bug bible:
   https://github.com/jbrick2070/comfyui-custom-node-survival-guide/blob/main/BUG_BIBLE.yaml
7. Git pushes are the user's responsibility. Hand them a PowerShell block with `cd <absolute path>` as line 1. Never assume they are in the right directory.
8. After every push, lockstep verify: local HEAD vs GitHub HEAD, scan for 0-byte files, BOM issues, truncation, missing node registrations. Wait and re-check until confirmed.
9. Do not make assumptions about file encoding on Windows. PowerShell mojibake is real. Verify with `Get-Content | Select-String` before declaring done.
10. If a tool or library is not available on Windows + CUDA 13 + torch 2.10 + Blackwell sm_120, do not chase it. Document the dead end so the next session does not repeat the search.

### First moves for the next session
```powershell
cd C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
git status
git log --oneline -5
git fetch origin
git log --oneline origin/main -5
```
Confirm local HEAD == origin/main HEAD before touching anything. Then read the v1.4 plan below and pick exactly one theme to start on. Do not start all three in parallel.

---

## v1.4 — Voice, Arc Enhancer 2.0, and Infrastructure Hardening

**Theme:** Convert v1.3's narrative wins into production-grade consistency.

Three focused themes. No new model stacks, no cloud, no 27B plans. Everything stays inside the 16 GB VRAM ceiling on the RTX 5080 Laptop.

### Theme A — Voice & Narration (highest user-visible win)

- **Kokoro Announcer. [IN PROGRESS — node landed, needs end-to-end episode test]** Dedicated non-Bark narrator track. Clean separation of announcer bus from character voices so ANNOUNCER has a broadcast-ready "Voice of God" cadence instead of sharing the Bark pool. Implemented as `nodes/kokoro_announcer.py` (`OTR_KokoroAnnouncer`, display name "🎙️ Kokoro Announcer"). Picks one voice per episode from a curated 4-voice British grab bag — `bm_george` (BBC authoritative male), `bm_fable` (documentary relaxed male), `bf_emma` (BBC authoritative female), `bf_lily` (documentary relaxed female) — seeded from the episode seed so runs are deterministic. `BatchBarkGenerator` now filters ANNOUNCER lines out of its dialogue pool (logged as "skipped N ANNOUNCER lines — routed to Kokoro bus") and `SceneSequencer` consumes the Kokoro output via a new optional `announcer_audio_clips` input with a separate `announcer_clip_idx`. Voice `.pt` files lazy-download from `1038lab/KokoroTTS` via `huggingface_hub` on first run. All three shipped workflow JSONs (`old_time_radio_scifi_full`, `old_time_radio_scifi_lite`, `old_time_radio_test`) have the node wired in. **Remaining:** full-episode regression pass on hardware with measured VRAM delta, then tick off "Kokoro Announcer integrated and tested on a full episode end to end" in the v1.4 ship criteria.
- **Bark female voice pool expansion.** Address the known "female characters all sound the same" issue (Bug Bible `10.06`). Add at least one additional distinct female preset while keeping the 3-female safety rule.
- **Optional stretch:** RVC voice-lock prototype as an **opt-in** node (post-BatchBark). Not default, not stabilization, just a low-risk experiment for cross-episode timbre locking. If it explodes scope, punt it to v1.5 without guilt.

### Theme B — Arc Enhancer 2.0 (build directly on v1.3 Arc Enhancer)

- **Phase A score floor retry.** If the arc score from Phase A is less than 3/5, auto-retry Phase B exactly once. Log both attempts.
- **Plot Spine visible in log.** Currently extracted and fed to Phase B but not surfaced. Add a one-line `_runtime_log` so the user can eyeball what the bookend rewriter actually saw.
- **OpenClose parallel evaluator re-enable.** Currently gated (`ENABLE_3_OUTLINE_EVALUATOR = False`) because of timeout risk. Re-activate using the v1.3 timeout/budget fixes (450 tok, 480s wall) as the per-outline contract.
- **Chunked context continuity.** Replace the old hard `acts[-1][:3000]` slicing with sentence-boundary-aware truncation.
- **Automatic scene transitions.** Programmatic `[TRANSITION: ...]` injection between scenes when the arc enhancer detects weak handoffs.

### Theme C — Infrastructure (observability first)

- **Project State JSON.** Per-series / per-session "bible" file. Stores character voice locks, forbidden patterns, tone contracts, locked decisions. **Read-only during generation.** Writable only between episodes. Prevents LLM context bloat and contradictory logic across a series.
- **VRAM profile test.** New `tests/vram_profile_test.py` that snapshots VRAM between major nodes and asserts ≤ 14.5 GB peak on the RTX 5080 Laptop.
- **Per-node VRAM snapshot logging.** Lightweight telemetry in `_runtime_log` so every run shows exact VRAM high-water marks per phase.

### v1.4 Ship Criteria
- All items in Themes A–C landed OR explicitly deferred to v1.5 with justification and VRAM numbers.
- Kokoro Announcer integrated and tested on a full episode end to end.
- OpenClose parallel evaluator back online and stable across a 10-batch run.
- Project State JSON schema finalized and wired into `Gemma4ScriptWriter` and `Gemma4Director`.
- `tests/vram_profile_test.py` passing (≤ 14.5 GB peak) on the RTX 5080 Laptop.
- Regression sweep passes: AST parse, Lemmy RNG check, arc coherence check, VRAM profile test.
- Tagged `v1.4` only when green.

### First moves for v1.4 work
1. `git checkout -b v1.4-voice-arc-infra`
2. Implement **Theme C first** (Project State JSON + VRAM test). Lowest risk, highest observability win. It tells you the truth about what Themes A and B are actually costing.
3. Then **Theme B** (Arc Enhancer 2.0). Re-uses the exact `_arc_check_and_rewrite_bookends` and `_score_arc_coherence` methods already shipped in v1.3.
4. Finish with **Theme A** (Kokoro + female pool + optional RVC stretch).
5. Full regression + VRAM profile test before any push.

---

## v1.5 — Deferred (do not start in v1.4)

These are genuinely interesting but out of scope for the 1.4 stability/consistency pass. Documented here so the next session knows they exist and does not stumble into them by accident.

- **Expanded TTS stack.** Fish Speech, XTTS. Mixing paradigms creates a voice identity crisis and pipeline fragmentation risk. Do not integrate without a clear separation of narrator vs character roles.
- **Shared latent SFX grafting.** Text-to-speech latents mixed with text-to-audio latents without a defined embedding contract leads to semantic mismatch artifacts. Research phase, not implementation phase.
- **Asynchronous expert scheduling.** Separate CUDA streams for LLM + TTS + TTA would require a ComfyUI queue refactor. Sequential is slower but radically more stable.
- **High-rank LoRA fine-tuning UI.** Training 7B+ models exceeds the 16 GB VRAM headroom. Inference-only for now.
- **AI-driven automated ducking.** Requires a full mixing stack rewrite.
- **Visual-audio lip-sync bridges.** Out of scope. This is an audio drama project, not video.

---

## Hard Rules (permanent)

- 16 GB VRAM ceiling, 14.5 GB real-world target.
- 100% local, open source, offline-first. No cloud. No API keys.
- Sequential execution only. ComfyUI manages the queue.
- Flash Attention 2 is not available on this platform. Stop asking.
- Git pushes via PowerShell blocks handed to the user with `cd` baked in.
- Lockstep verify local HEAD vs GitHub HEAD after every push.
- Encoding on Windows is UTF-8 no BOM. Verify before declaring done.
- Every change ships with a regression pass.

---

## Credits
Created by Jeffrey Brick. Built on top of Bark (Suno AI) and Gemma 4 (Google DeepMind). Patterns from the ComfyUI Custom Node Survival Guide.
