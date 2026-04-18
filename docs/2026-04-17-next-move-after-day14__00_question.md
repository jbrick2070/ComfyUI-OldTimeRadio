# Question -- 2026-04-17

# Round-robin consult — What's the highest-ROI next move?

## Current state (as of 2026-04-17)
- **Repo:** ComfyUI-OldTimeRadio
- **Branch:** v2.0-alpha-video-stack @ 15fe48e (pushed, lockstep clean)
- **Just shipped:** 14-day autonomous sprint. All Day 1-14 rows in ROADMAP marked DONE. v2.0 video stack is feature-complete in **stub mode** (CI-safe, no GPU weights). Backends: flux_anchor, pulid_portrait, flux_keyframe (FLUX+ControlNet Union Pro 2.0), ltx_motion (LTX-2.3 I2V), wan21_loop (Wan2.1 1.3B I2V), florence2_sdxl_comp, placeholder_test. Plus: VHS postproc, planner.py, wall_clock estimator, character_regression SSIM gate, LHM telemetry poller, 20-min episode dry-run gate.
- **Also just shipped today:** BUG-LOCAL-042 marked [FIXED] — stale Windows __pycache__ self-resolved; tests/test_core.py now 103/103 green. Bug Bible regression 24 passed + 2 xfailed.
- **ComfyUI Desktop:** Just restarted on :8000. Alive. Jeffrey is available.

## Platform constraints (immutable)
- RTX 5080 Laptop, 16 GB VRAM, Blackwell sm_120, single GPU, no cloud
- Windows, Python 3.12, torch 2.10.0, CUDA 13.0
- SageAttention + SDPA only (NO flash_attn — do not chase)
- 14.5 GB VRAM ceiling, 100% local, offline-first

## Immutable project rules
- C2: no CheckpointLoaderSimple in main graph (OOM on 16 GB)
- C3: all visual gen in spawn subprocesses
- C4: LTX/Wan clips ≤ 10 s @ 24 fps
- C5: FP8 e4m3fn on Blackwell
- C6: IP-Adapter environments-only
- C7: audio byte-identical to v1.5 baseline (ffmpeg -c:a copy everywhere)
- **Audio is king.** If video breaks audio, revert immediately.
- Only Jeffrey tags releases — `scripts/tag_v2.0-alpha-video-full.cmd` is ready, he'll run when ready.

## The three real candidates on the table

### Option A — WEDGE_PROBE instrumentation (task #32, unblocked by restart)
- Was held pending ComfyUI restart. Restart happened.
- This is diagnostic plumbing, not new features.
- Purpose: trace which node in the live ComfyUI graph is dropping frames / silent-lipping / drifting audio sync during episode runs.
- Risk: low. Plumbing only, no graph changes. Touches live nodes.
- Return: lets us debug the NEXT live episode run by reading probe output instead of guessing from timestamps.

### Option B — Start landing real weights and exercising deferred gates
- Every Day 2-7 backend has a real-mode code path behind OTR_*_STUB=1 envvars. Weights haven't landed yet on disk; stub mode was proving the plumbing works.
- Weights to land: FLUX.1-dev FP8 (~12 GB), PuLID-FLUX adapter (~2 GB), ControlNet Union Pro 2.0 (~2 GB), LTX-Video 2.3, Wan2.1 1.3B I2V, Florence-2, SDXL inpaint.
- Deferred gates that need exercising once weights land: FLUX ≤12.5 GB @ 1024², PuLID face-embed SSIM identity lock, Union Pro 2.0 ≤13.5 GB, LTX-2.3 + Wan2.1 ≤10-14.5 GB, Florence-2 mask quality.
- Risk: high. VRAM ceilings are theoretical until real weights hit the GPU. Kill criteria in ROADMAP say "FLUX FP8 peak > 14 GB on 1024²" → fallback to SDXL+LoRA.
- Return: actual visual output on a real episode. The entire 14-day sprint was staging for this.

### Option C — Repo hygiene
- `git status` shows ~25 untracked scratch files at repo root (`_bug042_*.txt`, `_run_*.cmd`, `_do_commit.log`, `day13_*.txt`, `_launch_comfy.bat`, stray `renderer.py`, `docs/LLM_SHOWDOWN.md`, several `scripts/_*.ps1` / `_probe*.py` / `_kill*.ps1` / `_restart*.ps1` / `yoga_watchdog.py`).
- Also `M config/episode_cast.txt` and `M scripts/watcher_overrides.json` in tracked-changes.
- Risk: near-zero. Just `.gitignore` scratch patterns + delete dead files + decide about `config/episode_cast.txt` drift.
- Return: clean `git status`, easier to see real changes at a glance. But doesn't move the product forward.

## Question for you
Which is the highest-ROI next move right now, and why? Rank A / B / C and give your reasoning. If you'd propose a fourth option I'm missing, name it.

Constraints on your answer:
- Respect the platform pins (no cloud, no flash_attn, 14.5 GB ceiling).
- "Audio is king" — if your recommendation touches the audio path at all, flag it.
- Jeffrey is awake and available for ~2 hours before likely getting pulled away. Favour moves that give concrete returns in that window.
- Be blunt. I'd rather have a wrong opinion defended crisply than a hedged "it depends."
