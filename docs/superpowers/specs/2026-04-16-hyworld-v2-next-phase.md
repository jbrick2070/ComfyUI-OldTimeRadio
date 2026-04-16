# HyWorld v2 Next-Phase Plan -- 2026-04-16

**Branch:** `v2.0-alpha`
**Status:** Active spec. Phase A in progress.
**Owner:** Jeffrey Brick
**Source:** Synthesis of three-round ChatGPT consultation (`docs/superpowers/consultations/2026-04-16-chatgpt/`) plus current state of `otr_v2/hyworld/*.py` and the prior creative-mapping doc (`2026-04-15-otr-to-hyworld-narrative-mapping.md`).

---

## 0. Scope and constraints

This plan covers what ships next on the HyWorld stack now that the motion-stub worker (Ken Burns clips driven by ffmpeg `zoompan`) is merged on `v2.0-alpha` (commit `e616d09`).

Hard rules carried forward from CLAUDE.md and the prior PoC design:

| ID | Rule |
|----|------|
| C2 | No `CheckpointLoaderSimple` in main graph (OOM on 16 GB Blackwell). |
| C3 | All visual generation in subprocesses via `multiprocessing.get_context("spawn")`. |
| C4 | LTX-2.3 clips max 10-12 s (257 frames @ 24fps) when used. |
| C7 | Audio output byte-identical to v1.7 baseline at every gate. |
| -- | Worker must never starve Bark TTS of VRAM during the audio render window. |
| -- | Bug Bible regression (~25 tests) must stay green after every phase merge. |
| -- | All v2 work on `v2.0-alpha`. Main is frozen. |

Hardware: RTX 5080 Laptop, 16 GB VRAM, Blackwell sm_120, single GPU, no cloud. Windows, Python 3.12, torch 2.10, CUDA 13.0 in main env.

## 1. Robustness baseline (current state)

ChatGPT's robustness scorecard for the merged code is **6/10**. The five Critical issues, in priority order:

1. **Orphaned worker subprocess** if ComfyUI is cancelled or crashes mid-run -- no PID liveness sweep.
2. **STATUS.json race condition** -- worker writes, poll reads, no atomic rename, partial JSON visible to readers.
3. **Schema-blind contract files** -- malformed `script_json` can crash the worker after spawn.
4. **No GPU coordination** between worker and audio pipeline -- a real-inference worker would fight Bark for VRAM.
5. **Audio file not validated** before mux in renderer -- type/sample-rate mismatch surfaces as a silent ffmpeg failure.

Phase A fixes the first four. (5) becomes a small assertion in renderer.py during Phase A as a stretch goal.

## 2. Phase ladder

Each phase = one mergeable commit on `v2.0-alpha`. No phase touches the audio path. Every phase ships with the workflow producing a valid MP4, even if the visuals stay stub-quality.

### Phase A -- Atomic writes + VRAM coordinator + sidecar liveness (THIS PHASE)

**Goal.** Eliminate the STATUS.json race, give the worker a one-line API to claim/release the GPU, and let the poll node detect dead workers.

**Files changed (planned).**
- New: `otr_v2/hyworld/_atomic.py` -- `atomic_write_json()` helper.
- New: `otr_v2/hyworld/vram_coordinator.py` -- file-lock-based GPU gate.
- Modified: `otr_v2/hyworld/worker.py` -- atomic STATUS.json writes; coordinator scaffold (used only when GPU work is added in Phase B+).
- Modified: `otr_v2/hyworld/poll.py` -- detect dead PID via `sidecar_pid.txt`, mark `WORKER_DEAD` so the workflow can fall back without timing out.
- Modified: `otr_v2/hyworld/bridge.py` -- atomic writes for the contract files; light schema validation on `script_json`.
- New: `tests/test_hyworld_phase_a.py` -- atomic-write race smoke test, dead-PID detection test, VRAMCoordinator acquire/release/timeout tests.

**Not changed.** Renderer (visuals path), shotlist, motion-stub. Phase A is purely robustness.

**Risk gate.**
- Bug Bible regression 23/23 + 2 xfailed.
- HyWorld smoketest 30/30.
- New phase-A unit tests pass.
- Manual: kill ComfyUI mid-run; reopen and verify the next workflow run is not blocked by a stale lockfile and that the prior worker's job dir is marked WORKER_DEAD.

**Rollback.** Revert the new files; the modified files restore from git.

**Estimated.** 2-3 hours.

**Why now.** Foundation for everything GPU-bound that comes next. Cannot land Phase B (SDXL anchor) safely without the coordinator.

### Phase B -- Local anchor-image generation (Diffusion360 or SDXL+CRT-LoRA)

**Goal.** Replace solid-color PNGs with one real anchor image per shot, gated by the Phase A coordinator so Bark TTS never collides.

**Module decision (deferred to Phase B kickoff).** Either:
- **Diffusion360_ComfyUI** -- text-to-360-pano, SDXL-based. Pano-shaped; aligns with Mapping B in the creative doc. Requires the `otr_pano` env (torch 2.4 + pinned diffusers).
- **SDXL or Flux + CRT/analog LoRA** in the existing `hyworld2` env. Square-shaped output; aligns with Mapping A. No new env needed.

ChatGPT recommends Diffusion360 first for the visual payoff. I'm going to verify the version-pin pain on Blackwell + torch 2.10 main env before committing -- if `diffusers <= 0.26.0` is too far behind to install cleanly, fall back to SDXL+LoRA in the existing sidecar env.

**Risk gate.** End-to-end MP4 produced; Bug Bible green; anchor images deterministic (same script_lines + episode_title -> same anchor hash -> same image bytes).

**Rollback.** Worker switches back to placeholder PNG mode if anchor generation fails or VRAM-coordinator deny.

**Estimated.** 4-6 hours.

### Phase C -- Image-to-splat + headless splat renderer

**Goal.** Take Phase B's anchor and lift it to 3D Gaussian Splats; render a real fly-through MP4 per scene driven by the shotlist `camera` adjective.

**Modules.** `ComfyUI-Sharp` for one-shot image-to-3DGS (sub-1s, very low VRAM). For rendering, evaluate `gsplat` library or `SplatFusion` for headless PLY-to-MP4 on Windows. The rendering side is the unknown -- this is where Windows splat tooling is least mature.

**Risk gate.** Per-shot fly-through MP4 produced; renderer.py mux still byte-identical on audio (C7); no OOM on 14.5 GB ceiling.

**Estimated.** 6-8 hours (heavy on Windows compatibility verification).

### Phase D -- CRT / broadcast-static post-FX pass

**Goal.** Wire the existing `crt_postfx` boolean in renderer.py to an actual ffmpeg filter chain (scanlines, vignette, chroma bleed, gentle interlace). Keeps the SIGNAL LOST aesthetic on every output.

**Implementation.** Pure ffmpeg filtergraph; no new dependencies. Toggle is already exposed as a node input.

**Risk gate.** Output MP4 visually shows CRT FX when toggled on; audio still byte-identical (`-c:a copy` preserved).

**Estimated.** 2 hours.

### Phase E -- Caching + disk hygiene

**Goal.** Hash-key cache for anchors and splats keyed by (`style_anchor_hash`, `scene_id`, `env_prompt`). Periodic sweep of `io/hyworld_in/<job_id>/` and `io/hyworld_out/<job_id>/` for jobs older than N days or in terminal failure states.

**Risk gate.** Cache hit on rerun of same episode. Disk usage bounded after 5 sequential episode runs.

**Estimated.** 2-3 hours.

### Phase F -- Aspirational (SplaTraj + HY-Pano-2.0 swap-in)

**Goal.** When HY-Pano-2.0 ships, the visual_backend strings in shotlist.py change from `"diffusion360"` to `"hy_pano_2"` and Phase B is replaced. SplaTraj adds semantic camera paths driven by `dialogue_line_ids` instead of camera adjectives.

**Status.** Frozen until upstream models ship. Tracking only.

## 3. Notes

**Biggest design risk across the whole plan:** Splat rendering on Windows (Phase C). If the only viable headless renderer is unstable or torch-version-pinned to something incompatible with the main env, Phase C either becomes its own conda env hop or gets deferred. Early signal: `gsplat` or `SplatFusion` headless render fails to produce a valid MP4 in a 30-min spike at the start of Phase C.

**What Jeffrey should manually verify after each phase:**
- Listen to the final MP4 audio against the v1.7 baseline. Any drift, any clipping, any sample-rate weirdness = revert.
- Visually judge the first 2-3 generated MP4s for the SIGNAL LOST aesthetic. Claude cannot judge "is this on-brand" -- only Jeffrey can.

**Dropping from the prior creative-mapping doc Section 11.** The "LLM-driven shotlist" idea (Lane 2 / Translated). Deterministic mapping is robust, fast, testable, and good enough for several phases. LLM pass adds VRAM pressure and non-determinism right when we're trying to add real generative video. Defer until Phase F or later.

**FIRST MOVE.** Phase A. Atomic STATUS.json writes + VRAMCoordinator + dead-PID detection in poll. After Phase A merges, every subsequent phase has a safe runway.

---

## Appendix -- ChatGPT consultation transcript

Saved under `docs/superpowers/consultations/2026-04-16-chatgpt/`:
- `00_context_bundle.md` -- full source bundle that was sent to gpt-4.1
- `01_round1_robustness.md` -- 15-issue critique + scorecard
- `02_round2_fidelity.md` -- 5-section graphical fidelity / replacement modules / VRAM coordinator design
- `03_round3_plan.md` -- 6-phase plan + risk notes + first-move recommendation
- `transcript.json` -- full multi-round message log

Model used: gpt-4.1 (gpt-5 not available on this account). Total ~52s API time, ~17K chars of replies.
