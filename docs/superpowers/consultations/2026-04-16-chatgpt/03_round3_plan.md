# 03_round3_plan.md -- model=gpt-4.1 elapsed=13.7s

## User prompt

ROUND 3 of 3 -- Prioritized v2.0 next-phase plan.

You've critiqued robustness (Round 1) and proposed graphical upgrades + missing modules (Round 2). Now synthesize.

YOUR TASK: Produce the v2.0 next-phase implementation plan as a sequence of PHASES. Each phase = ONE mergeable PR/commit on the v2.0-alpha branch.

Constraints to honor:
- Each phase must keep the audio pipeline byte-identical (C7) and the Bug Bible regression suite (~25 tests) green.
- Each phase must work end-to-end before merge: the workflow still produces a valid MP4, even if the visuals are still stub-quality.
- Worker must not contend for GPU with Bark TTS during the audio render window.
- VRAM ceiling 14.5 GB peak. No CheckpointLoaderSimple in main graph (C2). Subprocess + spawn for any GPU-heavy work (C3).
- Solo developer, async swimming breaks, prefers honest plans over ambitious ones.
- Branch is v2.0-alpha. Main is frozen.

DELIVERABLE:

```
PHASE A -- <one-line title>
  Goal:        <what this unlocks>
  Files:       <which files change>
  Risk gate:   <what test/probe must pass before merge>
  Rollback:    <how to revert cleanly if it breaks>
  Estimated:   <hours of human + Claude work>
  Why now:     <why this and not Phase B/C>

PHASE B -- ...
PHASE C -- ...
PHASE D -- ...
PHASE E -- ...
```

Aim for 4-6 phases. Order them strictly by lowest-risk-highest-value first. The first phase must be something that could realistically ship today. The last phase can be aspirational ("when HY-Pano 2.0 ships").

After the phase list, add a NOTES section covering:
- The single biggest design risk across the whole plan, and the early signal that would tell us we're hitting it.
- One thing Jeffrey should manually test/verify himself that Claude can't (e.g. visual quality judgment, listening for audio drift).
- One thing to drop from the original design doc Section 11 that you now think is wrong or over-scoped.

End with a one-line FIRST-MOVE recommendation for the very next coding session.

---

## Assistant reply

```
PHASE A -- Atomic STATUS.json + VRAM Coordinator (File Lock)
  Goal:        Eliminate race conditions on STATUS.json and guarantee no worker-GPU contention with Bark TTS.
  Files:       otr_v2/hyworld/worker.py, otr_v2/hyworld/poll.py, otr_v2/hyworld/bridge.py, (new) otr_v2/hyworld/vram_coordinator.py
  Risk gate:   All Bug Bible regressions pass; run a 2-episode test with Bark and worker in parallel, confirm no OOM or audio drift.
  Rollback:    Revert the new file and STATUS.json write logic; restore previous worker and poll.py.
  Estimated:   2-3 hours.
  Why now:     Fixes the most critical robustness holes, unblocks all future visual upgrades, and is safe to merge without touching visuals.

PHASE B -- Deterministic Anchor Image Generation via SDXL/ComfyUI Node
  Goal:        Replace solid-color PNGs with deterministic, locally-generated anchor images per scene using SDXL (e.g., DreamShaper XL + CRT LoRA).
  Files:       otr_v2/hyworld/worker.py, otr_v2/hyworld/shotlist.py, (possibly) new comfyui_anchor_image.py, docs/superpowers/specs/2026-04-15-hyworld-poc-design.md
  Risk gate:   End-to-end MP4 is produced with correct scene count; Bug Bible green; anchor images are visually scene-specific and deterministic.
  Rollback:    Switch worker back to placeholder PNG mode.
  Estimated:   3-4 hours.
  Why now:     Huge visual upgrade with minimal risk, no GPU contention if run after Bark, and keeps audio untouched.

PHASE C -- CRT/Analog Post-FX Pass in Renderer
  Goal:        Add CRT/broadcast-static post-processing to final MP4 using ComfyUI-VideoFX or Advanced-PostProcessing nodes.
  Files:       otr_v2/hyworld/renderer.py, requirements.txt, docs/README.md
  Risk gate:   Output MP4 passes visual inspection (CRT FX visible), Bug Bible green, and audio is byte-identical.
  Rollback:    Remove post-FX call, revert to plain video.
  Estimated:   2 hours.
  Why now:     Makes the visuals on-brand, can be done independently of 3D/splat work, and is trivial to toggle off if needed.

PHASE D -- Splat Generation and Rendering (ComfyUI-Sharp + SplatFusion)
  Goal:        Generate 3D Gaussian splats from anchor images and render fly-through MP4s for each scene.
  Files:       otr_v2/hyworld/worker.py, otr_v2/hyworld/renderer.py, (new) splat_render.py, docs/superpowers/specs/2026-04-15-hyworld-poc-design.md
  Risk gate:   End-to-end MP4 with splat-based visuals, no OOM, Bug Bible green, and audio byte-identical; fallback to anchor images if splat fails.
  Rollback:    Disable splat path, revert to anchor image video.
  Estimated:   4-6 hours (integration + Windows testing).
  Why now:     Unlocks real 3D scene motion, but only after anchor images and CRT FX are proven stable.

PHASE E -- Caching & Disk Hygiene
  Goal:        Implement caching for anchor images/splats and periodic cleanup of old job dirs to prevent disk bloat.
  Files:       otr_v2/hyworld/worker.py, otr_v2/hyworld/bridge.py, otr_v2/hyworld/renderer.py, (new) cache_utils.py
  Risk gate:   Confirm cache hits for repeated scenes, disk usage remains bounded after 5+ runs, Bug Bible green.
  Rollback:    Remove cache/disk sweep logic, revert to always-generate.
  Estimated:   2 hours.
  Why now:     Reduces recomputation, keeps disk usage sane, and is safest after core pipeline is stable.

PHASE F -- Aspirational: Semantic Camera Path Planning (SplaTraj) & HY-Pano 2.0 Integration
  Goal:        Add semantic camera trajectories and support for HY-Pano 2.0 when available.
  Files:       otr_v2/hyworld/worker.py, otr_v2/hyworld/shotlist.py, docs/superpowers/specs/2026-04-15-hyworld-poc-design.md
  Risk gate:   Only merge when HY-Pano 2.0 is locally testable and SplaTraj integration is proven not to break audio or VRAM ceiling.
  Rollback:    Keep as feature branch until models are stable.
  Estimated:   N/A (future-facing).
  Why now:     This is the endgame for full visual fidelity, but blocked on upstream model releases.

```

**NOTES**

- **Biggest design risk:** Integration of splat rendering (Phase D) on Windows—driver/VRAM issues or splat renderer instability could block the pipeline.  
  **Early signal:** SplatFusion or ComfyUI-3DGS fails to render MP4s reliably or causes OOM/crashes during test runs.

- **Manual test for Jeffrey:** After each phase, visually inspect the first 2-3 generated MP4s for scene fidelity, CRT FX, and—critically—listen for any audio drift or glitches that would break C7 (byte-identical audio).

- **Drop from Section 11:** Defer or drop "LLM-driven shotlist" for now—deterministic mapping is robust, testable, and sufficient for PoC; LLM pass adds unnecessary VRAM pressure and non-determinism at this stage.

---

**FIRST-MOVE RECOMMENDATION:**  
Implement atomic STATUS.json writes and add a file-based VRAM lock in the worker and poll nodes—this unlocks all safe visual upgrades and closes the most critical robustness gaps.
