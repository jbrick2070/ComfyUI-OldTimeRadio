# Synthesis -- 2026-04-17

**Question:** # Round-robin consult — What's the highest-ROI next move?

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

---

## ChatGPT (gpt-5.4)

**Blunt answer: B first, A second, C third.**

If Jeffrey is available for ~2 hours, the highest-ROI move is:

## 1) **Option B — land one real backend, not all of them**
Specifically: **FLUX.1-dev FP8 real-mode smoke test first**, with hard kill criteria.

That is the best next move because:

- **The biggest remaining project risk is no longer architecture; it’s reality.**  
  You already proved the plumbing in stub mode. The next thing that can actually invalidate v2.0 is: *does any of this run inside 14.5 GB on the real machine without breaking the audio-first contract?*
- **A is useful only after you have a real failure worth probing.**  
  WEDGE_PROBE is a force multiplier for debugging live graph behavior, but right now you still haven’t crossed the line from “feature-complete in stub mode” to “real model on real GPU.” Instrumenting before the first real-weight attempt is optimizing the wrong uncertainty.
- **C is good housekeeping, not product progress.**  
  Worth doing, but not while Jeffrey is online and the machine is warm.

## Recommended ranking

### **#1 — B**
But **narrow the scope aggressively**:
- Do **not** try to land FLUX + PuLID + Union Pro + LTX + Wan + Florence + SDXL in one session.
- Do **one backend at a time**, in this order:
  1. **FLUX anchor still-image path**
  2. **PuLID identity lock on top of FLUX**
  3. **Union Pro 2.0 control path**
  4. Only then touch **LTX/Wan**
  5. Florence/SDXL inpaint last

Why this order:
- It tests the **foundational visual path** first.
- It gives the fastest answer to the roadmap’s most important kill criterion:
  - **“FLUX FP8 peak > 14 GB at 1024² → fallback to SDXL+LoRA.”**
- It avoids burning the whole 2-hour window on video models before you know whether the base image stack is viable.

### **#2 — A**
Do this **immediately after the first real-weight smoke test**, especially if:
- frames drop,
- lip sync drifts,
- silent-lip appears,
- subprocess completion timing looks weird.

A becomes high-ROI the moment you have a real run exhibiting real pathology.

### **#3 — C**
Do this when Jeffrey leaves or while weights are downloading/copying/checksumming.
It’s ideal “background productivity,” not prime-time work.

---

# Why B beats A right now

You asked for the **highest ROI next move**, not the safest move.

Option A is safer.  
Option B has higher ROI because it answers the only question that matters for release viability:

> **Can the real stack produce usable visuals on this exact Blackwell laptop within the VRAM ceiling, without touching the audio path?**

Until you know that, everything else is secondary.

Stub mode proved:
- process boundaries work,
- orchestration works,
- CI-safe code paths work,
- telemetry/planner/gates are wired.

What stub mode did **not** prove:
- actual VRAM residency,
- actual model compatibility on `sm_120`,
- actual ComfyUI node behavior under load,
- actual subprocess stability with real tensors,
- actual wall-clock viability.

That’s the expensive uncertainty. Kill it first.

---

# The right way to do B in the next 2 hours

## Goal for this session
Not “make all video features work.”

Goal is:

> **Get one real image generation backend to complete deterministically enough for evaluation, while preserving audio byte identity by not touching audio at all.**

That means:
- **No ffmpeg audio re-encode**
- **No graph edits that touch muxing/audio**
- **No “quick fix” in post that could violate C7**
- Keep all testing on **visual subprocesses only**

## Concrete plan
### Phase 1 — FLUX real-mode smoke test
Use the smallest meaningful test:
- single still frame
- 1024² if that is the gate target, but be ready to step down to 896² purely for diagnosis
- one known-good prompt
- one subprocess invocation
- capture:
  - peak VRAM
  - wall-clock
  - success/failure mode
  - exact node where failure occurs if obvious

**Decision rule:**
- If **FLUX FP8 at 1024² stays under ~14 GB peak** and completes, proceed.
- If it lands in the **14.0–14.5 GB danger band**, I would already treat that as **operationally suspect** on Windows/ComfyUI, even if it technically passes once.
- If it exceeds **14.5 GB** or shows unstable allocator behavior, **stop** and invoke the roadmap fallback. Don’t spend the night “optimizing.” Your own constraints explicitly reject low-level VRAM heroics.

### Phase 2 — PuLID on top of FLUX
Only if Phase 1 passes comfortably.
- Test one portrait identity lock
- Run the SSIM identity gate you already built
- Confirm the adapter doesn’t push you over the ceiling

### Phase 3 — Union Pro 2.0
Only if FLUX + PuLID still looks viable.
- One controlled shot
- Measure whether the control path breaks the budget

### Stop there
Do **not** try to squeeze LTX/Wan into the same session unless the first three are unexpectedly smooth and fast.

---

# Why I would not start with LTX/Wan tonight

Because on a **16 GB laptop GPU**, video models are where time goes to die.

Even with your constraints:
- clips capped at ≤10s @ 24 fps,
- subprocess isolation,
- FP8 where applicable,

the likely failure modes are:
- VRAM fragmentation,
- scheduler/node incompatibility,
- long iteration loops,
- “almost works” instability.

That’s not the best first bite in a 2-hour window.

If FLUX anchor itself is not viable, then a lot of downstream assumptions change anyway.

---

# What A is worth, specifically

I do think **WEDGE_PROBE should happen soon**. Just not first.

Its value is strong if the first real-weight run produces:
- frame count mismatch,
- latent stall in one node,
- subprocess returns success but output is short,
- lip-sync drift from visual timing mismatch,
- VHS postproc timing weirdness.

In other words:
- **A is the best second move because B will likely create the first real evidence worth instrumenting.**

If you do A first, you risk spending your prime window adding observability to a system that still hasn’t crossed the real-weight threshold.

---

# What C is worth, specifically

C is worth doing, but it’s not the answer to “highest ROI next move.”

I would:
- add ignore patterns for obvious scratch artifacts,
- decide whether `config/episode_cast.txt` is intentional local state or accidental drift,
- probably keep `scripts/watcher_overrides.json` if it’s machine-local and document it,
- remove root-level junk that obscures real diffs.

But that is **maintenance ROI**, not **delivery ROI**.

---

# My recommended execution order for tonight

## Best use of Jeffrey’s 2 hours
1. **B1: Land FLUX.1-dev FP8 only**
2. **Run one real still-image smoke test**
3. If it fails ambiguously, **immediately do A**
4. If it fails clearly on VRAM > ceiling, **stop and call the fallback**
5. If it passes, optionally:
   - add PuLID
   - maybe Union Pro
6. Do C only with leftover time

---

# Hard truths / candid warnings

## 1) Your biggest risk is probably not “can FLUX run once”
It’s:
> **Can FLUX run repeatedly in ComfyUI subprocesses on Windows with enough headroom to be operationally boring?**

A one-off pass at 14.4 GB is not success.  
That’s a trap.

For a solo developer system, I’d want **practical headroom**, not just theoretical compliance. On Windows especially, “barely under cap” often becomes “mysterious intermittent failure.”

## 2) Don’t let sunk cost force FLUX if it misses the gate
You already wrote the kill criterion. Respect it.

If FLUX FP8 is over budget at the target resolution:
- **do not** start inventing bespoke memory tricks,
- **do not** spend days shaving hundreds of MB,
- **do** fall back to the simpler stack.

That is exactly in line with your stated preference:
> smallest change, biggest payoff; no low-level VRAM optimization rabbit hole.

## 3) Audio path should remain untouched tonight
Per **C7**, and your “audio is king” rule:
- do not modify muxing,
- do not alter ffmpeg flags,
- do not re-encode audio for convenience,
- do not “just test end-to-end” in a way that risks changing byte identity.

Tonight should be **visual-only validation**.

---

# Fourth option I’d propose: **B-lite**
Not a new feature, not full instrumentation, not hygiene.

## Option D — “Single-backend reality check”
A very narrow version of B:
- land only the **minimum real weights for FLUX anchor**
- run one scripted benchmark/smoke path
- record VRAM + timing + pass/fail
- make a **go/no-go decision tonight**

Why I’m naming it separately:
- Your current Option B is too broad for a 2-hour window.
- The real highest-ROI move is not “start landing real weights” in general.
- It is:
  > **force a decision on the base visual stack with the least surface area possible**

So if I’m being precise, my actual recommendation is:

## **D > A > C**, where D is a constrained subset of B.

But if forced to rank only your listed options:

## **B > A > C**

---

# Final recommendation

**Do Option B now, but ruthlessly scoped to a single real FLUX smoke test.**  
That gives the highest payoff in Jeffrey’s available window and attacks the biggest remaining uncertainty.

## Final ranking
1. **B** — highest ROI, if scoped to one backend
2. **A** — best immediate follow-up once a real failure exists
3. **C** — useful but not now

If you want, I can also give you a **tight 90-minute execution checklist** for the FLUX-first smoke test with explicit stop/go criteria.

---

## Gemini (gemini-3-pro-preview)

**1. AGREE / PARTIALLY AGREE / DISAGREE**
DISAGREE: You cannot execute a high-risk VRAM stress test (Option B) on a dirty git working tree after a 14-day sprint; Option C takes two minutes and is mandatory to protect your baseline before risking a Windows GPU hard-lock.

**2. FACTUAL ERRORS IN THE CHATGPT ANSWER**
*   **Dismissing Option C as "background productivity":** This is a dangerous junior-level mistake. Hitting a VRAM ceiling on Windows often causes hard system freezes or ComfyUI crashes. If you crash with uncommitted state (`config/episode_cast.txt`, `watcher_overrides.json`), you risk corrupting your working tree and losing the clean lockstep you just spent 14 days building.
*   **"A is useful only after you have a real failure":** False. In systems engineering, you instrument *before* the load test. If the first real-weight run OOMs or silent-lips, you want the WEDGE_PROBE data from *that* run. If you wait until it fails to add the probe, you have to run the failing, heavy workload a second time just to get the logs.
*   **Ignoring the Release Tag:** The prompt explicitly states `scripts/tag_v2.0-alpha-video-full.cmd` is ready and waiting for Jeffrey. ChatGPT completely ignored this critical milestone marker.

**3. IMPORTANT THAT WAS OMITTED**
*   **Windows VRAM Reality (Shared Memory Fallback):** On Windows, a 14.5 GB ceiling on a 16 GB card is basically zero headroom. The Desktop Window Manager (DWM) and OS easily eat 1–1.5 GB. If you exceed dedicated VRAM, Windows will silently spill into "Shared GPU Memory" (system RAM). Your ComfyUI won't crash, but your wall-clock time will explode by 10x, instantly violating your `planner.py` constraints.
*   **Download Time vs. 2-Hour Window:** ChatGPT assumes you can just "land" FLUX.1-dev (~12 GB). If those weights aren't already cached locally on the laptop, downloading them will consume Jeffrey's entire 2-hour window. 
*   **Audio Rule C7 Risk in Option A:** If WEDGE_PROBE touches or delays the subprocess muxing timing in any way, it could inadvertently cause the audio sync drift it's trying to measure.

**4. MY SHORT RECOMMENDATION**
*   **Step 1: Secure the Baseline (Option C + Tag) [10 mins].** `.gitignore` the scratch files, commit or revert `config/episode_cast.txt`, and have Jeffrey run `scripts/tag_v2.0-alpha-video-full.cmd`. Lock in the 14-day sprint success *now*.
*   **Step 2: Enable WEDGE_PROBE (Option A) [5 mins].** Turn the plumbing on while the graph is light. You want the telemetry running *before* the heavy weights hit the GPU.
*   **Step 3: FLUX FP8 Smoke Test (Option B-Lite) [Remaining Time].** Assuming weights are local, run a single 1024² still-image test in a spawn subprocess (C3). 
*   **Step 4: Watch Task Manager, Not Just ComfyUI.** During Step 3, keep Windows Task Manager open to the GPU Performance tab. If "Dedicated GPU memory" maxes out and "Shared GPU memory" spikes, you have failed the VRAM gate, regardless of whether ComfyUI throws an OOM. Fall back to SDXL immediately.

**5. UNCERTAINTIES TO VERIFY**
*   **Are the weights already on disk?** If not, abort Option B for this session and just do A, C, and start the downloads.
*   **Is `config/episode_cast.txt` drift intentional?** I need to know if this modified file is required for the real-weight run or if it's leftover test junk.
*   **Does WEDGE_PROBE alter timing?** I need to verify if the probe itself introduces I/O blocking or latency that could trigger the 20-min episode dry-run gate or affect the `wall_clock` estimator.

---

## To decide (Claude / human)

- [ ] Agree:
- [ ] Disagree:
- [ ] Facts to verify:
- [ ] Final grounded recommendation:
