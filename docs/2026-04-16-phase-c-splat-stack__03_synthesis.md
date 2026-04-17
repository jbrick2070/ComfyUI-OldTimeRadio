# Synthesis -- 2026-04-16

**Question:** # OTR v2.0 Phase C — Splat-Rendering Architecture Pick

## Context

Project: ComfyUI-OldTimeRadio (v2.0-alpha). An AI radio-drama pipeline where the
audio (~12-18 min episode Bark TTS output) is fully produced and is the ground
truth: visuals must stretch to the audio duration, never the reverse.

**Platform (immutable):**
- Windows 11, RTX 5080 Laptop, 16 GB VRAM, Blackwell sm_120, single GPU, no cloud
- Python 3.12, torch 2.10.0, CUDA 13.0, SageAttention + SDPA
- Flash Attention 2/3 not available on Blackwell - do not chase
- 100% local / offline-first / no API keys / open source only
- VRAM ceiling 14.5 GB real-world target
- Audio byte-identical to v1.7 baseline at every gate (C7)

**State of the pipeline:**
- Phase A (robustness): SHIPPED. _atomic.py (atomic JSON/text writes with
  Windows retry), vram_coordinator.py (file-lock GPU gate with dead-PID
  reclaim), worker.py wrapped in VRAMCoordinator.acquire(), poll.py detects
  dead sidecar via sidecar_pid.txt, bridge.py validates script_json schema.
  24/24 Phase A tests green.
- Phase B (SDXL anchor image): PARTIALLY DONE. SD 1.5 .ckpt sidecar loading
  resolved via a four-layer fix (torch.load weights_only override, pytorch_
  lightning sys.modules shim, local original_config + local_files_only, tqdm
  disable on both load and inference paths). SD 1.5 anchors visually rejected
  for SIGNAL LOST mood; pivoting to SDXL 1.0 base + period LoRA.
- Phase C (this question): SPLAT RENDERING. The active 2026-04-16 spec defines
  Phase C as image-to-splat + headless splat renderer. An older 2026-04-12
  spec proposed LTX-2.3 video diffusion instead; that path was abandoned
  because 10-12s clip cap + non-deterministic cross-fade chunks fight the
  narrative-first duration contract.

**The time-stretch contract:**
Per-scene audio duration is a given float (e.g., 4:32.1 sec). Visual output
for that scene MUST be exactly that duration. With splats the knob is
parametric: a camera trajectory through fixed geometry is scaled by a single
float target_duration / default_duration to produce exactly the right
frame count. This is the central artistic / data-science reason for picking
splats over video diffusion.

**Shotlist camera adjectives** already drive the stub ffmpeg zoompan. They
need to map to splat camera paths:
- "locked wide" -> zero translation, micro-jitter
- "slow push" -> forward translate along -Z
- "handheld drift" -> perlin-noise jitter around origin
- "pull out" -> backward translate along +Z
- "pan across" -> yaw rotation

## The four candidate stacks

| # | Image->3DGS | PLY->MP4 renderer | Install story (unknown) |
|---|-------------|-------------------|-------------------------|
| 1 | ComfyUI-Sharp (Apple SHARP wrapper) | gsplat (nerfstudio-project) direct Python loop | gsplat JIT/source build on py3.12 + CUDA13 + Blackwell |
| 2 | ComfyUI-Sharp | SplatFusion (ashawkey) as black-box renderer | SplatFusion Windows headless status |
| 3 | ComfyUI-3D-Pack (MrForExample umbrella) | same umbrella | Umbrella install is notoriously heavy |
| 4 | **SKIP SPLATS.** SDXL anchor + ffmpeg zoompan fly-through indefinitely | ffmpeg zoompan only | Zero new deps. Visual ceiling is Ken Burns. |

## Four questions to both models

**A.** Which of the four stacks is lowest risk for this exact platform
(Windows + Blackwell + Python 3.12 + torch 2.10 + CUDA 13) and has the
clearest install story? Consider: existence of pre-built wheels, whether
CUDA kernels JIT-build safely, whether the upstream repo has Blackwell
issues logged, dependency conflicts with torch 2.10.

**B.** Is a parametric-camera-path fly-through
(path_speed = default_speed / (target_duration_sec / default_duration_sec))
the right time-stretch model for narrative-audio-length-first MP4 output?
Or is there a better approach - e.g., variable-speed path segments keyed to
dialogue-line boundaries, easing curves per camera adjective, or a different
abstraction entirely? This is the artistic + data-science core of the
design.

**C.** What is the single biggest architectural risk in the Phase C design
as described (image -> PLY -> camera path -> gsplat rasterize per frame ->
ffmpeg assemble -> mux to byte-identical audio)? Not "list all risks" -
pick the one most likely to kill the phase.

**D.** What component is missing from the design that we should have named?
Candidates to consider but not limit to: depth-map pre-pass to condition
Sharp, MiDaS or Marigold for monocular depth, trajectory smoothing /
easing, keyframe interpolation, per-shot style consistency across a scene,
output frame color management / gamma / tone-map for CRT post-FX in Phase D,
deterministic seeding for reproducible per-frame output.

Please keep answers structured: label A, B, C, D. Be specific to this
platform and this project - no generic AI advice. If you disagree with the
framing itself, say so and explain why.

---

## ChatGPT (gpt-5.4)

A. **Lowest-risk stack: #4 now, #1 only as an R&D branch**

For **this exact machine/runtime** — Windows 11, **Python 3.12**, **torch 2.10**, **CUDA 13**, **Blackwell sm_120**, single RTX 5080 Laptop, no cloud, no appetite for kernel-chasing — the safest production pick is:

- **Production:** **#4 SDXL anchor + ffmpeg zoompan**
- **Experimental branch:** **#1 ComfyUI-Sharp + gsplat**, but only if you are willing to treat it as a research spike, not Phase C’s critical path

Why:

1. **#4 has the clearest install story: basically none**
   - You already have ffmpeg.
   - It preserves your current duration contract trivially.
   - It adds no CUDA extension risk, no pybind/JIT build risk, no torch ABI mismatch risk.
   - It is the only option I’d call compatible with your “smallest change, biggest payoff” rule.

2. **#1 is the most plausible splat path, but still risky on this platform**
   - **gsplat** is the least “mystery meat” renderer in your list because it’s a focused renderer library rather than a giant umbrella.
   - But your stack is hostile to CUDA-extension assumptions:
     - **Python 3.12** still breaks older build scripts surprisingly often.
     - **torch 2.10 + CUDA 13 + sm_120** is ahead of what many repos test.
     - If gsplat relies on custom CUDA kernels compiled against torch headers, you are in “maybe it builds, maybe it doesn’t” territory.
   - The key issue is not runtime VRAM; it’s **build/install certainty**.

3. **#2 SplatFusion black-box renderer is higher risk than #1**
   - I would expect more hidden assumptions around Linux, OpenGL/EGL, viewer stack, or training-time dependencies.
   - “Windows headless status” is exactly the kind of thing that kills solo-dev schedules.
   - Black-box renderer means less control when you need deterministic frame output.

4. **#3 ComfyUI-3D-Pack is the highest integration risk**
   - Heavy umbrella repos are where dependency conflicts go to breed.
   - On your stack, the likely failure mode is not one bug; it’s a week of environment archaeology.
   - This directly violates your stated preference to avoid low-level platform wrestling.

### My candid recommendation
If you must choose a splat architecture **today**, pick **#1** as the only splat candidate worth testing.  
If you must choose the **lowest-risk Phase C delivery path**, pick **#4** and explicitly reframe splats as **Phase C.5 / R&D**.

That may sound conservative, but your constraints are unusually strict:
- exact-duration visuals
- local-only
- deterministic audio gate
- Windows
- Blackwell
- no FA2/FA3
- no low-level optimization work

That combination strongly favors boring tools.

---

B. **Yes, parametric camera-path scaling is the right core model — but not as a single uniform speed curve**

Your core idea is correct:

> fixed scene representation + parameterized camera path + duration scaling

That is the right abstraction for a **narrative-audio-length-first** system.

It is much better than video diffusion for this project because:
- duration is exact by construction
- no chunk seam problem
- no cross-fade drift
- no “regenerate until timing fits” nonsense
- one scene can stretch from 20 sec to 4 min without changing the underlying visual asset class

But I would **not** use a single global formula like:

`path_speed = default_speed / (target_duration / default_duration)`

as the only timing model.

That works mechanically, but artistically it will fail on long scenes. A 4:32 scene with one constant push-in will feel dead.

## Better model: normalized path + deterministic retiming curve

Use this abstraction:

1. Define a camera path in normalized time `u ∈ [0,1]`
2. Define pose as `P(u)` for translation/rotation/FOV
3. Define a monotonic retiming map `u = R(t)` where `t ∈ [0, target_duration]`
4. Sample frames at exact timestamps:
   - `t_i = i / fps`
   - `u_i = R(t_i)`
   - render `P(u_i)`

This gives you:
- exact frame count
- exact duration
- adjective-driven motion design
- deterministic output
- freedom to add pauses, accelerations, and line-aware motion later

## What `R(t)` should be
Not constant speed. Use **piecewise easing**.

### Recommended baseline
For each shot adjective, define:
- base path shape
- micro-motion profile
- easing profile
- max displacement/rotation budget

Examples:

- **locked wide**
  - path: constant pose
  - micro-jitter: tiny deterministic noise
  - easing: none
- **slow push**
  - path: translate along `-Z`
  - easing: ease-in-out or very shallow sigmoid
- **pull out**
  - path: translate along `+Z`
  - easing: same
- **pan across**
  - path: yaw rotation
  - easing: cubic in/out
- **handheld drift**
  - path: low-frequency deterministic noise around anchor pose
  - optional tiny bias drift over scene duration

## Best next step: line-boundary-aware retiming
You mentioned dialogue-line boundaries. Yes — that is the first meaningful upgrade.

Not because you need complex cinematography, but because it solves the “long scene goes visually numb” problem while preserving exact duration.

### Practical model
Split scene audio into semantic beats:
- narration block
- dialogue line
- pause / stinger / SFX emphasis

Then allocate path progress by beat weights.

Example:
- intro ambience: 15%
- line 1: 20%
- pause: 5%
- line 2: 25%
- reveal line: 20%
- outro hold: 15%

Now the camera can:
- hold during important lines
- move during transitions
- accelerate into reveals
- settle on dramatic pauses

This is still deterministic and still exact-duration.

## My recommendation for Phase C timing model
Implement in this order:

### Phase C baseline
- normalized path `P(u)`
- deterministic retiming `R(t)`
- adjective-specific easing presets
- exact frame count from audio duration

### Phase C+1
- beat-aware piecewise retiming using dialogue boundaries from script JSON / subtitle timings
- optional “hold windows” around important lines

That gives you the artistic payoff without exploding complexity.

---

C. **Single biggest architectural risk: image-to-splat quality, not rendering**

The most likely phase-killer is:

> **The image→3DGS conversion will not produce geometry stable or coherent enough for multi-minute camera motion from a single anchor image.**

That is the risk I would put at #1 by a wide margin.

Not install. Not ffmpeg. Not muxing. Not even VRAM.

Why this is the killer:
- Your whole Phase C value proposition depends on **camera travel through fixed geometry**.
- If the generated splat is only good for tiny parallax, then:
  - slow pushes break
  - pull-outs reveal garbage
  - pans expose floaters / holes
  - handheld drift becomes nausea fuel
- A single-image 3D reconstruction often looks acceptable for:
  - short orbit demos
  - tiny dolly moves
  - social-media-length clips
- It often fails for:
  - **minutes-long sustained motion**
  - repeated scrutiny
  - moody low-key scenes with sparse detail and atmospheric ambiguity

And your use case is especially hard:
- old-time-radio mood
- dark interiors
- fog, shadows, practical lamps, corridors, voids
- scenes where monocular depth is ambiguous
- likely stylized SDXL anchors rather than photometrically grounded photos

That means the renderer could work perfectly and still the result is unusable.

### Why this is more dangerous than install risk
Install risk is binary and early: you know in a day or two.

Image→splat quality risk is worse:
- you may get it “working”
- then discover all camera adjectives except micro-push are artistically bad
- then you’ve sunk time into the wrong representation

That’s the kind of risk that kills a phase late.

### Implication
Before committing architecture, run a **content-validity spike**:
- 3 representative SIGNAL LOST anchors
- 5 camera moves each
- durations: 10s, 30s, 90s
- evaluate:
  - hole exposure
  - edge floaters
  - temporal shimmer
  - mood retention
  - whether motion remains watchable beyond 20–30s

If it only survives tiny motion, then splats are not your production Phase C, regardless of install success.

---

D. **Missing named component: deterministic camera-motion engine**

If I have to pick one missing component to explicitly name, it is:

> **A deterministic trajectory/retiming engine**  
> not just “camera path,” but a formal module that maps shot adjectives + scene duration + optional dialogue beats into exact per-frame camera poses.

This is more important than depth pre-pass in your architecture doc, because it is the thing that actually enforces your narrative-first contract.

Right now your description has:
- image -> PLY
- camera path
- render frames
- ffmpeg assemble
- mux audio

What’s missing is the **time-domain control layer** between “camera adjective” and “per-frame pose.”

## Why this should be a named subsystem
Because it owns:
- exact frame count
- exact duration fit
- easing
- hold segments
- deterministic jitter
- adjective mapping
- future beat alignment
- reproducibility

Without naming it, it tends to become ad hoc math scattered across render code.

## What it should output
Given:
- `scene_duration_sec`
- `fps`
- `shot_adjective`
- `seed`
- optional `beat_times[]`

It should emit:
- `frame_index`
- timestamp `t`
- normalized progress `u`
- camera transform matrix
- optional FOV
- optional post-FX metadata

That becomes the canonical truth for rendering.

## Determinism matters here
You explicitly care about reproducibility and byte-identical audio. Even if C7 is only about audio, your visual pipeline should still be deterministic enough to debug.

So the trajectory engine should:
- use fixed seeds
- avoid wall-clock randomness
- use deterministic noise tables, not ad hoc random calls
- quantize/snapshot camera poses before render if needed

## Second-place missing component: depth/geometry conditioning
If you wanted the most important *visual* missing component instead, I’d say:

> **A depth/geometry conditioning pre-pass**

Because single-image splat generation usually benefits from some explicit geometry prior. Whether that is MiDaS, Marigold, or whatever Sharp internally supports, the principle matters more than the brand.

But I still rank the deterministic trajectory engine higher architecturally, because without it you don’t actually have a robust duration-control system.

---

## Bottom line

### A
- **Lowest-risk overall:** **#4**
- **Only splat stack worth testing:** **#1**
- **Avoid for now:** **#2 and #3**

### B
- Your **parametric camera-path** idea is correct.
- Upgrade it from **uniform speed scaling** to **normalized path + deterministic retiming curve**.
- Add **beat-aware piecewise timing** as the first enhancement.

### C
- Biggest risk: **single-image image→splat quality won’t support sustained multi-minute camera motion**.

### D
- Missing named component: **deterministic trajectory/retiming engine**.
- Runner-up: **depth/geometry conditioning pre-pass**.

If you want, I can turn this into a concrete **Phase C decision memo** with:
1. a go/no-go recommendation,
2. a 2-day validation matrix for stack #1,
3. and a minimal module layout for the trajectory engine inside your current repo.

---

## Gemini (UNAVAILABLE -- quota-locked)

Gemini API returned HTTP 429 across the entire ladder today
(gemini-3.1-pro-preview, gemini-2.5-pro, gemini-2.0-flash). gemini-1.5-pro
returned 404 (deprecated). Free-tier quota was exhausted.

Round B was filled by **gpt-4.1 as a devil's-advocate second opinion**
(different training cutoff and behavior from gpt-5.4, instructed to
stress-test gpt-5.4's answer). See `02_gpt41_devils_advocate.md` for the
full response.

### gpt-4.1 summary (where it diverges from gpt-5.4)

- **Agrees** on every headline: #4 for production, #1 only as R&D,
  #2/#3 ruled out, image->splat quality as the #1 risk, trajectory engine
  as the critical missing component.
- **Adds** long-term maintainability as a meta-risk gpt-5.4 underweights.
- **Pushes back** on beat-aware retiming as a baseline: start with simple
  ease-in/out only; script/audio alignment work is non-trivial and should
  wait for signal that it is needed.
- **Adds missing components gpt-5.4 didn't name**: color management /
  tone-mapping pipeline (important for Phase D CRT FX), and an automated
  end-to-end determinism test harness (hash-equality on frame output).
- **Adds install-risk vectors gpt-5.4 missed**: Windows DirectML / DirectX
  / OpenGL-EGL issues on hybrid-graphics laptops, renderer-side
  non-determinism from AA sampling.

Both models are in agreement on every load-bearing conclusion.

---

## To decide (Claude / human) -- VOTE

### Where all three voices AGREE (lock in)

1. **Stack #4 (SDXL anchor + ffmpeg zoompan) is the only low-risk Phase C
   delivery path.** Zero new CUDA extensions, preserves C7 by construction.
2. **Stack #1 (ComfyUI-Sharp + gsplat) is the only splat stack worth
   testing**, and only as a research spike on a separate branch.
3. **Stacks #2 and #3 are ruled out.**
4. **Biggest risk is NOT install -- it is image->splat quality** failing to
   sustain multi-minute camera motion on moody low-key SIGNAL LOST scenes.
5. **A deterministic trajectory / retiming engine is the critical missing
   named component** and belongs in the codebase regardless of stack pick.
6. **Parametric camera-path time-stretch is the right abstraction**, but
   upgrade from a single-speed scale to normalized path P(u) + retiming
   R(t) + per-adjective easing.

### Claude's grounded vote

**Phase C production path:** stack #4. Accept the "Ken Burns with better
easing" visual ceiling. The honest read of the consultation is that nobody
-- not even the splat optimists -- thinks #1 is a responsible production
commitment on this platform today.

**In parallel:** 2-hour time-boxed gsplat install spike on a throwaway
branch. Binary outcome: clean install -> keep as R&D for v2.1+; fails ->
shelve splats until upstream supports Blackwell + torch 2.10 without
build archaeology.

**Build now regardless of stack:** `otr_v2/hyworld/camera_path.py` --
deterministic trajectory engine. Inputs: `scene_duration_sec`, `fps`,
`shot_adjective`, `seed`, optional `beat_times[]`. Outputs: deterministic
per-frame `(frame_idx, t, u, pose_matrix, fov)`. Serves both stack #4
(richer ffmpeg motion than today's zoompan stub) and stack #1 (if the
spike lands).

**Determinism harness:** `tests/test_camera_path_determinism.py` -- render
a canned path twice, assert frame-hash equality. Cheap insurance against
the color / gamma / AA-sampling drift both models flagged.

**Skip for v1** (gpt-4.1's correct pushback):
- Beat-aware retiming (Phase C+1 only)
- Depth pre-pass (MiDaS / Marigold)
- Color-management pipeline (enters with Phase D)
- Script-audio beat alignment

### Punch-list

- [ ] Accept stack #4 as Phase C production path? (Claude vote: YES)
- [ ] Write `otr_v2/hyworld/camera_path.py` -- deterministic trajectory engine
- [ ] Wire camera_path into worker.py, replace ffmpeg zoompan arg-gen with
      adjective -> camera_path pose sequence -> ffmpeg frame-by-frame
- [ ] Add `tests/test_camera_path_determinism.py` with hash-equality assert
- [ ] Run 2-hour gsplat install spike on a throwaway branch (not v2.0-alpha)
- [ ] If spike is green: start `v2.0-phase-c-splat-spike` branch for R&D.
      Keep v2.0-alpha on stack #4 only.
- [ ] Bug Bible regression + Phase A tests + new determinism test must stay
      green before commit.

### Final recommendation

Ship stack #4 with a proper trajectory engine and a determinism harness.
Treat splats as a time-boxed R&D experiment, not a phase. Re-evaluate the
splat path in Phase E (caching) or later when either (a) gsplat gets
Blackwell CI coverage, or (b) upstream HY-Pano 2.0 ships and the creative
ceiling on anchors raises enough that splats become worth the engineering
cost.
