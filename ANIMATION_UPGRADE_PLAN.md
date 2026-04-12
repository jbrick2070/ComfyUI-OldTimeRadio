# Animation Upgrade — Call to Action
## ComfyUI-OldTimeRadio v2.0 Visual Drama Engine

**Status:** Alpha 2 pre-release planning document  
**Format:** Round-robin annotation — add your initials + date in brackets after any item  
**Purpose:** Decide what "animated visual output" means for Alpha 2.0 before we tag RC1  

---

## Why This Document Exists

The current v2 ProductionBus produces a **keyframes-mode slideshow**: one SD3.5-generated still per scene, held as a static frame for `audio_duration / num_scenes` seconds. For a 3-scene episode that is three JPEGs cross-faded by FFmpeg. It works. It is not 2.0.

The hardware on this system (RTX 5080 / 16 GB VRAM) can do better. The question is *what to commit to* before tagging Alpha 2.0 RC1, what to ship as v2.1+, and where the VRAM ceiling forces hard constraints.

---

## Current Baseline (Keyframes Mode)

| Metric | Value |
|--------|-------|
| Mode | Keyframes (still slideshow) |
| Frames per episode | 1 per scene (typically 1–3) |
| Resolution | 1280x720 |
| Image model | SD3.5 Large FP8 |
| Image gen time | ~4 seconds per frame |
| Total visual gen time | ~15 seconds for 3 frames |
| Output | MP4 matching audio duration, ~4 MB |
| VRAM at visual phase | ~12 GB peak |

**What it looks like:** A period-art postcard flips to the next scene every 30–40 seconds. Radio drama feel. Not unwatchable. Not a release.

---

## Option A — Dense Keyframes (Low Effort, Ships in Alpha 2.0)

**Idea:** Instead of 1 frame per scene, generate 1 frame per *dialogue line* or per *beat*. A 23-line episode would get 23 still frames, each held for ~4–5 seconds. More visual variety with zero new infrastructure.

**Pros:**
- Zero new models needed. SD3.5 already working.
- Total gen time: 23 frames × 4s = ~92 seconds. Acceptable.
- Each frame prompted from the dialogue line text — contextually driven.
- Fits inside VRAM budget (no change).

**Cons:**
- Still a slideshow. Frames are not animated.
- SD3.5 has no temporal consistency — adjacent frames may look completely different (different lighting, different character appearance).
- Character continuity would require ControlNet or IP-Adapter (not yet wired).

**VRAM estimate:** 12 GB peak, same as now.  
**Effort:** ~2 days. Director and VisualCompositor prompt changes only.  
**Blocks RC1?** No — could be a day-1 patch after tagging.

> **[ ANNOTATE HERE ]** — Is dense keyframes enough to call 2.0, or does it still feel like 1.5 with more frames?

---

## Option B — LTX-Video Animated Scenes (Medium Effort, Strong Upgrade)

**Idea:** Replace SD3.5 still images with short LTX-Video clips per scene. Each scene gets a 2–4 second looping animation. ProductionBus assembles the clips with audio.

**Models already on system:**
- `ltx-2.3-22b-distilled-fp8.safetensors` (22B distilled, fastest)
- `ltx-2-19b-distilled-fp8.safetensors` (19B distilled, alternate)

**How it would work:**
1. ScenePainter generates a still image (SD3.5) as the "anchor frame."
2. New node `SceneAnimator` takes the still + scene description text → LTX-Video → 2–4s clip (25fps = 50–100 frames).
3. VisualCompositor assembles clips end-to-end instead of stills.
4. ProductionBus receives a pre-assembled video instead of a frame list.

**Pros:**
- Actual motion. Characters walk, environments breathe, camera pans.
- LTX-Video distilled is fast: ~25–40 seconds per 2s clip on RTX 5080.
- Output looks like a real animated drama, not a slideshow.
- Both LTX models are already downloaded — no network needed.

**Cons:**
- VRAM: LTX-Video 22B FP8 peaks at ~12–13 GB. SD3.5 anchor + LTX-Video cannot run simultaneously — must fully unload SD3.5 before loading LTX-Video.
- Total gen time for 3 scenes: 3 × ~35s = ~105 seconds. Same order as audio gen. Manageable.
- No temporal consistency between scenes (each LTX clip is independent). Characters may change appearance between scenes. Requires IP-Adapter or a seeded anchor approach to fix — deferred to v2.1.
- Requires new `SceneAnimator` node (~150 lines) and ProductionBus video-concat mode.

**VRAM estimate:** 12–13 GB peak during LTX-Video phase. Within 14.5 GB ceiling.  
**Effort:** ~1 week. New node + ProductionBus changes + workflow JSON wiring.  
**Blocks RC1?** This *would be* RC1 if it works. Recommend targeting this.

> **[ ANNOTATE HERE ]** — LTX-Video or bust for 2.0? Or ship dense-keyframes as 2.0 and hold LTX for 2.1?

---

## Option C — AnimateDiff Motion on SD3.5 Stills (Alternate Animation Path)

**Idea:** Post-process each SD3.5 still with AnimateDiff to add subtle motion (parallax, breathing, ambient camera). Each still becomes a 2–3 second looping GIF-style clip.

**Pros:**
- Character appearance stays consistent — same SD3.5 output, just animated.
- AnimateDiff runs at ~8 GB VRAM for standard motion modules.
- Subtle motion is very appropriate for the old-time radio aesthetic (vintage film grain, slight camera drift).

**Cons:**
- AnimateDiff v3 / SparseCtrl are the current state of the art but require specific model files not confirmed on this system.
- Motion quality is limited — good for "living photo" effects, not for full character animation.
- Less impressive than LTX-Video on a demo reel.

**VRAM estimate:** ~10 GB peak (SD3.5 done before AnimateDiff loads).  
**Effort:** ~3 days if AnimateDiff models are present; ~1 week if they need to be sourced.  
**Blocks RC1?** No — could be a v2.1 option alongside LTX-Video.

> **[ ANNOTATE HERE ]** — Is AnimateDiff worth evaluating, or is LTX-Video the clear winner?

---

## Option D — Hybrid: LTX for Action Scenes, Stills for Dialogue Close-ups

**Idea:** Director tags each scene with a motion budget: `"motion": "high"` for action scenes, `"motion": "static"` for intimate dialogue scenes. High-motion scenes get LTX-Video clips; static scenes keep SD3.5 stills.

**Pros:**
- VRAM-aware: static scenes don't burn LTX-Video time.
- Aesthetically interesting — animated action, painterly dialogue. Could be a signature style.
- Reduces total render time compared to full LTX-Video on every scene.

**Cons:**
- Director prompt changes needed to reliably emit `motion` tags.
- More logic in VisualCompositor and ProductionBus to handle mixed input types.
- More moving parts = more bugs.

**VRAM estimate:** Same as Option B, just fewer LTX-Video invocations.  
**Effort:** ~2 weeks. Depends on Option B landing first.  
**Blocks RC1?** No — this is a v2.1+ feature.

> **[ ANNOTATE HERE ]** — Good idea? Should Director scene tags include motion budget now, even if we don't use it yet?

---

## Open Questions (For Team Annotation)

These are unresolved design questions. Add your initials and a short answer next to each.

**Q1.** What is the minimum bar for "animated" that justifies calling this v2.0 vs v1.5?
> [ ]

**Q2.** Should character continuity (same face across scenes) be required for Alpha 2.0, or is it acceptable that the same character looks different between scenes?
> [ ]

**Q3.** LTX-Video generates T2V (text-to-video) or I2V (image-to-video). Should we use I2V anchored on the SD3.5 portrait, or pure T2V from the scene description?
> [ ]

**Q4.** What should happen to the v1.5 SignalLostVideo node during Alpha 2.0? Keep it in the workflow as a fallback, or remove it?
> [ ]

**Q5.** The current Director visual_plan produces 1 scene even for a 23-line script. Should we force a minimum scene count (e.g., 3 scenes for any script longer than 10 dialogue lines)?
> [ ]

**Q6.** At 14.5 GB VRAM ceiling, can we run SD3.5 (anchor still) + LTX-Video (animation) sequentially in the same episode without an OOM? Needs a live T-test.
> [ ]

**Q7.** Should the keyframes-mode slideshow be kept as a fast-preview fallback (e.g., `runtime_preset = [EMOJI] preview`) even after LTX-Video is wired?
> [ ]

---

## Recommended Path to Alpha 2.0 RC1

Based on current system state and VRAM budget, the recommended sequence is:

1. **Immediate (this sprint):** Confirm Run #3 end-to-end on keyframes mode. Validate T1-T7 test checklist. This closes the current bug cycle.

2. **Alpha 2.0 RC1 target:** Implement Option B (LTX-Video SceneAnimator node). Three milestones:
   - M1: `SceneAnimator` node scaffolded, LTX-Video 22B FP8 loads cleanly, 1 clip generated.
   - M2: ProductionBus video-concat mode replaces frame concat. 3-scene episode generates 3 clips + audio in one run.
   - M3: T2-T4 VRAM stress tests pass. No OOM. Peak VRAM ≤ 14.5 GB.

3. **After RC1:** Option C (AnimateDiff) and Option D (hybrid motion budget) as v2.1 candidates.

> **[ ANNOTATE HERE ]** — Agree with this path? Revise?

---

## Technical Constraints Summary

| Constraint | Value | Source |
|------------|-------|--------|
| VRAM ceiling | 14.5 GB real-world target | CLAUDE.md |
| Flash Attention 2 | NOT available (torch 2.10 + CUDA 13 + sm_120, no Windows wheel) | startup log |
| Async CUDA streams | Not permitted | CLAUDE.md |
| LTX-Video 22B FP8 VRAM | ~12–13 GB peak | estimated |
| SD3.5 Large FP8 VRAM | ~12 GB peak | observed Run #1 |
| Sequential execution | Required | CLAUDE.md |
| Models already on system | SD3.5 Large FP8, LTX 2.3-22B distilled FP8, LTX 2-19B distilled FP8 | confirmed |

---

## How to Use This Document

1. Each section has an annotation block `> **[ ANNOTATE HERE ]**`. Reviewers: replace the bracket with `> **[YourInitials, YYYY-MM-DD]:** Your note.`
2. Questions section: answer inline next to each Q.
3. When a decision is made, move the item to a `## Decisions` section at the bottom and note who decided and when.
4. This document lives in the repo root until decisions are finalized. After RC1, archive it to `docs/decisions/`.

---

*Draft: 2026-04-11 — Jeffrey Brick / ComfyUI-OldTimeRadio v2.0 visual engine sprint*
