# Problem Statement — The Simplest Viable 8GB OTR Workflow for v2 Ship

**Date:** 2026-05-25
**Component:** target workflow JSON variant for 8GB-tier users
**Goal:** ship v2 with an 8GB tier that runs end-to-end on a sub-$300 consumer GPU, zero install help required, with a watchable result.
**Status:** for operator confirmation. Architecture is mostly already locked in ROADMAP B2 (2026-04-30); this document consolidates the simplest-possible distillation of that lock plus the May 2026 LTX 2.3 GGUF update, so the v2 ship decision can be made in one read.

---

## 1. Summary

We want the **simplest** 8GB workflow that:

- Produces a watchable OTR episode end-to-end (audio + visuals + final mp4).
- Fits in 8GB VRAM with realistic OS + ComfyUI overhead headroom.
- Has zero exotic dependencies that break for non-Jeffrey users.
- Holds Prime Directive 1 (audio byte-identical to baseline) and PD4 (SFW, good narrative arc).
- Ships **without** HuMo, LTX-2, LTX 2.3 video, or any model that demands more than 6GB resident.

**Simplicity is the optimization.** Quality is the floor, not the ceiling. The target is "watchable on a phone screen, recognizably an OTR episode" — not "festival-grade." Festival-grade is the 16GB tier's job.

---

## 2. Constraints (hard)

- **VRAM ceiling:** 8GB. Realistic working budget ~6.5GB after OS + ComfyUI + browser overhead.
- **Target hardware class:** RTX 3060 8GB, RTX 4060 8GB, RX 7600. Used-market floor ~$200.
- **OS:** Windows primary (Jeffrey), Linux + Mac welcome (HuggingFace Space). Cross-platform paths required (B1 release blocker).
- **Audio pipeline:** Bark + Kokoro + MusicGen + AudioGen. Non-negotiable — this IS OTR.
- **Workflow file count:** one master JSON with bypassable groups, NOT a separate `_8gb_safe.json` sibling. Per standing memory rule "keep workflow JSONs to minimum — no _v2/_safe variants" and the B2 lock "Single master JSON with bypassable video-stack groups."
- **Zero paid API dependencies.** 100% local, open source, offline-first per CLAUDE.md.

---

## 2b. Vendor-agnostic consideration (added 2026-05-25 mid-draft)

**Operator raised:** *"maybe the 8GB can be CUDA / NVIDIA agnostic so anyone can run it."*

This is a bigger ask than 8GB-fit alone, but it mostly **simplifies** the architecture rather than complicating it. The 8GB stack is already gravitating toward portable components (GGUF + PyTorch defaults); going vendor-agnostic mostly means **not adding** NVIDIA-only accelerations rather than replacing things.

### What works on every vendor today

| Component | NVIDIA CUDA | AMD ROCm | Apple MPS | CPU fallback |
|---|---|---|---|---|
| Bark / Kokoro / MusicGen / AudioGen (PyTorch) | yes | yes (Linux strong, Windows weak) | yes | yes (slow) |
| Mistral-Nemo / Gemma writer (HF Transformers) | yes | yes | yes | yes (slow) |
| GGUF LLM writer (llama.cpp port) | yes | yes | yes | yes |
| **FLUX.1-dev Q4_K_S GGUF** (city96) | yes | yes | yes | yes (very slow) |
| Ken Burns pan/zoom (compositing math) | n/a | n/a | n/a | n/a (CPU) |
| Frame interpolation (RIFE) | yes | yes | yes | yes |
| Real-ESRGAN upscale (cross-platform) | yes | yes | yes | yes |
| SDPA attention (PyTorch built-in) | yes | yes | yes | yes |

### What we LOSE going vendor-agnostic

- **RTX VSR ULTRA upscale.** NVIDIA-only. The roadmap's locked B2 upscale pick is NVIDIA HW-accelerated. Vendor-agnostic substitute: **Real-ESRGAN** (mature, cross-platform, ~1-2GB VRAM, slower than RTX VSR but works everywhere). OR drop the upscale entirely in default 8GB mode — output at native 720p, let users upscale externally if they care.
- **SageAttention.** NVIDIA Blackwell-tuned. Substitute: PyTorch SDPA (built-in, cross-platform, ~10-30% slower depending on workload).
- **NVFP4 quantization.** RTX 50 Series only, irrelevant on 8GB anyway (was a 16GB-tier optimization).
- **Flash-Attention.** NVIDIA-only. SDPA covers the same role cross-platform.

### What we GAIN

- **Apple Silicon support.** MacBook M-series with 16GB unified memory has ~12-14GB usable for ML — practically the same headroom as a discrete 8GB card. M2/M3/M4 Pro MacBooks become a target. This is a huge wow-audience expansion.
- **AMD on Linux support.** ROCm on Linux is solid in 2026; opens RX 7600 (8GB) and RX 7700 XT (12GB) as target hardware. Used-market floor for AMD 8GB cards is even lower than NVIDIA.
- **HuggingFace Space portability.** HF Spaces don't guarantee NVIDIA; vendor-agnostic deploys to more Space templates.
- **Simpler dependency tree.** No NVIDIA-only kernels to manage, no `--force-fp16` / `--force-fp8` launcher branches, no SageAttention install dance.

### Cost honestly named

Non-CUDA is slower. Order of magnitude on FLUX Q4_K_S still rendering (very rough, depending on model + resolution):

| Platform | FLUX Q4_K_S 1024px still | Notes |
|---|---|---|
| RTX 4060 8GB (CUDA) | ~15-25s | baseline |
| RX 7600 8GB (ROCm Linux) | ~25-40s | ~1.5-2x slower |
| M2 Pro 16GB (MPS) | ~30-50s | ~2-3x slower |
| M3 Pro 18GB (MPS) | ~20-35s | ~1.5-2x slower |
| CPU-only (modern 8-core) | ~5-15 min | unusable for full episode |

**Recommendation:** target NVIDIA + AMD + Apple Silicon. **Skip CPU-only as a supported tier** — it works but a 3-4 min episode becomes a multi-hour render and the user-support burden isn't worth it. Document it as "technically possible, not recommended; use a GPU."

### Architectural impact on this problem statement

- Section 3 pipeline order **unchanged** — every component listed already has a cross-platform implementation.
- Section 4 locked picks: **swap RTX VSR ULTRA for Real-ESRGAN** as the default 8GB upscale. RTX VSR stays in the 16GB tier as a NVIDIA-only optimization (no quality loss for 16GB users who have it).
- Acceptance criteria: add **"workflow loads and renders on a non-NVIDIA reference machine"** (MacBook M2 Pro or AMD RX 7600 Linux box).
- Validation strategy: Runpod option in Section 6 expands to include AMD ROCm + Apple Silicon (Mac mini M2) test instances. Total validation cost still under $50.

---

## 3. The simplest workflow — pipeline order

```
[Audio pipeline]
  Mistral-Nemo writer (or Gemma alt)
    -> Bark dialogue + Kokoro announcer
    -> MusicGen theme + bed
    -> AudioGen SFX (when SFX dedicated pass lands)
    -> EpisodeAssembler master mix

[Visual pipeline]
  FLUX.1-dev Q4_K_S still per beat (~5-6GB resident)
    -> Ken Burns pan/zoom (deterministic, CPU/GPU-light)
    -> Frame interpolation to 24fps (RIFE or equivalent, ~1GB)
    -> Optional RTX VSR ULTRA upscale (~0GB, HW-accelerated)

[Mux]
  FFmpeg combines audio master mix + interpolated still sequence
    -> final .mp4 in output/episodes_for_obs/<ep>/
```

**What's NOT in the simplest path:**

- No HuMo (any tier). Physically does not fit with FLUX + text encoders + audio stack co-resident.
- No LTX-2, LTX 2.3, Wan 2.2 5B B-roll in DEFAULT mode (Wan 5B remains an opt-in toggle for users who want generative B-roll on non-dialogue beats and accept slow render).
- No diffusion video at all in default 8GB mode. Visuals = still + camera motion + interpolation only.

---

## 4. Locked picks from ROADMAP B2 (2026-04-30) — confirming

| Component | 8GB pick | Status |
|---|---|---|
| **Stills** | FLUX.1-dev Q4_K_S city96 GGUF (~5-6 GB) | LOCKED |
| **Motion** | Still + Ken Burns parallax + frame interpolation | LOCKED |
| **Optional B-roll** | Wan 2.2 5B TI2V (opt-in toggle) | OPTIONAL |
| **Upscale (speed)** | RTX VSR ULTRA (~0 GB, HW-accel) | LOCKED |
| **Upscale (quality)** | n/a on 8GB — quality upscale is 16GB-tier only | LOCKED |
| **Audio** | Bark + Kokoro + MusicGen + AudioGen | UNCHANGED |

**Architecture decision (locked 2026-04-30):** single master JSON with bypassable video-stack groups. Toggle groups via Ctrl+B in ComfyUI Desktop. README screenshot shows the "8GB mode" group state.

---

## 5. The single open question (per the May 2026 lip-sync survey)

A May 2026 web search surfaced viable 8GB lip-sync options that did not exist when ROADMAP B2 was locked:

- **LTX 2.3 GGUF (Q2-Q4) + lip-sync ID-LoRA.** Q5_K_M was the floor when B2 locked; Q2-Q4 quants now ship for sub-16GB cards, with a dedicated 8GB workflow + LoRA that holds character identity while syncing the mouth.
- **InfiniteTalk (MultiTalk backend).** 6GB+ for 480p talking video from one portrait + audio.
- **Wan 2.2 S2V.** Audio-driven talking head with body gestures.

**Question for the v2 ship decision:** does adding ONE opt-in lip-sync lane belong in v2, or v2.1?

**Recommendation: v2.1.** The simplest 8GB v2 ship lane is still+parallax+interpolation, proven and deterministic. LTX 2.3 GGUF + lip-sync LoRA becomes the "want better motion on 8GB? Opt in to this lab tier" path in v2.1, lab-isolated in `otr-tts-lab` ComfyUI Portable first per the standing engine-adoption rule.

**Why deferral is the right call:** v2 has been in flight for months and is gated on shipping, not on feature parity with 16GB. Adding lip-sync to the 8GB ship lane reopens render-time + quality + OOM-risk variables that are settled in the still+parallax path. Lock the simplest possible thing, ship, then iterate.

---

## 6. Operator decisions needed (round-robin candidates)

1. **Vendor-agnostic 8GB tier — adopt as the ship target?** (Claude recommends: yes. Costs are small — drop RTX VSR for Real-ESRGAN, accept SDPA instead of SageAttention. Gains are large — Apple Silicon + AMD Linux + HF Space portability + simpler dependency tree. See Section 2b for the full tradeoff.)
2. **Adopt this overall architecture as the 8GB v2 ship target?** (Claude recommends: yes — most of it is already locked, this just confirms + adds the vendor-agnostic axis.)
3. **Validation strategy.** Jeffrey has no 8GB hardware. Options: (a) simulate 8GB VRAM cap locally on the RTX 5080 via PyTorch memory limits, (b) rent ~$10-50 of cloud time across an NVIDIA RTX 4060 8GB + AMD ROCm + Apple Silicon Mac mini instance and validate end-to-end across all three vendors. (Claude recommends: b — actual hardware truth beats simulation, cost is trivial, validates the vendor-agnostic claim in real conditions, doesn't violate "100% local" because it's test-rig use not deployment.)
4. **Defer LTX 2.3 GGUF lip-sync LoRA to v2.1?** (Claude recommends: yes — lab-isolate in `otr-tts-lab` first, A/B against still+parallax, decide after operator listen test.)
5. **Confirm Wan 2.2 5B stays as the ONLY opt-in B-roll lane in 8GB v2 default mode.** No second exotic toggle until v2.1.
6. **CPU-only tier: explicit non-goal.** Document as "technically works, not recommended." Operator: confirm or push back?

---

## 7. Acceptance criteria for the 8GB v2 ship

- Full audio pipeline runs unchanged (PD1 byte-identical against 16GB-tier baseline on identical seed).
- FLUX Q4_K_S still renders complete with <7GB peak VRAM during the visual phase.
- Ken Burns + interpolation pipeline produces a 24fps mp4 with no dropped frames.
- Total wall-clock for a 3-4 minute episode on an RTX 4060 8GB: target <60 minutes, document upfront in README.
- Output mp4 lands at `output/episodes_for_obs/<ep>/<ep>.mp4` identically to 16GB tier.
- ComfyUI Desktop loads the workflow with zero missing-node warnings on a clean ComfyUI install (cross-platform, not Jeffrey-machine-specific paths).

---

## 8. Out of scope for v2 (deferred to v2.1 or later)

- LTX 2.3 GGUF + lip-sync ID-LoRA (v2.1 lab-isolated track)
- InfiniteTalk / MultiTalk (v2.1+ lab eval)
- Wan 2.2 S2V audio-driven talking head (v2.1+ lab eval)
- SeedVR2 v2.5 NVFP4 quality-tier upscale on 8GB (locked as 16GB-only; revisit only if a Q4 SeedVR2 ships)
- Stable Audio 3 SFX cue-to-audio (batched with the parked SFX spotting pass on the AUDIO QUALITY TRACK)

---

## 9. Why this matters

v2 ships when 8GB-class users can run an episode end-to-end. The HuggingFace Space wrapping the 8gb_safe preset is named in ROADMAP S24.4 as the *"single highest-impact item — every person who can't install ComfyUI becomes a possible user."* That's the wow-people distribution mechanism. The simplest possible 8GB workflow is the gate between OTR-as-Jeffrey's-private-project and OTR-as-something-strangers-can-use.

The decision to ship the simplest thing is the decision to ship.

---

## 10. UI tiering addendum — Green/Yellow/Red group architecture (added 2026-05-25, second pass)

This section captures a UI-partitioning proposal raised after the initial draft, the LTX Micro counter-argument it surfaced, and the operator-pending synthesis. NOT yet integrated into the go-forward plan — review-stage only.

### 10.1 The Green/Yellow/Red proposal (as raised)

Three color-coded groups in the master JSON, replacing the binary "8GB safe vs 16GB" framing with a three-tier UI:

- **Green — "BASELINE — Always On."** Audio pipeline end-to-end (writer LLM → Bark → Kokoro → MusicGen → AudioGen → EpisodeAssembler), the locked visual baseline (FLUX Q4_K_S still + Ken Burns + interpolation), FFmpeg mux. Nothing bypassable without breaking the episode.
- **Yellow — "OPTIONAL — More VRAM Recommended."** Wan 2.2 5B B-roll, Real-ESRGAN upscale, heavier resolutions. All nodes ship bypassed (Ctrl+B). User opts in consciously.
- **Red — "LAB — 16GB+ ONLY — Will OOM on 8GB."** HuMo, longer LTX clips, lip-sync LoRAs, SeedVR2, anything experimental. All bypassed. Loud labeling.

**Default shipped state:** Green enabled, Yellow + Red bypassed. User has to consciously toggle anything optional. Drops accidental OOM near zero.

**Supporting mechanisms:**

- Per-group MarkdownNote header in plain English: *"This group requires 12GB+ VRAM. If you have 8GB, do not unbypass these nodes."* No jargon.
- `OTR_WorkflowValidator` extended to detect "Yellow or Red group active AND no VRAM override set" → emit pre-run warning. Belt + suspenders.
- Friendly OOM messaging: *"Lab group X OOM'd at scene Y. Bypass that group (Ctrl+B) and re-queue. Your baseline path is unaffected."* — not the default ComfyUI stack trace.
- Node title prefix convention: `[BASELINE] Audio Pipeline`, `[OPTIONAL 12GB] Generative B-Roll`, `[LAB 16GB+] LTX Hero Shots`. Bracketed tag is the user's anchor.
- HF Space mirrors the tiering with three radio buttons (**Quick / Better / Best**) that toggle the groups behind the scenes — non-technical users never see the graph. README opens with *"8GB users: do not touch Yellow or Red groups."*

### 10.2 The LTX Micro-Theater counter-argument

The proposal surfaced the LTX Micro-Theater path as the strongest existing counter to still+KB as the 8GB default:

- Community-validated 8GB LTX 2.3 GGUF workflows exist (May 2026 confirmed via web search — see Section 5).
- "Real AI video" framing matters for mass-audience excitement in a way slideshows do not. Slideshow aesthetic reads 2018; even small-res LTX reads 2026.
- The graph already has the LTX branch infrastructure (`OTR_BatchLTXRender`, deferred Gemma encoder).

**Counter-counter-argument (the case for still+KB shipping first):**

- Simplicity is the stated optimization. LTX Micro adds a shot planner, budget-mode router, hero-line selector, fallback logic — new architecture, not just new weights. That's a sprint of work, not a swap.
- 60-minute render budget on a 4060 8GB: LTX Q2/Q3 at 384px × 49 frames × 6 hero shots is *probably* feasible but **unmeasured**. Still+KB is measured.
- Standing engine-adoption rule: lab-isolate first in `otr-tts-lab`. The 8GB LTX path has not done that pass yet.

**Sequencing call (proposal recommendation):** still+KB ships as v2 baseline; LTX Micro becomes the v2.1 headline upgrade. Gives a marketing arc instead of one flat release.

**The deciding measurement:** a real render-time pass of LTX 2.3 GGUF Q2/Q3 at 384px on a 4060 8GB rig. If 6 hero shots fit under the 60-minute episode budget, LTX Micro becomes the right v2 ship target. If not, still+KB ships v2 and LTX Micro becomes v2.1.

### 10.3 Claude's synthesis critique (review-stage notes)

The proposal is structurally sound. Two weak joints to fix before integration:

1. **Replace static GB labels with relative tier labels.** "12GB recommended" / "16GB+ only" age badly and don't translate across vendors (Apple Silicon unified memory, AMD GB-per-tier). Better: **Green = always safe / Yellow = depends on your headroom / Red = 16GB+ only.** Let the validator do a runtime VRAM probe and decide whether to display warnings, instead of pinning to specific GB numbers in the UI text.

2. **Wan 2.2 5B is mis-tiered.** On an 8GB card with FLUX Q4_K_S already resident for stills (5-6GB), Wan 5B's 6-9GB will OOM. Yellow implies "might work with clean stack" — but the stack is not clean, because FLUX is the always-on still renderer in Green. Wan 5B belongs in **Red**, OR in a Yellow sub-tier with explicit "FLUX must be offloaded between still and video" wiring. Current placement is a footgun.

**What the proposal misses:**

3. **The HF Space "Quick / Better / Best" radio buttons need a mapping spec.** What does each button enable in terms of group state? Without that mapping, the buttons are decorative. Suggested mapping:
   - Quick → Green only.
   - Better → Green + Yellow (Real-ESRGAN upscale, no Wan B-roll).
   - Best → Green + Yellow + selected Red items (16GB-tier territory).

4. **The Runpod $50 validation pass should measure THREE things, not one:**
   - (a) LTX 2.3 Q2/Q3 on RTX 4060 8GB — the LTX Micro viability gate.
   - (b) Full Green-tier episode on AMD RX 7600 Linux (ROCm) — the vendor-agnostic claim test.
   - (c) Same Green-tier episode on Apple Silicon M2 8GB unified — the cross-platform breadth test.

   If (b) or (c) fails, the vendor-agnostic claim from Section 2b is rhetoric and needs scoping back to NVIDIA-only-on-8GB.

5. **Audio-pipeline VRAM residue is not addressed.** Even in Green-only mode, Bark / Kokoro / MusicGen leave residue between phases unless the lever-1 unload runs cleanly (this is the BUG-LOCAL-231 mechanism on a different card). The 8GB visual-phase budget is 8GB MINUS audio residue. One sentence in the doc + a guarantee that `free_otr_pipeline_residue()` fires between the audio and visual phases on the 8GB path.

6. **Failure-mode messaging is good but incomplete.** "Lab group X OOM'd at scene Y" assumes the OOM is recoverable mid-run. If FLUX OOMs in Green, the episode is dead. Need a second message variant: *"Your baseline OOM'd. Your hardware doesn't meet the 8GB-tier requirements. See README hardware table."* — distinct from *"Your opt-in group OOM'd, baseline is unaffected."*

### 10.4 Integration recommendation (when ready)

When the operator is ready to integrate this into the go-forward plan:

1. Take the Green/Yellow/Red proposal **verbatim** as the UI architecture.
2. Apply fixes 1, 2, 3, 5, 6 above before writing the workflow JSON.
3. Expand the Runpod $50 pass to the three-vendor measurement (fix 4) as the gating action.
4. Hold the structural call (still+KB v2 baseline, LTX Micro v2.1 headline) unless the Runpod LTX measurement comes back fast enough to revise it.

### 10.5 Open operator decisions added by this addendum

7. **Adopt Green/Yellow/Red as the workflow UI architecture?** (Claude recommends: yes, with the 5 fixes from §10.3.)
8. **Approve the three-vendor Runpod measurement pass** (LTX 8GB viability + AMD ROCm + Apple Silicon) as the next gating action before any v2 ship JSON work? (Claude recommends: yes.)
9. **Confirm still+KB v2 baseline / LTX Micro v2.1 headline sequencing?** (Claude recommends: yes, contingent on Runpod LTX measurement not surfacing surprisingly favorable numbers.)

---

## 11. Captions — burn-in is a ship requirement (DECIDED 2026-05-25)

**Operator decision:** ship v2 with burned-in captions, accessibility-first framing for hearing-impaired viewers. NOT optional. Not Yellow tier. **Green tier, mandatory baseline.**

### 11.1 Why the constraint forces burn-in

Episode distribution path is **OBS Studio → live broadcast / RTMP out** (per the `output/episodes_for_obs/<ep>/<ep>.mp4` ship-folder convention). OBS Media Source does NOT read:

- Embedded soft subtitle tracks (`mov_text` codec in mp4 container) — ignored.
- Sidecar SRT files paired with the mp4 — ignored.
- Any external caption format that requires player-side parsing.

OBS broadcasts pixels. Captions must be in the pixels by the time the mp4 reaches the OBS scene. Burn-in is the only path that survives the OBS broadcast layer intact.

### 11.2 Architecture

**New node:** `OTR_BurnInCaptions`, inserted between `OTR_RTXUpscale` and the final mp4 emit. Burn-in goes on the **highest-resolution frame** (post-upscale) for crispness — burning on 1472×832 then upscaling to 1920×1080 would blur the text. Per-frame ffmpeg subtitle filter pass on the upscaled mp4.

```
[EpisodeAssembler] -> [VideoComposite] -> [RTXUpscale] -> [OTR_BurnInCaptions] -> final.mp4
```

**Tier placement:** Green (always-on, baseline). Updates §10.1 — caption burn-in joins the locked baseline visual stack alongside FLUX still + Ken Burns + interpolation + RTXUpscale + FFmpeg mux.

**ffmpeg surface:** `subtitles=<srt_path>:force_style='...'` filter with ASS styling injected. Optionally read a styling config from an OTR node widget so the style is repo-controlled, not hardcoded into ffmpeg flags.

**Source data:** read from in-memory ledger (passed via socket), NOT from disk (BUG seen 2026-05-25 with ledger lock + WinError 5 atomic-rename failures). The node accepts `led` as input, walks `lines[]` for `text` + `start_s` + `dur_s`, emits a temporary SRT to the temp dir, runs the ffmpeg filter, deletes the temp SRT.

### 11.3 Sidecar SRT still ships always-on

The on-disk SRT sidecar is the BACKUP, not the deliverable. Costs nothing to emit (same data the burn-in node already reads). Reasons to keep it:

- Accessibility tools that DO read sidecar SRT (VLC, web players, archival systems).
- Future YouTube/Vimeo upload path where SRT can override auto-captions.
- Text-searchable archive of every episode's dialogue (script-search across the catalog).
- Translation pipeline target — translate the SRT, ship a `<ep>.es.srt` / `<ep>.ja.srt` next to the mp4 for any platform that does support paired sidecars.

Implement as either a second node `OTR_EmitSRT` (Green tier, runs before `OTR_BurnInCaptions`) or as a side-effect of `OTR_BurnInCaptions` itself (it generates the SRT internally for ffmpeg anyway — write it out before deleting).

### 11.4 Caption style — to be locked in a sub-document

Initial Claude recommendations (NOT yet operator-confirmed):

- **Font:** sans-serif, accessibility-tuned. Atkinson Hyperlegible (open-source, designed by Braille Institute for low-vision readers) or Inter as fallback.
- **Color:** white text with strong black drop-shadow OR black 2-3px outline (readable on any background — critical given dynamic FLUX still backgrounds).
- **Position:** bottom-third, safe-area padding (~5% margin from frame edge).
- **Speaker prefix:** *STANLEY:* "The pressure hull..." — italicized character name distinguishes who's speaking when voices sound similar; standard radio-drama caption convention.
- **Non-dialogue cues:** `[♪ tense musical interlude]` / `[radio static]` / `[explosion]` — standard accessibility convention for SFX + music beats. Required for the hearing-impaired use case (the whole point — they need to know what's audibly happening, not just what's said).
- **Timing:** show 200ms before line start, hold 200ms after line end (perceptual breathing room, prevents flash-cuts).

**Style spec lives inline in ROADMAP.md §B5** (operator directive 2026-05-25: bake the decision into the roadmap, no separate style-spec doc).

### 11.5 Updated open operator decisions

10. **Confirm burn-in captions as Green-tier baseline?** (DECIDED 2026-05-25: yes.)
11. **Always-on SRT sidecar as accompanying ship artifact?** (Claude recommends: yes, free + accessibility + future translation pipeline.)
12. **Caption style spec — lock in separate doc before node implementation?** (Claude recommends: yes, 200 lines of config that ages well.)
13. **`OTR_EmitSRT` separate node OR side-effect of `OTR_BurnInCaptions`?** (Claude recommends: side-effect, fewer moving parts, single source of truth for caption data.)
