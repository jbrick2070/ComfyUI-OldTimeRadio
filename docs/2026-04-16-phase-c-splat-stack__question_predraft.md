# OTR v2.0 Phase C — Splat-Rendering Architecture Pick

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
