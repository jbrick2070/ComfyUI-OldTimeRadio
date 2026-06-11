# Comfy-native 3D engine options for the OTR video dropdowns — first 3D test case

**Question for the panel:** which 3D (or 2.5D) generative models are ComfyUI-native or
near-native ENOUGH that they could be wired into our video engine dropdowns as the FIRST
working 3D test case in DAYS, not weeks — explicitly WITHOUT the parked heavy-toolkit lane
(no compiler toolchain, no source builds)? Evaluate, correct, extend, and rank the candidate
classes below; recommend ONE first test case + one runner-up; declare explicit NO-GOs.

## Context (the app)

OTR (Old Time Radio) generates complete radio-drama episodes in ComfyUI: script (local LLM)
→ frozen byte-identical audio master → per-beat VIDEO clips → mux. Video engines are
selected per dropdown SLOT in the production workflow: `announcer_visual`, `music_visual`,
`other_beats_visual` (cast/character beats). Each engine is an adapter registered in a
capability registry (grounding file 1) with role-compat, VRAM class, and enable-set reasons;
a capability profile (grounding file 3) resolves which engines a tier may use. Cheap CPU
families already exist (grounding file 2): still_kenburns (FLUX still + pan/zoom),
station_card, visualizer, abstract. Heavy engines shipped and working: HuMo 1.7B/14B
(audio-driven talking character, 480x832), LTX-Video (1472x832 landscape scene/radio-open),
Wan 2.2 i2v, LatentSync (lip-sync over a base clip). Fallback chains are LOUD and
ledger-stamped (e.g. humo -> humo_1.7B -> latentsync -> still_kenburns).

A SEPARATE, parked "character_3d" lane (TripoSG mesh -> ARKit-52 wrap -> Rhubarb visemes ->
Blender alpha render) is the full talking-3D-head pipeline; its sidecar needs a cu128
compile toolchain the machine does NOT have (probe NO-GO: no ninja/nvcc/VS BuildTools).
THIS roundtable is NOT that lane. We want the cheap on-ramp: a 3D-flavored visual engine
that drops into the dropdowns next to the 2D options.

## Hard constraints (binding; reject any candidate that violates them)

- Windows 11, RTX 5080 Laptop 16 GB (Blackwell sm_120), torch 2.10.0+cu130, Python 3.12,
  ComfyUI 0.24.1 (Desktop + headless). NO compiler toolchain: wheel-only installs
  (`--only-binary=:all:` mindset). Anything needing nvdiffrast / diff-gaussian-rasterization
  / spconv / flash-attn / kaolin source builds on sm_120 = NO-GO unless a real Blackwell
  wheel demonstrably exists.
- V-12 dependency isolation: a candidate's pip deps must NOT downgrade torch / numpy /
  transformers in the main venv (the indextts2/chatterbox lesson — those went sidecar).
  In-process preferred; a python sidecar venv is acceptable ONLY if wheel-clean.
- Single resident heavy engine <= 14.5 GB VRAM (NVML, dynamic ceiling); engines must
  free/reclaim after use.
- 100% local/offline RENDER path. ComfyUI Partner/API nodes (Hunyuan 3D 3.0 API, Rodin,
  Tripo API) violate this for rendering — list them only as a labelled "cloud lane"
  footnote, not a recommendation. (Cloud is sanctioned for the writer LLM only.)
- Determinism: seed-keyed reproducibility (OS-entropy seeds at episode level, but a given
  request seed must reproduce). Licensing: commercial-clean strongly preferred (MIT /
  Apache-2 / permissive; flag anything NC or research-only).
- Output contract: an engine renders a CLIP (or frame directory) at an exact ledger
  target_frame_count; canvases 1472x832 landscape (or padded); silent video, audio muxed
  LAST from the frozen master. Talking/lip-sync NOT required for announcer_visual or
  music_visual — motion/parallax/orbit visuals are fine there.

## Candidate classes (panel: correct facts, add missing candidates, rank)

- **A. 2.5D depth parallax ("still_parallax"):** DepthAnythingV2 (or Comfy-core depth) on
  the existing FLUX still -> displacement/camera-warp pan (the kenburns upgrade). Pure
  wheels, tiny VRAM, deterministic. Not "real" 3D — does it count as a credible first
  3D test case for the operator?
- **B. Native mesh-gen + orbit render:** ComfyUI core ships native Hunyuan3D-2 (2mv)
  support (geometry, no texture in core?); generate mesh from the episode still/portrait,
  render a turntable/orbit clip (Load 3D node camera? export path to frames?). What is the
  REAL no-compile path from mesh to an mp4/frame-dir on 0.24.1, and what texture options
  exist wheel-only (Hunyuan3D-2.1 paint? vertex color)?
- **C. Image-to-multiview/orbit video models:** SV3D / Zero123-class / newer 2025-26
  open-weight orbit-video models that output frames directly (skip meshes entirely).
  Which are wheel-only on sm_120, <= 14.5 GB, and license-clean?
- **D. Gaussian splatting lanes:** LGM / TRELLIS(.2) / splat renderers — most need compiled
  rasterizers. Is there ANY wheel-only splat gen+render path on Windows sm_120 (e.g.
  gsplat wheels?), or is this class a NO-GO today?
- **E. World models (HyWorld / WorldMirror 2.0):** stack-drift risk on torch 2.10/cu130 —
  assumed NO-GO for the first test case; confirm or refute briefly.
- **F. Anything we missed:** 2026-era comfy-native 3D-ish video options (depth-warp video
  nodes, camera-control video LoRAs on engines we ALREADY run — e.g. LTX/Wan orbit-style
  camera LoRA = "3D feel" with zero new deps?).

## What the panel must deliver

Per candidate: (1) wheel-only feasibility on THIS stack (cite the actual dep that compiles,
if any); (2) VRAM envelope; (3) slot fit (announcer/music/other_beats) given no-lip-sync
slots exist; (4) integration cost in days against the adapter/registry pattern above;
(5) determinism + license; (6) failure-mode fit with the LOUD fallback chain. Then: ONE
recommended first 3D test case, one runner-up, explicit NO-GOs. Be adversarial: kill
hand-wavy candidates; cite concrete package/wheel realities over vibes.
