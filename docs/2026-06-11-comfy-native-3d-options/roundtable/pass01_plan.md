# Easiest LOCAL 3D add for the OTR dropdowns — converged recommendation (pass 1)

**Scope pin (operator, 2026-06-11): ADDITIVE ONLY.** Nothing is removed from the 3D plan.
The toolkit lane (TripoSG -> ARKit-52 -> Rhubarb -> Blender; the hunyuan3d_talk /
trellis_talk dark scaffolds; S-3D-0 and its gates) is UNTOUCHED and remains the real
talking-3D destination. This doc only adds the cheapest LOCAL on-ramp so a 3D-flavored
option can be tested in the dropdowns in days. Everything below is 100% local render.

## The pick (unanimous, 4 reviewers)

### 1. FIRST TEST CASE: `ltx_orbit` — camera-orbit preset on the LTX engine we already run
- **What:** a new registry row + prompt-template preset on the EXISTING LTX-Video adapter:
  episode still in (init_image), slow orbit/dolly/parallax camera language appended from a
  fixed preset table (era-tail compatible), same seed plumbing, same 1472x832 landscape.
  v0 = prompt-directed camera only (ZERO new files, zero new deps). v1 (optional) = a
  published LTX camera-control LoRA through the existing LoRA loader — gated on naming the
  exact asset + license first (Grok's condition; UNVERIFIED today).
- **Why it wins:** zero new pip deps (wheel question vacuous), VRAM already proven
  (12.5 GB row, fits 14.5 ceiling), deterministic via existing seed plumbing, license
  already accepted (we ship LTX), LOUD fallback slots in unchanged
  (ltx_orbit -> still_kenburns), and it shows a camera moving THROUGH a scene — the
  "3D feel" — in every dropdown (announcer_visual, music_visual, other_beats via
  character_video role-compat).
- **Cost:** ~1 day. **Honest label:** 3D-feel (diffusion prior), not a 3D asset — name it
  `ltx_orbit`, not `3d_*`.

### 2. RUNNER-UP: `still_parallax` — 2.5D depth-parallax upgrade of the kenburns family
- **What:** depth map from the existing FLUX still (PIN the Apache-2.0 model:
  DepthAnythingV2-SMALL — Base/Large are CC-BY-NC; Marigold or DA-V1 are Apache
  alternates) -> ~200-line torch grid_sample camera warp, in-repo, no new deps beyond the
  depth checkpoint. Registers as a NEW engine id in the EXISTING `static_motion` family
  (reuses the cheap-family base; new id = its own dropdown row).
- **Why second:** real parallax from OUR stills, <4 GB VRAM, CPU-degradable, deterministic;
  slightly more code than ltx_orbit (warp + depth inference + edge-fill fallback to plain
  kenburns when depth is low-confidence).
- **Cost:** 1–2 days. Label: 2.5D parallax.

### 3. FIRST REAL 3D ASSET (later, still no toolchain): `hy3d_orbit` — native Hunyuan3D-2mv
  mesh + stylized orbit
- ComfyUI core ships hy3d-2mv GEOMETRY natively (wheel-only, ~6–10 GB transient). Core has
  NO texture stage (texture lives in wrapper code whose rasterizer compiles CUDA = out).
  v1 look = OWNED stylization: bronze/hologram/wireframe matcap orbit of the untextured
  mesh — fits the sci-fi radio brand; music_visual first.
- TWO verify-at-build items before promising it: (a) offscreen mesh->frames renderer
  (pyrender/OpenGL is wheel-only but Windows-GL fiddly; Load 3D node is preview-only, not
  a render path); (b) Tencent community license review. Park behind the test of #1/#2;
  it does NOT gate on, and does NOT modify, the toolkit lane.

## Explicit NO-GOs for the FIRST test (the 3D plan itself is unchanged)
- **SV3D / Zero123 orbit-video class:** square low-res output violates the 1472x832
  contract (postage-stamp padding); NC-tier licenses. (Gemini, grounded.)
- **Splats / TRELLIS(.2) / LGM:** rasterizer + spconv/flash-attn class deps have no
  sm_120/cu130/Windows wheels; registry already pins the trellis_talk scaffold to
  cu128_toolkit. Trellis2-wrapper "wheels" claim = verify-at-build, expect NO-GO.
- **HyWorld / WorldMirror world models:** stack-drift risk on torch 2.10/cu130 + no clean
  fixed-frame-count clip contract. Revisit post-toolkit.
- **Cloud/API 3D (Hunyuan 3.0 partner nodes, Rodin, Tripo API):** violates the local
  render invariant — awareness footnote only, never a build item.

## Build contract for the coder window (folded from GPT's must-fixes)
1. Registry row per new engine: engine id, family (`image_to_video` for ltx_orbit;
   `static_motion` for still_parallax), roles + role_compat (announcer_visual,
   music_visual, character_video), vram_class/estimate, required_toolchain=None,
   requires_sidecar=False, cpu_ok (parallax degraded mode), model_requirements (exact
   depth checkpoint for parallax; LoRA file IF v1), commercial_clean per engine.
2. Canvas + frames: adapters set 1472x832 + ledger target_frame_count EXPLICITLY (the
   cheap-family default is 832x480 when canvas is absent — do not inherit it silently).
3. Acceptance smoke per engine: frame_count == target_frame_count; silent stream; yuv420p;
   1472x832; ledger carries seed + engine id; LOUD fallback fires on depth-fail /
   missing-still / VRAM refusal and restamps the ledger.
4. Determinism: request-seed reproducibility on the existing plumbing; record backend
   nondeterminism caveats in the engine docstring.
5. Profile: both engines enter the 16gb_full enable-set as SELECTABLE (no default
   role_override flip without operator say-so).
6. Wheel proof ritual (one line, per V-12): `pip install --only-binary=:all:` transcript
   (or "no new deps" statement) recorded in the ticket before registration.

## Sequencing vs the live build
This work is dropdown/engine-lane code, parallel-safe with 3D Track 3, but it touches the
same registry/profile files the coverage sweep exercises — land it AFTER the sweep
finishes (or coordinate with the coder window) and re-run one sweep leg per new engine as
its acceptance render. Section-0 ordering is unaffected: this is a GATE-A-adjacent
dropdown addition, not a new sprint.

Raw reviews: `pass01/` (gpt-5.5, gemini-3.1-pro, grok-4.3, claude-panelist).
Judgment log: `pass01_judgment.md`. Spend: ~$0.12 of the OpenRouter budget.
