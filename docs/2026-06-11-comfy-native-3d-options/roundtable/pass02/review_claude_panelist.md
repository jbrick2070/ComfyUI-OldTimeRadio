# Claude panelist review, pass 2 (written BEFORE reading the panel — independent)

## 1. Mesher menu (ranked)

**#1 Hunyuan3D-2mv, ComfyUI core native — the v1 mesher.** Zero install (it ships in the
core we already run = the most "comfy-friendly" possible), wheel-only by construction,
~6–10 GB transient, GLB/mesh out, geometry-only (texture stage = compiled rasterizer,
stays out). Quality on human character portraits: the best of the no-compile field (4/5
geometry). LICENSE FLAG the panel must not skip: Tencent community licenses carry
commercial-use thresholds AND territory exclusions — needs an operator-visible record,
not a vibe. Mitigation: the MIT fallback below.

**#2 TripoSR (MIT) with the marching-cubes swap — the license-clean fallback.** Stock
TripoSR pulls torchmcubes (COMPILES — name the dep). Swap extraction to
`skimage.measure.marching_cubes` (pure wheels) — the SAME swap our parked plan's S-3D-0
pre-step already specifies for TripoSG, so the technique is plan-blessed. ~4–6 GB, fast.
Honesty score on humans: 2/5 (blobby) — fine as fallback + license hedge, not the demo.

**#3 SF3D / Stable-Fast-3D — verify-at-build, expect NO-GO.** The one-shot TEXTURED GLB
is tempting (kills the clay look) but its uv-unwrap/texture-bake extensions are compiled
C++/CUDA on Windows sm_120; Stability community license has revenue thresholds. Only
worth a probe if the clay look fails operator look-QA.

**Step1X-3D (Apache-2.0):** right license, but the texture stage compiles and the geometry
stage's dep tree needs a real probe; UNVERIFIED — record as candidate, not v1.
**NO-GO (name the dep):** InstantMesh/CRM/Unique3D (nvdiffrast), TRELLIS(.2)
(spconv/flash-attn), Era3D (multiview-only, needs a recon stage that compiles),
3DTopia-XL (compile-heavy).

## 2. Blender stage (the part pass-1 undersold)

- **Engine pick for v1: WORKBENCH, not EEVEE.** EEVEE headless needs a GL context —
  historically flaky in `--background` (verify-at-build per-version); Cycles is reliable
  headless but slow. Blender's Workbench engine renders headless, FAST, deterministic,
  and its matcap/cavity look IS the stylized bronze/clay/hologram aesthetic we already
  chose for untextured meshes. v1 = Workbench matcap turntable/orbit/dolly presets;
  v1.5 quality tier = Cycles (fixed seed, fixed samples, adaptive sampling OFF,
  denoise OFF or pinned OIDN) when we want lit virtual sets with shadows.
- **Determinism pins:** PINNED portable Blender zip (exact version recorded; lives under
  the env-pointed tools dir, e.g. C:\ComfyUI-Models\tools\blender-<ver>, env var
  OTR_BLENDER_EXE, fail-closed LOUD if missing); fixed thread count; scripted scene only
  (no .blend state); frame_start/end set from ledger target_frame_count = exact frames
  by construction.
- **Windows gotchas:** render to a short tmp dir under output\otr\tmp then atomic
  dir-rename publish (the plan's existing Windows atomic-publish caveat applies); spawn
  with the existing sanitized-env pattern (build_sidecar_env); long-path safety = keep
  frame dirs shallow.
- **GPU sequencing:** mesher is transient (free-after-use, like FLUX portraits); Blender
  is a SEPARATE PROCESS — take the existing GPU lease (OTR_GPU_LEASE_DIR seam) around the
  render when using Cycles-GPU; Workbench v1 makes this nearly moot (CPU/GL-light).
- **Auto-rig (tier-2 motion): correctly stays parked.** Mixamo = cloud (out). Rigify =
  manual placement (not automatable v1). UniRig-class 2025 auto-riggers = research-watch,
  unverified deps. v1 animation = camera paths + object rotation + floating-idle ONLY;
  articulated/talking motion remains the parked ARKit lane's job. This keeps the bridge
  clean: same Blender seam, meshes swap in, rig arrives with the toolkit lane.

## 3. Contracts

Mesh cache keyed by PORTRAIT CONTENT HASH per character (one mesh per character reused
across beats/episodes — matches the plan's character-level granularity lock + the
ImageDirector must-fix); LOUD chain mesh_stage -> still_parallax -> still_kenburns with
ledger restamp; registry row family image_to_video (required_inputs init_image),
selectable-not-default in 16gb_full; texture-free mesh + matcap = commercial_clean
depends ONLY on the mesher license (record per-mesher).

## 4. Ticket cut (6)

E-1 mesher adapter (hy3d-2mv core path) + content-hash mesh cache + LOUD fail-down.
E-2 Blender stage kit: pinned portable Blender + stage.py (GLB import, matcap preset
    table, camera-path presets, Workbench, exact frames, frame-dir out, atomic publish).
E-3 `mesh_stage` video-engine adapter wiring portrait->mesh(cache)->Blender->frame-dir
    clip; registry+profile rows; fallback chain.
E-4 determinism + acceptance smoke (frames==ledger, 1472x832 canvas contract, seed
    reproducibility, ledger engine stamps, byte-identical audio untouched).
E-5 one coverage-sweep leg per new engine as the acceptance render.
E-6 license record (Tencent thresholds/territories; TripoSR-MIT hedge) + docs + tracker.

Sequencing: rides Track-3's directory-clip plumbing; zero contact with the parked lane's
files; land after the running sweep (or coordinate) per the pass-1 note.
