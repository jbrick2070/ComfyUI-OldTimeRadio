# Pass 2 — TRADITIONAL 3D, comfy-friendly, LOCAL: harden the portrait->mesh->Blender chain

Pass-1 converged on quick "3D-feel" wins (ltx_orbit prompt preset; still_parallax 2.5D).
The OPERATOR now wants the TRADITIONAL 3D workflow hardened as a first-class test case:

    character portrait / full-figure still (we already generate these per episode)
      -> LOCAL image-to-3D MESHER (wheel-only, no compiler toolchain)
      -> 3D object (GLB/OBJ)
      -> ANIMATED in a 3D scene / virtual background
      -> frames at exact ledger frame count -> our compositor muxes (audio frozen)

This is ADDITIVE to an existing parked heavy lane (TripoSG -> ARKit-52 -> Rhubarb ->
Blender talking-head pipeline, blocked on a cu128 compile toolchain probe). Nothing is
removed; this chain is the no-toolchain on-ramp that must BRIDGE to that lane later
(same Blender render seam; mesher swappable; faces gain ARKit animation later).

## Stack (binding, same as pass 1)
Windows 11, RTX 5080 Laptop 16 GB (sm_120), torch 2.10.0+cu130, Python 3.12, ComfyUI
0.24.1. Wheel-only python deps (no nvcc/ninja/VS BuildTools, no nvdiffrast/spconv/
flash-attn/kaolin source builds). Blender as a STANDALONE app binary IS allowed (it is
not a pip dep; the parked lane already plans headless Blender for its render stage).
Prebuilt binaries (Rhubarb-style) allowed. 100% local. <=14.5 GB resident VRAM,
free-after-use. Deterministic per request seed. Commercial-clean strongly preferred.
Output: silent clip OR frame directory, 1472x832 landscape, exact target_frame_count.

## What the panel must evaluate (adversarial; cite package/wheel realities)

1. **The MESHER menu (the core question).** For image->mesh on THIS stack, wheel-only:
   - Hunyuan3D-2 / 2mv native in ComfyUI core (geometry only; texture stage lives in
     wrapper code that compiles CUDA rasterizers — confirmed out). VRAM? mini variants?
     Tencent community license implications?
   - TripoSR (MIT): does it still need torchmcubes/compiled marching cubes, or is a pure
     wheel path (skimage marching_cubes swap) viable? Quality vs hy3d?
   - Stable-Fast-3D / SF3D (Stability): textured GLB in one shot — but which deps compile
     (uv-unwrapper? texture baker?), and what is the license tier for commercial use?
   - Step1X-3D (Apache-2.0), Craftsman, CRM, Unique3D, Era3D, 3DTopia-XL, or any
     2025-26 mesher we have missed: wheel-only? quality? license?
   - For EACH: dep that compiles (name it), VRAM, output format, texture or geometry-only,
     license, and a 1-5 "mesh quality on a human character portrait" honesty score.
2. **The ANIMATION/STAGE tier (Blender headless).** v1 animation WITHOUT rigging: orbit /
   dolly / floating-idle camera paths, turntable, simple object rotation, virtual set
   (HDRI or stylized void), lights. Confirm: EEVEE headless on Windows GPU vs Cycles
   (determinism pins? GL context pitfalls in a spawned headless render?), GLB import of
   untextured meshes + stylized material assignment (matcap/bronze/hologram) by script,
   exact-frame-count rendering, frame-dir output. What breaks first in a spawned
   `blender --background --python stage.py` on Windows? Auto-rigging for tier-2 motion:
   is there ANY local, scriptable, license-clean auto-rig worth naming (Mixamo is cloud =
   out), or does rigged motion correctly stay with the parked ARKit lane?
3. **Pipeline contracts.** Per-CHARACTER mesh caching (mesh keyed by portrait content
   hash; one mesh reused across that character's beats + episodes — the existing plan
   already mandates character-level 3D granularity); LOUD fallback chain
   (mesh-gen fail -> still_parallax/kenburns); VRAM sequencing (mesher transient, Blender
   render after free); where Blender lives (machine-level install path pinned in config?
   portable zip in tools dir?); determinism (Blender seed/sampling pins, fixed thread
   counts); Windows long-path/atomic-publish gotchas for frame dirs.
4. **Sequencing sanity.** This chain should ride the SAME frame-directory plumbing the
   repo's Track-3 work is building now (directory-clip read path + CPU fixture) and must
   not modify the parked heavy lane. Flag anything in the chain that would conflict.

Deliver: a ranked mesher pick (+ runner-up) for the v1 traditional test case, the v1
Blender stage spec (animation tier-1 only), explicit NO-GOs with the compiling dep named,
and the 5-8 build tickets you would cut this into. Kill hand-wavy candidates.
