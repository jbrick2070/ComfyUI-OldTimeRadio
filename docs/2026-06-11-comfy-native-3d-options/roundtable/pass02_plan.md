# TRADITIONAL LOCAL 3D — converged build spec (the easy on-ramp lane, pass-2 final)

**One-plan rule:** this lane is INTEGRATED into `docs/2026-06-09-3d-toolkit/
3D_TOOLKIT_PLAN.md` (section 0-E) — there is ONE 3D plan. This file is the detailed spec
the plan points at. ADDITIVE: the parked toolkit lane (TripoSG/ARKit/Rhubarb/Blender
talking heads; S-3D-0 gates) is untouched and remains the destination for rigged/talking
3D. Campaign: 2 passes, 4 reviewers each, ~$0.22 total, grounded on the live install.

## The traditional chain (operator-requested shape)

    character portrait / full-figure still (existing per-episode FLUX assets)
      -> Hunyuan3D-2mv NATIVE ComfyUI-core mesh gen (in-process, compile-free —
         nodes verified PRESENT in the running install: EmptyLatentHunyuan3Dv2,
         Hunyuan3Dv2Conditioning[MultiView], VAEDecodeHunyuan3D, VoxelToMesh, SaveGLB)
      -> GLB mesh, cached per character (canonical-portrait key)
      -> Blender headless stage (pinned portable Blender, WORKBENCH matcap v1):
         virtual set/void, turntable-orbit camera preset, exact ledger frame count
      -> frame directory -> existing compositor/mux (frozen audio untouched)

Mesher runner-up (license hedge): TripoSR (MIT) in a wheel-clean sidecar venv with the
`skimage.measure.marching_cubes` swap (the S-3D-0-blessed technique); vertex-color via
Blender Color Attribute. Lower human-likeness (2/5) — hedge only.
NO-GO (dep named): SF3D/InstantMesh/CRM/Unique3D (nvdiffrast JIT = runtime nvcc/ninja),
TRELLIS(.2) (spconv/flash-attn), Era3D (recon stage compiles), 3DTopia-XL (compile-heavy).
Step1X-3D (Apache) = recorded candidate, unprobed. EEVEE = banned v1 headless.

## The three on-ramp engines (one family of work, three dropdown rows)

| Engine id (working) | What | Cost | Slots |
|---|---|---|---|
| `ltx_orbit` | camera-orbit prompt preset on the EXISTING LTX adapter (pass-1 winner; zero new deps; optional camera-LoRA v1.1 gated on naming the exact asset+license) | ~1 day | all three |
| `still_parallax` | 2.5D depth parallax on existing stills; DepthAnythingV2-SMALL (Apache) pinned; extends the static_motion cheap family; CPU-degradable | 1-2 days | all three |
| `mesh_stage` | the traditional chain above — the first REAL 3D OBJECT in the workflow | 3-5 days | music + announcer first; character beats after look-QA |

## Build tickets (E-1..E-7; coder-window ready)

- **E-1 mesher adapter:** hy3d-2mv core-node path wrapped portrait->GLB; model_requirements
  recorded (checkpoint list verify-at-build); VRAM probe on the 5080; free-after-use +
  reclaim (BUG-291 pattern) + `torch.cuda.empty_cache()` BARRIER before any Blender spawn;
  never concurrent torch-mesher + Blender-GPU (GPU lease seam).
- **E-2 mesh cache:** key = character_id + CANONICAL reference portrait content-hash +
  mesher id/version (NOT the per-episode portrait hash — regen trap); manifest JSON
  sidecar (source hash, character id, mesher version, seed, license, schema ver) for
  later ARKit-lane bridging; episode-scoped casts reuse within episode, stable cast
  (announcer) across episodes.
- **E-3 Blender stage kit:** pinned PORTABLE Blender zip under the env-pointed tools dir
  (`OTR_BLENDER_EXE`, fail-closed LOUD if missing; no dev-path hardcode); `--background
  --factory-startup --python stage.py`; stage.py: GLB import, bbox-normalize/center/
  scale, camera framed from bounds, ONE v1 camera preset (turntable-orbit, fixed
  radius/elevation), ONE material mode (matcap; Color Attribute when vertex colors
  exist), WORKBENCH v1 (Cycles v1.5 tier: fixed seed `cycles.seed=request_seed`,
  use_animated_seed=False, fixed samples, adaptive+denoise OFF), fixed thread count,
  frame_start=1/frame_end=N inclusive/step=1.
- **E-4 publish + paths:** render to short-ID tmp dir under the otr tmp tree -> validate
  count==N + dims -> ATOMIC dir-rename publish; stale-tmp cleanup; `\\?\` long-path
  prefix defensive; spawn via the sanitized-env pattern.
- **E-5 engine adapter + registry:** `mesh_stage` row (family token verified against
  schemas.FAMILIES at build — never overload triposg_talk), required_inputs init_image,
  role_compat (music_visual, announcer_visual, character_video), vram_class/estimate from
  E-1 probe, required_toolchain=None, requires_sidecar=False (hy3d core; TripoSR hedge
  would be True), commercial_clean per license record; SELECTABLE in 16gb_full, never
  default without operator say-so; LOUD fallback mesh_stage -> still_parallax ->
  still_kenburns with ledger restamp + structured reason codes.
- **E-6 acceptance:** cube-GLB 3-frame headless SELF-TEST probe (gates first use); per
  engine: frames==ledger target, 1472x832 contract (set EXPLICITLY — cheap-family default
  is 832x480), seed reproducibility, ledger engine stamps, `test_audio_byte_identical`
  untouched; ONE coverage-sweep leg per new engine as the acceptance render; visual
  contact sheet (first/mid/last frames).
- **E-7 license + docs + tracker:** Tencent community license review (thresholds +
  territory clauses) recorded operator-visible BEFORE default-on; TripoSR-MIT hedge
  documented; tracker rows; plan cross-links.

## Hard sequencing gates

1. Track-3 directory-clip read path + CPU fixture MERGED (in flight now) before E-5
   registers anything.
2. The running coverage sweep finishes (or coordinate file ownership with the coder
   window) before registry/profile edits land.
3. Section-0 operator gates (restart-side look-QA etc.) keep their order; this lane is
   PARALLEL-SAFE and touches none of the parked toolkit files.

Raw reviews: `pass02/`. Judgment: `pass02_judgment.md`. Pass-1 (quick wins): `pass01_plan.md`.
