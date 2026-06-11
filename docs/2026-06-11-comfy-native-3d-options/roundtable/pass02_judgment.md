# Pass-02 judgment — traditional local 3D chain (Claude judge; 3-model panel + Claude panelist)

Panel: gpt-5.5, gemini-3.1-pro, grok-4.3 + Claude panelist (independent, pre-read).
Spend pass 2: ~$0.10 (campaign total ~$0.22). Verdict: CONVERGED after grounding —
remaining panel disputes were settled by reading the ACTUAL installed ComfyUI, not by
another pass.

## Decisive grounding (done on the live machine, this session)

`C:\Users\jeffr\ComfyUI-Installs\ComfyUI\ComfyUI\comfy_extras\nodes_hunyuan3d.py` EXISTS
in the running install: EmptyLatentHunyuan3Dv2, Hunyuan3Dv2Conditioning(+MultiView),
VAEDecodeHunyuan3D, VoxelToMeshBasic/VoxelToMesh; plus nodes_load_3d.py + nodes_save_3d.py
(GLB export path). Core mesh-gen is in-process, compile-free (VoxelToMesh = torch-side
surface extraction). => **Hunyuan3D-2mv native = the v1 mesher.** Gemini's "TripoSR is
the ONLY viable wheel-only pick" and its claim that registry cu128 rows cover core
hy3d-2mv: **MISREAD** (those rows are the parked TALKING-toolkit scaffolds
hunyuan3d_talk/trellis_talk, a different thing). TripoSR (MIT, +skimage marching-cubes
swap) demotes to the LICENSE-HEDGE runner-up — kept, since Tencent license review is open.

## Accepted (CONFIRMED) panel items folded into the spec

- **Cache-key flaw (Gemini M2 + GPT M4):** per-episode portrait hash would regen meshes
  per episode. FIX: mesh keyed by character_id + CANONICAL reference portrait hash +
  mesher id/version; honest scope note: OTR casts are mostly per-episode entities, so
  cross-episode reuse applies to stable cast (announcer) — key supports both. Mesh
  manifest JSON sidecar (GPT/Grok) adopted (provenance + later ARKit bridging).
- **VRAM barrier (Gemini M3 + GPT M5):** torch's allocator retains the pool — explicit
  reclaim (the BUG-291 wrapper_bridge.reclaim_idle_models pattern) + empty_cache barrier
  BEFORE the Blender spawn; rule: never run torch mesher + Blender GPU concurrently
  (the existing OTR_GPU_LEASE_DIR seam wraps the render).
- **Render engine (adjudicated 3-way):** v1 = WORKBENCH headless (fast, deterministic,
  matcap = the stylized look we already chose; vertex-color via Color Attribute when the
  mesher provides it — Gemini's cut of UV texturing adopted). Cycles = the v1.5 lit-set
  tier (fixed seed/samples, adaptive+denoise OFF). EEVEE = BANNED v1 on Windows headless
  (GL-context flake — Gemini M4 right about EEVEE, wrong that Cycles is the only option;
  GPT's "EEVEE unless probe fails" rejected as flake-prone). Startup SELF-TEST probe
  (GPT M9/OPTIONAL): render a known cube GLB 3 frames headless before first use.
- **Frame contract (GPT M6):** frame_start=1, frame_end=N (inclusive), step=1; validate
  rendered count == N pre-publish; fail LOUD.
- **Atomic publish + paths (GPT M10 + Gemini SF2):** render to short-ID tmp dir under the
  otr tmp tree, validate, atomic dir-rename; short stable IDs primary, `\\?\` prefix
  defensive.
- **Track-3 gate (GPT M7):** the engine does NOT register until the directory-clip read
  path + CPU fixture land (in flight in the coder window NOW). Hard sequencing gate.
- **Registry/naming (GPT M8 + Grok M1/M4):** NEW engine id `mesh_stage` (working name) —
  never overload triposg_talk; full CAPABILITIES row + family token verified against
  schemas.FAMILIES at build; commercial_clean enforced per-mesher (policy: NC/unclear
  licenses cannot be default — GPT SF1 adopted).
- **v1 scope cuts (GPT):** ONE camera path preset (turntable-orbit, fixed radius/
  elevation) + ONE material mode for the seam-hardening build; frame-dir is the ONLY v1
  output (clip canonicalization later); broad mesher survey cut from build scope
  (Step1X-3D et al recorded as candidates only); auto-rig fully out of v1 (all four
  reviewers) — articulated/talking motion stays with the parked ARKit lane.

## Rejected / downgraded (with reason)

- Gemini "TripoSR is the only pick" — killed by live grounding (above).
- Gemini "Hunyuan3D-2 carries cu128_toolkit per registry" — misread of the talking-lane
  rows; core mesh nodes verified present and compile-free.
- Grok SF2 "--enable-cycles-denoiser-off flag" — no such Blender CLI flag; the intent
  (denoise off) is set in stage.py scene settings instead. --factory-startup adopted.
- GPT cut #4 "prefer EEVEE, cut Cycles" — inverted by adjudication (Workbench v1,
  Cycles v1.5, EEVEE banned).
- Verify-at-build (UNVERIFIABLE, recorded): hy3d-2mv checkpoint file list + VRAM on THIS
  card; SaveGLB node coverage for the voxel-mesh output; Workbench `--background`
  behavior on this exact Blender build (the cube self-test answers it); Tencent license
  territory/threshold reading (operator-visible record before default-on).
