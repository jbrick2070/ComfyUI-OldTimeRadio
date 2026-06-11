<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: no. The plan mandates wheel-only (no compilers) but suggests meshers that strictly require JIT compilation, relies on a flawed caching key that defeats its own reuse goal, and will OOM during the Blender handoff.

MUST-FIX BEFORE BUILD:

1. [Section 1] The Mesher Menu NO-GOs. SF3D, Unique3D, and CRM strictly depend on `nvdiffrast` and/or `spconv`. Even if installed via wheel, `nvdiffrast` invokes `ninja` and `nvcc` at runtime to JIT-compile CUDA kernels, violating your "no compiler toolchain" constraint. Hunyuan3D-2 and Trellis are already correctly flagged in `registry.py` as requiring `cu128_toolkit`.
   *Fix:* TripoSR (MIT) is the ONLY viable 100% wheel-only pick. You must use the `skimage.measure.marching_cubes` fallback to bypass `torchmcubes` compilation. It outputs vertex colors, not UV textures.

2. [Section 3] Broken Caching Contract. You state: "mesh keyed by portrait content hash; one mesh reused across that character's beats + episodes". But you also state: "character portrait... we already generate these per episode". If the portrait is generated per episode, its content hash changes per episode, meaning the mesh will regenerate every episode, destroying cross-episode consistency.
   *Fix:* Key the 3D mesh cache by `Character_ID` + `Mesher_Version`, using a single canonical reference portrait per character, NOT the per-episode portrait hash.

3. [Section 3] VRAM Sequencing OOM. You state "mesher transient, Blender render after free". Unloading the mesher model from ComfyUI does NOT free the VRAM to the OS; [ASSUMPTION] PyTorch's caching allocator will hold the 14.5GB VRAM pool. When the standalone Blender subprocess spawns, it will immediately OOM on the RTX 5080.
   *Fix:* The pipeline must explicitly call `torch.cuda.empty_cache()` (and ideally `gc.collect()`) *before* invoking the `subprocess.run()` for Blender.

4. [Section 2] Headless EEVEE on Windows. Running `blender --background` using EEVEE on Windows GPU often fails or falls back to CPU because [ASSUMPTION] EEVEE requires an active desktop/display server to initialize the OpenGL/Vulkan context. It will crash in an isolated background process.
   *Fix:* Force the Blender python script to use the `CYCLES` engine with the `OPTIX` or `CUDA` compute device, which correctly initializes headlessly without a display server.

5. [Section 4] Missing Registry Declaration. The plan states it "must not modify the parked heavy lane" (`triposg_talk` in `registry.py`). But to ride the same plumbing, the new mesher needs its own capability declaration, which is missing from the plan.
   *Fix:* Add a new row to `CAPABILITIES` in `registry.py` (e.g., `"triposr_stage": {"vram_class": "heavy", "vram_estimate_mb": 6000, "required_toolchain": None, "requires_sidecar": False, "cpu_ok": False}`).

SHOULD-FIX:

1. [Section 2] Auto-rigging fantasy. There is no local, scriptable, commercial-clean auto-rigging Python library that doesn't rely on heavy compiled ML dependencies (like SMPL-X fitting).
   *Fix:* Explicitly scope v1 animation strictly to object-level transforms (orbit, dolly, turntable). Defer all skeletal rigging to the parked ARKit lane.

2. [Section 3] Windows Long Paths. Frame directory outputs with deep hash paths will hit the Windows 260-character `MAX_PATH` limit during Blender's atomic write.
   *Fix:* Prefix the output directory path passed to Blender with `\\?\` (e.g., `\\?\C:\path\to\frames`) to force Windows to bypass the legacy path length limit.

OPTIONAL / NICE-TO-HAVE:
- [Section 2] Determinism: In the Blender script, explicitly set `bpy.context.scene.cycles.seed = request_seed` and set `bpy.context.scene.cycles.use_animated_seed = False`.
- [Section 1] TripoSR output is `.obj` or `.ply`. GLB export requires an extra conversion step in Python (via `trimesh`) before handing off to Blender, or just import the `.obj` directly into Blender.

CUT THESE (over-engineering):
1. [Section 1 & 2] "Textured GLB in one shot" / "stylized material assignment". TripoSR generates vertex colors, not UV-mapped textures. Cut the requirement for UV texture mapping.
   *Why it's safe:* You can assign a Blender material that uses the `Color Attribute` (vertex colors) node directly to the Principled BSDF, achieving the exact same visual result without needing a UV unwrapper or texture baker dependency.