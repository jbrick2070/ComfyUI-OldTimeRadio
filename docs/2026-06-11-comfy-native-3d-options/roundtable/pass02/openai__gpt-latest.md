<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. This is a research/evaluation brief, not a build-ready spec: it lacks a selected mesher, exact contracts, adapter/profile integration, frame-count rules, cache keys, and failure sequencing.

MUST-FIX BEFORE BUILD:
1. [1 MESHER menu] No ranked v1 mesher is selected, but the rest of the chain depends on its install path, model assets, output format, VRAM envelope, license, and failure modes. Concrete fix: choose exactly one v1 mesher and one runner-up before implementation; freeze for each: package/source, install command, Python/torch compatibility, required model files, output format, texture support, license flag, peak VRAM target, and known no-go deps.

2. [1 MESHER menu] The spec requires “wheel-only” and “NO-GOs with the compiling dep named,” but several candidates are left as open questions. This is not buildable because implementation cannot know what to exclude. Concrete fix: create a candidate table where every rejected candidate names the specific blocking dep/toolchain, e.g. CUDA rasterizer / marching-cubes / UV unwrapper / texture baker package as applicable. If the exact dep is not verified, mark candidate “unapproved pending install probe,” not eligible for v1.

3. [Stack] The bound runtime is extremely specific: Windows 11, RTX 5080 Laptop sm_120, torch 2.10.0+cu130, Python 3.12, ComfyUI 0.24.1. The document does not prove the chosen mesher has wheels compatible with that exact tuple. Concrete fix: add a pre-build install-probe ticket that creates the sidecar venv, installs the selected mesher from wheels only, imports it, loads the model, and runs one portrait to GLB/OBJ. Fail the build if pip attempts local compilation or if torch/CUDA/sm_120 compatibility is not available. [ASSUMPTION] This compatibility cannot be asserted from the provided excerpts.

4. [3 Pipeline contracts] “Per-CHARACTER mesh caching keyed by portrait content hash” contradicts “one mesh reused across that character’s beats + episodes.” If portraits are regenerated per episode, a content hash alone creates different meshes per portrait and will not guarantee character-level reuse. Concrete fix: cache key must include stable character_id plus a canonical portrait_asset_id/hash, and the mesh manifest must also include mesher name/version, model checkpoint id, seed, settings, license status, and output schema version.

5. [3 Pipeline contracts] VRAM sequencing is underspecified. The profile budget is 14500 MB in [16gb_full.json], and existing 3D capabilities in [registry.py CAPABILITIES] use 14000 MB estimates for character_3d lanes, leaving only ~500 MB margin. Running mesher and Blender in the same process/session risks allocator retention and overlap. Concrete fix: require the mesher to run in an isolated sidecar subprocess that exits before Blender starts; add an explicit unload/free barrier and a “no concurrent Blender + torch mesher” rule.

6. [2 ANIMATION/STAGE tier] Exact frame count is not defined at the Blender seam. Blender frame_end is inclusive, so naïvely setting frame_start=1 and frame_end=target_frame_count can work, but any off-by-one or FPS-derived duration logic will break the ledger. Concrete fix: define render contract as: input target_frame_count N and fps; set frame_start=1, frame_end=N, frame_step=1; render exactly N numbered frames; validate count before publish; fail loud if missing/extra frames.

7. [4 Sequencing sanity] The chain depends on “the SAME frame-directory plumbing the repo’s Track-3 work is building now,” but that plumbing is not shown as shipped. Concrete fix: add a hard sequencing gate: do not build/register this engine until directory-clip read path plus CPU fixture are merged and tested. If building earlier, include a local temporary frame-dir reader fixture and mark it disposable.

8. [registry.py / 16gb_full.json / 4 Sequencing sanity] There is no integration point for this new no-toolchain traditional 3D lane in the shown registry/profile. [registry.py CAPABILITIES] contains `triposg_talk`, `hunyuan3d_talk`, and `trellis_talk`, but no generic portrait-mesh-to-Blender engine; [16gb_full.json] defaults `video_render_engine` and `other_beats_visual` to `humo`. Concrete fix: add a new adapter/capability row only if intended to be selectable, with family `character_3d` or the repo’s accepted equivalent, role compatibility, flag gate, `requires_sidecar`, VRAM estimate, model requirements, and commercial_clean metadata. Do not change existing heavy-lane rows.

9. [2 ANIMATION/STAGE tier] Blender headless behavior on Windows GPU is treated as a question, not a contract. `blender --background --python stage.py` can fail first on GPU/OpenGL device selection, missing GLB import assumptions, file paths, or renderer/device availability. Concrete fix: define the Blender binary source/path, Blender version, render engine for v1, device policy, environment variables if any, and a startup self-test that imports a known GLB and renders 3 frames headlessly on the target machine.

10. [3 Pipeline contracts] Atomic publish for frame dirs is mentioned but not specified. A compositor may read partially rendered frames if Blender writes directly into the final directory. Concrete fix: render into a temp directory on the same volume, validate exact count and image dimensions, then atomically rename/move to final output; include cleanup of abandoned temp dirs.

11. [2 ANIMATION/STAGE tier] GLB/OBJ import is underspecified. OBJ brings path/MTL/texture ambiguity; untextured GLB/OBJ needs deterministic material fallback, scale, origin, camera framing, and coordinate normalization. Concrete fix: for v1 require GLB as normalized handoff if possible; if OBJ is accepted, require colocated MTL/textures or force a generated material. Script must normalize bounding box, center origin, set scale, assign fallback material, and frame camera from bounds.

12. [3 Pipeline contracts] “LOUD fallback chain” is not operational. It says mesh-gen fail -> still_parallax/kenburns, but not which errors fall back, where the failure is recorded, or how ledger frame count survives. Concrete fix: define fallback trigger classes, emit structured reason codes, write manifest status, and require fallback renderer to receive the same target_frame_count/fps/output-dir contract.

SHOULD-FIX:
1. [Stack / 1 MESHER menu] “Commercial-clean strongly preferred” is too weak for build selection. Concrete fix: define policy: either commercial_clean=true required for default/profile use, or non-commercial candidates require explicit opt-in flag and cannot be default. The engine protocol in [registry.py] has `commercial_clean`; the new adapter should set it and `assert_usable` should enforce the policy.

2. [2 ANIMATION/STAGE tier] Determinism is underdefined. Blender seed/sampling pins are mentioned, but the mesher and renderer may still vary across versions/devices. Concrete fix: define deterministic scope as “same machine, same versions, same model assets, same seed”; pin Blender version, script version, render settings, samples, resolution, fps, threads, and random seeds; write all to manifest.

3. [2 ANIMATION/STAGE tier] Tier-1 animation scope should be narrowed. The doc lists orbit, dolly, floating-idle, turntable, object rotation, HDRI/stylized void, and lights. Concrete fix: pick one default camera path for v1, e.g. turntable/orbit with fixed radius/elevation, plus one fallback material. Add more paths only after the seam is stable.

4. [3 Pipeline contracts] Blender location policy is unresolved. Concrete fix: choose one: machine-level configured executable path or portable zip under tools. Add config key, version check, and friendly error if missing. Do not hardcode a developer path.

5. [2 ANIMATION/STAGE tier] Auto-rigging is correctly out of scope for v1, but the spec still asks to name local auto-rig options. Concrete fix: explicitly mark rigged character motion as non-v1 and blocked to the parked ARKit lane unless a separate license-clean local rigging spike passes.

6. [Stack] “Output: silent clip OR frame directory” conflicts with [4 Sequencing sanity], which says ride frame-directory plumbing. Concrete fix: make frame directory the only v1 output. If a silent clip is needed later, implement it as a separate canonicalization step after frame validation.

7. [3 Pipeline contracts] Long-path handling is named but not specified. Concrete fix: cap generated path component lengths, use stable short IDs in cache/output paths, and test paths under Windows with deep episode/character names. Avoid relying on global Windows long-path policy.

8. [registry.py CAPABILITIES] Existing comments say `triposg_talk` is “the v1 NO-COMPILE character_3d lane” and still flag-gated dark. The new document says this is additive to the parked heavy lane and must bridge later. Concrete fix: clarify whether the new v1 lane reuses `triposg_talk`, adds a separate `portrait_mesh_blender` engine, or is only a test harness outside registry. Do not overload `triposg_talk` if it means TripoSG/ARKit/Rhubarb semantics.

OPTIONAL / NICE-TO-HAVE:
- Add a golden CPU fixture: tiny known mesh or cube GLB rendered through the same Blender script to prove frame-dir output without running the mesher.
- Add a mesh manifest JSON next to each cached mesh: source image hash, character id, mesher version, seed, dimensions, material mode, license, and provenance.
- Add a visual QA contact sheet for rendered frames to catch blank renders, microscopic meshes, inverted normals, or off-camera imports.
- Add a cleanup command for stale sidecar envs, model cache, temp frame dirs, and Blender crash leftovers.

CUT THESE (over-engineering):
1. [1 MESHER menu] Cut the broad survey list from the v1 build scope: Step1X-3D, Craftsman, CRM, Unique3D, Era3D, 3DTopia-XL, and “any 2025-26 mesher we missed.” Safe to cut because v1 needs one wheel-only mesher plus one runner-up, not an open-ended benchmark matrix.

2. [2 ANIMATION/STAGE tier] Cut auto-rigging from v1. Safe to cut because v1 explicitly says animation without rigging, and the parked ARKit lane owns face/rigged motion later.

3. [Stack / 4 Sequencing sanity] Cut silent clip output from v1. Safe to cut because the repo seam being targeted is frame-directory plumbing and the compositor muxes frozen audio later.

4. [2 ANIMATION/STAGE tier] Cut Cycles from v1 unless EEVEE headless fails the startup probe. Safe to cut because the goal is exact-frame stylized staging, not physically based rendering; Cycles adds sampling/device/determinism/VRAM variables.

5. [2 ANIMATION/STAGE tier] Cut multiple material modes/HDRI sets for v1. Use one deterministic fallback material and one simple light rig. Safe to cut because import/render seam hardening is the goal, not look-development breadth.