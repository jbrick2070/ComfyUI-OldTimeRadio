<!-- requested_model: deepseek/deepseek-v4-pro | resolved_model: deepseek/deepseek-v4-pro-20260423 -->

VERDICT: no. The plan is missing critical specifications: subject selection for non-character beats, fork location, composite modifications, prompt templates, and aspect reconciliation. Without these, the build cannot proceed.

MUST-FIX BEFORE BUILD:
1. [Open questions / Subject selection] The plan does not define how the mesh fodder subject is chosen for beats without a character (announcer/music slots). Without a subject, no mesh fodder can be generated, causing the 3D engine to fail. Fix: define a rule—e.g., if no char_id and no story-object register, the beat must not use a 3D engine; the image stage should fall back to a single 2D image, and the video phase should use a non-3D engine for that beat.
2. [Open questions / Fork location] The fork location is undecided. The image stage must be modified to produce two images for 3D beats, but the seam is not chosen. Fix: select OTR_ImageDirector (or another component that has access to engine capabilities) and define the interface for issuing two image generation requests per beat.
3. [Design / Aspect reconciliation] The mesh fodder and background plate require different aspect ratios (portrait vs. 16:9), but the plan does not specify how to request these from the image engines. Fix: define exact dimensions (e.g., 832×1216 for fodder, 1472×832 for plate) and ensure the image engines can produce them; add aspect parameters to the image generation request.
4. [Design / Composite change] The plan states the composite must place the subject opaquely instead of ghosting, but does not detail the changes to `_silent_procgen_blended_final` or the composite engine. Fix: specify the composite modification—e.g., use straight-alpha compositing without blending for 3D clips, or introduce a new composite mode.
5. [Design / Prompt templates] The plan relies on prompt scaffolds that “reliably yield isolated subjects + subject-free plates” but provides none. Without proven prompts, the generated images may not meet requirements. Fix: develop and test prompt templates for mesh fodder (isolated subject) and background plate (subject-free environment) on the target image engines, and include them in the design.
6. [Design / Engine capability flag] The plan says gating on engine capability, but the current engine code (eng_mesh_stage.py) lacks a flag like `requires_mesh_portrait`. Fix: add a capability attribute (e.g., `requires_mesh_fodder = True`) to MeshStageEngine and any other 3D engines, and implement the check in the chosen fork location.
7. [Design / Ledger taxonomy] The plan proposes `mesh_fodder` and `background_plate` kinds but does not specify how the video phase will consume them. `build_request_from_shot` and `build_clip_manifest` must be updated. Fix: update `build_request_from_shot` to set `init_image` to the mesh_fodder image for engines with `requires_mesh_fodder`, and update `build_clip_manifest` to set `bg_still_path` to the background_plate image for those beats.
8. [Design / Fallback handling] If mesh fodder generation fails, the video phase would receive no init_image, causing a hard failure. The plan does not address this. Fix: define fallback behavior—e.g., if mesh fodder generation fails, fall back to generating a single 2D image and mark the beat to use a non-3D engine; or in the video phase, if init_image is missing, the engine should fail closed and walk its fallback chain.

SHOULD-FIX:
- Specify the exact aspect ratios and dimensions for mesh fodder and background plate, and validate that the image engines support them.
- Include a migration strategy for existing cached meshes (old cinematic-portrait meshes must be invalidated).
- Detail how the background plate will be passed to the composite (e.g., via manifest field `bg_still_path` or a new field).
- Clarify how the image stage will know the planned engine for each beat (reading the video section of the ledger) and ensure the video section is available at image stage time.

OPTIONAL / NICE-TO-HAVE:
- Provide an environment variable to override the mesh fodder subject for testing.
- Add a debug mode to visualize the mesh fodder and background plate.

CUT THESE (over-engineering):
- The “story-object register” for subject selection is premature if not yet implemented. The initial implementation can rely on char_id only; announcer/music slots can fall back to a generic subject or be excluded from 3D.
- The extensive discussion of mesh cache key mechanics is already handled in the code; no need to re-specify.