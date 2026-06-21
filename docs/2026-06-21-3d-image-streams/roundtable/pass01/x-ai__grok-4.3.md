<!-- requested_model: x-ai/grok-4.3 | resolved_model: x-ai/grok-4.3-20260430 -->

VERDICT: no. The plan rests on unspecified seams (fork timing, capability exposure, subject choice, composite integration) and false assumptions about early knowledge of 3D routing + engine flags that do not exist in the grounded code.

MUST-FIX BEFORE BUILD:
1. [Capabilities / "The design"] Design gates fork exclusively on `requires_mesh_portrait` / `character_3d` / `mesh_stage` capability fields; eng_mesh_stage.py:MeshStageEngine exposes none of these (only `roles`, `family="image_to_video"`, `required_inputs=("init_image",)`, `uses_still=True`). Fix: add explicit capability attr to the engine class (and registry query path) before any image-phase gating logic; do not fall back to name checks.
2. [Sequencing / "Open questions for the panel" + "The design"] Fork must occur where image phase already knows both per-beat role *and* the eventual video engine capability, but dispatcher.py only sees `image_policy["video_models"]` for `_still_needed_for_role` / `engine_consumes_still` (via `accepts_still` or `required_inputs`); video engine assignment for a beat can occur later in render_driver.py:build_request_from_shot / run_real_episode. Fix: make ImageDirector policy the single source that writes the chosen video engine per role/beat into the policy before any prompt/dispatch step; reject policies that leave it unset for 3D-capable roles.
3. [Subject selection / "Open questions for the panel"] No rule for choosing fodder subject on announcer/music beats (no `char_id`, no story-object register in ledger). Fix: define explicit fallback (e.g., first `beat_id`-linked object in ledger["images"] or a synthetic "scene_object" row) and the corresponding prompt source; announcer/music must not silently reuse cinematic portrait.
4. [Ledger taxonomy / "Open questions for the panel"] Design requires new `mesh_fodder` + `background_plate` kinds + distinct cache keys, but dispatcher.py:dispatch_images and render_driver.py:_still_index / _portrait_index only key on existing `kind` (scene_* or portrait) + `beat_id`/`object_id`; no collision handling or dual-row association for the same beat. Fix: extend request_cache_key + row stamping to carry both kinds per beat_id with separate prompt_hash/seed entries; update _still_index to prefer `mesh_fodder` when engine capability is 3D.
5. [Aspect contract / "Open questions for the panel" + eng_mesh_stage.py:DEFAULT_W/H] Mesh fodder specified as near-square/portrait for reconstruction; mesh_stage hard-codes landscape canvas (1472x832) and passes the same `still` both to mesher and Blender texturing. Fix: add explicit `fodder_aspect` vs `plate_aspect` to the engine request and reconcile in build_request_from_shot before calling the mesher; reject portrait fodder into landscape-only 3D path.

SHOULD-FIX:
1. [Prompt templates / "Open questions for the panel"] No scaffolds or negative prompts supplied for fodder (isolated subject, diffuse light) vs plate (empty environment); dispatcher.py only forwards `obj["prompt"]`. Fix: add versioned templates in the prompt-gen seam with role+engine gating so cinematic prompt never reaches a 3D beat.
2. [Composite integration / Problem + "The design"] Design changes from ghosting (`_silent_procgen_blended_final`) to opaque placement over `background_plate`; render_driver.py only stamps `bg_still_path` for mesh_stage rows and C1 manifest. Fix: extend build_clip_manifest to emit distinct `bg_plate_path` (from `background_plate` kind) and require composite to read it when engine=mesh_stage.
3. [Cache migration / "Open questions for the panel"] Switching source image changes `portrait_content_hash` (good) but old cinematic meshes remain in `_cache_root()`; no invalidation step. Fix: add one-time migration that drops meshes whose manifest `source_portrait_sha256` no longer matches any current fodder row.

OPTIONAL / NICE-TO-HAVE:
- Add per-beat `mesh_fodder_seed` override in ledger so operator can force a specific clean view without changing the main request seed.
- Unit test the exact capability query path end-to-end (dispatcher -> engine attr) with a mock 3D engine.

CUT THESE (over-engineering):
- None. The dual-stream split is the minimal change that satisfies the reconstruction requirement and the "no engine-name checks" invariant; everything else is additive ledger keys only.

[ASSUMPTION] Image phase (director/dispatcher) will always run after video engine assignment is written into policy; verify against actual call order in saved workflows.
[ASSUMPTION] Composite code (not in excerpts) will consume `background_plate` rows exactly as `bg_still_path` is consumed today.