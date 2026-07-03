<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: no. The design assumes `mesh_stage` currently receives the portrait and that adding new image kinds will automatically route them, but `render_driver.py` currently feeds `mesh_stage` the 16:9 scene still, and will ignore any new `mesh_fodder` kind without explicit wiring.

MUST-FIX BEFORE BUILD:
1. [Problem] False premise: `mesh_stage` does NOT receive the portrait. Because its family is `image_to_video`, `render_driver.py` (`build_request_from_shot`) overwrites `init_image` with the 16:9 scene still via `_SCENE_INIT_FAMILIES`. This is the actual root cause of Hunyuan3D meshing the whole room. Fix: Update `render_driver.py` to bypass the `_SCENE_INIT_FAMILIES` override for `mesh_stage`, routing the new fodder image to `init_image` instead.
2. [Ledger images taxonomy] `render_driver.py` currently only extracts `_portrait_index` and `_still_index`. A newly minted `mesh_fodder` kind will be silently ignored. Fix: Add a `_fodder_index` helper to `render_driver.py` and modify `build_request_from_shot` to assign it to `init_image` when the engine is `mesh_stage`.
3. [Where does the fork live?] The design proposes gating the fork on `requires_mesh_portrait` or `character_3d` capability. `MeshStageEngine` lacks `requires_mesh_portrait` and its family is explicitly `image_to_video` (not `character_3d`). The fork will never trigger. Fix: Add `requires_mesh_portrait = True` to `MeshStageEngine` in `eng_mesh_stage.py`.
4. [BACKGROUND PLATE] `build_clip_manifest` in `render_driver.py` populates `bg_still_path` using `_still_index(ledger)`, which only matches `kind.startswith("scene_")`. A new `background_plate` kind will be ignored, leaving the composite with a black background. Fix: Mint the background plate with a kind like `scene_plate` so `_still_index` catches it, OR update `_still_index` to match `background_plate`.
5. [Subject selection] Announcer/music beats have no `char_id`. If the fork doesn't synthesize a subject, `init_image` will be empty and `mesh_stage` will crash (`FileNotFoundError` in `render_clip`). Fix: The prompt generator must synthesize a story-object (e.g., "vintage microphone") for non-character 3D beats, ensuring a `mesh_fodder` image is always minted.

SHOULD-FIX:
6. [Aspect] The design notes mesh fodder wants a near-square/portrait aspect. `cv_encode` in `eng_mesh_stage.py` hardcodes `"crop": "center"`. Fix: Ensure the image generator mints the `mesh_fodder` in a square or portrait aspect (e.g., 1024x1024) so the subject isn't clipped by the center crop.

OPTIONAL / NICE-TO-HAVE:
- Cache key migration: Since `mesh_cache_key` uses the content hash of the input image (`portrait_sha256`), switching the input to `mesh_fodder` automatically changes the hash. No explicit cache migration is needed; old blobs will just be orphaned safely.

CUT THESE (over-engineering):
- None. The stream split is strictly necessary to prevent Hunyuan3D from fusing the subject and environment into a single blob.