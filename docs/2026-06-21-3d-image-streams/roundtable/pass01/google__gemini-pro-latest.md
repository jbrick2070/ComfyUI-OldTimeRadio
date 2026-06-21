<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: no. The design fundamentally misunderstands what image `mesh_stage` is currently receiving, and the proposed gating capability does not exist on the engine.

MUST-FIX BEFORE BUILD:
1. [Problem] The premise that `mesh_stage` receives the per-character PORTRAIT is false. Because its family is `image_to_video`, `render_driver.py`'s `_SCENE_INIT_FAMILIES` logic overrides `init_image` with the SCENE STILL. This is why Hunyuan meshes the whole room.
   Fix: In `render_driver.py` `build_request_from_shot`, bypass the `_SCENE_INIT_FAMILIES` override for engines that declare `getattr(eng, "requires_mesh_portrait", False)`, and feed them the new `mesh_fodder` image instead.
2. [The design] The design proposes gating the fork on the `requires_mesh_portrait` capability to avoid hardcoding engine names, but `MeshStageEngine` in `eng_mesh_stage.py` does not declare this attribute. The gate will fail-closed.
   Fix: Add `requires_mesh_portrait = True` to the `MeshStageEngine` class definition in `eng_mesh_stage.py`.
3. [Ledger images taxonomy] If the background plate is given `kind="background_plate"`, `render_driver.py`'s `_still_index` will ignore it because it strictly filters for `kind.startswith("scene_")`. The composite will receive no `bg_still_path` and fall back to black.
   Fix: Name the new kind `scene_background_plate` so `_still_index` picks it up, or update `_still_index` to explicitly match `background_plate`.
4. [Ledger images taxonomy] `_still_index` maps `beat_id` to a single path. If the image prompter mints both a `scene_still` and a `scene_background_plate` for the same beat, `_still_index` will blindly return whichever appears last in the ledger array (a race condition).
   Fix: Ensure 3D beats mint ONLY the background plate (no generic `scene_still`), or update `_still_index` to prioritize `scene_background_plate` over `scene_still`.

SHOULD-FIX:
1. [Subject selection] Announcer and music slots have an empty `char_id`. If they route to `mesh_stage`, looking up `mesh_fodder` by `char_id` will yield nothing.
   Fix: For non-character beats, key the `mesh_fodder` on `beat_id` or a story object ID, and update `render_driver.py` to fall back to `beat_id` when resolving the fodder.
2. [Where does the fork live?] The fork must live in `OTR_MetaBriefImagePromptGen`. If `OTR_ImageDirector` handled it, it would have to synthesize the distinct prompts without the LLM context.
   Fix: Implement the fork in `OTR_MetaBriefImagePromptGen` so it can use the LLM to generate the specialized "mesh fodder" and "background plate" prompts.

OPTIONAL / NICE-TO-HAVE:
- [Aspect] `mesh_stage` renders at 1472x832 because it is not in `_face_excl`. This is fine for the 3D stage, but ensure the `mesh_fodder` image prompt explicitly requests a square or portrait aspect ratio so Hunyuan3D gets an isolated subject.

CUT THESE (over-engineering):
1. [Mesh cache] "Keep cache keys distinct (the fodder must not collide...)": Extra cache key logic is unnecessary. `otr_image_gen_dispatcher.py`'s `request_cache_key` already includes the `kind` parameter, so passing `kind="mesh_fodder"` automatically isolates the cache key from the cinematic portrait.