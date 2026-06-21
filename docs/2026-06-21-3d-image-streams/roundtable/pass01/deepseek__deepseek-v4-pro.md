<!-- requested_model: deepseek/deepseek-v4-pro | resolved_model: deepseek/deepseek-v4-pro-20260423 -->

VERDICT: no. Critical design gaps (missing prompt templates, unresolved subject selection, integration points between image fork, render driver, and composite) make it unbuildable as-is.

MUST-FIX BEFORE BUILD:
1. [Composite Opaque Placement] The plan states the mesh must be placed opaquely, not ghosted, but provides no design for the composite change. The current `_silent_procgen_blended_final` ghosts the mesh. Without specifying how the composite will use the background plate and place the mesh opaquely (e.g., alpha-composite over plate), the entire visual goal is unimplementable. Concrete fix: add a design section detailing the composite pipeline modification, including how the background plate path from the manifest replaces the current ghosted background, and ensure the composite logic respects straight-alpha.

2. [Prompt Templates Missing] The “Prompt templates” open question is unanswered. The plan relies on mesh-fodder and background-plate prompts that must “reliably yield isolated subjects” and “subject-free plates,” but no concrete strings, negative prompts, or validation against the listed image engines (flux_gen1, z_image_turbo, flux2_klein, lumina_image) are provided. Without tested templates, the core fix (improved 3D fodder) remains speculative. Concrete fix: provide exact prompt scaffolds (including negative prompts) proven on each supported engine, with example outputs, before build.

3. [Subject Selection Unresolved] The plan asks how to choose the 3D subject when the slot has no character (announcer/music), whether to use a story-object, and how objects are identified in the ledger. This is an open question with no resolution, leaving the fork logic undefined for non-character beats. Concrete fix: decide subject selection rules – e.g., use `char_id` if present; if absent, look for a `story_object_id` field on the shot or fall back to a generic placeholder object; document the ledger field(s) used.

4. [Ledger Taxonomy and Render-Driver Integration] The plan proposes adding `mesh_fodder` and `background_plate` kinds but never specifies how the render driver and composite will read them. Currently `build_request_from_shot` assigns `init_image` from scene still for `image_to_video` family; that must change to use the mesh fodder. The composite manifest builder (`build_clip_manifest`) extracts `bg_still_path` from `_still_index`, which only finds `scene_*` kinds. Concrete fix: define:
   - The exact `kind` strings (`mesh_fodder`, `background_plate`) and their ledger row structure (include `beat_id`, `char_id`).
   - In `build_request_from_shot`, for engines requiring mesh fodder (detected via new capability), set `init_image` to the path of the `mesh_fodder` row for the beat (lookup by beat_id and kind).
   - In `build_clip_manifest`, for `mesh_stage` rows, set `bg_still_path` to the `background_plate` row path (lookup by beat_id and kind).

5. [Dispatcher Skip for Background Plate] The dispatcher’s `_still_needed_for_role` checks only the video engine’s consumption of stills. A background plate is not consumed by the video engine; it would be skipped, making the plate never generate. Concrete fix: either bypass `_still_needed_for_role` for kinds `background_plate` or extend the dispatcher to unconditionally generate any object whose kind is `background_plate` when the beat’s video engine requires a mesh (as indicated by the new capability field).

6. [Mesh Cache Key for Non-Character Subjects] `mesh_cache_key` uses `character_id`. If the subject is an object (e.g., artifact), a generic or empty `character_id` will cause cache collisions. Concrete fix: extend `mesh_cache_key` to accept a `subject_id` that defaults to `character_id` but can be set to an object identifier, ensuring unique cache keys per distinct object.

SHOULD-FIX:
7. [Aspect Reconciliation Documentation] The plan notes mesh fodder wants near-square/portrait, plate wants 16:9. Ensure that the prompt generator emits `w`/`h` fields consistent with these goals, and the dispatcher passes them through. Already feasible, but explicitly document the expected dimensions.

8. [Capability Field Definition] Define a proper `requires_mesh_portrait` (boolean) class attribute on the engine and gate all fork decisions (prompt generation, dispatcher, render driver) on it, never on engine name. This is the “gate on capability” invariant.

OPTIONAL / NICE-TO-HAVE:
- None identified given the current state; basic completeness is required first.

CUT THESE (over-engineering):
- Nothing clearly over-engineered; generating two targeted images per 3D beat appears necessary to fix the documented blob issue.

[ASSUMPTION] The plan assumes the composite pipeline can be altered to support opaque placement; no evidence of feasibility is provided.
[ASSUMPTION] The plan assumes the validated image engines can produce the mesh fodder and background plate as described with the yet-undefined prompts.