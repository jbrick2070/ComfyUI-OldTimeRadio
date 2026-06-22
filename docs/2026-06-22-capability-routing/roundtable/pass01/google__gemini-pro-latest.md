<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: no. Dropping the whitelist entirely destroys semantic routing for specialized engines, allowing them to silently fill inappropriate roles just because they meet the input requirements.

MUST-FIX BEFORE BUILD:
1. [Proposed direction] Semantic Over-matching. 
   - Defect: The plan states "drop the separate `roles` whitelist gate" entirely. This means specialized engines (e.g., `station_card` in `cheap_families`, which likely only requires a `text_prompt` or `init_image`) will suddenly match *every* general role (like `scene_broll` or `music_visual`) purely because their input requirements are low. This violates the "no-silent-swap safety" by introducing semantic mismatches.
   - Fix: Do not drop the `roles` gate entirely. Make it an *optional strict override*. Update `engine_fits_role`: if `descriptor.get("roles")` is explicitly defined and non-empty, enforce the whitelist. If it is empty or `None` (like `wan_i2v` or `ltx_video`), bypass the whitelist and rely *purely* on the capability match.

2. [Current state / Proposed direction] The `roles` vs `default_roles` Descriptor Bug.
   - Defect: The grounding asks "where does the director descriptor's `roles` come from?" If the descriptor builder is currently populating `descriptor["roles"]` by falling back to `engine.default_roles` (which is `()` for `wan_i2v`), that is the root cause of the bug. The plan fails to address how the descriptor is built.
   - Fix: Explicitly decouple `default_roles` (auto-selection preference) from `roles` (capability/semantic restriction) in the director descriptor builder. Do not inject an empty `roles` list into the descriptor just because `default_roles` is empty.

3. [Proposed direction] Render-side Dual Source of Truth.
   - Defect: The plan says "Single source -> both gates derive from it" but fails to specify how `render_driver._assert_family_inputs_satisfiable` will be updated. Currently, it checks `FAMILY_REQUIRED_INPUTS`, which is a separate hardcoded mapping that will drift from the engine's `required_inputs`.
   - Fix: Route `_assert_family_inputs_satisfiable` to read from the exact same `descriptor["required_inputs"]` (or the engine class attribute directly) rather than maintaining a separate `FAMILY_REQUIRED_INPUTS` map.

SHOULD-FIX:
1. [Proposed direction] `character_video` vs `character_3d` mismatch.
   - Defect: `character_3d` requires `audio_ref`. `ROLE_AVAILABLE_INPUTS["character_video"]` does *not* supply `audio_ref`. Therefore, `character_3d` will fail to fit the `character_video` role. This is technically true today as well, but the naming strongly implies a gap in the role's available inputs.
   - Fix: Either add `audio_ref` to `character_video`'s available inputs, or explicitly document that 3D characters are strictly announcer/music models.

OPTIONAL / NICE-TO-HAVE:
- Rename `roles` to `restricted_roles` or `allowed_roles_override` in the engine declarations to make its semantic-locking purpose obvious and prevent future engineers from confusing it with `default_roles`.

CUT THESE (over-engineering):
- N/A. The plan is currently under-engineered regarding the render-side gate and semantic restrictions.

[ASSUMPTION] The plan assumes image engines have a parallel `ROLE_AVAILABLE_INPUTS` structure and identical descriptor logic that can be cleanly unified. The grounding provides zero visibility into the image engine routing code, roles, or capabilities.