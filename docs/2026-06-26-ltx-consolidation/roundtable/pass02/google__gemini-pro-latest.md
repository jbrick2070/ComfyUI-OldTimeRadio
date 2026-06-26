<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: yes-with-fixes. The role-driven routing is the right architectural fix, but the proposed `still_route` helper regresses a documented aspect-ratio bug and lacks the context to perform the checks it claims.

MUST-FIX BEFORE BUILD:
1. [Part B1] **Landscape engines on character beats (BUG 1 regression).** The rule `_is_character_face_beat AND engine consumes a still -> portrait` ignores the engine's native aspect ratio. Grounding explicitly states `flux_still` and `flat_still` are landscape (`render_aspect="wide"`) and MUST NEVER condition on the vertical portrait, even on character beats, to avoid pillarboxing.
   *Fix:* Condition the portrait route on the engine's `render_aspect`. If `getattr(engine, "render_aspect", "portrait") == "wide"`, it must route to `scene_still`, not `portrait`.

2. [Part B0] **`_is_character_face_beat` misses line-level `char_id`.** The classifier checks `char_id present`, but `char_id` is often stored on the ledger `line`, not the `shot` dict (as seen in `build_request_from_shot`: `shot.get("char_id") or line.get("char_id")`).
   *Fix:* Change the signature to `_is_character_face_beat(shot, line)` and check for `char_id` on both dicts.

3. [Part B1] **`still_route` signature lacks ledger context.** The plan expects `still_route(engine, shot)` to enforce "Missing-still terminality". Without the `ledger` to resolve the actual image paths, the helper cannot know if the still is missing. Furthermore, raising an exception inside the request builder bypasses the `run_episode` trace and fallback logging.
   *Fix:* Pass `ledger` (or the pre-resolved `portrait`/`_still`/`_fodder` paths) to `still_route`. To fail loud properly, return a missing-source string (e.g., `init_image=""`, `init_source="missing_scene_still"`) and let `_assert_family_inputs_satisfiable` catch the empty `init_image` and raise `FamilyInputGap`, which preserves the trace stamp.

SHOULD-FIX:
1. [Part B2] **`_uses_ambient_master_audio` signature.** Passing the `shot` into `_uses_ambient_master_audio` is fine, but since you are already computing `_is_character_face_beat(shot, line)` in the builder, it is cleaner to just pass that boolean: `_uses_ambient_master_audio(engine_id, family, is_char_face)`.

OPTIONAL / NICE-TO-HAVE:
- [Part C.3] When updating the canvas clamp (1082) to `("ltx_audio_in",)`, also update the log message in that block to reflect the unified engine name.