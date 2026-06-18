<!-- model: gemini-pro-latest (direct API) -->

VERDICT: no. The document is an options paper, not a finalized spec, and Candidate B introduces a fatal data-loss bug while Candidate A leaves `role_compat` blind to optional inputs.

MUST-FIX BEFORE BUILD:
1. [Candidate B] Fixed-path overwrite risk. If engines blindly read `<episode>/stills/<role>.png`, multiple beats using the same role will overwrite each other's stills before rendering. Fix: Reject Candidate B's path convention. The dispatcher must pass the specific beat's image path to the video engine explicitly (e.g., via `init_image=path` kwargs), preserving beat-level granularity.
2. [Candidate A / Grounded facts] `ltx_video` optional input blindness. If you add `accepts_still=True` to fix the `ltx_video` dispatch bug, `role_compat.engine_fits_role` (which checks `required_inputs`) remains blind to it. Fix: Introduce `optional_inputs=("init_image",)` to the engine protocol. Update both `_still_needed_for_role` and `role_compat` to check `required_inputs + optional_inputs`.
3. [Grounding: `_ROLE_TO_VIDEO_SLOT`] The dictionary maps roles to slots but is missing `announcer` and `character_3d` [ASSUMPTION: these roles exist based on the Operator Ask mentioning "music / announcer / beats" and "character_3d talkers"]. Missing roles fall through to `return True`, bypassing the procedural optimization. Fix: Add all valid roles to `_ROLE_TO_VIDEO_SLOT`.
4. [Grounding: `_still_needed_for_role`] The bare `except Exception: return True` directly violates the "no silent fallback (every skip/degrade is LOUD)" invariant. Fix: Catch `KeyError`/`ImportError` specifically, and emit a loud logger warning before returning `True`.

SHOULD-FIX:
1. [The decision to harden] Choose a Hybrid "Opt-Out" approach. Default `accepts_still = True` on the base classes. Only add `accepts_still = False` (or `optional_inputs=()`) to the 3 procedural/audio-only engines (`abstract`, `visualizer`, `ltx_av_music`). This avoids modifying all 20 engines while preserving the accessible no-image floor.
2. [Candidate A] 3D dark scaffolds lack a base class. If you rely on base class defaults for `accepts_still`, the dark 3D talkers (`triposg_talk`, etc.) will crash or misbehave. Fix: Ensure the dark scaffolds explicitly declare the new attributes or inherit from a minimal protocol base.

OPTIONAL / NICE-TO-HAVE:
- Rename `_still_needed_for_role` to `_role_consumes_still` to better reflect that it asks the engine about its capabilities, not the workflow about its requirements.

CUT THESE (over-engineering):
1. [Candidate A] `image_engines.registry.usable(name, role)`. Safe to cut because `role_compat.py` already handles the role-to-engine matrix. Do not build a redundant capability matrix in the registry.
2. [Candidate A] `still_kind in {init_image, mesh_portrait}`. Safe to cut because `requires_mesh_portrait=True` already exists, works perfectly for the 3D lock, and avoids migrating 20 engines to a new string enum just to unify a boolean. Keep the boolean.