VERDICT: yes-with-fixes. Multiple interface contract mismatches, sequencing conflicts in chunk ordering, and circular dependency risks need to be resolved first.

MUST-FIX BEFORE BUILD:
1. [C3] Role Mapping Mismatch / Dead Code: `input.md` maps `"scene"` to `Role.SCENE_BROLL.value` in `SPEAKER_TO_VIDEO_ROLE`. However, the outline schema `_otr_outline.SpeakerRole` and `_otr_speaker_role.VALID_SPEAKER_ROLES` do not define `"scene"` as a valid beat speaker role. They only contain `"sfx"` for non-voiced sound effects/b-roll beats. The only ledger lines with `speaker_role == "scene"` are scene header section markers (e.g., `=== SCENE 1 ===`) which have no duration/audio and should not map to a video shot. Actual b-roll beats carry the `"sfx"` role and are currently left unmapped, falling back to `BACKGROUND_ABSTRACT`. [ASSUMPTION] We assume that the b-roll beats are mapped from `"sfx"` speaker role in production outlines/ledgers because `"sfx"` is the only non-voiced, non-music, non-announcer role in `VALID_SPEAKER_ROLES` and `_NEVER_HUMO_ROLES`.
Concrete Fix: Map `"sfx"` to `Role.SCENE_BROLL.value` in `SPEAKER_TO_VIDEO_ROLE` in `nodes/otr_shot_lock.py` so that actual sound effect/b-roll beats are routed to the `scene_broll_video_model`.

2. [C1] Circular Dependency / Layering Violation: `nodes/_otr_video_engines/render_driver.py` plans to use `engine_consumes_still(eng)`. However, `engine_consumes_still` is currently defined in `nodes/otr_image_gen_dispatcher.py` (an image-phase component that in turn imports from `_otr_video_engines`). Having the video driver import from the top-level image dispatcher creates a circular import risk and violates clean layered architecture.
Concrete Fix: Move `engine_consumes_still` to a shared, low-level module such as `nodes/_otr_shared/role_compat.py` or `nodes/_otr_video_engines/registry.py` and import it from there in both modules.

3. [C2] Base Class Pollution / Tight Coupling: Modifying `nodes/_otr_shared/engine_registry_base.py` to import and call `role_compat.engine_fits_role` breaks the dependency-free, reusable design of the base registry (which is designed for reuse by the image platform in C1).
Concrete Fix: Keep `engine_registry_base.py` dependency-free. Instead, pass the capability check function as a callback during registry instantiation, or override the registry methods (`engines_for_role`, `assert_usable`) in `nodes/_otr_video_engines/registry.py` (which is video-specific).

4. [C4 / C5] Sequencing Defect in Chunk Order: The `CHUNK ORDER` specifies that C1-C4 land and undergo QA before C5. However, C4's matrix test plans to assert that the soak fills the pick ("and (post-C5) the canonical soak fills it"). If C1-C4 land and are tested in QA before C5 is implemented, this soak assertion will fail.
Concrete Fix: Defer the soak-filling assertion in C4's matrix test to C5, or modify the chunk order so C4 and C5 land and undergo QA together.

5. [C1 / Transition] Transition Coverage Gap for B-Roll Beats: The plan leaves `still_motion` and `station_card` without `accepts_still = True` because they are retirement candidates. However, `still_motion` remains the default video engine for the `scene_broll` role (in `cheap_families.py`). During the transition phase, any beat running on `still_motion` will have `engine_consumes_still` return `False`, meaning the image dispatcher will skip generating the still. Consequently, `still_motion` will render a dark procedural floor instead of the b-roll still.
Concrete Fix: Either change the default engine for `scene_broll` in `cheap_families.py` to `still_pan` (which has `accepts_still = True` post-C1), or temporarily set `accepts_still = True` on `StillMotionFamily` until its retirement.

SHOULD-FIX:
1. [C5] Hardcoded Luma Floor: The luma threshold (ffmpeg signalstats YAVG) is mentioned but not specified as configurable. Hardcoding this value might cause false positives if different style profiles produce naturally darker scenes.
Concrete Fix: Make the luma floor threshold configurable via an environment variable (e.g., `OTR_SOAK_LUMA_FLOOR`) defaulting to a safe minimum.

2. [C4] Helper Location: The plan proposes adding `descriptor_for_engine` helper to "registry", but if placed in `engine_registry_base.py`, it couples the generic class to video-specific metadata assumptions (like `required_inputs`).
Concrete Fix: Add the `descriptor_for_engine` helper to `nodes/_otr_video_engines/registry.py` rather than the base registry file.

OPTIONAL / NICE-TO-HAVE:
- [C5] Cache-friendly temporal variance check: A simple frame diff sum or structural similarity index (SSIM) can be used to implement the temporal variance check efficiently to avoid CPU overhead during soak tests.

CUT THESE (over-engineering):
1. [C5] Complex temporal variance checks for visualizer: The procedural visualizer engine (procedural CRT scope) does not need motion checking, since it's procedurally generated from audio. If temporal variance checks are slow/complex, they can be bypassed for "visualizer" family engines entirely.
