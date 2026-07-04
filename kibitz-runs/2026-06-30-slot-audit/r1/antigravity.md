VERDICT: yes-with-fixes. The plan correctly identifies the mismatch between the capability-only validation and the legacy roles whitelist, but it misses key details about the still-dispatcher capability lookup, has speaker-role routing omissions, and makes incorrect assumptions about the video engines' capability declarations.

MUST-FIX BEFORE BUILD:
1. [CONFIRMED DEFECT 2 / SPRINT FIX 3] Still-carrier engines (still_pan, still_motion, station_card) render black because they lack the `accepts_still` capability attribute and do not list `"init_image"` in `required_inputs`. In nodes/otr_image_gen_dispatcher.py, _still_needed_for_role calls engine_consumes_still which resolves to False and skips generating the still. Fix: Add `accepts_still = True` to the StillPanFamily (still_pan), StillMotionFamily (still_motion), and StationCardFamily (station_card) classes in nodes/_otr_video_engines/cheap_families.py.
2. [THE 3 USER SLOTS / OPEN QUESTIONS 1] B-roll beats (speaker_role = "scene" in the ledger) cannot route to the scene_broll slot because SPEAKER_TO_VIDEO_ROLE in nodes/otr_shot_lock.py completely lacks the "scene" mapping, causing it to fall back to background_abstract. This renders scene_broll_video_model in nodes/otr_video_director.py entirely unused. Fix: Add `"scene": Role.SCENE_BROLL.value` to SPEAKER_TO_VIDEO_ROLE in nodes/otr_shot_lock.py.
3. [SPRINT FIX 1 / OPEN QUESTIONS 1] EngineRegistry.assert_usable in nodes/_otr_shared/engine_registry_base.py checks role in getattr(eng, "roles", ()). If the legacy roles whitelist is bypassed, assert_usable will raise EngineUnusable at runtime. Fix: Refactor assert_usable to use role_compat.engine_fits_role(descriptor, role) for compatibility verification.

SHOULD-FIX:
1. [SPRINT FIX 4] The proposed parametrized matrix test asserting eligibility in all 3 slots must be capability-grounded. Some engines are legitimately incompatible with certain slots (e.g. humo requires audio_ref and init_image, which are not supplied by background_abstract). Fix: Assert that eligibility matches role_compat.engine_fits_role, rather than expecting a flat True for all combinations.
2. [OPEN QUESTIONS 3] Aspect-ratio handling is already resolved. In nodes/_otr_video_engines/render_driver.py (line 1125), non-face engines fill the landscape canvas while audio_driven_face keeps its portrait pillarbox. Fix: No new aspect handling code is needed.

OPTIONAL / NICE-TO-HAVE:
1. Keep the legacy per-engine roles attribute solely for UI sorting/ordering of the combo boxes in OTR_VideoDirector, but completely decouple it from runtime eligibility and validation.

CUT THESE (scope / over-engineering):
1. [SPRINT FIX 1] Deprecating default_roles: Decoupling eligibility check from roles is sufficient. [ASSUMPTION] default_roles remains necessary for UI sorting in OTR_VideoDirector so that standard defaults (like humo for announcer and still_motion for b-roll) sort first.
