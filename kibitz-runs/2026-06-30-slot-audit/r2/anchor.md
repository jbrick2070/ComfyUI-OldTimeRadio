# Claude anchor review -- r2 (coding plan / implementability)

VERDICT: F1-F6 implementable as ordered chunks; 4 must-fix implementability points.

## CHUNK ORDER (each green+pushed; suite+BugBible+B7)
C1 = F1 black fix: cheap_families.py add `accepts_still = True` to StillPanFamily / StillMotionFamily /
   StationCardFamily + a test that `engine_consumes_still` is True for each. (Quick win; unblocks the
   image legs.)
C2 = F2 drift kill: video registry `engines_for_role` + `assert_usable` -> `role_compat.engine_fits_role`.
C3 = F3 scene_broll: SPEAKER_TO_VIDEO_ROLE += "scene".
C4 = F6 matrix test (guards C2): parametrized eligibility == engine_fits_role.
C5 = F4 soak on canonical JSON (the big one) + F5 content oracle.

## MUST-FIX (implementability)
M1. F2 SCOPE -- `engine_registry_base` is the SHARED base for BOTH the video AND image registries.
    `role_compat` defines the FIVE VIDEO roles only. Applying it in the shared base would break the
    image registry (different roles). FIX: override `engines_for_role`/`assert_usable` in the VIDEO
    registry subclass (registry.py) to delegate to role_compat; leave the base generic (or inject a
    per-registry compat fn). GROUND: confirm the image registry subclasses the same base + its role set.
M2. F4 SEAM -- set the node-87 picks on the canonical JSON via the ONE production applier
    (`otr_api.apply_profile_to_workflow` -> `_otr_workflow_apply.apply_profile`), NOT ad-hoc
    patch_widget. Build a profile carrying the per-slot engine pick + apply to the loaded
    otr_scifi_16gb_full.json, submit via /prompt. This is the exact production resolution (role_compat
    runs inside the director node) -> soak eligibility == production BY CONSTRUCTION.
M3. F1 SIDE-EFFECT -- adding accepts_still to station_card/still_motion makes the dispatcher MINT a
    scene still for those beats too. Desired (they show the image), cheap (image-gen already runs), no
    VRAM surprise. Confirm no test asserts "still NOT minted for station_card" that would flip.
M4. F2 CONSUMER MIGRATION -- enumerate + migrate EVERY caller of engines_for_role / the roles gate:
    the soak build_profile (subsumed by C5), the director combos (already role_compat for validation;
    confirm), image director, ShotLock assert, capability_profiles, and the tests asserting the old
    whitelist exclusions (test_capability_routing etc. -- update to capability-expected cells).

## SHOULD-FIX
S1. C5 content oracle: luma via ffmpeg signalstats (YAVG > floor threshold) + frame variance via a
    2-frame diff or freezedetect (motion). Reuse the S-A freezedetect parser already shipped.
S2. Keep C1-C4 SMALL + independently green so the black fix + drift kill land before the big C5 soak
    rebuild (operator can QA the image legs after C1+C4 without waiting for the soak rebuild).

## VERIFY-AT-BUILD (-> r3)
The image-registry base sharing (M1); the exact apply_profile seam for node-87 per-slot picks (M2);
the speaker_role "scene" value (C3); which tests encode the old whitelist exclusions (M4).
