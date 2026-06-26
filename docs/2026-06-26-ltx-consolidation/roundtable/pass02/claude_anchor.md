CLAUDE ANCHOR REVIEW -- R2 (coding plan / implementability)
Grounded against render_driver.py (still-routing 797-970, _uses_ambient_master_audio
723-730, scene-prompt 1089-1190), eng_ltx_av.py, registry.py, cheap_families.py,
eng_visualizer.py.

VERDICT: yes-with-fixes. The role-driven design is implementable and the pieces
exist, but three implementation details are under-specified and one ordering risk
remains.

MUST-FIX BEFORE BUILD:
1. [Part B0] The classifier must reuse the EXISTING `_is_char_face_beat` logic, not
   invent a parallel one. render_driver already computes `_is_char_face_beat =
   (_fam == "audio_driven_face" and _shot_role not in (announcer_visual,
   music_visual))` at ~1100 for the prompt path. CONFIRMED. Defect: that definition
   keys on family==audio_driven_face, which ltx_audio_in is NOT. Fix: BROADEN the
   single shared classifier to `role=="character_video" OR (char_id present and role
   not in announcer/music) OR family==audio_driven_face-non-open`, define it ONCE
   (a module helper), and replace BOTH the ~1100 use and the new still/audio uses
   with it. One definition, three call sites -- otherwise the axes drift.
2. [Part B1] The still-route rewrite must be ADDITIVE-then-subtractive to stay
   green: the three existing branches (_SCENE_INIT_FAMILIES@842, flux_still/
   flat_still@869, ltx_video@906) each set init_image/init_source with bespoke LOUD
   logs and the `_i2v_still_missing` stamp consumed downstream. Replacing them with
   one helper risks dropping the stamp or a log a test asserts. Fix: implement
   `still_route()` as pure, then route ALL FOUR engine classes through it while
   PRESERVING the exact init_source strings ("scene_still"/"missing_scene_still")
   and the `_i2v_still_missing` flag the trace + tests key on. Grep tests for
   init_source / missing_scene_still / _i2v_still_missing before deleting branches.
3. [Part B1 / required-still] ltx_audio_in failing LOUD "before GPU" needs a
   concrete site. Today render_clip raises GraphExecutionError mid-build. The plan
   wants a pre-GPU input-gap. CONFIRMED there is `_assert_family_inputs_satisfiable`
   (referenced @956) that already FamilyInputGap-crashes audio_conditioned_video
   without audio_ref. Fix: route the missing-required-still through that SAME
   assertion path (extend it to required init_image for the audio-in lane) so the
   loud failure is at the existing gap-check, not a new ad-hoc raise.

SHOULD-FIX:
1. [Part C.3] _LTX_OPEN_ENGINES + _LTX_OPEN_ROLES feed check_ltx_open_health; after
   swapping ltx_av_music->ltx_audio_in confirm the health check still treats a
   music/announcer open rendered on ltx_audio_in as HEALTHY (not a soft-open
   degrade). One-line set edit + the test_ltx_open_health update.
2. [Part D] The routing-matrix test should call the REAL build_request_from_shot
   (not a reimplementation) with synthetic ShotRows + a minimal ledger, asserting
   req init_image/init_source/text_prompt source + the audio path per row. Reuse
   the fixtures in test_ltx_av_driver_wiring + test_video_render_driver_perbeat_audio.

[ASSUMPTION] `_assert_family_inputs_satisfiable` can be extended to the required
init_image case without disturbing the audio_ref gap it already guards -- verify by
reading it before relying on it.
