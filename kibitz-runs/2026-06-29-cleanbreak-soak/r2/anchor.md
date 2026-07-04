# Claude anchor review -- r2 (coding plan / implementability)

VERDICT: implementable as 5 ordered chunks; 3 must-fix sequencing/verification gaps.

## CHUNK ORDER (each = its own green+pushed commit; suite+BugBible+B7 after each)
C1. Fallback rip-out: render_driver.py (delete make_fallback_of/FLOOR_NAMES/UNIVERSAL_FLOOR/
    SYNTH_FALLBACKS/FamilyInputGap+its classify_failure case; prune ENGINE_FAMILY drop soak_oom_3d;
    strip the `fallback_of` param from run_episode/run_real_episode/render_single) + every engine's
    fallback_engine -> None + the A3 test migration/deletes. KEEP engine_family/classify_failure/
    RenderError.
C2. Gut the soak: remove build_soak_fixture/run_gpu_soak/assert_soak_ok/_PROFILES/_CHAR3D/OOM_ENGINES/
    EXPECTED_OOM_TRAIL/OomSignal/SoakError/RenderFloorError; OTR_VideoRenderBatch mode combo
    ["soak","single","episode"]->["single","episode"] default "episode" + node-92 JSON rebaseline +
    its soak tests.
C3. Retire abstract + station_card (registry/cheap_families/dropdown-options-JSON/tests); still_motion
    DEFERRED behind the operator flag.
C4. allow_auto_fallback clean-delete + VideoDirector/schemas + workflow-JSON widget-vector rebaseline.
C5. Combo soak: extend otr_visual_smoke (--start-boundary image_gen, keep {91,92,composite,mux}, bake
    script/policy/prompts/master, clear stills) + _otr_cov_runner bake-and-replay + B4 acceptance +
    offline unit tests. (Live fixture needs the reference run -> the GPU batch.)

## MUST-FIX
M1. PRODUCTION-CALLER SWEEP before C1 (CONFIRMED needed). Grep PRODUCTION (not tests) for
    make_fallback_of / SYNTH_FALLBACKS / FLOOR_NAMES / UNIVERSAL_FLOOR / run_episode(fallback_of=) /
    run_gpu_soak / mode=="soak". A hidden non-test/non-soak consumer (a script, the watchdog, a CI
    gate, OTR_VideoRenderBatch.render soak branch) changes C1/C2 scope. Do this FIRST.
M2. C2 node-92 mode rebaseline: the saved node-92 mode VALUE is already "episode" (safe), but dropping
    the "soak" OPTION + changing the default reshapes the COMBO. Re-validate the node-92
    widgets_values vector via otr_api + OTR_WorkflowValidator in the SAME commit; the mode widget is
    positional (BUG-LOCAL-097) -- confirm no later widget shifts.
M3. C5 is GATED on the live reference run (node_episode_input.json exists only post-render). The
    OFFLINE C5 (prune-shape / stills-clear / acceptance-verdict unit tests) lands now; the LIVE combo
    soak runs in the GPU batch. Don't block C1-C4 on C5.

## SHOULD-FIX
S1. C1 fallback_engine removal: some engines may READ their own fallback_engine in assert_usable
    messaging -- grep each before nulling so a log line doesn't NameError.
S2. C3 retirement re-baseline: the director combos are registry-DERIVED (_video_model_combo), so
    options auto-drop, but the SAVED workflow node-87/director option arrays + any test pinning the
    option SET must update. Verify which test pins the set.
S3. Keep each chunk SMALL enough that the suite delta is reviewable; C1's test migration is the
    largest -- split the test deletes from the code if the diff balloons.

## UNVERIFIABLE (-> r3 wiring)
The exact node ids + forceInput wiring for the image_gen prune boundary; the stills-clear mechanism
inside the baked ledger vs the dispatcher cache.
