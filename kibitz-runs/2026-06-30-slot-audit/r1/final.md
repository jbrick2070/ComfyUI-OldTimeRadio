# ALL-ENGINES x ALL-SLOTS -- r1-HARDENED AUDIT + SPRINT PLAN (Claude-synthesized, grounded)

Panel r1 (Codex + Antigravity + Claude anchor), all claims grounded vs the real code. Architecture
invariant: any video model incl stills usable CORRECTLY (eligible AND renders real content) in all 3
user slots; no preferred path. Capability is the ONE rule.

## THE FOUR GROUNDED DEFECTS
D1. ELIGIBILITY DRIFT (two rules disagree). `role_compat.engine_fits_role` = capability-only
    (model-agnostic, production). BUT `engine_registry_base.engines_for_role` (:169-177) AND
    `assert_usable` (:214-218) BOTH still gate on the stale per-engine `roles` whitelist. PROVEN:
    ltx_video fits character_video by capability=TRUE but engines_for_role/assert_usable EXCLUDE it.
    The soak `build_profile` uses engines_for_role -> still_flat on character (the QA "stills").
D2. BLACK STILLS -- ROOT CAUSE FOUND (grounded). In `cheap_families.py` ONLY `still_flat` declares
    `accepts_still = True` (:241); `still_pan`, `still_motion`, `station_card` LACK it (and inherit
    nothing -- they are _CheapFamilyBase, not MotionEngineBase). The dispatcher's `engine_consumes_still`
    (:287-307) reads `accepts_still`; `_still_needed_for_role` (:307-339) -> if False the dispatcher
    SKIPS minting the scene still (:437). So still_pan gets NO scene still -> `init_image=""` ->
    dark-floor BLACK. (That is exactly why still_flat showed the image but still_pan image legs were
    black.)
D3. SOAK NOT ON THE CANONICAL JSON (the drift's root + CLAUDE.md S0 violation). The soak synthesizes a
    profile + uses engines_for_role instead of loading `workflows/otr_scifi_16gb_full.json` through the
    real director/role_compat path.
D4. scene_broll SLOT IS DEAD. `otr_shot_lock.SPEAKER_TO_VIDEO_ROLE` (:55) lacks a `"scene"` key, so
    b-roll beats fall to `_DEFAULT_VIDEO_ROLE` (background_abstract) and `scene_broll_video_model` is
    never exercised. (Verify the map keys at build.)

## THE AUDIT MATRIX = CAPABILITY MATRIX (NOT "all TRUE" -- Codex/Antigravity corrected)
Per `role_compat.ROLE_AVAILABLE_INPUTS`: announcer/music/character supply
{text_prompt, init_image, audio_ref, base_clip_ref} -> EVERY engine is eligible in the 3 MAIN slots
(this is where "any model any slot" must hold, and where D1 wrongly excluded engines). scene_broll
supplies {text_prompt, init_image, base_clip_ref} (NO audio_ref); background_abstract supplies ONLY
{text_prompt}. So audio/init-requiring engines (humo* need audio_ref+init_image; visualizer needs
audio_ref; wan_i2v/still_parallax need init_image) are LEGITIMATELY excluded from background_abstract
(and audio engines from scene_broll) -- by CAPABILITY, not whitelist. The audit GENERATES the matrix
from each engine's `required_inputs` (no hand tables).

## SPRINT FIX (chunks; each green+pushed; suite+BugBible+B7)
F1. BLACK FIX (D2): add `accepts_still = True` to StillPanFamily, StillMotionFamily, StationCardFamily
    (cheap_families.py). Dispatcher then mints their scene still -> they render the image, never the
    dark floor. (still_flat already correct.) Add a test: `engine_consumes_still` True for every
    still-carrier engine.
F2. KILL THE DRIFT AT SOURCE (D1): route `engine_registry_base.engines_for_role` AND `assert_usable`
    through `role_compat.engine_fits_role` (capability). DECOUPLE the per-engine `roles` from
    eligibility -- keep `roles`/`default_roles` ONLY for combo UI sorting (Antigravity). Guard test:
    no video eligibility path reads the stale `roles` gate; eligibility == engine_fits_role everywhere.
F3. scene_broll ROUTING (D4): add `"scene": Role.SCENE_BROLL.value` to
    `otr_shot_lock.SPEAKER_TO_VIDEO_ROLE` so the scene_broll slot is reachable.
F4. REBUILD THE SOAK ON THE CANONICAL JSON (D3): the soak LOADS otr_scifi_16gb_full.json, sets each
    node-87 OTR_VideoDirector slot pick via the REAL director/role_compat/profile-applier path, runs
    the real graph (per-role over all 5 roles, summarized to the 3 UI slots). SUBSUMES build_profile +
    engines_for_role. Merge with `docs/2026-06-29-coverage-soak/COMBO_SOAK_CONVERGED_PLAN.md`
    (bake story+audio, vary node-87 picks, stills+video+upscale->obs).
F5. CONTENT ORACLE ("used CORRECTLY"): acceptance per beat = (a) the required `scene_*` ledger row
    exists keyed by the render lookup (`still_pool_key`/`beat_id`) BEFORE OTR_VideoRenderBatch (the D2
    invariant); (b) the emitted clip is NOT the dark floor (sample luma > floor) AND has frame variance
    (motion, not a frozen still). Add as the soak's per-leg gate.
F6. MATRIX TEST: parametrized over every registered video engine x 5 roles, asserting eligibility ==
    `role_compat.engine_fits_role` (capability-grounded -- NOT flat True; the background_abstract/
    scene_broll legit exclusions are expected), the director accepts an eligible pick, and (post-F4)
    the canonical-JSON soak fills it (no silent still/floor swap).

## CUTS / NON-GOALS (panel consensus)
Aspect handling already exists (render_driver:1125 landscape-vs-portrait); no aspect redesign. Do NOT
deprecate `default_roles` (keep for UI sort). "audio byte-identical" stays an existing regression gate,
not this sprint's defect. No manual roles-gap tables (automated drift report instead). registry IS the
menu (combo unchanged); workflow-JSON only if a widget/schema actually changes (F1-F6 don't).

## VERIFY-AT-BUILD (-> r2/r3)
Confirm SPEAKER_TO_VIDEO_ROLE keys (D4); enumerate EVERY consumer of engines_for_role/the roles gate to
migrate (image director, ShotLock, capability_profiles, tests); the canonical-JSON soak seam (node-87
pick via director vs profile-applier); the exact per-engine required_inputs for the matrix.
