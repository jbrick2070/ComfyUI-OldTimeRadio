# ALL-ENGINES x ALL-SLOTS -- r2-HARDENED CODING PLAN (Claude-synthesized, grounded)

Panel r2 (Codex + Antigravity + Claude anchor), grounded. Build as ordered chunks; suite+BugBible+B7
green AND push per chunk. The capability rule (role_compat) is the ONE eligibility source; `roles`
becomes UI-sort metadata only.

## C1 -- BLACK FIX (D2), capability-DERIVED still binding
- cheap_families.py: add `accepts_still = True` to **StillPanFamily** (the keeper / IMAGE_CARRIER).
  still_motion + station_card are RETIREMENT candidates (clean-break) -- do NOT add to them here;
  their fate is the retirement chunk, not this fix.
- render_driver.py: the scene-still binding branch (~928) currently hardcodes
  `("still_pan","still_flat","ltx_audio_in")` and EXCLUDES station_card (:910-918). Make the binding
  CAPABILITY-DERIVED: bind the beat's scene still for any engine where `engine_consumes_still(eng)`
  (accepts_still) is True (covers still_pan now + any future still-consumer), instead of a hardcoded
  tuple. A still-ignoring engine (visualizer, accepts_still=False) is untouched.
- TESTS: `engine_consumes_still` True for still_pan; AND an integration test asserting
  `init_source == "scene_still"` for each still-consuming engine (proves the still is actually USED,
  not just minted -- Codex).

## C2 -- KILL THE DRIFT (D1), FAIL-SOFT capability routing
- `engine_registry_base.engines_for_role` (:169-185) + `assert_usable` (:193-218): try
  `role_compat.engine_fits_role(descriptor, role)` (build a `{engine_id, roles, required_inputs}`
  descriptor; preserve default_roles SORT; wrap `RoleCompatError` -> `EngineUnusable`). FAIL-SOFT
  (Antigravity): if the engine declares NO `required_inputs`, OR the role is not a known
  `role_compat` role (RoleCompatError), FALL BACK to the legacy `roles` whitelist. This keeps the
  IMAGE registry (non-video roles) + stub adapters + non-canonical-role tests working, and applies
  capability only where it is well-defined (the 5 canonical video roles + real adapters).
- `roles` is now UI-sort metadata only (decoupled from eligibility); `default_roles` kept for sort.
- TESTS: update the membership-rejection assertions to capability reasons (e.g. test_video_motion
  ~77-82: wan_i2v excluded from background_abstract because `init_image` unavailable, NOT "role").

## C3 -- scene_broll ROUTING (D4)
- `otr_shot_lock.SPEAKER_TO_VIDEO_ROLE` (:55): add `"scene": Role.SCENE_BROLL.value` (VERIFY the
  b-roll beat's speaker_role string at build). Makes `scene_broll_video_model` reachable.

## C4 -- MATRIX TEST (F6, guards C2)
- Parametrized over `vreg.all_engine_names()` x the 5 roles: assert eligibility ==
  `role_compat.engine_fits_role` (capability-grounded -- NOT flat True; background_abstract/scene_broll
  legit exclusions are EXPECTED), the director accepts an eligible pick, and (post-C5) the canonical
  soak fills it. Add a shared `descriptor_for_engine(engine_id)` helper (registry) so director /
  registry / sweep / tests stop rebuilding the descriptor by hand.

## C5 -- REBUILD THE SOAK ON THE CANONICAL JSON (D3) + CONTENT ORACLE (F5)
- The soak LOADS `workflows/otr_scifi_16gb_full.json` and sets each per-slot engine pick via the ONE
  production applier `otr_api.apply_profile_to_workflow` -> `_otr_workflow_apply.apply_profile`
  (patches by node TYPE via `config/profiles/widget_mapping.json`; do NOT hardcode "node-87" -- the
  applier rejects raw node ids). Enumerate `all_engine_names()` x the 5 roles (NOT the stale 3-slot
  `engines_for_role`). SUBSUMES `build_profile`; merge with COMBO_SOAK_CONVERGED_PLAN.md (bake
  story+audio, vary picks, stills+video+upscale->obs).
- CONTENT ORACLE (per beat), SPLIT BY CAPABILITY:
  - ledger-row invariant: assert the `scene_*` row keyed by `still_pool_key`/`beat_id` exists BEFORE
    render ONLY when `engine_consumes_still(eng)` is True (visualizer etc. exempt).
  - non-floor luma (ffmpeg signalstats YAVG > floor threshold) for EVERY rendered beat.
  - temporal variance (motion) ONLY for motion engines (still_pan/still_motion/wan/ltx/humo...);
    EXEMPT static stills (still_flat) -- a flat hold is CORRECT, not a failure (Codex+Antigravity).
- KEEP OFFLINE CI: retain a mock-based unit test that loads the canonical JSON, mocks the node
  executions, and asserts the applier -> shotlock -> dispatcher paths produce the expected per-role
  engine + still decisions (the live HTTP soak alone loses offline coverage -- Antigravity).

## CHUNK ORDER
C1 (black, quick win, unblocks image legs) -> C2 (drift kill) -> C3 (scene_broll) -> C4 (matrix test,
guards C2) -> C5 (soak rebuild + oracle, the big one). C1-C4 land + QA before C5.

## NON-GOALS / CUTS
No aspect redesign (render_driver:1125 already handles landscape/portrait). No unconditional
frame-variance (wrong for still_flat). Don't deprecate default_roles. registry IS the menu. No
workflow-JSON change for C1-C4 (C5 only loads/patches the canonical JSON via the applier, no schema
edit). still_motion/station_card accepts_still + retirement = the separate clean-break chunk.

## VERIFY-AT-BUILD (-> r3)
SPEAKER_TO_VIDEO_ROLE key (C3); the fail-soft branch's exact condition (C2); the apply_profile per-slot
seam for the 5 roles (C5); the full consumer list of engines_for_role/the roles gate.
