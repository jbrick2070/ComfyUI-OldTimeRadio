# ALL-ENGINES x ALL-SLOTS -- r3-HARDENED (WIRING) PLAN (Claude-synthesized, grounded)

Panel r3 (Codex + Claude anchor; agy r3 pending, non-blocking -- corrections grounded). Wiring locked.

## C1 -- BLACK FIX (CORRECTED: mint only; do NOT generalize the binding)
- cheap_families.py: add `accepts_still = True` to **StillPanFamily** ONLY. That makes the DISPATCHER
  mint the scene still for still_pan (`engine_consumes_still` reads accepts_still). still_pan is ALREADY
  in the render-driver scene-still BINDING branch (`render_driver.py:928` tuple
  `("still_pan","still_flat","ltx_audio_in")`), so once minted it binds correctly.
- DO NOT make the scene-still binding "capability-derived via engine_consumes_still" (r2 error, Codex
  M1): `engine_consumes_still` is True for HuMo (`accepts_still=True`, audio_driven_face) whose
  `init_image` is a PORTRAIT, not a scene still -- generalizing would wire b-roll stills into HuMo.
  Keep the explicit scene-still engine set (+ the separate LTX-I2V branch at :959); exclude
  audio_driven_face. A FUTURE non-still_pan scene-still engine gets added to that explicit set (or a
  dedicated `scene_still_init=True` flag), NOT via accepts_still.
- station_card: still NOT in the binding branch (:910-918) + retirement candidate -> out of scope here.
- TESTS: `engine_consumes_still(still_pan)` True; integration: `init_source=="scene_still"` for the
  still-consuming bind set (proves the minted still is USED, not just minted).

## C2 -- KILL THE DRIFT (CORRECTED fail-soft; VIDEO-scoped)
- LOCUS: override `engines_for_role`/`assert_usable` in the VIDEO registry subclass
  (`nodes/_otr_video_engines/registry.py`), NOT the shared `engine_registry_base` (which serves IMAGE
  + AUDIO registries with non-role_compat roles).
- Use `role_compat.engine_fits_role(descriptor, role)` (capability). FAIL-SOFT fall back to the legacy
  `roles` whitelist ONLY when `required_inputs` is MISSING/None OR the role is unknown to role_compat
  -- NEVER for `required_inputs=()` (Codex M2): `()` is a VALID capability that fits every role
  (e.g. AbstractFamily required_inputs=() must become eligible everywhere by capability, not be
  re-restricted to its legacy 2 roles). Wrap `RoleCompatError` -> `EngineUnusable`; preserve
  default_roles SORT. `roles` -> UI-sort metadata only.
- TESTS: update membership-rejection assertions to capability reasons (test_video_motion etc.);
  registry-is-the-menu guard stays green.

## C3 -- scene_broll ROUTING (verify the token)
- VERIFY the actual `speaker_role` token b-roll/scene beats emit (the writer/sequencer); the
  SPEAKER_TO_VIDEO_ROLE map (otr_shot_lock.py:55, ALSO imported by otr_meta_brief_image_prompt.py:290)
  has no "scene" -> unmapped falls to background_abstract. Map the verified token (or emit "scene") so
  `scene_broll_video_model` is reachable; add a test that the token -> role=="scene_broll". Harmless if
  no scene beats exist yet (makes the slot routable for when they do).

## C4 -- MATRIX TEST (pure eligibility contract)
- Parametrized `vreg.all_engine_names()` x 5 roles: eligibility == `role_compat.engine_fits_role`
  (capability-grounded, NOT flat True). Shared `descriptor_for_engine(engine_id)` helper in the
  registry layer (used by director + registry + this test). Do NOT duplicate canonical-render coverage
  here (Codex CUT) -- C4 is the contract; C5 owns render coverage.

## C5 -- SOAK ON THE CANONICAL JSON + ORACLE
- CONVERGE ALL THREE soak entry points (Codex M3): `scripts/_otr_cov_runner.py`,
  `scripts/otr_coverage_sweep.py`, `scripts/_otr_combo_soak.py` -- ONE canonical-JSON soak path; the
  others retire or delegate (do not leave a 2nd/3rd runner on the stale `engines_for_role`/3-slot
  model).
- The soak LOADS otr_scifi_16gb_full.json + sets ALL FIVE role keys (`announcer_visual`,
  `music_visual`, `character_visual`, `scene_broll_visual`, `background_abstract_visual`) via
  `apply_profile_to_workflow` (node-TYPE via widget_mapping.json; no node-id hardcode). Fill
  non-under-test roles with KNOWN-COMPATIBLE baselines so a fallback to `other_beats_video_model`
  cannot hide a broken slot. Enumerate `all_engine_names()` x role_compat (not the stale gate).
- ORACLE -- SPECIFY THE INTERFACE before building (Codex SHOULD): clip source = the per-beat clip /
  the obs final; ffmpeg `signalstats` YAVG > floor threshold (non-dark) for every beat; temporal
  variance (e.g. frame-diff / freezedetect window) ONLY for motion engines; EXEMPT static stills
  (still_flat) + still-ignoring engines (visualizer). Ledger-row invariant (scene_* keyed by
  still_pool_key/beat_id before render) ONLY when `engine_consumes_still`. Keep an OFFLINE mock soak
  test (load canonical JSON, mock node exec, assert applier->shotlock->dispatcher decisions).

## NON-GOALS / OPTIONAL
No aspect redesign (render_driver:1125). No default_roles deprecation. OPTIONAL: a workflow-audit test
that every widget_mapping.json key lands on a real OTR_VideoDirector widget in the canonical JSON.

## CONVERGENCE STATE
r1 (defects) -> r2 (coding) -> r3 (wiring) all grounded + corrected (HuMo scene-still mis-bind; ()-
capability; 3-soak convergence; video-scoped fail-soft). r4 = confirm no new must-fix; expected
BUILD-READY.
