# ALL-ENGINES x ALL-SLOTS -- SPRINT-READY PLAN (kibitz r1-r4 CONVERGED, 2026-06-30)

Hardened across the full kibitz arc (Codex + Antigravity + Claude code-grounded anchor; r1 arc -> r2
coding -> r3 wiring -> r4 convergence). Every claim grounded against the real repo. Raw rounds +
judgments: `kibitz-runs/2026-06-30-slot-audit/`. This is the coder-window contract.

## ARCHITECTURE INVARIANT
Any video model INCLUDING STILLS is usable CORRECTLY in any of the 3 user slots (announcer / music /
other-beats). No preferred path. "Correctly" = ELIGIBLE (capability) AND RENDERS REAL CONTENT (the
right still/video, never a black floor; static is OK for static engines, not a failure). CAPABILITY
(`role_compat`) is the ONE eligibility rule; `roles` becomes UI-sort metadata only.

## ROOT-CAUSE FINDINGS (grounded)
- D1 DRIFT: production validates by CAPABILITY (`role_compat.engine_fits_role`); the registry's
  `engines_for_role` + `assert_usable` (`engine_registry_base`) still gate on the stale per-engine
  `roles` whitelist. PROVEN: ltx_video fits character_video by capability=TRUE, whitelist=FALSE -> the
  soak (which uses the whitelist) filled `still_flat` on character ("stills not video").
- D2 BLACK: only `still_flat` has `accepts_still=True`; `still_pan` lacks it -> the dispatcher's
  `engine_consumes_still` returns False -> SKIPS minting its scene still -> dark-floor BLACK (the image
  legs). (still_flat showed the image; still_pan didn't -- exact match to QA.)
- D3 SOAK NOT CANONICAL: the soak synthesizes a profile + uses the stale gate instead of loading
  `workflows/otr_scifi_16gb_full.json` through the real director path (CLAUDE.md S0 violation; the root
  of the drift's visibility).
- D4 scene_broll DEAD: `SPEAKER_TO_VIDEO_ROLE` has no "scene" key -> b-roll beats fall to
  background_abstract -> `scene_broll_video_model` never exercised.

## SPRINT CHUNKS (each green: suite+BugBible+B7; push per chunk)

### C0 -- ENGINE-SET DECISION (operator gate, do FIRST)
The all-engines soak (C5) enumerates `all_engine_names()`. Decide concretely (NOT "candidate"):
- KEEP + make-correct the 3 still options: `still_pan`, `still_flat`, `still_motion` (pan / flat /
  ken-burns) -- all 3 are valid user choices in every slot.
- RETIRE (deregister + capability row + dropdown + tests): `station_card` (broken black card, excluded
  from scene-still binding) and `abstract` (redundant with visualizer). [OPERATOR CONFIRM: retire these
  two now, or keep+fix them too?] -- this is the only open decision; everything else is determined.

### C1 -- BLACK FIX (mint, don't generalize the bind)
- cheap_families.py: add `accepts_still = True` to `StillPanFamily` AND `StillMotionFamily` (so the
  dispatcher MINTS their scene still). still_pan binds via the explicit scene-still branch
  (`render_driver.py:928`); still_motion (family `static_motion`) binds via `_SCENE_INIT_FAMILIES`
  (`render_driver.py:476,895-903`) -- both already bind once minted.
- DO NOT make the bind "capability-derived via `engine_consumes_still`": it is True for HuMo
  (audio_driven_face, portrait init) and would wire b-roll stills into faces. Keep the EXPLICIT
  scene-still bind set {still_pan, still_flat, ltx_audio_in} + the separate LTX-I2V branch (:959);
  exclude audio_driven_face.
- TESTS: `engine_consumes_still` True for still_pan + still_motion; integration `init_source ==
  "scene_still"` for the explicit scene-still bind set.

### C2 -- KILL THE DRIFT (capability everywhere, fail-soft, VIDEO-scoped)
- Override `engines_for_role` + `assert_usable` in the VIDEO registry subclass
  (`nodes/_otr_video_engines/registry.py`) -- NOT the shared `engine_registry_base` (it serves image +
  audio). Delegate to `role_compat.engine_fits_role` via a shared `descriptor_for_engine(engine_id)`.
- FAIL-SOFT to the legacy `roles` ONLY when `required_inputs` is MISSING/None, OR the role is unknown
  to role_compat. NEVER for `required_inputs=()` (that is a valid capability fitting all roles --
  AbstractFamily). `engines_for_role`: unknown role -> legacy filter; `assert_usable`: wrap
  `RoleCompatError`/unknown -> `EngineUnusable` (preserve the public contract). default_roles SORT kept.
- TESTS: update membership-rejection assertions to capability reasons (test_video_motion etc.);
  registry-is-the-menu guard green.

### C3 -- scene_broll ROUTING
- VERIFY the actual b-roll `speaker_role` token (the writer/schema prompt that constrains
  speaker_role); map it in `SPEAKER_TO_VIDEO_ROLE` (otr_shot_lock.py:55, ALSO imported by
  otr_meta_brief_image_prompt.py:290) so it resolves to `scene_broll`. Test the token -> scene_broll.
  Harmless if no scene beats exist yet (makes the slot routable).

### C4 -- MATRIX TEST (pure eligibility contract)
- Parametrized `vreg.all_engine_names()` x 5 roles: eligibility == `role_compat.engine_fits_role`
  (capability-grounded, NOT flat True; background_abstract/scene_broll legit exclusions EXPECTED). Use
  the shared `descriptor_for_engine`. Do NOT duplicate canonical-render coverage here.

### C5 -- SOAK ON THE CANONICAL JSON + ORACLE
- CONVERGE all 3 stale soak entry points to ONE canonical-JSON path; retire/delegate the others:
  `scripts/_otr_cov_runner.py:50-55`, `scripts/otr_coverage_sweep.py:86-91`, `_otr_combo_soak.py:67-89`.
- LOAD otr_scifi_16gb_full.json; set ALL FIVE role keys (announcer/music/character/scene_broll/
  background_abstract) via `apply_profile_to_workflow` (node-TYPE via widget_mapping.json; no node ids).
  NAMED baselines for non-under-test roles: `still_flat` for video roles + `flux_gen1` for image roles
  (override only if the tested role needs a different carrier); the profile sets all 5 keys, NEVER the
  legacy other-beats fallback. Enumerate `all_engine_names()` x role_compat.
- ORACLE (per-beat, on the MANIFEST `clips[].path` -- NOT the obs final): ffmpeg `signalstats` YAVG >
  a NAMED luma floor (e.g. > 16/255) = non-dark, for every beat; temporal motion (frame-diff or
  freezedetect over a defined window) ONLY for motion engines; EXEMPT static stills (still_flat) +
  still-ignoring engines (visualizer). Ledger-row invariant (scene_* keyed by still_pool_key/beat_id
  before render) ONLY when `engine_consumes_still`. Reserve the obs final for a publish smoke. Keep an
  OFFLINE mock soak test (load canonical JSON, mock node exec, assert applier->shotlock->dispatcher).

## CHUNK ORDER
C0 (operator decision) -> C1 (black, quick) -> C2 (drift) -> C3 (scene_broll) -> C4 (matrix) -> C5
(soak rebuild + oracle). C1-C4 land + QA before the big C5.

## NON-GOALS
No aspect redesign (render_driver:1125 already landscape/portrait). No default_roles deprecation.
registry IS the menu. No workflow-JSON edit for C1-C4 (C5 only loads/patches the canonical JSON via the
applier). audio byte-identical stays an existing regression gate.

## VERIFY-AT-BUILD CHECKLIST (r4)
1. C3: the real b-roll speaker_role token resolves to scene_broll (+ image-prompt import path).
2. C2: required_inputs=() does NOT fall back to legacy roles (AbstractFamily fits all 5 by capability).
3. C5: all 3 stale soak entry points retired/delegated; canonical JSON patched via apply_profile
   (widget_mapping node-type, not node ids); the matrix generated from descriptors + engine_fits_role.
4. C5: oracle reads node_episode_manifest.json / clip_manifest_json rows + checks per-beat clip files
   before publish; named luma floor + motion window + static/visualizer exemptions.
5. C1: still_pan + still_motion mint + bind (init_source==scene_still); HuMo NOT wired to scene stills.
