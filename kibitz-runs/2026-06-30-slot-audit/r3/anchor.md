# Claude anchor review -- r3 (wiring / integration) -- GROUNDED

VERDICT: wiring is sound; 3 integration refinements + 1 confirm. Grounded vs the real code.

## CONFIRMED (grounded)
- D4: `otr_shot_lock.SPEAKER_TO_VIDEO_ROLE` (:55-64) keys = announcer / music{,_open,_close,_inter} /
  character / char_voice / dialogue; NO "scene". `_video_role_for_line` (:76) -> unmapped ->
  `_DEFAULT_VIDEO_ROLE = background_abstract`. So scene_broll is unreachable via speaker_role.
- `engines_for_role` lives in the SHARED `engine_registry_base` and is consumed by the VIDEO, IMAGE,
  AND AUDIO registries (+ otr_coverage_sweep.py, _otr_cov_runner.py, _otr_enum_engines.py,
  otr_video_probe.py, + ~14 tests). role_compat defines the 5 VIDEO roles only.

## WIRING REFINEMENTS (update r2)
W1. C2 LOCUS -- put the capability+fail-soft override in the VIDEO registry subclass
    (`nodes/_otr_video_engines/registry.py`), NOT the shared `engine_registry_base`. The base serves
    image + audio too (their roles are NOT role_compat roles) -- editing the base risks them. The
    video subclass overrides `engines_for_role`/`assert_usable` to use `role_compat.engine_fits_role`
    with FAIL-SOFT (engine missing `required_inputs` OR non-canonical role -> legacy `roles`), which
    also covers VIDEO stub adapters in tests. Image/audio base unchanged -> zero blast radius.
W2. C5 has TWO soak entry points to converge -- `scripts/_otr_cov_runner.py` AND
    `scripts/otr_coverage_sweep.py` BOTH enumerate via the stale `engines_for_role` over 3 slots.
    The canonical-JSON rebuild must REPLACE/retire BOTH (or one delegates to the other) so there is a
    single soak path; otherwise the drift survives in the second runner.
W3. C3 scope -- adding `"scene"` to SPEAKER_TO_VIDEO_ROLE makes the scene_broll SLOT reachable, but
    only matters if the writer emits a `speaker_role` that should map to scene_broll. VERIFY what
    speaker_role b-roll/scene beats carry (the writer/ledger): if none distinct from background_abstract
    today, C3 makes the slot SELECTABLE+routable for when they do, and is harmless now. Note it; do not
    over-engineer a scene-beat classifier in this sprint.

## CONFIRM (not changes)
- C5 apply seam: `apply_profile_to_workflow` patches OTR_VideoDirector widgets by node TYPE via
  `config/profiles/widget_mapping.json` (the 5 per-role video-model slots) -- the correct production
  seam; no node-id hardcoding. The soak builds a profile `role_overrides[slot]=engine` and applies it
  to the loaded canonical JSON.
- C1 capability-derived still binding + C4 matrix test + C5 oracle-split: unchanged, correct.

## VERIFY-AT-BUILD (-> r4 convergence)
The video-subclass override keeps the registry-is-the-menu guard test green; the two-soak convergence
doesn't strand otr_coverage_sweep; the writer's scene speaker_role (C3) -- a no-op-now-but-correct.
