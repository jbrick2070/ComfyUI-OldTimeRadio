# PLAN: delete the opt-in flag entirely (video + image engines)

## End goal (operator 2026-06-29)
No opt-in, no switches, no behind-the-scenes "promotions". A model in a dropdown
is there BECAUSE it is good + tested; untested models are removed from the
registry before release. Selecting a model in OTR_VideoDirector (or the image
slots) just renders it -- the only runtime gate is "are its model files on disk".
There must be NO `OTR_ENABLE_*` env, no `requires_flag`, no hidden enable step.

## Current state (grounded)
There are TWO flag gates, both keyed on `requires_flag` == `os.getenv(flag)=="1"`:
1. Registry-level: `nodes/_otr_shared/engine_registry_base.py` `assert_usable`
   L222-228 raises `GATED_BY_FLAG` for a non-default engine whose `requires_flag`
   is unset. This base is shared by the VIDEO + IMAGE registries. AUDIO keeps its
   OWN frozen copy (docstring L23-24) -- audio is OUT OF SCOPE (frozen spine).
2. Adapter-level: each video/image engine's own `assert_usable` re-checks the
   flag THEN checkpoint-on-disk (e.g. `eng_humo.py` L163-173: flag first, ckpt
   second). The render path calls the ADAPTER `assert_usable` (this is what fired
   `gated_by_flag` on the live b000 HuMo bookend today).

The interim "drive the flag from the dropdown selection" change (commit 1c73aec +
`apply_selection_enable_set`/`_restore_enable_set` + `tests/test_video_selection_
enable_set.py`) is SUPERSEDED by this plan and will be REVERTED in step 0.

## Scope (video + image only; audio frozen + untouched)
Engines with a flag gate to clear (adapter `assert_usable` + registry row):
- video: humo, character_3d, ltx_av, wan_i2v, wan_ti2v, ltx_video, still_parallax,
  visualizer, mesh_stage, triposr, cheap_families.
- image: flux2_klein, z_image_turbo, flux_gen1, sd35_large, qwen_image,
  lumina_image, hidream_i1.

## The change (proposed)
0. REVERT the interim option-B change (apply_selection_enable_set + restore + its
   test + the run_real_episode try/finally). Back to HEAD pre-1c73aec behaviour.
1. Base: in `engine_registry_base.py` `assert_usable`, DELETE the `GATED_BY_FLAG`
   block (L222-228). The registry gate becomes: registered? serves role? -> ok.
2. Adapters: in each video/image engine `assert_usable`, DELETE the
   `requires_flag` check; KEEP the checkpoint/dep-on-disk check (the real gate)
   and any genuine capability check (e.g. wan_ti2v VAE-2.2, ltx_video BUG-070).
3. Field: remove `requires_flag` from `EngineCore` (base) and from every
   video/image registry row + adapter attribute. (Audio EngineCore is separate;
   leave it.)
4. Reason code: `GATED_BY_FLAG` -- remove from the SHARED enum ONLY if audio does
   not import the shared enum; otherwise keep the enum member (harmless, unused by
   video/image) to avoid touching audio. VERIFY which.
5. Dep-verify harness re-point: `otr_video_dep_pilot.py`, `otr_image_dep_pilot.py`,
   `otr_video_gpu_smoke.py`, `otr_coverage_sweep.py` currently use `requires_flag`
   to know which engines to GPU-verify / which env to set. Re-point them to derive
   "needs GPU verification" from the CAPABILITIES table (`vram_class != cpu` or a
   non-empty `model_requirements`) instead of the flag. This is the one real
   ripple -- get it right so the soak/dep-pilot keep working.
6. Curation = the dropdown: `OTR_VideoDirector._video_model_combo` /
   `_image_model_combo` already list ONLY `validated_engine_names()`. That stays
   the single "is this model good+tested" gate. Confirm every engine we want
   selectable is in the validated set, and any not-good engine is REMOVED from it
   (operator removes pre-release).
7. Tests: update the ~20 video/image tests that assert `gated_by_flag` /
   `requires_flag` to the new contract (a registered, role-fitting engine is
   usable; the only refusal left is missing-model / incompatible-role).

## Invariants preserved
- Files-on-disk is still enforced (missing ckpt -> LOUD MISSING_MODEL, no silent
  swap; no-fallback render still RAISES on a real failure).
- AUDIO spine FROZEN + untouched (separate registry/enum copy).
- No workflow-JSON node/widget change (V-11).
- Determinism, single resident heavy <= 14.5GB, LOUD behaviour all unchanged.

## Open questions for the panel
1. Is deriving "needs GPU dep-verify" from `vram_class`/`model_requirements` a
   sound replacement for `requires_flag` in the dep-pilot + gpu-smoke, or is there
   a hidden coupling (a verify run that EXPECTS the engine gated-off first)?
2. Does the shared `EngineUsabilityReason` enum get imported by the audio
   registry? If yes, keep `GATED_BY_FLAG` as a dead member; if no, delete it.
3. Any consumer that treats "not in default_roles + no flag" as "available
   everywhere" in a way that floods a role with engines it should not serve?
   (role_compat still filters per-role -- confirm it is independent of the flag.)
4. After deletion, is there ANY remaining hidden enable/promotion path (env,
   profile, force-map, OTR_LSYNC_BASE_ENGINE) that could still beat the dropdown?
5. Sequencing: base + adapters + field + harness + tests -- the safest commit
   chunking so the suite stays green at each step.
