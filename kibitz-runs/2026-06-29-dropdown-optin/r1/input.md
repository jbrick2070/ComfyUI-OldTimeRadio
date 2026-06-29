# QA: dropdown-driven opt-in for video engines (commit 1c73aec)

## Problem
OTR's opt-in video engines (humo / wan_i2v / wan_ti2v / ltx_av / character_3d /
still_parallax) each declare a runtime `requires_flag` (e.g. `OTR_ENABLE_HUMO`).
`assert_usable` checks that flag FIRST, then checkpoint-on-disk. Until now the
flag had to be set as an environment variable at ComfyUI launch. Result: the
operator selects an engine in the `OTR_VideoDirector` dropdowns, but the render
dies LOUD with `gated_by_flag` unless they also remembered to export the env at
launch. The music-open bookend (`shot_b000_music_open`, Route-A radio-face) is
stamped to `humo`, so EVERY episode dies at beat b000 unless `OTR_ENABLE_HUMO=1`,
regardless of the dropdown pick.

Operator directive: the three video dropdowns must be the ONLY switch -- no
hidden launch-time opt-in. It is acceptable for the dropdown selection to DRIVE
the opt-ins behind the scenes, as long as nothing hidden beats the dropdown.

## The change (root fix, option B)
New helper `apply_selection_enable_set(ledger)` in
`nodes/_otr_video_engines/render_driver.py`, called from `run_real_episode`
immediately AFTER `apply_engine_override(ledger)` and BEFORE `run_episode`.

It collects the distinct `engine_id`s already stamped on `ledger["video"]["shots"]`
(by OTR_VideoDirector -> OTR_ShotLock, post force-override), looks up each
engine's `requires_flag` via the video registry, and sets `os.environ[flag]="1"`
for any that is unset. LOUD log of the derived set. Best-effort; never raises.

## Invariants it must preserve
- dep-on-disk half of `assert_usable` UNTOUCHED -- a missing checkpoint still
  fails LOUD (no silent enable of an uninstalled model).
- The soak / dep-pilot / gpu-smoke harnesses drive `run_episode` DIRECTLY (not
  `run_real_episode`), so they keep their own explicit-flag path -- this avoids
  the "flag-gated == needs-dep-verification" conflation that reverted a prior
  attempt.
- No new workflow-JSON node/widget (V-11); pure code reading existing shot ids.
- Determinism, LOUD fallbacks, single resident heavy <= 14.5GB unaffected.

## Files to read
- `nodes/_otr_video_engines/render_driver.py` -- `apply_selection_enable_set`,
  `run_real_episode`, `run_episode`, `render_shot`, `apply_engine_override`.
- `nodes/_otr_video_engines/eng_humo.py` -- `assert_usable` (flag then ckpt).
- `nodes/_otr_video_engines/registry.py` -- the `requires_flag` field.
- `nodes/otr_video_director.py` -- the three dropdowns -> `video_policy_json`.

## QA questions for the panel
1. Correctness: does deriving from `ledger["video"]["shots"]` after
   `apply_engine_override` reliably cover EVERY engine that will actually render,
   including the `humo` music bookend and any force-mapped engine?
2. Placement: is `run_real_episode` the right chokepoint vs `run_episode`? Does
   any production render path reach engines WITHOUT going through
   `run_real_episode`?
3. Side effects: `os.environ` mutation persists for the server lifetime (never
   unset). Is "once enabled, stays enabled this session" a problem (e.g. a later
   render that deselects the engine)? Should it be scoped per-render instead?
4. Harness safety: confirm the soak/dep-pilot/gpu-smoke truly bypass this (drive
   `run_episode` directly). Any path where this would wrongly auto-enable during
   a dep-verification run and mask a real gating bug?
5. Test gaps: what unit/integration test should lock this contract?
6. Any cleaner seam already in the codebase (capability_profiles enable-set,
   role_slots) that this should reuse instead of setting env directly?
