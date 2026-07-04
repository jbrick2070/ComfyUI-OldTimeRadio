VERDICT: no. Deleting the named ceiling symbols as described leaves real call sites and test/smoke ceiling gates behind, and D8 leaves a live frame-budget clamp unresolved.

MUST-FIX BEFORE BUILD:
1. [D2 / RIP ORDER 4] Incomplete call-site list for removed `motion_common` symbols. `VramPeakProbe` / `assert_peak_within_ceiling` are still used at `nodes/_otr_video_engines/eng_ltx_av.py:979,996`, `nodes/_otr_video_engines/eng_wan_i2v.py:294,319`, and `nodes/_otr_video_engines/eng_wan_ti2v.py:450,486`. Fix: update the plan to remove/replace every probe/assert call, and convert remaining peak values to telemetry-only logging.

2. [D2 + D8] `compute_real_frame_budget` is not telemetry; it gates frame count with `budget_mb = min(free_vram_mb_value, ceiling_mb) * margin` at `nodes/_otr_video_engines/motion_common.py:370-411`. `MotionEngineBase.teardown` also waits below `dynamic_vram_ceiling_mb()` at `nodes/_otr_video_engines/motion_common.py:476-487`. Fix: decide before implementation whether frame budgeting uses live free VRAM only, no clamp, or another non-policy threshold; replace teardown with reclaim/stability telemetry or remove the threshold wait.

3. [D2 / RIP ORDER 5] The smoke/soak scripts still enforce hard ceilings, contradicting “NO ceiling assert.” `scripts/_otr_soak_capstone.py:66,654-659` aborts over `VRAM_CEILING_MB`; `scripts/run_ltx_av_q_bakeoff.py:103,614-616,727` hard-fails / marks over-ceiling. Fix: remove those aborts or explicitly mark the scripts archived/not used by the required live smoke. [ASSUMPTION] These are candidates for the “B7 / live smoke” path.

4. [D2] `wrapper_bridge` has a separate runtime ceiling constant/export at `nodes/_otr_video_engines/wrapper_bridge.py:36-37,642-644`. It is not in the rip list. Fix: either remove `VRAM_CEILING_MB` and its `__all__` export, or explicitly scope it out with proof it is dead.

5. [D4] LFC watchdog removal must include exports and telemetry shape, not just the function body. `_otr_lfc_watchdog.py` exports `VRAM_DEFAULT_CEILING_GB` and `vram_over_ceiling` at `nodes/_otr_lfc_watchdog.py:38-46`, defines them at `:55` and `:226-242`; `_otr_freeze_cascade.py:727-743` stamps `lfc_vram_ceiling_gb` and warns over ceiling. Fix: keep only `vram_at_cascade_entry_gb` telemetry via a direct allocation read, drop the ceiling stamp/warn, and update `__all__`.

6. [RIP ORDER 4] Test update list is incomplete. Removing `assert_vram_within_ceiling` breaks `tests/test_video_motion.py:287-298` and `tests/test_video_motion_common_additive.py:99-104`; `tests/test_clip_fill.py:29` still treats `OTR_VRAM_CEILING_MB` as an env var to clear. Fix: rewrite these as telemetry/no-ceiling tests or delete them in the same chunk.

SHOULD-FIX:
1. [D2 / D3] Validator wording is ambiguous: “rip host-fit tier suggestion” could remove CUDA/platform stamp checks that should remain. `nodes/_otr_workflow_validator.py:291-310` mixes CUDA absence, VRAM budget, and platform suggestions. Fix: state explicitly to delete only the `profile["vram_budget_mb"]` branch and env export at `:297-333`, while keeping no-CUDA and platform mismatch checks.

2. [D1 / D3] Registry/profile test rewrites need exact new invariants. Current tests assert removed fields directly at `tests/test_cloud_image_adapters.py:30-31`, `tests/test_cloud_video_adapters.py:42-43`, `tests/test_wan_capability_row.py:21,43-44`, `tests/test_video_ltx_av.py:51`, and others. Fix: replace with `required_toolchain`, `requires_sidecar`, `cpu_ok`, `model_requirements`, and registry-consistency assertions.

3. [D2] Verify whether dark 3D engine code is in scope. There are hard 3D VRAM ceilings in `nodes/_otr_video_engines/eng_character_3d.py:74,257,324,393` and `nodes/_otr_video_engines/eng_triposr.py:52,121`. [ASSUMPTION] If these engines are unreachable because registry rows are removed, document that; otherwise rip or convert them too.

4. [RIP ORDER] The plan pushes full suite + Bug Bible to the final step, while the repo rule says run regression suite + Bug Bible after every code change/chunk. Fix: make each chunk’s green gate include the required suite/Bug Bible, or explicitly define a smaller approved “green chunk” gate plus final full gate.

OPTIONAL / NICE-TO-HAVE:
- Add a repo-wide post-change grep target for `vram_ceiling`, `VRAM_CEILING`, `OTR_VRAM_CEILING_MB`, `assert_peak_within_ceiling`, `assert_vram_within_ceiling`, `VramPeakProbe`, `vram_class`, and `vram_estimate_mb`.

CUT THESE (over-engineering):
1. [D8] Cut “VERIFY-AT-BUILD” as a separate later decision. The source already shows `compute_real_frame_budget` is a gate at `nodes/_otr_video_engines/motion_common.py:370-411`; resolving it up front is simpler than pausing mid-build.