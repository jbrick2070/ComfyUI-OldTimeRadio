# Overnight session log -- 2026-06-10/11 (GATE B foundation, autonomous)

## SESSION LOG HEADER -- ticket order
1. S0 profile foundation: schema + config/profiles/ (16gb_full EXTRACTED from master, 8gb_lite, cpu_floor) + shape validator + checked-in widget MAPPING JSON + image-lane gate topology check.
2. S1 registry capability declarations + derived enable-set + cross-validation (two-heavy-roles regression).
3. S2 the ONE applier (apply_profile, offline INPUT_TYPES adapter tested vs _serialized_slot_names, coverage test) -- CPU-tested against FIXTURE COPIES of the workflow only.
4. Code defects: _load_workflow/IS_CHANGED repo-root fix + node-63 empty-validator-path fix.
5. latentsync OTR_LSYNC_BASE_ENGINE=still_kenburns base-engine support, CPU-tested.
6. (operator, mid-session) dispatcher stale `pending_*` stills dir: title rename to `signal_lost_rapid_roots_*` happens AFTER the dispatcher keys the stills dir; give the dispatcher the same re-resolve MasterAudioMux uses.

## BASELINE (before any work)
- HEAD == origin/v2.0-alpha @ 6a1b716. Full suite: 3903 passed / 28 skipped / 0 failed.
- Bug Bible: **5 PRE-EXISTING failures at baseline** (not caused by this session; bible rules last changed 2026-06-07, the flagged code landed in the still-spine/capstone commits): BUG-01.03 wrapper_bridge.py deep dirname chain; BUG-01.02 otr_video_render_batch.py folder_paths; BUG-07.03 unload_all_models w/o empty_cache (otr_video_render_batch, flux_gen1, eng_humo, motion_common); BUG-09.02 Popen cleanup (eng_latentsync, _otr_soak_marathon, or_probe); BUG-09.02 communicate-with-ffmpeg (wrapper_bridge). Those files belong to parallel-window work (except eng_latentsync, in tonight's scope -- will fix its Popen cleanup). Operator confirmed mid-session: treat the 5 as KNOWN failures (likely already in BUG_LOG). Session green bar = suite 0-fail + Bug Bible NO NEW failures beyond these 5. Operator eyeball gates TAGS only (pushes to v2.0-alpha proceed).
- Uncommitted edits to the two plan docs (planner window) left unstaged and untouched.

## BLOCKERS
(none yet)
