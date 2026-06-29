VERDICT: yes-with-fixes. Main episode-path idea is coherent, but the plan contradicts “dropdown-only” around force-map env, session-global opt-ins, and the stated b000/HuMo premise.

MUST-FIX BEFORE BUILD:
1. [Problem / The change] Hidden `OTR_FORCE_ENGINE_MAP` can still beat the dropdown. `run_real_episode` calls `apply_engine_override` before `apply_selection_enable_set`, and the override rewrites `shot["engine_id"]` from env: `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\nodes\_otr_video_engines\render_driver.py:1776-1780`, `:1817-1865`. Fix: either explicitly carve out force-map as an operator override, or derive enablement from pre-override dropdown selections and require explicit flags for force-mapped engines.

2. [The change / Side effects] Session-global `os.environ` mutation breaks the claimed tight scope. The helper sets `os.environ[flag] = "1"` and never restores it: `...\nodes\_otr_video_engines\render_driver.py:1721-1723`. In a resident ComfyUI server, “once selected, enabled for session” is the real contract. Fix: bracket `run_episode` with restore-on-exit, or state session persistence as intended and log previous/new state.

3. [Problem] “EVERY episode dies at beat b000 unless `OTR_ENABLE_HUMO=1`” is false as stated for the current workflow. Synthetic open is `music_visual`, and ShotLock stamps the engine from policy: `...\nodes\otr_shot_lock.py:282-309`, `:711-714`, `:750-772`. Current workflow node 87 has `music_video_model = visualizer`, not `humo`: `...\workflows\otr_scifi_16gb_full.json` widgets index 1. Fix the premise to “any selected flag-gated engine fails without derived opt-in,” or cite the specific workflow state where music is HuMo.

4. [The change] “Best-effort; never raises” undercuts the root fix. The helper swallows all failures and allows the old `gated_by_flag` path to reappear: `...\nodes\_otr_video_engines\render_driver.py:1707`, `:1729-1730`. Fix: fail early with a clear selected-engine enable-set error for registered engines; only defer unknown custom engines to `assert_usable`.

SHOULD-FIX:
1. [Invariants] Harness wording is inaccurate. `run_gpu_soak` uses `run_episode`: `...\nodes\_otr_video_engines\render_driver.py:2264-2269`, but dep-pilot probes subprocess imports, not `run_episode`: `...\scripts\otr_video_dep_pilot.py:512-587`, and GPU smoke drives adapter lifecycle directly: `...\scripts\otr_video_gpu_smoke.py:111-130`. Fix wording to “these bypass `run_real_episode`.”

2. [Test gaps] Existing `run_real_episode` tests cover request shape/determinism, not opt-in env behavior: `...\tests\test_video_render_driver_additive.py:386-404`. Add tests for selected engine sets flag before `assert_usable`, missing checkpoint still fails, force-map behavior, env restore/persistence contract, and direct `run_episode` bypass.

3. [Missing pieces] [ASSUMPTION] If `run_otr_30word_smoke.py` is considered a production smoke path, it still pre-requires launch env before render: `...\scripts\run_otr_30word_smoke.py:224-232`. Either update it for dropdown-driven opt-in or explicitly classify it as an explicit-flag harness.

OPTIONAL / NICE-TO-HAVE:
- Add the derived enable-set to the render report/manifest for auditability.

CUT THESE (scope / over-engineering):
1. [Invariants] Cut fallback-chain/floor language from this opt-in plan. Current render path says no fallbacks and raises loud: `...\nodes\_otr_video_engines\render_driver.py:1526-1536`. Keeping fallback claims here is stale bloat.

2. [QA questions] Cut `capability_profiles enable-set` as an implementation dependency unless profile gating is explicitly in scope. That module is hardware/profile fit, not runtime consent: `...\nodes\_otr_shared\capability_profiles.py:16-24`, `:311-319`.