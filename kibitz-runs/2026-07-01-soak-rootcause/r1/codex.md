VERDICT: yes-with-fixes. The two root causes are real, but the plan is not build-ready until it chooses one output authority and one force-map/image-phase contract.

MUST-FIX BEFORE BUILD:
1. [ROOT CAUSE 1] The proposed “launcher exports env” fix cannot by itself fix the driver, because `scripts/_otr_visual_soak_6leg.py:32-33` imports `_otr_soak_capstone` before any per-leg launcher env is written, and `scripts/_otr_soak_capstone.py:57-60` freezes `SERVER_OUTPUT` at import time. Child `.cmd` env cannot flow back into the parent Python process. Concrete fix: set the parent driver output env before importing capstone, or replace capstone’s import-time constants with a resolver called at leg start; then make the default the real Documents tree, not `C:\Users\jeffr\ComfyUI-Installs\...`.

2. [ROOT CAUSE 1] “Retry-until-obs” still treats harness misconfiguration as render failure. `scripts/_otr_visual_soak_6leg.py:109-120` retries any exception while `_obs_set()` watches the same wrong capstone obs dir; the log proves three successful HuMo renders were repeated after orphan-report failures (`scripts/_otr_soak_capstone_results/visual6_20260701_130355/soak.log`, leg2). Concrete fix: classify output-tree mismatch/orphan-report/no-obs-under-watched-tree as non-retryable harness failures and abort the leg before re-rendering.

3. [ROOT CAUSE 2] The plan does not choose the semantic contract for `OTR_FORCE_ENGINE_MAP`. Current render-time force is video-scoped in `nodes/_otr_video_engines/render_driver.py:1938-1999`, while mesh fodder is computed earlier from saved `video_policy` in `nodes/otr_image_director.py:185-197` and `:376-381`. Concrete fix: choose one contract before coding: either force-map becomes a cross-phase visual override and `mesh_fodder_roles_from_video_policy` must resolve the effective forced engine, or mesh-stage soak must use a real policy/profile so the image phase sees `mesh_stage` normally.

4. [ROOT CAUSE 2] The plan says the leg “needs mesh_fodder minted” but does not state the music-open/no-character subject policy. The failing shot is `shot_b000_music_open`; render code explicitly supports beat-id fodder for no-char beats in `nodes/_otr_video_engines/render_driver.py:973-980`, and the log shows missing fodder for empty subject at `server_leg4_mesh_stage.log:1059-1062`. Concrete fix: require tests/acceptance for both character and no-character forced mesh roles.

SHOULD-FIX:
1. [Ask 3] Add an explicit acceptance matrix. The current plan lists fixes but not proof. Minimum: unit test capstone output resolution, unit test visual driver parent env/import behavior, unit test force-map mesh-fodder roles, then one real `workflows/otr_scifi_16gb_full.json` headless leg. Capstone already loads the real workflow at `scripts/_otr_soak_capstone.py:53-54` and `:456-459`.

2. [Non-issues to CONFIRM] Do not leave “VRAM gate separate question” vague. `scripts/_otr_soak_capstone.py:641-656` gates on render-phase `vram_peak_mb`, not whole-run NVML, so the plan should either keep that gate unchanged or explicitly defer any ceiling change. Otherwise a real successful render may still be marked fail for a policy reason unrelated to these root causes.

OPTIONAL / NICE-TO-HAVE:
- Add a startup log line printing all resolved output roots: capstone `SERVER_OUTPUT`, `REPORT_PATH`, `_obs_dir()`, launcher `OTR_REAL_OUTPUT`, and ComfyUI `--output-directory`.

CUT THESE (scope / over-engineering):
1. [ROOT CAUSE 1] Cut “shared resolver shared by launcher and capstone” if it means a cross-language abstraction. A `.cmd` launcher and Python capstone do not need a shared library; a single parent-side env contract plus capstone fail-loud check closes the bug.

2. [ROOT CAUSE 2] Cut any general redesign of force-map/profile precedence. The narrow missing piece is forced-engine visibility to the image director’s mesh-fodder decision, or using a real policy for mesh-stage. Reopening global selection architecture is not needed for this soak failure.