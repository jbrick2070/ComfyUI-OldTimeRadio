# 6-Leg Visual Soak -- ROOT CAUSE + FIX PLAN (for kibitz)

The 2026-07-01 six-leg 45-word visual soak reported all legs FAIL / 0 obs. Log review
shows the RENDERS ACTUALLY SUCCEEDED -- the failures are two harness/integration bugs,
NOT the feature code and NOT VRAM. Harden these fixes.

## Evidence (from the run logs)
- Leg 2 (`humo_1.7B`) server log tail: `[OTR_MasterAudioMux] obs_publish OK -> ...`,
  `audio_byte_identical OK`, `Prompt executed in 00:44:06`. A 50 MB final mp4 WAS published
  to `C:\Users\jeffr\Documents\ComfyUI\output\otr\obs\`.
- Yet the driver logged: `SoakFail: render report node_episode_report.json is OLDER than
  this leg's start -- orphan report rejected (run identity)` and `NO new obs mp4 (LOUD)`.
- Six real finals exist in `C:\Users\jeffr\Documents\ComfyUI\output\otr\obs\` from the run
  (leg 2 x3 + leg 3 x3 -- the driver retried 3x each because it thought they failed).
- Leg 4 (`mesh_stage`) server tail: `RenderError: shot shot_b000_music_open engine
  'mesh_stage' ... requires input(s) ['init_image'] the request does not carry`.

## ROOT CAUSE 1 -- soak output-tree split source of truth (the "orphan report" + "no obs")
- `scripts/_otr_soak_capstone.py:57`
  `SERVER_OUTPUT = os.environ.get("OTR_SOAK_SERVER_OUTPUT", r"C:\Users\jeffr\ComfyUI-Installs\ComfyUI\ComfyUI\output")`
  -> `EPISODES_DIR` / `REPORT_PATH` (:60/:63) / `_obs_dir()` (:214) all hang off it.
- The launcher `scripts/_otr_soak_server_launch.cmd` boots the server with
  `set OTR_REAL_OUTPUT=C:\Users\jeffr\Documents\ComfyUI\output` + `--output-directory` +
  `OTR_OUTPUT_DIR` -> the server writes obs/report/episodes to the **Documents** tree.
- The DRIVER process (which runs the capstone gate) did NOT set `OTR_SOAK_SERVER_OUTPUT`,
  so the gate defaulted to the stale **ComfyUI-Installs** tree and read an OLD
  `node_episode_report.json` (orphan) + saw zero new obs -> false FAIL -> 3x wasteful
  re-renders (~44 min each).
- The two paths are a SPLIT SOURCE OF TRUTH: the launcher's canonical `OTR_REAL_OUTPUT`
  vs the capstone's hardcoded default disagree.
- Proposed fix (harden): ONE canonical output-tree resolver shared by the launcher AND the
  capstone gate (e.g. capstone reads `OTR_REAL_OUTPUT`/`OTR_OUTPUT_DIR` first, or the
  launcher exports `OTR_SOAK_SERVER_OUTPUT`, or both derive from a single constant).
  A mismatch must fail LOUD at leg start (assert the obs/report tree the gate watches is the
  tree the server was booted with), never silently check the wrong tree. Also: the
  visual-soak driver `scripts/_otr_visual_soak_6leg.py` must set the same env before it
  imports the capstone, and its retry-until-obs loop should not re-render on a
  wrong-tree miss.

## ROOT CAUSE 2 -- mesh_stage forced via OTR_FORCE_ENGINE_MAP has no mesh_fodder minted
- `OTR_FORCE_ENGINE_MAP=*=mesh_stage` re-routes the VIDEO engine at render time
  (`render_driver.apply_engine_override`), but the IMAGE phase decides whether to mint
  clean `mesh_fodder` (vs a cinematic scene still) from the SAVED video policy via
  `otr_image_director.mesh_fodder_roles_from_video_policy` -- which does NOT consult
  `OTR_FORCE_ENGINE_MAP`. So no fodder is minted; at render `mesh_stage` (family
  image_to_video, `requires_mesh_fodder`) finds no `init_image` -> `FamilyInputGap`
  (DEPENDENCY_MISSING) on `shot_b000_music_open` -> hard fail (no fallbacks).
- Proposed fix (choose one, harden): (a) make the image-phase fodder-role computation honor
  `OTR_FORCE_ENGINE_MAP` (same `_effective_engine_after_force_map` seam
  `_still_needed_for_role` already uses at :359), OR (b) the soak must drive mesh_stage via
  the real video POLICY (a profile / policy patch), not the env force, and document that the
  env force is video-only and does not fork the image phase.

## Non-issues to CONFIRM (not the cause)
- The brief-driven-radio-host feature + ltx A/B addendum code (shipped, 5941 tests green).
- The HuMo-host `audio_ref` fix (commit 4046a50c) worked -- leg 2/3 (hosts-on) rendered
  b000 and published byte-identical audio.
- VRAM: humo_1.7B video ran ~11.5 GB; the ~16 GB peak is the transient FLUX image phase,
  not the resident video model. (The capstone's <=14.5 GB gate measuring the FLUX transient
  is a separate question the panel may weigh in on.)

## Ask for the panel
1. Confirm ROOT CAUSE 1 + the strongest single-source-of-truth fix (fail-LOUD on tree
   mismatch vs shared resolver vs launcher-exports-env).
2. Confirm ROOT CAUSE 2 + whether honoring the force map in the image phase is correct or a
   scope trap (should forcing a mesh engine imply forcing its fodder fork?).
3. Any other latent bug in the soak driver / capstone the logs imply.
