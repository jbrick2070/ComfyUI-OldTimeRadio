# Fable -- OTR overnight-soak review + recommendations (read-only analysis)

You are reviewing the results of two overnight coverage soaks on the OTR build
(branch v2.0-alpha, repo `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio`).
This is a READ-ONLY analysis + recommendation pass. Do NOT change code, do NOT
push, do NOT enable Wan, do NOT touch the 16gb_full enable-set or GO_FORWARD_PLAN
section 1A. Produce a written review only.

## What the soak did (context)
- Harness: `scripts/_otr_overnight_soak_run.py` (a thin throwaway wrapper that
  reuses `scripts/_otr_soak_capstone.py::run_leg` + the coverage sweep's
  `scripts/otr_coverage_sweep.py::profile_for`/`enumerate_options`, filtering out
  the parked `wan_i2v`).
- Each leg = one FULL ~70-word end-to-end episode (writer -> audio -> visuals ->
  SilentComposite -> MasterAudioMux, credits + subtitles) on a live headless
  ComfyUI :8000, RTX 5080 (16 GB), torch 2.10+cu130, normal VRAM.
- Env: OTR_SMOKE_WORDS=70; 3D engines enabled (OTR_ENABLE_LTX_ORBIT,
  OTR_ENABLE_STILL_PARALLAX, OTR_ENABLE_MESH_STAGE); unique story per leg (no
  C7/cast/style seed); OTR_SOAK_SERVER_OUTPUT pinned to the live server output.
- 25 legs per pass = every confirmed VIDEO + 3D + AUDIO dropdown permutation
  across the announcer_visual / music_visual / other_beats_visual (character_video)
  slots. Wan (2 legs) excluded; hunyuan3d_talk + trellis_talk skipped (no cu128).
- Two passes run back-to-back (no code change between) for reproducibility.

## Inputs to read (all under the repo)
- `scripts/overnight_soak_report.md` -- my human-readable report incl. pass-1 and
  pass-2 sections, per-leg PASS/FAIL table, beat histograms, and findings R1/R2/R3.
- `scripts/overnight_soak_run_summary.json` -- machine per-leg verdicts + errors +
  elapsed (currently holds the most recent pass; pass-1 detail is in the report).
- `scripts/overnight_soak_run.log` -- full run log incl. every leg's
  "EXPERIMENT histogram (informational)", "audio byte-identical OK",
  "obs viewing audio OK", and the failing-gate messages.
- `scripts/sweep_monitor_digest.md` -- live digest.
- Harness + logic to assess: `scripts/_otr_soak_capstone.py` (gates:
  assert_no_stray_writes, byte-identity, obs playable, VRAM ceiling),
  `scripts/otr_coverage_sweep.py`, `scripts/_otr_overnight_soak_run.py`.
- Engine + capability code: `nodes/_otr_video_engines/` (esp. eng_ltx_video.py,
  eng_wan_i2v.py, eng_humo.py, eng_latentsync.py, eng_character_3d.py,
  eng_still_parallax.py, the cheap-family / floor render path, and wherever the
  floor writes `otr_floor_*` / `otr_parallax_*` / `otr_ltx_*` temp mp4s),
  `nodes/_otr_shared/capability_profiles.py` (availability/fit logic),
  `nodes/_otr_shared/fallback.py` (the humo->latentsync->still_motion chain),
  and the writer post-validator that raises `PostValidationError: V1: key_term ...
  not in source` (search `OTR_LedgerScriptWriter` / "key_term" / "PostValidation").
- Rendered output to spot-check (open a few): the live server output tree
  `C:\Users\jeffr\ComfyUI-Installs\ComfyUI\ComfyUI\output\otr\episodes\` and the
  playable finals in `...\output\otr\obs\*_final.mp4`. Watch 2-3 finals per slot
  (announcer / music / other_beats) and judge look quality, not just pass/fail.

## The three findings to confirm / root-cause / fix
- **R1 (PRIMARY, hit every full-render leg both passes):** the floor + cheap-family
  video render path leaks `.mp4` files into the system temp dir
  (`%LOCALAPPDATA%\Temp`) and never cleans them up (`otr_floor_still_motion_*`,
  `otr_floor_<engine>_*`, `otr_parallax_*`, `otr_ltx_*`; ~6/leg). The strict
  hygiene gate (`assert_no_stray_writes`) fails on this even though the episode
  rendered playable with byte-identical master audio. Likely a NamedTemporaryFile
  (or equivalent) never unlinked. THIS is the one fix that flips most legs green.
- **R2 (behavioral):** at 70 words the audio-driven / motion / 3D engines render 0
  native beats and fully demote to the still floor in the character_video slot --
  including the DEFAULT `humo_1.7B` (0/6) and `humo` 14B (0/6), `latentsync` (0/6),
  `triposg_talk` (0/6). Still-image engines (`flux_still` 3/3, `still_parallax`
  3/3) and the music engines (`visualizer`/`abstract`/`ltx_orbit` 1/1 music beat)
  DO render. Peak VRAM ~16.1 GB NVML. Determine whether HuMo is hitting the
  in-render VRAM lease and demoting, an input/eligibility gate, or a 70-word
  timing/aspect issue (cf. CS-4: umt5 text-encoder starving the 14B budget).
- **R3 (intermittent):** the writer post-validator aborts an episode after 3
  retries with `PostValidationError: V1: key_term '<x>' not in source` -- 1 leg in
  pass 1, 2 legs in pass 2 (~1-2 per 25, story-RNG dependent; the same leg passes
  with a different story). Decide whether to relax the validator or re-prompt to
  draw key_terms from source.

## Deliverables (write a report; group exactly like this)
1. **Fixes (must-fix for a green sweep):** root-cause + concrete fix for R1
   (name the file/function and the cleanup approach), R3 (validator policy), and a
   verdict on R2 (is the HuMo/default-char flooring a bug to fix or expected
   budget behavior at 70 w? if fixable, how). For each: risk, blast radius, and a
   regression test to add.
2. **Minor improvements:** harness/quality-of-life items -- e.g. the output-tree
   auto-resolver picking the stale Documents tree (should prefer the live server's
   OTR_OUTPUT_DIR), a `--exclude` flag so parked engines (Wan) aren't enumerated
   as runnable, quieter heartbeat cadence, hygiene-gate allowlist scoping, etc.
3. **Best permutations for release (the ask):** given the two passes, recommend
   the per-slot engine choices to ship as defaults for a 70-word (and a longer)
   episode -- which announcer_visual / music_visual / other_beats_visual engines
   give the best look + reliability + VRAM headroom on a 16 GB 5080, and which to
   keep selectable-not-default. Justify from the histograms + the finals you
   watched. Note any engine you'd gate off at release until R1/R2 land.
4. **Bigger-than-a-breadbox future updates:** larger directions worth considering
   -- e.g. a real character-motion path that survives at 70 w within 14.5 GB,
   bringing Wan online once unparked, a VRAM-budget-aware scheduler so heavy
   engines don't silently floor, longer-episode soak tiers, multi-GPU/offload,
   look-QA automation, etc. Rank by impact vs effort.

## Constraints
- UTF-8 no BOM, ASCII-only, SFW, no profanity. Use "placeholder", not other words.
- Single resident heavy must stay <= 14.5 GB; frozen audio byte-identical within a
  leg; 100% local. Honor these in any recommendation.
- This is analysis only -- no code edits, no pushes, no Wan, no enable-set changes.
  If you want to propose code, write it as a diff/snippet in the report for a
  separate coder window to apply.
