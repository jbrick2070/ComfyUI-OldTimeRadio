# MORNING NOTE -- 2026-06-30 (overnight while you slept)

## What's rendering right now
The proven coverage runner is rendering the **18 pending video x image combos -> `output\otr\obs`**
against your live :8000 server (gemma writer + all engines were up; verified the first leg `cov_fill_humo`
queued + rendering). It is **resumable** and **per-leg fault-tolerant** (a failed leg is recorded and the
run continues to the next), so by morning you should have a set of `*_final.mp4` combos in obs.

- Watch progress: `scripts\_otr_overnight_combos.log`
- Matrix + dashboard: `scripts\_otr_coverage_matrix.json` -> the `otr-coverage-soak` artifact
- The 20 legs: 15 video engines (humo / humo_1.7B / humo_14B_169 / ltx_video / ltx_audio_in / wan_i2v /
  wan_ti2v / mesh_stage / still_* / station_card / abstract ...) + 5 image engines (flux_gen1 /
  flux2_klein / z_image_turbo / qwen_image / lumina_image). 2 were already done (1 pass, 1 fail).

## IMPORTANT CAVEAT
That server was booted **before** today's S-A clip-fill fix, so the **HuMo combos still show the
held-frame murk** (S-A is committed + pushed but loads only on a ComfyUI restart). The other engines
render at current quality. For a clean S-A-loaded eyeball, restart ComfyUI Desktop and re-run the
overnight runner. I did NOT restart the server overnight because a single-lane headless boot risks not
covering all engines, and the running server was proven (it had just rendered combos at 02:32).

## The BAKED combo soak you asked for (modified workflow)
I ran your `/kibitz` on the design -- Claude Code + Codex + Antigravity, all grounded against the real
repo, r1->r2->r3 CONVERGED. The full spec is `docs\2026-06-29-coverage-soak\COMBO_SOAK_CONVERGED_PLAN.md`.

Key grounded finding: the per-beat engine_id is baked into the ledger by **ShotLock (node 90)**, which
is upstream of the image dispatcher -- so "start at image gen" cannot vary engines. The correct
"start from story + audio" boundary is **upstream of the directors**: bake node-62 `script_json` (the
frozen story) + the node-7 audio, keep `directors -> shotlock -> imagegen -> video -> composite ->
upscale -> mux (-> obs)` LIVE, and vary the node-87/88 engine picks per combo. That gives ONE story +
byte-identical audio for every combo (apples-to-apples) and skips the writer + TTS (minutes per leg, not
~28). I did NOT ship an untested modified JSON overnight -- building + validating it unattended risked
producing nothing; the converged spec is build-ready for us to wire + test together when you're back.

## Today's shipped work (all green: suite 5766p/0f + Bug Bible, pushed to v2.0-alpha)
S-F smoke fixture (c6c50579) - S-A clip-fill + legibility floor (4e13a692) - S-B ltx_audio_in
observability (eb8c3781) - S-D gemma {RadioEditPlan} unwrap (5a50fa40) - S-E5 ledger recipe-stamp
(9e4f3a33) - BUG-411 verified done. The clean-break fallback rip-out (E1/E3) is designed + converged
(same kibitz doc) but NOT yet built -- it's a large cleanbreak best done with your eyeball on the
A-ship gate change.

## First thing to check
`output\otr\obs` for the new `*_final.mp4` combos + `_otr_overnight_combos.log` for the per-leg verdicts.
