# Talking-radio Sub-plan C -- probe criterion + verdict

Contract: `kibitz-runs/2026-07-01-talking-radio/r1/final.md` (Sub-plan C).
Probe legs + batch driven by `scripts/_otr_talking_radio_night.py` (local
throwaway harness per the `scripts/_*.py` gitignore convention; results:
`night_results.jsonl`, probe stamps: `probe_manifest_*.json`, live class_type
capture: `object_info_ltx_capture.json`).

Durable re-probe (no manual dispatch edit): boot headless via
`scripts/_otr_soak_server_launch.cmd` with `OTR_ENABLE_LTX_AV=1
OTR_ENABLE_ZIMAGE=1 OTR_ZIMAGE_UNET=z_image_turbo_nvfp4.safetensors
OTR_LTX_AV_UNET=distilled-1.1\ltx-2.3-22b-distilled-1.1-Q3_K_M.gguf
OTR_LTX_RADIO_FACE=<0|1>` then run `scripts/run_otr_30word_smoke.py`
(forces every role to ltx_audio_in, no-fallback, histogram-asserted); score
the pair with `scripts/otr_talking_radio_probe_eval.py <face0.mp4> <face1.mp4>`.

## Context (why this probe exists)

Our `ltx_audio_in` is DOCUMENTED as AMBIENT motion, not lip-sync
(render_driver). The official LTX-2.3 template lip-syncs a NON-human face, and
`LTXVImgToVideo` has no face/landmark detector -- it drives whatever READS as a
mouth. Sub-plan B shipped a mouth-forward still (huge rubbery grille-mouth,
`style="ltx_radio_mouth"`) for the `OTR_LTX_RADIO_FACE` mint. This probe tests
whether that still actually gets DRIVEN by the audio -- BEFORE any engine
surgery (Sub-plan A) or routing promotion.

## Probe design (matched pair)

Two 30-word all-`ltx_audio_in` episodes, both booted `OTR_C7=1` (cast/style
seeds pinned to 42 so the pair is matched); the ONLY delta is the flag:

* `probeA_face0` -- `OTR_LTX_RADIO_FACE=0`: bookends animate the FACELESS
  scene still (today's production look). CONTROL leg.
* `probeB_face1` -- `OTR_LTX_RADIO_FACE=1`: bookends animate the Sub-plan-B
  mouth-forward radio-face still. TEST leg.

Manifest stamp (`ambient-vs-lipsync-expectation`) rides each probe manifest so
a non-articulating face is never misread as a broken render.

## WRITTEN CRITERION (pre-registered BEFORE viewing any render)

The talking radio is PROVEN ("talks") iff, on the OPEN bookend segment of the
face1 leg:

1. **Articulation**: the grille-mouth region visibly opens/closes (aperture /
   shape change), not merely global drift, camera sway, or glow pulsing.
2. **Transient correlation**: mouth open/close events align with speech/music
   transients across the clip. Programmatic aid: Pearson correlation between
   (a) per-frame motion energy in the mouth region (lower-central face box,
   frame-differenced) and (b) the audio onset-strength envelope of the same
   segment, computed at matched timestamps:
   * face1 correlation r1 >= 0.35, AND
   * r1 - r0 >= 0.15 (face1 must clearly EXCEED the face0 control r0 -- this
     kills the "everything wobbles with the music" false positive).
3. **Sanity**: the face0 control shows the expected ambient-only behavior
   (drift OK, no consistent syllable-level articulation).

VERDICT = GO (talks) only if 1 AND 2 AND 3 hold. Anything else = NO-GO
(only-drifts): keep the moving-console look; HuMo stays the face path;
Sub-plan A is NOT built. "Retire OTR_ENABLE_HUMO_HOSTS for bookends" stays OUT
of scope either way until C passes (Codex CUT #2).

GO/NO-GO is OPERATOR-GATED: the numbers + clips below are evidence; Jeffrey's
morning eyeball decides.

## Results

_(pending -- filled by the probe analysis after the pair renders)_

| leg | obs mp4 | mouth-region r | verdict aid |
|-----|---------|----------------|-------------|
| probeA_face0 | pending | pending | control |
| probeB_face1 | pending | pending | test |

## Verdict

_(pending operator eyeball)_
