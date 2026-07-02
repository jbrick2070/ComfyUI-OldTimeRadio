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

## Results (2026-07-02 ~01:50, analysis run same night)

Both legs rendered clean: 6/6 clips `ltx_audio_in`, obs finals exist
(`night_results.jsonl`). Probe stamps: `probe_manifest_probe{A,B}*.json`.
Eval JSONs: `probe_eval_open14s.json` (0-14s window) +
`probe_eval_b001_announcer.json` (the CLEAN window: b001 announcer bookend,
~9.6s+11.5s -- pure LTX clip, speech transients; the 0-14s window is
contaminated by the procgen title blend on b000).

| leg | obs mp4 | b001 r (mouth motion vs onsets) | mouth motion mean | verdict aid |
|-----|---------|--------------------------------|-------------------|-------------|
| probeA_face0 | signal_lost_recorded_mysteries_20260702_005445_..._final.mp4 | **0.009** | 1.77 | control: ambient as expected |
| probeB_face1 | signal_lost_jazz_code_cracker_20260702_012739_..._final.mp4 | **0.047** | 1.06 | r1 << 0.35; delta 0.037 << 0.15 |

**What the frames show** (`probeB_t10p5/t12/t15/t19p5.png`): the face-still
bookend IS being animated with REAL mouth articulation -- closed (10.5s) ->
open "oh" (12s) -> closed (15s) -> parted (19.5s), slow push-in -- so this is
NOT pure drift. But the articulation runs on its OWN rhythm: correlation with
the announcer's speech transients is ~zero on the very window where the voice
plays. Classic "dubbed film" mouthing, not lip-sync. Criterion 1 partially
holds (aperture changes), criterion 2 decisively FAILS, criterion 3 holds
(control leg ambient).

**Strengthening observation:** probe B's still was the PRE-material-fix mint
-- a literal human face in the radio (`probe_face1_mouth_still.png`; see
BUG-note in the d87f8fc5 commit). A human mouth is the STRONGEST mouth prior
LTX could get; if the distilled recipe does not couple THAT to the audio, the
corrected appliance grille-mouth (shipped d87f8fc5, live in the overnight
batch_face1 legs) will not couple either.

**Honest caveat / the one re-probe knob:** our recipe is
`distilled-1.1 Q3_K_M / distilled_native / 8-step`. The official comfy.org
LTX-2.3 lip-sync demo may lean on the DEV unet's fuller audio coupling. A
re-probe is ONE env change on the same harness
(`OTR_LTX_AV_UNET=ltx-2.3-22b-dev-Q3_K_M.gguf` -> sharp_lora recipe), at
~1.4x step cost. NOT run tonight -- criterion + budget say stop at the
pre-registered evidence.

## Verdict (pre-registered criterion applied)

**NO-GO on lip-sync as measured** -- r1 = 0.047 (threshold 0.35), delta =
0.037 (threshold 0.15). Per the contract: keep the moving-console look;
**HuMo stays the face path; Sub-plan A (upsampler) is NOT built.**
"Retire OTR_ENABLE_HUMO_HOSTS" stays out of scope.

**OPERATOR EYEBALL STILL DECIDES**: watch both obs mp4s (b001 segment,
~10-21s). If the uncorrelated mouthing reads as charming "old-dub" talking
radio and you want it as a LOOK (not lip-sync), say so -- that is a creative
GO on the A/B toggle, separate from the technical lip-sync claim, and A can
be revisited. If you want true sync, the dev-unet re-probe above is the next
cheap step; otherwise HuMo hosts remain the talking path.

_(operator: fill in your morning call here)_
