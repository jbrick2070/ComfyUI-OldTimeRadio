# Ghost Signal -- the publish smoke, and the promotion it earned

**Date:** 2026-08-22
**Lane:** `animatediff15_video` ("AnimateDiff -- Ghost Signal")
**Profile:** `config/profiles/otr_ghost_signal.json`, promoted `draft` -> `shipping`
**Plan:** `docs/2026-08-22-GHOST-SIGNAL-CODING-PLAN.md` (sha256 `06104e91...`)

This is the ONE allowed GPU acceptance run from plan section 10. It is a
pass/fail publish smoke and it mints no performance or quality claim.

## The run

Driven by `scripts/otr_headless_canonical.ps1 -Profile otr_ghost_signal -Acts 1`,
which resets the box, boots the UTF-8 launcher and submits the **real**
`workflows/otr_canonical.json` -- never an ad-hoc graph or a generated copy.

* Episode: `signal_lost_the_constables_knock_20260822_050116`
* `RESULT SUCCESS`, `obs_publish OK`, prompt executed in **00:21:33**
* Published: `output/otr/obs/signal_lost_the_constables_knock_20260822_050116_silent_procgen_blended_captioned_with_credits_final.mp4`
  -- 126 MB, 1920x1080, 25 fps, 2433 frames, 97.32 s

## Every section-10 pass condition

| Condition | Result |
|---|---|
| No OOM, no silent fallback | PASS |
| Every beat routed to `animatediff15_video` | PASS -- 8/8 |
| Every per-clip receipt `delivery_scale_mode=lanczos_clean_full_frame` | PASS |
| Every per-clip receipt `cadence_mode=hold_2` | PASS |
| Every row `extension_mode="none"` | PASS |
| `cadence_source_frame_count == ceil(target/2)` | PASS -- all 8 |
| `cadence_tail_trim` only 0 or 1 | PASS |
| Distinct shot-derived clip path per beat, none shared | PASS -- 8 distinct |
| Every native clip exactly 512x288 @ 25 fps | PASS |
| Every native clip SILENT (V-1) | PASS |
| Pre-mux silent master carries no audio stream | PASS |
| Final output 1920x1080 @ 25 fps | PASS |
| Canonical assets on disk + final file in the LIVE `otr/obs` | PASS |

Per-beat cadence, as the frozen ledger recorded it:

| shot | model frames | cadence source | delivered | tail trim |
|---|---:|---:|---:|---:|
| `shot_music_opening_001` | 125 | 125 | 250 | 0 |
| `shot_b001` | 250 | 250 | 500 | 0 |
| `shot_b002` | 140 | 140 | 280 | 0 |
| `shot_b003` | 86 | 86 | 171 | 1 |
| `shot_b004` | 123 | 123 | 246 | 0 |
| `shot_b005` | 58 | 58 | 115 | 1 |
| `shot_b006` | 93 | 93 | 185 | 1 |
| `shot_music_closing_001` | 100 | 100 | 200 | 0 |

Odd delivered counts trim exactly one frame, even ones trim none. That is what
duration-preserving hold-2 looks like when the claim is true. The `by_engine`
rollup reported `varied: []`, so every clip agreed on recipe, delivery mode and
cadence mode.

The applied-profile audit passed on the SUBMITTED prompt (all three director
roles, the render engine, both canvases, fps, dtype, upscale-off) while the
source workflow stayed exactly **23 nodes / 57 links / 140 widget slots** with
node 87 still reading `still_flat` -- `apply_profile` is pure and nothing was
committed back.

## THE FIRST LEG FAILED, AND THAT IS THE MOST USEFUL LINE IN THIS FILE

Attempt 1 died at node 90 (`OTR_ShotLock`) before a single weight loaded:

> Ghost Signal protected slots compose to 344 chars, over the 320 ceiling, with
> nothing droppable left (slots=pack_cue,subject,action,shot_law).

**The defect the unit tests could not see.** Every budget test handed the
composer an AUTHORED motion clause, which short-circuits the visual-style pack's
own register entirely. A real bookend beat with the optional motion pass OFF
does the opposite: it pulls the pack's own `announcer_subject_face` (163-178
characters across the nine shipped packs) AND its `motion_registers` value
(130-209, budgeted at 240 by the loader). On `recur_frac` that is
29 + 178 + 209 + 58 = 474 against a 320 ceiling.

`_trim_to` could not shrink either surface, because both are largely COMMA-FREE
prose and a comma-phrase trimmer is a no-op on them: it finds one phrase, cannot
drop it, and returns the text at its original length.

**Fixed at the root, not papered over.** `_trim_to` now falls back to a WORD
boundary and never ends on a dangling function word; step 4 of the published
trim order shrinks the PACK-derived surfaces proportionally, floored at 55
characters. The character sigil, an authored motion clause, the mid-shot floor
and the shot law remain untouchable -- runtime truncation is still not their
preservation mechanism. 72 new cases cover every role x every shipped pack x
every register key with no clause.

The refusal itself was correct behaviour and is worth keeping in mind: the lane
declined to ship an over-budget prompt, said exactly why, and cost nothing but
the script it had already written.

## What this receipt does NOT claim

No VRAM, wall-time, quality, luma, SSIM, PSNR, tokenizer or canvas measurement
was collected or published. The operator declined the measurement campaign, so
`animatediff15_video` remains **admission-unenforced** in
`docs/evidence/video_evidence_manifest.json` and makes **no VRAM-fit claim**.
The lane may OOM on a smaller card.

One number was OBSERVED in passing and is recorded here as an observation, not
a qualification: `nvidia-smi` read 5872 MiB at 100% GPU during sampling. That is
a single instantaneous reading, not a `VramPeakProbe` maximum, and it may not be
used to seed a cost row.

## Operator acceptance of the look (2026-08-22)

The delivered motion reads FAST. The operator watched the published episode and
accepted it as-is: *"i was expecting experimental vj"* and *"its perfect"*. The
speed is inherent to the recipe -- 12.5 fps of AnimateDiff motion held to 25 --
and it is **not** to be chased. This is an accepted look, not an open defect.
