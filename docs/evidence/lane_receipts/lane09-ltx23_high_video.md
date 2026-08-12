# VIDEO_LANE_PREFLIGHT receipt -- lane 9, `ltx23_high_video` (`ltx_video`)

`VIDEO_LANE_PREFLIGHT receipt: ltx_video | 2026-08-11 | smoke receipt
output/otr/episodes/_lane_smokes/lane09_ltx23_high_video/ | verdict PASS`

The lane whose preflight rows were ALREADY 7/7 green before it opened, so its
work was never gate-flipping -- it was two measurements the corpus forbade
guessing at, and what those measurements then obliged the code to do.

## Matrix row

7/7 GREEN before and after. No `EXPECTED_RED` entries to delete -- this lane
never had any.

## What the lane found BEFORE spending a single second of GPU time

**The lane could not have reported an honest number, and nothing said so.**
`7afe40e5` landed the L4 receipt fields "because lane 9 cannot measure without
them" and routed them to `render_clip` -- the SINGLE-PASS path. The installed
unet on this box is `ltx-2.3-22b-dev-Q3_K_M.gguf`, and `_detect_recipe` routes a
dev-family unet to `hq_two_stage`. So the receipt landed on the path this box
does not run, and `_render_clip_hq` had **no `VramPeakProbe` and no
`_clip_telemetry` call at all**.

That is worse than a missing number, because `render_driver` reads:

```python
clip_peak = clip.get("vram_peak_mb")
return clip, out_shot, [eng], (clip_peak or _mc.vram_used_mb())
```

An absent peak is silently replaced by an INSTANTANEOUS post-render sample --
shaped exactly like a peak, readable as one. **This leg would have reported
4,124 MB instead of 15,916 MB: a 3.9x understatement**, and by L10 PROVENANCE a
sample is a lower bound that must never seed a cost row, because a row built on
one under-predicts and admits renders that then OOM. Lane 9's entire deliverable
is a measured marker; it would have shipped a fabricated one.

**And the second seam, which is `eng_ltx_8gb`'s bug repeated inside the file
that quotes it.** `_clip_from_raw` copied `native_frame_count` and
`extension_mode` and dropped the other five, under a comment stating in as many
words that a field this method does not copy is a field `render_beat_coverage`
never sees. So even the single-pass path's correctly-probed peak died one method
short of the ledger.

Found by reading `docs/LANE_BUILD_LESSONS.md` top to bottom before writing code,
which is step 1 of the per-lane loop and has now paid for itself on lanes 2 and
9 both.

Pinned by `tests/test_ltx_video_receipt_seam.py` -- 11 tests, every one of which
fails against the pre-fix file (verified by reverting and re-running, not
assumed). The peak assertion pins the SENTINEL a probe spy returns rather than
`is not None`, because `is not None` passes on the GPU box against the very
fallback sample it exists to forbid.

## LEG 2 -- the marker (run FIRST, production-shaped)

Consent act CLOSED, stock `default` boot on the `LTX` token.

| Item | Value |
|---|---|
| Harness | `_otr_single_engine_smoke.py --engine ltx_video --frames 169` |
| Prompt id | `0bee9763-78fa-4995-b9e3-d99c4d81ff13` |
| Recipe | `hq_two_stage`, quant `Q3_K_M`, **no `+prequalification` suffix** |
| Wall time | **147.5 s** |
| Canvas PROBED | **1024x576** -- equals the declaration |
| Frames PROBED | **169**, duration 6.760 s = 169/25 exactly |
| Rate / codec | 25/1, h264, yuv420p |
| Audio | **zero audio streams** -- silence PROVED on the emitted file |
| Trim | none; `extension_mode: none`, native 169 |
| Peak, ABSOLUTE | **15,916 MB** device-total, COLD, `VramPeakProbe` MAXIMUM |
| Peak, NET | **13,313 MB** (minus this leg's own 2,603 MB pre-queue baseline) |
| Artifact | `.../lane09_ltx23_high_video/ltx_video_default_marker_f169.mp4` |
| sha256 | `665389a1b7b586ad5ab5be0121b68b2cac1fda7c7519bc0915867b41afb443d8` |

**State the surface or the number is not evidence (L7).** 15,916 MB absolute is
**OVER** the 14.5 GiB working ceiling and reaches 97.6% of this 16 GB card.
13,313 MB net is **under** it. The cost-row surface is NET by the 2026-08-11
ruling, so the marker rests on 13,313 -- but the absolute figure is the honest
answer to "does this lane have headroom at f169", and the answer is **no
measured headroom**, with no diet boot yet tried on it.

`high` is measured against its own SIBLING, not against a card: the same LTX 2.3
22B stack at the same 1024x576 canvas costs 11,872 MB net on
`ltx23_low_audio_in` and 13,313 MB net here. That is what the token states.

## LEG 1 -- the decode band, and S8b-14 answered at its ROOT

Consent act OPEN (`OTR_LTX_VIDEO_PREQUALIFICATION=1`, `OTR_LTX_MIN_DECODE_FRAMES=9`).
Every clip stamped `+prequalification[min_decode_frames=9]`.

| rung | wall | peak abs | baseline | **net** | frames PROBED |
|---:|---:|---:|---:|---:|---:|
| 9 | 60.5 s | 14,670 | 1,566 | **13,104** | 9 |
| 49 | 75.4 s | 14,823 | 1,590 | **13,233** | 49 |
| 97 | 95.7 s | 14,823 | 1,596 | **13,227** | 97 |
| 121 | 110.5 s | 15,139 | 1,528 | **13,611** | 121 |
| 137 | 120.5 s | 15,431 | 1,528 | **13,903** | 137 |
| 169 | 147.5 s | 15,916 | 2,603 | **13,313** | 169 |

**THE BAND IS OPEN. The 169 decode floor is not required at 1024x576.** The
decisive rungs are 121 and 137 -- the exact pair that raises the wrapper's
tensor 256-vs-128 (dim 1) mismatch at 1472x832. Both decode clean here, and so
does f9, the ladder's bottom rung.

**A decode that RETURNS is not a decode that produced a picture**, so every clip
was probed for content and motion rather than trusted for exiting zero: mean
luma 16.8-19.6 (a dim interior, plausible), mean inter-frame delta 0.27-0.83
with no frozen span anywhere. No false passes.

**Rung order was chosen cheapest-information-first**: f9 (if the bottom decodes
the band is wide open), then the two known-bad-elsewhere rungs, then the middle
to give the band interior evidence rather than two endpoints and an assumption.

## What the measurement then obliged

The floor was **never a look choice** -- it was a measured decode constraint at
a canvas this lane no longer uses. Removing a constraint that does not exist is
the root fix S8b-14 asked for, so:

* `_LTX_DECODE_FLOOR_DEFAULT` 169 -> **9**
* `frame_contract` `min_frames` 169 -> **9** (max 169, quantum 8 unchanged)

`min_frames=169` was TRUE while the runtime raised every short ask to 169, and
the 2026-08-01 note explaining that is still exactly right. What changed is the
RUNTIME, not the doctrine: the honest 8n+1 declaration is true again **because
the runtime was fixed, not because the declaration was relaxed to fit**.

**An independent confirmation nobody wrote by hand.** Regenerating
`docs/ENGINE_MATRIX.md` moved the multi-clip partition for a 442-frame beat from
`3: 169, 169, 169` (507 rendered, 65 surplus) to `3: 169, 169, 113` (451
rendered, 9 surplus) -- computed by the real `partition_beat`, not asserted by
this lane.

## G8.1 solo smoke -- the SHORT-BEAT A/B (operator condition)

The floor move changes the delivered look on short beats, so the operator gated
it on eyes rather than arithmetic and made the lane's solo smoke a ~2 s beat
delivered as an A/B. Clean boot, consent act CLOSED.

| Item | Value |
|---|---|
| Harness | `_otr_single_engine_smoke.py --engine ltx_video --frames 50` |
| Prompt id | `05726a44-8bfc-4d0c-8eff-3c80db9b7dc2` |
| Ask / delivered | 50 asked, **49 delivered** (snapped DOWN to 8n+1) |
| Recipe | `hq_two_stage`, `Q3_K_M`, **no `+prequalification` suffix** |
| Wall time | **75.4 s** against the old behaviour's 147.5 s |
| Canvas PROBED | **1024x576**; frames PROBED **49**, duration 1.960 s = 49/25 |
| Audio | zero audio streams |
| Peak | 15,231 MB absolute / **13,330 MB net** (baseline 1,901) |
| Artifact | `.../ltx_video_smoke_short_beat_f50.mp4` |
| sha256 | `50e0bb305506faa3de77234ae8c794d5ff2c338e083dab86f184dc3395b5984c` |

**That sha256 is BYTE-IDENTICAL to the f49 clip from the consent-act sweep.**
One was produced with the floor forced by environment under prequalification;
the other with the floor shipped in code and no consent act at all. Same bytes.
That proves two things at once: the sweep measured the real thing rather than
some measurement-only path, and V-7 determinism holds on this lane.

### The A/B, and what it actually shows

* `ab_BEFORE_truncated_from_f169_f49.mp4` -- the first 49 frames of the real
  f169 render, which is literally what the composite delivered for a 2 s beat.
* `ltx_video_smoke_short_beat_f50.mp4` -- the same beat rendered natively.
* `ab_BEFORE_vs_AFTER_f49.mp4` -- 2048x576 labelled side-by-side, 49 frames.

Both sides share the same seed (`render_single` keys the seed to
`shot_id="single_0000"` -> seed 7, constant), the same still, the same prompt,
the same canvas and the same recipe. **Only the rendered length differs.**

BEFORE shows the opening slice of a 6.76 s motion arc; AFTER shows a complete
arc in the beat's own window. Measured rather than described: mean inter-frame
delta 0.418 at f169 against 0.828 at f9 -- LTX paces whatever arc it is given
into the length it is given, so a short render is **not** the truncation of a
long one. That is the whole visible change, and one constant reverses it.

**A false alarm worth recording so the next reader does not re-raise it.** The
A/B looks heavily pillarboxed. That is the INIT STILL, not the lane: the still
is a portrait padded into an 832x480 frame by the WAN smoke that produced it,
and only 39.4% of its width carries picture. The render reproduces 39.6%.
Measured, not eyeballed. It makes the A/B a fair comparison and a poor showcase.

## The naming MOVE

`ltx23_16gb_video` **MOVED** into `_LEGACY_ENGINE_ALIASES`; `ltx23_high_video`
is the public row. A MOVE, never an ADD -- two public ids on one internal id
collapses `_INTERNAL_TO_PUBLIC` and trips the module-scope bijection assert at
IMPORT time, which empties most of the ComfyUI menu rather than failing one lane
(L5's wider shape, proved on a real rename in lane 5).

This retires the **last `16gb` token** in the table, and it was wrong in the
opposite direction from `8gb`: `16gb` read as a comfortable fit for a 16 GB card
while the render peaks at 97.6% of one.

Measurement came BEFORE the naming, per lane 8's order.

## Deliberately NOT done here

**No cost row.** `QUALIFIED_COST_ROWS` stays empty and the manifest says
"admission NOT enforced" for this lane, in words. One cold leg at one rung is a
datapoint; the band sweep measured decode LEGALITY, not budget fit.

**No diet boot tried.** This lane has no measured headroom at f169 and the
obvious next question is whether a diet contract buys any. It is not this lane's
question -- lane 7b proved the reserve half of that lever is inert on the LTX-AV
adapter, so the answer here is genuinely unknown and deserves its own leg.

**The 169 ceiling is not re-derived.** The contract still declares max 169 and
this leg rendered exactly that, so the top of the ladder is measured at the
declared canvas -- but only the top and the rungs swept.
