# VIDEO_LANE_PREFLIGHT receipt -- lane 3, `humo17_high_audio_in_portrait` (`humo_1.7B`) and its landscape twin

`VIDEO_LANE_PREFLIGHT receipt: humo_1.7B + humo_1.7B_169 | 2026-08-11 | smoke
receipt output/otr/episodes/_lane_smokes/lane03_humo17_portrait/ | verdict PASS`

Two lanes closed together because they are ONE checkpoint at two aspects: same
weights, same VRAM class, and the aspect is the entire difference -- which is
why the aspect is now in both public ids rather than only in a label suffix.

## Matrix rows

Both GREEN on all seven gates; both `EXPECTED_RED` G2 entries deleted.

| Gate | `humo_1.7B` | `humo_1.7B_169` |
|---|---|---|
| G1 weights | PASS (inherited lane 2's shared resolver) | PASS |
| G2 canvas | PASS -- declares 480x832, /32-legal, profile corrected to agree | PASS -- declares 832x480 |
| G3 contract | PASS -- 33..177 q4, 25 == 25, `soft_reference` | PASS |
| G4 admission | PASS -- manifest says NOT enforced, in words | PASS |
| G5 audio law | PASS -- probed live, zero audio streams | PASS (shared canonicalize) |
| G6 guards | PASS | PASS |
| G7 surface | PASS | PASS |

## S8b item 3 -- the honesty check that a VRAM knob was gating

The exact-fit guard read `if cap is not None and target_fc > 0`, which tied a
question about HONESTY to a question about MEMORY. An uncapped tier declares
`safe_render_frames = None`, so it skipped the fit entirely: a beat asking for
more than the 177-frame ceiling rendered 177 and returned them stamped
`extension_mode: "none"` with `native_frame_count == frame_count` --
indistinguishable from an honest clip on any path reaching `render_shot`
without a stamped coverage plan. The video ran out before the audio and nothing
said so.

The guard is now unconditional. `fit_frames_to_target` trims a long render
(always safe, never reverses time) and raises on a short one; both answers are
right for every tier, and only the MESSAGE needs to know whether a cap was
involved -- so an uncapped tier's refusal now says "the beat is longer than any
single HuMo render, split it through the coverage planner" instead of naming a
cap that does not exist.

This can fail a beat that previously "succeeded". That is the same trade the
14B took on 2026-07-25, and those episodes shipped a face that stopped moving
while the line kept going.

## The profile that was lying

`otr_w45_humo_1_7b.json` declared `render.canvas_w/h = 832x480` -- landscape --
on the tier whose entire identity is the pillarbox talking head. The engine
renders 480x832 and the declaration is what wins, so the profile was not
causing a bad render; it was misinforming whoever read it. Corrected to
480x832, and G2.3 now enforces the agreement.

## G8.1 solo smoke

| Item | Value |
|---|---|
| Boot contract | `humo_diet` (`--reserve-vram 2.921 --disable-pinned-memory`, confirmed on the command line) |
| Harness | `_otr_single_engine_smoke.py --engine humo_1.7B --frames 129` |
| Prompt id | `8ea9093d-128d-4fb7-ab8a-544385d160c0` |
| Wall time | 210.9 s |
| Canvas PROBED | **480x832** -- equals the declaration, and portrait, which is the point |
| Frames PROBED | **129** counted, duration 5.160 s = 129/25 exactly |
| Rate / colour | 25/1, `yuv420p`, `bt709` |
| Audio | **zero audio streams** |
| Trim | none; 129 = 4*32+1, on the ladder, delivered exactly |
| Artifact | `output/otr/episodes/_lane_smokes/lane03_humo17_portrait/humo17_480x832_f129_diet_smoke.mp4` |
| Artifact sha256 | `86035ba476074290118b416dde30abc1a52c39d453003343bdeabba1fc0bc097` |

### The peak, and why it is higher than the 14B's

**OTR-side render-window peak: 15,261 MB (14.90 GiB), COLD, absolute.** Net of
the ~1,940 MB idle server baseline that is roughly 13.01 GiB, next to the lab's
12.84 GiB warm at the same canvas and rung.

It is HIGHER than the 14B landscape lane's 14,604 MB, which reads backwards
until you notice the rungs: this is **129** frames where that was **97**, at
the same pixel budget (480x832 and 832x480 are the same area). A third more
latent, on a checkpoint roughly eight times smaller. The tier's cost is
dominated by the frame count, not the weights -- which is exactly why it is the
LONG-BEAT lane and why it cannot be treated as "the cheap one" for admission
purposes.

Neither number qualifies anything. `QUALIFIED_COST_ROWS` is still empty, the
manifest says so per lane, and one cold leg is not an envelope.

## Not measured, and labelled so

`humo_1.7B_169` has NO receipt of its own at any rung. Its canvas declaration
fixes the request-versus-render channel and its public label says
**UNMEASURED at this aspect** rather than borrowing the portrait tier's number.
Same checkpoint is a reason to expect similarity, not evidence of it.

## Still open

`humo` (the portrait 14B) remains undeclared and is lane 4. It now carries the
"declares NOTHING" differential control that `humo_1.7B` used to hold -- the
control has moved once already and will move again when lane 4 closes, which is
the point: the invariant outlives every occupant.
