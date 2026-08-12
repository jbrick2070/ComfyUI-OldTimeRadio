# VIDEO_LANE_PREFLIGHT receipt -- lane 18, `still_word`

`VIDEO_LANE_PREFLIGHT receipt: still_word | 2026-08-11 | smoke receipt
output/otr/episodes/_lane_smokes/lane18_still_word/ | verdict PASS -- 7/7`

**The last cheap lane. With this, the whole cheap shelf -- four visualizers and
four still families -- is green.** And it is the lane where the right answer was
mostly "this is already done, verify it and say so".

## Matrix row

| Gate | Before | After | What moved |
|---|---|---|---|
| G1 | PASS | PASS | no weights |
| G2 canvas truth | **RED** | **PASS** | profile canvas channel declared INERT -- the last G2 row on the shelf |
| G3 | PASS | PASS | already green -- lane 10's shared-base fix |
| G4 / G6 | n/a | n/a | exempt -- CPU/ffmpeg lane |
| G5 | PASS | PASS | already probes its own emitted mp4 |
| G7 | PASS | PASS | matrix unchanged (nothing declared) |

Pre-lane `scripts/build_variants.py --check`: **46 variants, 0 failures**.

## Two of the three assigned items were ALREADY CLOSED -- verified, not rebuilt

The corpus assigns this lane: *"Preserve its existing missing-still refusal,
add/verify the ffmpeg and single-authority contract, and close its own row and
smoke."* Lane 14's rule is to run the acceptance check BEFORE implementing an
assigned defect, because a spec written weeks earlier may describe a hole
someone has since filled -- and re-implementing is how a second, divergent copy
of a guard gets born.

Ran it. Both were closed:

| item | state | closed by |
|---|---|---|
| missing-still refusal | `_require_still = True` | **Sprint B, 2026-07-03** -- this family was FIRST |
| ffmpeg PREFLIGHT gate | refuses by name | **lane 15's shared-base fix**, inherited |

So nothing was written for either. What WAS added is the acceptance check
itself, kept as a test on the lane that owns the contracts
(`test_lane_18s_two_assigned_defects_were_ALREADY_CLOSED`), exercising both
rather than reading flags: the refusal is fired through `render_clip`, and the
gate is fired with `find_ffmpeg` stubbed empty.

**"Already done" is only a safe conclusion if something now holds it that way.**
Before this lane, the ffmpeg gate had no test on `still_word` at all -- it was
inherited behaviour nobody asserted here.

## G2 -- INERT, the last row on the shelf

One profile sets `render.canvas_w/h` on this lane. It renders through
`ffmpeg_still_static_cmd` like `still_flat` (`_still_motion = False`): the
caller's width/height, scale to FIT plus pad, nothing cropped. No native canvas;
the `even_dim()` snap is a yuv420p mod-2 codec requirement.

Declaring would overrule `OTR_VIDEO_LANDSCAPE_CANVAS` for this lane alone -- and
it would be **especially** wrong here, because a word card's whole job is
legibility at whatever size the deliverable actually is. Pinning it to one
canvas is the opposite of what a title card needs.

## The builder partition, completed -- four lanes, two builders, two digests

Lane 16 showed two lanes sharing a builder produce byte-identical output. Lane
17 showed a different builder produces different output. This lane closes the
partition, and all four smokes were rendered from the same still at the same
canvas and frame count:

| lane | engine | builder | sha256 (first 16) |
|---|---|---|---|
| 15 | `still_motion` | `ffmpeg_still_motion_cmd` (cover+crop) | `3692f155b93b5f87` |
| 16 | `still_pan` | `ffmpeg_still_motion_cmd` | `3692f155b93b5f87` |
| 17 | `still_flat` | `ffmpeg_still_static_cmd` (fit+pad) | `56d48f215d58868c` |
| 18 | `still_word` | `ffmpeg_still_static_cmd` | `56d48f215d58868c` |

**Two builders, two digests, exactly partitioned.** The G2 reasoning across all
four lanes rests on "these lanes share a render path and neither path has a
native canvas", and that is now demonstrated in both directions rather than
asserted -- shared paths match, distinct paths differ. It cost nothing: the
renders already existed.

It also says something the code alone does not: `still_word` differs from
`still_flat` ONLY in the PROMPT its still was minted from, and its render is
byte-identical given the same still. That is exactly what its docstring claims
("the ONLY delta vs still_flat is the PROMPT"), now measured.

## The solo smoke -- LIVE PASS, two legs

Stock `default` boot; box reset per CLAUDE.md section 4 first.

| | LEG A -- real still | LEG B -- the REFUSAL |
|---|---|---|
| Harness | `--engine still_word --frames 100 --portrait <png>` | `--engine still_word --frames 100 --expect-fail "requires a base still"` |
| Prompt id | `430452e2-b15a-417d-8e65-09e9c5a462ae` | fail-closed, NAMED |
| Canvas PROBED | **832x480** | n/a |
| Frames PROBED | **100** exactly | n/a |
| Rate / codec | 25/1, h264, yuv420p | n/a |
| Audio | **zero audio streams** | n/a |
| sha256 | `56d48f215d58868c924b9346c175b90b03ecfd3fa260f63392bff1b638393921` | n/a |

Leg B is Sprint B's refusal, from 2026-07-03, fired on the live server. It has
presumably been correct for six weeks; this is the first time it has been
PROVED in production shape rather than in a unit test.

## The cheap shelf, finished

| lane | engine | G2 closed by | G3 |
|---|---|---|---|
| 11 | `viz_green` | channel INERT | declared |
| 12 | `viz_camera` | channel INERT | declared |
| 13 | `viz_mxc_cpu` | channel INERT | declared |
| 14 | `viz_mxc_mandala` | channel INERT | declared |
| 15 | `still_motion` | channel INERT | inherited (lane 10) |
| 16 | `still_pan` | channel INERT | inherited |
| 17 | `still_flat` | channel INERT | inherited |
| 18 | `still_word` | channel INERT | inherited |

**Eight for eight on INERT.** No cheap lane declares a `render_canvas`, and that
is a finding rather than a habit: none of them has a native canvas, and every
one of them would have lost the `OTR_VIDEO_LANDSCAPE_CANVAS` operator lever by
declaring one (L19). A future procedural lane that wants to declare owes the
argument explicitly.

## Deliberately NOT done here

**No VRAM number, no cost row** -- CPU/ffmpeg lane, G4 exempt.
**`still_plan` still audit-only** -- unchanged since lane 15 recorded it (S8b-15,
lesson L6). A green G7 row does NOT mean the plan is wired.
**No profile, variant, workflow or `ENGINE_MATRIX.md` change** -- nothing declared.
