# THE MOTION FLOOR -- operator direction 2026-07-29, and what it costs

## The ruling, in the operator's words

> "For video models, there needs to be video for every beat. We can't have a
> still for fifty-two seconds. Maybe a still for two seconds or three or four,
> but if the minimum is, like, four seconds, then we should have video for four
> seconds."

Read as three claims, in force from now:

1. **Every beat a VIDEO model renders is covered by real moving video.**
2. **A still is a BOUNDED exception -- seconds, not scenes.** Somewhere around
   2-4 s is tolerable; 52 s is not.
3. **Where an engine CAN render, it renders.** If the adapter's minimum legal
   render is ~4 s of frames, produce 4 s of real video rather than freeze a
   still for that span. A still is for where the model cannot go, never for
   where it merely was not asked.

This SHARPENS the 2026-07-28 still-floor ruling ("a still floor is legal ONLY
where the partition math is impossible, never where an engine refused"). That
one said WHEN a still is allowed. This one bounds HOW LONG it may last and
says that a legal minimum render outranks a still even inside the exception.

## WHERE IT BITES #1 -- the credits backdrop (this is the live regression)

`a14ecdfa` (WIRE-W6) changed the credits console backdrop from **the last drama
clip LOOPED** to **the body video's final frame HELD**, for the whole roll --
measured at 52.0 s on the last live leg. That is precisely the 52-second still
the operator is refusing. It is on HEAD now.

**Why it was changed, because the reason still has to hold after the fix.**
`plan_backdrop` searched the CLIP MANIFEST for the last loopable *file* clip
and raised `CreditsDataError` when it found none. `mesh_stage` writes a frame
DIRECTORY rather than an mp4, so an episode rendered entirely by `mesh_stage`
had no file clip at all and the TERMINAL node of the graph refused a finished
episode -- 7 of 7 shots rendered, nothing published. True since 2026-07-03,
found by the 2026-07-28 campaign. Any fix must keep that closed.

**The candidate to attack:** cut the last N seconds of the ASSEMBLED BODY VIDEO
as a real clip and loop THAT under the console. `otr_silent_composite` has
already flattened every directory clip into the body, so the body is a
complete, always-present, always-mp4 record of the same pixels and it is
already `OTR_CreditsRoll`'s own input. That keeps the motion, keeps ONE source
(no manifest read, so `mesh_stage` still publishes), and needs no new authority.

Open questions for the panel:

- Is a LOOP of the last N seconds honest, or is repeated motion its own lie
  over a 52 s roll? Is there a better tail (last N seconds played once then
  cross-faded to a slow drift, a longer slice, a reverse-and-forward)?
- What is N, and is it derived or declared? The roll's duration is
  `compute_credits_duration_s`, which is scroll-length-driven and capped by
  `_MAX_HOLD_S` -- so N and the loop count are knowable at render time.
- What happens when the body is SHORTER than N?
- Does the presentation-only failure boundary from r4/A7 still hold if the
  backdrop cut can fail? (Terminal = unreadable body; presentation = anything
  that merely makes glass.)

## WHERE IT BITES #2 -- the still floor on a beat

`coverage_plan.partition_beat` refuses rather than drifting, and the
2026-07-28 ruling allows a still ONLY in the three arithmetically-impossible
cases: target below `min_frames`; a cover needing more segments than the
ceiling; a `discrete_frames` menu with no exact cover.

Under the new ruling, case 1 is the one that changes character. A beat whose
target is BELOW an engine's `min_frames` is exactly the case where the engine
CAN still render -- it just renders LONGER than the beat asked for. So the
honest answer looks like "render `min_frames` of real video and trim to the
beat" rather than "hold a still", which is what `allow_tail_trim` already
does at the single-clip boundary.

Questions for the panel:

- Is any still floor left at all after this, or does "render the minimum and
  trim" absorb case 1 entirely? Which engines actually have a `min_frames`
  large enough for this to matter? (`humo` is 33 @ 25 fps = 1.3 s;
  `wan_i2v` 33; `ltx_8gb` 9; the Veo/Pixverse DISCRETE menus are the big ones
  -- 100 frames = 4 s minimum.)
- Where a still IS unavoidable, what BOUNDS it, and what happens when the beat
  is longer than the bound? Several short stills? A refusal?
- Does this collide with the audio-in rulings (every audio-in beat gets a still
  with a mouth; the lips may be a person OR a radio)? Those stills are INIT
  IMAGES for a video render, not a substitute for video -- confirm that reading
  is right, because if it is, those rulings are untouched by this one.
- `eng_wan_ti2v` floors its render to what live VRAM affords and PING-PONG
  EXTENDS it to the beat length. That is motion, but it is repeated motion from
  a short native render. Does the new ruling permit ping-pong fill, forbid it,
  or bound it? It is load-bearing for the shipped 8 GB tier
  (`PBUG-20260723-02`) and `coverage_plan` already forbids it for a planned
  multi-clip beat -- so the answer decides WIRE-W3b.

## What is already built, so the panel grounds against the right tree

HEAD `f6977e3d` on `v2.0-alpha`. Landed this session: WIRE-W1 `5efd2baf`
(partition takes the fewest legal clips), WIRE-W2 `a218b1f7` (typed cast-time
image gap), WIRE-W6 `a14ecdfa` (the change under review), WIRE-W3a `3e89d6b2`
(wan_i2v beat session). Suite 7561 / 27 skipped / 1 xfailed; Bible 17.

Still open in the wiring block: WIRE-W3b (wan_ti2v session + the ping-pong
question above), WIRE-W4 (HuMo session + per-segment audio slicer), WIRE-W7
(mouth-still ownership), WIRE-W5 (acceptance grader). None of it is
live-proven; the 45-word run over all 18 local engines is the real proof.

## What the panel is being asked for

A build-ready answer to: **what does "video for every beat, and no long
stills" mean concretely in this tree** -- which files change, in what order,
what stays untouched, and what the acceptance is for each. Not a philosophy;
a chunk list with the same shape as the r3/r4 arc this block is already
executing.
