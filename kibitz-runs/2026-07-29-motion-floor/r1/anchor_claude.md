# r1 ANCHOR -- Claude, code-grounded, written BEFORE the panel returns

VERDICT: the ruling is right and the brief under-claims how much of it the
tree ALREADY does. The credits half is a real regression I introduced. The
beat half is probably a NO-OP in code and a DOC fix -- and if that is right,
the dangerous outcome of this arc is a window "implementing" a motion floor
that `coverage_plan` has enforced since 2026-07-25.

## CONFIRMED -- read in the files today

1. **The credits duration is known before the backdrop is built.**
   `compute_credits_duration_s(roll_px, view_h)` returns
   `_LEAD_HOLD_S + roll_px/pps + _TAIL_HOLD_S`, capped at `_MAX_HOLD_S` by
   speeding `pps` up rather than truncating. So "how long must the backdrop
   last" is answerable at render time, and any loop count is derivable rather
   than guessed. (`otr_credits_roll.py`, `compute_credits_duration_s`.)

2. **Restoring a MOVING backdrop is a two-token change, not a redesign.**
   `render_credits_clip` feeds input 0 through
   `scale=...,crop=...,eq=brightness=-0.32,fps=...`. That chain is identical
   for a still or a clip. My WIRE-W6 change swapped ONLY the input flags:
   `-stream_loop -1 -i <clip>` became `-loop 1 -framerate <fps> -i <png>`.
   Going back to motion is swapping them again -- the filter graph, the
   overlays, the fades and the col-3 scroll are untouched.

3. **The body video is a legitimate source for a moving backdrop.**
   `otr_silent_composite` flattens every directory clip into the assembled
   body, and `roll()` already validates the body exists and probes it
   (`_probe_video`) before anything else. Cutting a tail from it needs no new
   authority and keeps the manifest read deleted -- which is what let an
   all-`mesh_stage` episode publish. The mesh_stage fix and the motion floor
   are NOT in tension.

4. **THE BIG ONE: "render the minimum and trim" is already the shipped
   behaviour, and it is not a still.** `join_mode_for` returns `JOIN_SINGLE`
   when `contract.allow_tail_trim and contract.smallest_legal_at_least(target)
   is not None` -- i.e. a beat SHORTER than an engine's `min_frames` renders
   the smallest legal length and trims the surplus. That is exactly the
   operator's "if the minimum is four seconds, we should have video for four
   seconds", and `partition_beat` has done it since 2026-07-25.

5. **...and essentially every engine opts in.** `allow_tail_trim=True` is
   declared on both HuMo contracts, both WAN, `ltx_video`, `ltx_av`,
   `ltx_8gb`, all four `viz_*`, `cheap_families`, both Google lanes and five
   `eng_cloud_video` rows. I did not find a live video engine that declares
   it False.

   **Consequence the panel should test hard:** case 1 of the 2026-07-28
   still-floor ruling ("target frames < the engine's `min_frames`") may be
   UNREACHABLE for every shipped video engine. If so, the ruling's still
   floor has no live caller and this arc's beat half is documentation, not
   code.

6. **The audio-in stills are orthogonal and must not be swept up.** The
   2026-07-28 rulings (every audio-in beat gets a still with a mouth; the lips
   may be a person or a radio) are about the INIT IMAGE a video render
   conditions on -- `render_driver` feeds it to LTX/HuMo i2v. They are not a
   substitute for video and this ruling does not touch them. A panel that
   "simplifies" them into the motion floor is wrong.

## MUST-FIX in the brief

1. **The brief treats the beat half as new work. Prove it is not first.**
   The first chunk of any plan here is a READ-ONLY audit: for every registered
   video engine, does a target below `min_frames` reach a still, or does it
   reach a trimmed minimum render? Write it as a roster test that fails BY
   NAME for any engine that can still reach a long still. If the audit comes
   back clean, the beat half ships as a test + a doc correction and nothing
   else -- which is the cheapest possible outcome and the one I expect.

2. **The brief asks "is a loop honest?" without naming the alternative that
   costs nothing.** The old behaviour looped the last drama CLIP. The
   candidate loops a tail cut from the BODY. A third option exists and is
   strictly better on this canvas: play the tail FORWARD, then loop, so the
   first pass is genuine motion and repetition only starts once the viewer is
   reading col 3. Cheap in ffmpeg, and it makes the honest half of the roll
   the half that plays first.

3. **N is not free.** A tail cut needs a length. The brief should pin it to
   something derived -- the roll duration is known (fact 1), so N can be
   "the whole roll if the body is long enough, else the whole body looped".
   A hard-coded N is the kind of constant this build keeps paying for.

## SHOULD-FIX

- The ti2v ping-pong question is the one place the ruling genuinely bites
  code, and it is already WIRE-W3b's blocker. Do not let the panel re-scope
  it; the answer only has to be "permitted / forbidden / bounded" for a
  coverage-planned segment, because `coverage_plan` already forbids it there
  and the 8 GB tier already depends on it everywhere else.

## UNVERIFIABLE from here -- flag as verify-at-build

- Whether a looped tail READS better than a held frame at 832x480 under a
  52 s scroll. No panel can answer this; it is an operator eyeball, and the
  plan should say so rather than assert a taste outcome.
- Whether any CLOUD lane's `min_frames` (Veo 100 frames = 4 s) makes a short
  beat pay for 4 s of provider time it then trims. That is a spend question,
  not a correctness one, and the cloud lanes are parked pending spend approval.
