# Problem statement: frames, timing, and beat matching in Ghost Signal

**Date:** 2026-08-22
**Status:** problem statement. Nothing is built, no constant is changed, and no
graph change is authorised by this document.
**Scope:** the Ghost Signal video lane's cadence. `workflows/otr_canonical.json`
is untouched. The golden lane (`animatediff15_video`) is untouched.

## Where this came from

The operator, watching a published Ghost episode:

> the animation of animatediff is heavily fast paced, you could probably bring
> it down to 5fps and might even look better, it goes really fast

> like stop action animation

> it turns old old school OTR stills and slow moving LTX into a wild
> rollercoaster -- and the cool thing, it does seem to match the beat audio

Three observations, and the third is the one that constrains every answer to the
first two. **The beat matching is not a coincidence and it must survive whatever
we do here.**

---

## 1. What the lane does today, exactly

Every number below is read from `nodes/_otr_video_engines/eng_ghost_signal.py`
at commit `87bb547d`.

| Constant | Value | Meaning |
|---|---|---|
| `GHOST_TARGET_FPS` | 25 | delivered frame rate |
| `GHOST_SOURCE_FPS` | 12.5 | rate of FRESH generated frames |
| `GHOST_CONTEXT_LENGTH` | 16 | AnimateDiff sliding window |
| `GHOST_CONTEXT_OVERLAP` | 4 | window overlap, so **stride = 12** |
| `GHOST_CONTEXT_FUSE_METHOD` | `pyramid` | how overlapping windows are blended |
| `GHOST_CONTEXT_USE_ON_EQUAL_LENGTH` | `False` | at exactly 16, the module runs DIRECTLY -- no sliding |
| `GHOST_SOURCE_FLOOR` | 16 | a short beat still gets a full window |
| `GHOST_CANVAS_W/H` | 512x288 | render canvas, delivered at 1920x1080 |
| `GHOST_STEPS` / `GHOST_CFG` | 20 / 8.0 | sampler; CFG 8 keeps the negative live |

The cadence chain, in order:

1. A beat's **audio duration** sets `target_frame_count` (**T**), the delivered
   frame count at 25fps. See `render_driver.py:1949` -- `ShotRow` is
   `extra="forbid"` and carries no `start_s`/`dur_s`, so T is the single
   duration authority.
2. `U = ceil(T / 2)` -- the FRESH frames actually generated
   (`ghost_unique_source_count`).
3. `source_request = max(U, 16)` -- the structural floor
   (`ghost_source_request`).
4. Hold-2 selector `[0,0,1,1,...][:T]` expands U back to T; `tail_trim = 2U - T`
   is always 0 or 1.
5. Ghost declares `max_frames=0` -- **unbounded**. Nothing splits a beat.

**This is why the audio matches.** One AnimateDiff timeline is generated to span
one beat's whole audio budget. The gesture and the line start and end together
by construction, not by tuning. Any proposal that breaks this is dead on
arrival.

---

## 2. The finding: motion character is an accident of line length

`GHOST_CONTEXT_USE_ON_EQUAL_LENGTH = False` with a 16-frame window and stride 12
means a beat traverses `1 + ceil((U - 16) / 12)` sliding windows, each
contributing its own motion, pyramid-fused into its neighbours.

| Beat length | T | U (hold-2) | **windows traversed** |
|---|---|---|---|
| 1.0s | 25 | 16 | **1** -- module runs directly, one clean gesture |
| 2.0s | 50 | 25 | 2 |
| 4.0s | 100 | 50 | 4 |
| 8.0s | 200 | 100 | 8 |
| **12.0s** | **300** | **150** | **13** |

**Real episodes sit at the bottom row.** The 2026-08-22 bakeoff legs ran
2444-2944 delivered frames across 8 beats -- roughly **12 seconds per beat**, so
roughly **13 fused motion windows in every beat.**

So the "wild rollercoaster" is quantified: a short line gets ONE coherent
AnimateDiff gesture, and a long line gets thirteen of them fused end to end.
Nobody chose that. It falls out of how long the writer's sentence happened to be.

**This is the actual problem.** It is not "the fps is too high" -- it is that
motion energy per beat is an uncontrolled function of dialogue length, and the
show's beats are long.

---

## 3. The levers, and what each one costs

### 3.1 The hold factor (`U = ceil(T / hold)`)

Currently 2, as a module-level assumption rather than a per-lane class
attribute. Changing it changes U, therefore the window count, therefore how much
motion is traversed in a fixed duration.

| Hold | Fresh fps | U at a 12s beat | Windows | Render cost |
|---|---|---|---|---|
| 1 | 25.0 | 300 | 25 | 200% |
| **2 (today)** | **12.5** | **150** | **13** | **100%** |
| 3 | 8.33 | 100 | 8 | 68% |
| **5 (operator's suggestion)** | **5.0** | **60** | **5** | **40%** |

**T never moves, so the audio sync is untouched at every value.** What changes is
motion distance traversed (slower) and temporal resolution (steppier). The
operator has already said the stepped look reads as stop-action and might be
*better* -- for a 1930s radio drama that may be a feature rather than a defect.

Cost note: hold-5 makes a 27-minute leg roughly 11 minutes. That is a real
side-benefit but it is NOT the reason to do it. Recipes here are not traded for
speed.

### 3.2 Context length and overlap

`context_length=16` is the pinned upstream SD1.5 window; `overlap=4` gives
stride 12. Raising the overlap raises coherence between windows and the window
count; the pyramid fuse method is the blend. None of these have been swept.

### 3.3 The source floor

`max(U, 16)` means very short beats generate surplus frames that are discarded
before cadence conversion. Already accounted for honestly in
`ghost_cadence_receipts` (`model_frame_count` vs `cadence_source_frame_count`).
Not a defect; listed for completeness.

### 3.4 Frame interpolation

Not currently used anywhere in OTR, and no RIFE/VFI custom node pack is
installed. **`ffmpeg minterpolate` IS available in the existing toolchain**
(confirmed in `ffmpeg -filters`), as are `framerate` and `tblend`. This is the
only lever that can deliver *slower motion AND smoother playback at once* --
every other option trades one for the other.

---

## 4. What is NOT known

Stated plainly so no proposal pretends otherwise.

* **AnimateDiff SD1.5's native training frame rate is not documented upstream.**
  The `guoyww/AnimateDiff` README states no fps, and the ADE pack hardcodes
  none. The commonly repeated figure is 8fps, but this build has NOT verified
  it, so no proposal may rest on it as fact. If it matters, it is measurable:
  render the same beat at hold-2/3/5 and look.
* **No cadence value here has ever been swept and judged by eye.** Hold-2 was
  chosen because 12.5 into 25 is exact integer arithmetic with no resampling --
  a correctness argument, not an aesthetic one.
* **Whether the window count is what the eye actually reads** as "fast" has not
  been isolated from the other things a longer beat changes.

---

## 5. Constraints any answer must respect

1. **The beat/audio sync survives.** T is set by audio and nothing may change
   that relationship.
2. **The canvas does not move.** 512x288 render, 1920x1080 delivery. The
   operator has ruled on this directly: *"i don't want to mess with canvas."*
3. **The golden lane is untouched.** It rendered the published episode.
4. **Additive, as a peer.** The pattern is settled: a class attribute a sibling
   overrides, per preflight G1.3. A cadence peer should cost about ten lines.
5. **Every leg publishes to `otr/obs/`.** A leg that does not reach it did not
   pass.
6. **No new dependency without a reason** -- `minterpolate` is free, a RIFE pack
   is an install.

---

## 6. Open questions worth ideas

1. Should the hold factor be **fixed per lane**, or **derived from beat length**
   so that a 12-second line and a 2-second line traverse a comparable number of
   windows? The second is more work and would make motion energy consistent
   across an episode for the first time.
2. Is the stop-action look a **defect to fix** or a **house style to commit to**?
   The operator's own reaction suggests the latter is live. If it is the style,
   the answer is hold-4/5 and no interpolation at all.
3. If we want slow AND smooth, is `minterpolate` good enough at 512x288, or does
   its warping artefact cost more than the stepping it removes?
4. Does raising `context_overlap` above 4 reduce the seam-to-seam jolt on long
   beats more cheaply than reducing the window count?
5. Is there a case for **capping beat length** upstream instead -- splitting a
   12-second line into two shots -- so no beat ever traverses 13 windows? This
   is a shot-planning answer rather than a cadence one, and it interacts with
   the audio slicing.
6. Should the cadence receipts already emitted (`ghost_cadence_receipts`) also
   record the **window count**, so this is visible in a receipt rather than
   recomputed by hand?

---

## 7. The cheapest first step

One episode, one style, one bank, rendered at hold-2 (today) and hold-5, and
looked at. It costs about 38 minutes of GPU total because the hold-5 arm is 40%
of a normal leg, and it answers questions 1 and 2 directly by eye -- which is the
only instrument that has ever settled a look question on this project.
