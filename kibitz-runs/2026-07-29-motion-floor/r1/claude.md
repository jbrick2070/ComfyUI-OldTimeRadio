# Kibitz R1 -- Motion Floor, 2026-07-29
**Reviewer:** Claude (Sonnet 4.6), grounded against real Windows files via Read tool.
**Review type:** High-level arc / creative coherence (pre-build adversarial).

---

## VERDICT

The design is substantially correct and the two bite points are real, but the document has
specification gaps that will force implementers to invent numbers rather than derive them.
WHERE IT BITES #2 is partially a phantom: `allow_tail_trim=True` already routes case 1
through the tail-trim path for wan_ti2v, ltx, and humo -- the new ruling does not change
their code behavior, it only provides the philosophical rationale that was previously
implicit. The genuine open questions are: (a) what is N for the credits tail cut and what
happens when the body is shorter than N; (b) whether ping-pong fill satisfies the motion
floor for WIRE-W3b (the code says yes; the document leaves it open when it should close it).
Close those two gaps in writing before any code is touched.

---

## MUST-FIX BEFORE BUILD

### M1. Declare N for the credits tail cut, with a derivation and a floor.

**Claim:** The document proposes "cut the last N seconds of the assembled body video as a
real clip and loop THAT under the console" but never declares N, its derivation, or what
happens when the body is shorter than N. Without this, the implementer invents a number.

**Evidence:** `nodes/otr_credits_roll.py` has `_MAX_HOLD_S = 120.0`, `_LEAD_HOLD_S = 3.0`,
`_TAIL_HOLD_S = 4.0`, and `compute_credits_duration_s` drives the roll from scroll length.
None of these constants constrain N. The assembled body from `otr_silent_composite.py` is
always a complete mp4 (guaranteed by ffmpeg flatten), but its duration is not bounded below
by any constant in the visible interface -- a very short episode body is legal.

**Specific ask:** Add to the document, before any chunk list:
- A constant `_BACKDROP_CLIP_S` (suggested: 8.0 s) with a rationale (long enough for the
  roll to not feel repetitive, short enough to always exist in a normal episode body).
- Derivation rule: `clip_s = min(body_duration_s, _BACKDROP_CLIP_S)`.
- Floor: if `body_duration_s < 1.0` this is a terminal error (the body must have content;
  a 0-duration body means `otr_silent_composite` itself failed, which is already a terminal
  path). If `1.0 <= body_duration_s < _BACKDROP_CLIP_S`, use all of body -- loop a short
  clip rather than refusing.
- The loop count is then `ceil(credits_duration_s / clip_s)` and is knowable at render time
  (the document already notes this).

This keeps the failure boundary from r4/A7: a body that exists but is short is presentation
(loop a short clip); a body that does not exist at all is terminal (already `CreditsDataError`
on the old path; this preserves the same terminal status under the new path).

---

### M2. Close the ping-pong policy ruling in the document so WIRE-W3b has a spec.

**Claim:** The document frames the ping-pong question as open ("does the new ruling permit
ping-pong fill, forbid it, or bound it?") and defers to WIRE-W3b. But the code already
implements a specific behavior, and "is this behavior permitted" is a policy ruling that must
be made in the document before W3b is specced.

**Evidence:** `nodes/_otr_video_engines/eng_wan_ti2v.py` lines 725-733 (`render_clip`):

```python
target_frames = int(plan.get("target_frame_count") or 0)
n_native = len(frames)
if target_frames > n_native:
    frames = _wb.extend_frames_to_target(frames, target_frames)
    _LOG.warning(
        "[OTR video] wan_ti2v CLIP-FILL: rendered %d frame(s) -> "
        "ping-pong extended to %d (beat target %d) @ %dx%d so the beat "
        "is FILLED with motion (no hold-last-frame freeze)",
        ...)
```

Ping-pong triggers when VRAM forces a shorter native render than the coverage plan requests
(the 8 GB tier path, PBUG-20260723-02). Native render is always `>= _TI2V_MIN_FRAMES = 17`
(the VRAM floor enforces this). Ping-pong then extends to the beat target.

The motion-floor ruling says "video for every beat." Ping-pong produces moving pixels -- not
a held frame. The prior accepted production baseline for the credits backdrop was a looped
drama clip, which is also repeated motion. The standard "repeated motion is acceptable" is
already in force. There is no logical reason ping-pong fill of a VRAM-limited native render
is less honest than a looped drama clip.

The document should add, explicitly:
- Ping-pong fill of a VRAM-bounded native render IS permitted under the motion floor because
  it produces moving pixels, not a still.
- Ping-pong is NOT permitted as a substitute for a multi-clip plan: if `coverage_plan`
  schedules CHAIN or JUMP mode (multiple segments), the engine must render multiple clips,
  not ping-pong a single short render to fill the beat. This is already the existing
  `coverage_plan` constraint; the document should confirm it is preserved.

Without this ruling written down, W3b has no spec and any implementer will re-open the
philosophical debate mid-build.

---

### M3. WHERE IT BITES #2 is only a real bite for discrete-menu engines -- say so.

**Claim:** The document implies case 1 (target below `min_frames`) is the general case that
changes character under the new ruling. For trim-capable engines it is already implemented
and requires no code change. Conflating "the ruling changes" with "the code changes" will
cause the implementer to touch code that does not need touching.

**Evidence:** `nodes/_otr_video_engines/coverage_plan.py` `join_mode_for` (lines 166-171):

```python
if contract.allow_tail_trim \
        and contract.smallest_legal_at_least(target) is not None:
    return JOIN_SINGLE
```

And the single-clip path (lines 292-297):

```python
if contract.allow_tail_trim:
    render = contract.smallest_legal_at_least(target)
    if render is not None:
        return CoveragePlan(target, mode,
                            (CoverageSegment(0, render, trim_tail=render - target),))
```

`eng_wan_ti2v.py` declares `FrameContract(min_frames=17, ..., allow_tail_trim=True, ...)`.
So for a beat target of, say, 12 frames (0.48 s at 25 fps), `smallest_legal_at_least(12)`
finds the smallest legal value >= 12 given min=17, quantum=4 -- which is 17. The plan says
render 17 frames, trim 5. Real video, no still. The ruling is already satisfied.

Same logic applies to ltx (min 9) and humo (min 33, also likely `allow_tail_trim=True` --
this needs a one-line grep to confirm but the pattern is consistent across the engines that
have a defined `FrameContract`).

The genuine open case is discrete-menu engines where `allow_tail_trim` may be False:
Veo/Pixverse at 100 frames (4.0 s minimum). For those engines, if a beat target is below
100 frames, `smallest_legal_at_least(target)` may return 100 anyway IF `allow_tail_trim` is
True (render 4s, trim to beat). The document should declare: "For discrete-menu engines,
if `allow_tail_trim` is False and the smallest legal choice exceeds the beat target, the
correct fix is to set `allow_tail_trim=True` in the FrameContract, not to introduce a still."

A single grep for `FrameContract` instantiations across `nodes/_otr_video_engines/` will
reveal which engines have `allow_tail_trim=False` and whether any of them are live production
engines that would fire on a real beat. Do this grep before writing the chunk list.

---

## SHOULD-FIX

### S1. Answer "is a loop honest?" in the document and close the question.

The document raises "is a LOOP of the last N seconds honest, or is repeated motion its own
lie over a 52 s roll?" This is already answered by the prior production baseline: the old
credits backdrop was a looped drama clip -- repeated motion over the same roll. The operator
did not object to that; the objection was to a STILL for 52 seconds, not to looping. A loop
of the body tail is at least as honest as a looped drama clip and more coherent (same pixels
the audience just saw). Write "yes, a loop is honest; the prior baseline was also a loop"
and remove the open question so it does not resurface as a review comment mid-build.

### S2. Confirm the presentation-only failure boundary from r4/A7 explicitly for the new path.

The document asks "does the presentation-only failure boundary from r4/A7 still hold if the
backdrop cut can fail?" The answer is yes, because:
- The only new failure mode introduced by the tail-cut approach is "body video unreadable" --
  which maps to the same terminal category as `CreditsDataError` already was.
- A body video that exists but is short (body_duration_s < _BACKDROP_CLIP_S) maps to
  presentation: loop a shorter clip, the roll still plays.
- The document should state this mapping explicitly so the implementer knows where to
  classify each failure path in the error hierarchy.

### S3. Audio-in stills -- confirm they are not touched by this ruling and close that question.

The document asks "does this collide with the audio-in rulings?" and correctly states that
those stills are init images for video renders, not substitutes. But it hedges with "confirm
that reading is right." The reading IS right -- an init image is an INPUT to a video render
that produces moving frames; it is not a still substituting for video. WIRE-W7 (mouth-still
ownership) is open, but the motion floor does not change WIRE-W7's scope. Add one sentence
to the document: "Audio-in stills are init images and are untouched by this ruling. WIRE-W7
scope is unchanged." This prevents W7 from getting re-scoped mid-build by someone reading
this document.

---

## OPTIONAL

### O1. The roll play-once-then-hold approach.

"Last N seconds played once then cross-faded to a slow drift" is out of scope for this
ruling. The operator said "video for every beat" -- a looped clip satisfies that. A
slow-drift cross-fade is a production quality enhancement, not a motion-floor fix. Log it
as a future idea and cut it from the build spec.

### O2. Loop count derivation.

The document notes loop count is knowable at render time (`ceil(credits_duration_s / clip_s)`).
This is a nice-to-have sanity check in the implementation, not a design decision. Mention it
in the implementation chunk as a log line, not in the spec.

---

## CUT THESE

### C1. "Is repeated motion its own lie over a 52 s roll?"

This question is answered by the prior production baseline (looped drama clip = repeated
motion over the same 52 s roll, accepted). Cut it. It adds no value to the design and will
be re-litigated by every reviewer who reads the document.

### C2. "Several short stills? A refusal?" for beats longer than a still bound.

The motion floor answers this: if the engine can render, it renders. The still is bounded to
the minimum legal render plus a trim. There is no "beat longer than the still bound" scenario
under the new ruling -- either the engine renders (trim-capable) or the existing
`CoveragePlanError` raises (true impossibility). The "several short stills" alternative does
not exist under the motion floor. Cut the question.

### C3. "A longer slice" as an alternative to N seconds for the credits backdrop.

If "longer slice" means a larger N, that is subsumed by declaring N with a rationale (M1).
If it means something architecturally different, it is out of scope. Cut the open-ended
alternatives list and replace with a single declared value for N plus the derivation rule.

---

## Summary table for the chunk list

| # | Where | Change needed | Blocking? |
|---|-------|---------------|-----------|
| M1 | Spec / `otr_credits_roll.py` | Declare `_BACKDROP_CLIP_S`, derivation rule, floor/terminal split | YES |
| M2 | Spec / WIRE-W3b section | Write ping-pong policy ruling (permitted as VRAM-fill, not as multi-clip substitute) | YES |
| M3 | Spec / WHERE IT BITES #2 | Scope case 1 correctly to discrete-menu engines only; grep FrameContract allow_tail_trim status | YES |
| S1 | Spec | Answer "loop is honest" and close the question | No |
| S2 | Spec | Confirm presentation-only boundary mapping for new backdrop path | No |
| S3 | Spec | Confirm audio-in stills are untouched; close WIRE-W7 scope question | No |
| O1 | Spec | Cut "slow drift" alternative; log as future idea | No |
| C1-C3 | Spec | Remove open questions that are already answered | No |

No code needs to be read that has not already been read. The three must-fix items require
no new files -- they require a decision and a sentence written into the design document
before any implementation begins.
