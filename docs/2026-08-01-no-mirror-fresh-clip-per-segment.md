# Kill shot-and-mirror: a FRESH render for every second of audio

**Operator decision, 2026-08-01 (explicit, twice):** "a fresh clip of video for
every segment of audio and not re-using video / ping-pong -- will make it more
consistent for all video models." No mirror fill. Bite the render cost.

This document is the plan for that, plus the blocker that makes it affordable or
punishing. Everything is measured on this box.

---

## 1. WHAT HAPPENS TODAY

A beat's length comes from ITS OWN audio (`otr_shot_lock.py:572`,
`frame_at = (cum*fps)//sample_rate`). Real beats measured on the shipped FastWan
episode: **8-21 s = 200-530 frames at 25 fps.** One render can produce at most the
tier cap (81 frames = 3.24 s for `fastwan_8gb`).

Two paths exist and the engine picks between them in `render_clip`
(`eng_wan_ti2v.py:994-1009`):

* **multi-clip** -> `_planned_length` -> render each planned segment WHOLE, join.
  This is the behaviour the operator wants.
* **single-clip** -> `_floor_length` -> render short, then
  `wrapper_bridge.extend_frames_to_target` PING-PONGS it (mirror cycle
  `[0,1,..,N-1,N-2,..,1]`, period `2N-2`) up to the beat length. This is the
  behaviour the operator is rejecting.

Which path runs is decided by `multi_clip` in `session_ctx`, which comes from the
STAMPED PLAN's segment count -- and the planner only narrows a beat for engines
listed in `frame_contract.PLANNING_CAP_ENGINES`.

**That list is `("ltx_8gb", "fastwan_8gb")`.** Every other chain-capable engine
therefore falls to render-short-and-mirror. That is the inconsistency across video
models the operator is seeing.

Chain capability (from `docs/ENGINE_MATRIX.md`, all verified):

| can chain | mode |
|---|---|
| `wan_ti2v`, `fastwan_8gb`, `wan_i2v`, `ltx_video`, `ltx_8gb` | `strict_first_frame` |
| `humo`, `ltx_audio_in` | `soft_reference` |
| `still_*`, `viz_*`, `mesh_stage` | `none` -- procedural, generate any length directly, never mirror |

**Every real video engine can chain.** There is no engine that structurally requires
the mirror.

## 2. THE BLOCKER: THE BEAT HOIST IS NOT HOLDING THE MODEL

Measured on the shipped `fastwan_8gb` episode (62 real renders, 7 beats):

    real renders (DMD sampler)          :  62
    "Requested to load"                 : 391      <- ~6 model loads PER render
    "unload_all"                        :  69
    "detached 0 resident model(s)"      :  65      <- the hoist held NOTHING

`BeatSession` exists precisely to load once per beat and keep the handle for that
beat's segments (`4fa992e6`, "chunk 5: the beat session -- one load per beat").
`MotionEngineBase._detach_patchers` walks `prepared["patchers"]`; detaching **0**
means that bucket was empty, so nothing was held and every segment reloaded.

**This is why "fresh per segment" currently looks expensive.** 62 renders paying ~6
loads each instead of 7 beats paying one each. Fix the hoist and the SAME 62 renders
get dramatically cheaper. Do not evaluate the mirror decision against today's churn.

Note the ordering: `session_ctx` was being DROPPED by `MotionEngineBase.prepare`
until `f5021f4b` (2026-08-01, fixed). That fix is in but UNVERIFIED on a live leg --
it may or may not be related to the empty patcher bucket.

## 3. THE MATH (what the operator asked me to own)

    segments_per_beat = ceil(beat_frames / tier_cap)
    each segment a legal 4n+1 rung, summing to the beat's frame count
    beat_frames = the beat's own audio duration x fps

Measured example, the shipped FastWan episode (cap 81):

| beat | seconds | frames | segments |
|---|---:|---:|---:|
| b001 | 17.68 | 442 | 6 |
| b002 | 8.92 | 223 | 3 |
| b003 | 10.48 | 262 | 4 |
| b004 | 10.48 | 262 | 4 |
| b005 | 21.20 | 530 | 7 |
| music_opening | 10.00 | 250 | 4 |
| music_closing | 8.00 | 200 | 3 |
| **total** | **86.8** | **2169** | **~31** |

The episode actually logged **62** renders, roughly double the arithmetic -- unexplained
and worth the panel's attention (retries? per-segment re-renders? the broken hoist
causing repeats?).

Audio-driven engines (`humo`, `ltx_audio_in`) already derive segmentation from the
audio itself. Non-audio engines derive it from beat frames, which came from that
beat's audio anyway -- so both land on the same number. There is no separate math.

## 4. THE PROPOSED CHANGE

1. **Fix the hoist** so one load serves a whole beat. Prerequisite; everything else
   is priced wrong until this is true.
2. **Add every chain-capable engine to `PLANNING_CAP_ENGINES`** with a per-engine
   cap, so the planner splits beats for all of them, not just two.
3. **Stop using `extend_frames_to_target` for coverage.** Keep it only where a beat
   genuinely cannot be split (if any such case survives), and make that case LOUD.
4. **Close the admission hole:** `_planned_length` never consults the VRAM predictor
   (`eng_wan_ti2v.py:795-805`), so planned segments render unchecked. Extract a pure
   `assert_frame_affordable(...)` and call it on BOTH branches.
5. **Scratch hygiene:** per-segment renders go to `episodes/_shared/tmp` with
   anonymous names (`otr_beat_<rand>.mp4`) and the janitor only runs AFTER a
   successful publish -- so failed legs leak. Measured now: **893 files, 5.9 GB**,
   never swept last night. More segments per beat multiplies this.

## 4B. THE CHAIN ALREADY EXISTS -- THIS IS A ROLLOUT, NOT A BUILD

The operator asked how a long beat's later segments get their first frame, and
whether new still generation is needed. Neither is a new problem: the design was
written on 2026-07-25 and the operator's requirement is quoted verbatim in
`coverage_plan.py:3-6`:

> "a beat be covered by enough REAL rendered clips to be moving video -- chain
> (segment N+1 begins on segment N's terminal frame) preferred, jump cut
> acceptable, and NEVER a mirror/ping-pong or a held last frame."

Wired, both halves:

* `render_driver.py:3288` -- `extract_terminal_frame(...)` pulls segment N's real
  last frame.
* `render_driver.py:3212` -- the successor's `init_source` becomes
  `"chain_terminal_frame"`, overwriting `asset_refs["init_image"]`.
* A chained successor **mints no still at all** -- `coverage_plan.jump_still_requests`
  returns nothing for a CHAIN plan. So NO new still generation is required.
* Seam arithmetic (`coverage_plan.py:16-20`): the successor drops its head frame so
  the shared frame is not shown twice --
  `sum(render_frames - drop_head - trim_tail) == target_visible_frames`.

So the mirror is not load-bearing and the replacement is not hypothetical. The task
is to make the coverage path REACHABLE for every chain-capable engine, not to invent
continuity.

## 5. WHAT THE PANEL MUST RULE ON

1. **Why is the patcher bucket empty?** 65 teardowns reporting "detached 0" against
   62 renders. Is the hoist not registering, not surviving, or not being reached?
   This is the load/unload churn the operator flagged, and it is the whole cost case.
2. **Why 62 renders for ~31 planned segments?** Double the arithmetic. Find it.
3. **Is adding 5 more engines to `PLANNING_CAP_ENGINES` safe?** Its own comment calls
   it "a deliberate allowlist of ONE, not a rollout" and warns that narrowing WAN's
   contract turns every beat into a pile of tiny renders. That warning was written
   against a cap of 17. Does it still hold at a real cap?
4. **What cap per engine**, given each has a different canvas and cost profile, and
   the caps must serve BOTH the planner and the render ceiling?
5. **Is there any beat that cannot be split** -- a minimum segment length, a
   continuity constraint, an audio-alignment constraint -- where removing the mirror
   leaves the beat short of its audio? The operator's hard bar is that every second
   of audio has video.
6. **Segment count vs quality:** does a 7-segment beat with `strict_first_frame`
   chaining actually look continuous, or does it visibly seam? A mirror is ugly in a
   known way; many joins may be ugly in a new way.

## 6. CONSTRAINTS

- Every second of audio gets video. Non-negotiable.
- No mirror / no re-used frames.
- `wan_ti2v`'s frozen RECIPE does not move.
- Fail loud, never silently degrade.
- The only workflow JSON is `workflows/otr_canonical.json`.
- 16 GB RTX 5080 laptop; models at `C:\ComfyUI-Models`.

---

## 7. STATUS CORRECTION, 2026-08-02 -- THE WAN MIRROR IS STILL LIVE

**Found by the kibitz r4 codex lane, and it is a FALSE CLAIM I made, not a new
defect.** My r2 and r3 kibitz `final.md` documents both ended with "INVARIANTS
HELD: ... no mirror/ping-pong/re-used frames". That was wrong. The path is live:

    eng_wan_ti2v.py:1070
        frames = _wb.extend_frames_to_target(frames, target_frames)
        extension_mode = "ping_pong"

The LTX boomerang was retired (01155e10) and the compositor loop-fill was
measured as a 1-frame artifact on one music beat. This third one was never
closed, and I should not have written that the invariant held.

### What is actually true, measured

The mirror is SINGLE-CLIP ONLY, and it is a **VRAM-pressure fallback**, not a
default:

* A beat longer than the declared `max_frames` (177) already splits -- coverage
  planning handles it and `_planned_length` never mirrors.
* For a beat that FITS one render, `_floor_length` asks the VRAM predictor how
  much it can afford. When that comes back SHORTER than the beat's audio-derived
  target, the short render is ping-pong-extended up to the target rather than
  freezing on the last frame (the 0.68s-then-freeze bug, PBUG-20260723-02).

So the mirror fires exactly when VRAM cannot afford the whole short beat. That
is narrower than "every WAN beat", and wider than "never".

### Why it was NOT fixed tonight

`PLANNING_CAP_ENGINES` is the obvious-looking lever and it is the WRONG one --
it governs whether a *profile ceiling* narrows the planning contract, not
whether a beat splits. Splitting is already driven by the declared `max_frames`.

The real fix is: when `_floor_length` would return less than the beat target,
route the beat to COVERAGE PLANNING (several affordable native segments) instead
of rendering short and mirroring. That is a planning-topology change on the
operator's primary engine, and `frame_contract.py:289` states the rule for this
class of change in as many words:

> Adding an id here is a per-engine decision with a LIVE PROOF attached, never
> a convenience.

There is no live proof available at 01:00 with the GPU mid-campaign and the
operator asleep, and a wrong fix here degrades every WAN beat. Recorded instead
of guessed.

### What the next session should do

1. Reproduce: force a WAN beat whose target exceeds the VRAM-affordable length
   and confirm `extension_mode == "ping_pong"` in the receipt.
2. Route that case to `partition_beat` instead, with the segment count coming
   from the affordable length rather than the contract max.
3. Prove on a live leg, then add a test asserting `extend_frames_to_target` is
   unreachable from any production render path.
4. Only then may any document claim the no-mirror invariant holds.
