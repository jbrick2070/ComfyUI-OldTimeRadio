# B4 QA record -- deleting the ltx_8gb ping-pong

Two fan-outs, before and before the push: 2 Sonnet lenses + 1 agy
(`Gemini 3.6 Flash (High)`) each round, every lens launched in ONE block. $0
external, no codex spend. Claude judged.

## THE PRE-CODE PANEL BROKE THE PLAN, AND IT WAS RIGHT

The brief proposed what the arc judgment prescribed: refuse when the ask exceeds
the cap, delete the CLIP-FILL block, and let an off-grid ask render short. Two
seats independently showed that ships a REGRESSION, not a fix.

**The old block padded whenever the decode came up short FOR ANY REASON** -- a
cap disagreement, an OOM recovery, an early stop -- and it logged a warning when
it did. Delete it with only a cap refusal in its place and a short clip flows
out unflagged into `otr_silent_composite`, whose `_should_loop_fill` hard-loops
it with `-stream_loop -1` **and suppresses its own underrun warning once
loop-fill activates** (`if frac <= 0 or will_loop ...: return`). That trades a
logged mirror for a silent jump-cut repeat, on the majority path. One seat put
it plainly: *"this is worse than relocation -- it's a coverage gap that didn't
exist before."*

**So the shipped change is not the one that was reviewed.** Three parts:

1. **`_ltx8_frame_length` is DELETED.** It snapped an ask DOWN to the nearest
   `8n+1` and clamped it to the env cap -- which is precisely WHY the ping-pong
   had to exist: something had to put the missing frames back. Removing the pad
   while keeping the snap-down would have shipped short clips. Its two jobs got
   explicit owners rather than being left dead: the LADDER moved to
   `Ltx8gbEngine.frame_contract.smallest_legal_at_least` -- the same object the
   coverage planner partitions against, so the adapter and the planner can no
   longer disagree about what a legal length is -- and the CAP became a refusal.
2. **The length now snaps UP and the surplus is TRIMMED in real frames.** An
   off-grid ask of 100 renders 105 real frames and keeps 100. Every frame
   delivered is a rendered frame in order: no mirror, no loop, no held frame.
   An ask below the declared minimum of 9 is left at 9 -- cutting below the
   floor would invent a length the declaration forbids.
3. **Two invariants replace the pad.** A pre-render REFUSAL when no legal rung
   reaches the ask (past the declared 161, or past `OTR_LTX_8GB_MAX_FRAMES`),
   ordered above `_materialize_init_image` and `_build_graph` so it costs
   nothing to discover; and a post-render `len(frames) != length` refusal for
   the under-delivery the pad used to absorb.

**This deviates from the arc judgment**, which says "delete it and any such
segment becomes a hard RenderError". A hard error on every off-grid ask would
fail nearly every non-planned beat, since audio-derived targets are essentially
never exactly `8n+1`. Rendering the next rung up and trimming real frames is
what the adapter's own `allow_tail_trim` declaration already promises. Recorded
as a deliberate deviation, not an oversight.

## The post-code panel cleared the sharp question and found two real defects

**The highest-value question was the chain seam, and it is CLEAN** -- verified
two independent ways, which is worth stating because it is the failure this
build has hit before. `ltx_8gb` declares `strict_first_frame`, so a chained
successor begins on its predecessor's TERMINAL frame. Could the trim cut the
frame the successor chains from?

No, and not by luck: on the multi-segment path every segment length is already
a legal `8n+1` value (the partitioner emits nothing else and
`validate_coverage_plan` re-asserts it at the render boundary), so
`smallest_legal_at_least(target) == target` and the trim's strict inequality is
false. The trim can fire ONLY on the single-clip historical path, which never
chains -- `extract_terminal_frame` is called only inside the multi-segment
loop. And the trim keeps the HEAD, which is the correct end twice over: it
preserves the init image that `LTXVImgToVideo(strength=1.0)` pins at frame 0,
and the emitted file's last frame is a real rendered frame in original order.

Found and fixed before the push:

1. **The module's own docstring still advertised the ping-pong** ("a short
   render is ping-pong-extended (CLIP-FILL) to the beat window"), as did the
   comment above `_LTX8_MIN_FRAMES`. All three seats flagged it. The highest-
   visibility stale doc in the change, in the file the change is about.
2. **`_LTX8_MIN_FRAMES` and `frame_contract.min_frames` are two declarations of
   one number** with no coupling. The constant now has exactly one job -- the
   lower bound `_resolve_render_config` range-checks `OTR_LTX_8GB_MAX_FRAMES`
   against -- so if the contract's floor ever moved alone, that config check
   would start accepting a cap below the real minimum and the refusal would
   fire for a reason nobody could trace. Pinned equal by test.

## Mutation proof

9 mutants: 7 DEFECT all red, 2 CONTROL green, baseline and restore green. The
defect set includes restoring the ping-pong, reverting the ladder to snap-DOWN,
dropping either invariant, and trimming below the declared floor.

## Declined, with reasons

- **agy: "define `_LTX8_MIN_FRAMES` from the class attribute."** The constant is
  defined before the class in the module; the equality pin closes the drift
  without a reordering that buys nothing.
- **agy: a test that `extract_terminal_frame` reads frame `target-1` from a
  TRIMMED clip.** Both Sonnet seats proved the trim cannot fire on a chained
  segment, so that test would assert something production cannot construct.
  Writing it would be theatre -- exactly the decorative-test shape this build
  keeps finding.

## Recorded, not fixed here

- **Two shipped diagnostic entry points flip green to red for an over-cap ask.**
  `POST /otr/video_render_single` takes `frame_count` straight from the request
  body and `OTR_VideoRenderBatch`'s `frame_count` widget allows up to 240
  against a 161-frame engine. A caller asking `ltx_8gb` for 200 frames used to
  get a padded clip and now gets a named error. That is the fix working: 161
  real frames wearing a 200-frame count was never a render of the ask.
  `render_single` already wraps it into `{"ok": false, "error": ...}`.
- **`encode_frames_to_silent_mp4` returns the size of the array it piped into
  ffmpeg, not a re-probe of what ffmpeg wrote.** So an ffmpeg-side frame drop
  could still under-report and reach the composite's loop-fill. PRE-EXISTING,
  untouched by B4, and worth closing when the assembly boundary is next opened.
- **`docs/GO_FORWARD_PLAN.md`'s "THREE silent coverage mechanisms" entry** now
  overstates the first one: `eng_ltx_8gb` no longer calls the extender. Updated
  in this session's handoff.
- **The soak harness never routes to `ltx_8gb`** (`_PROFILES` does not include
  it), so B4 cannot turn a soak leg red.

## The lane split, stated once

`wrapper_bridge.extend_frames_to_target` STAYS. `eng_wan_ti2v` STAYS on it.
WAN renders a deliberately short native clip and fills the beat with it -- the
shipped 8 GB tier contract that `PBUG-20260723-02` exists to protect. Ripping
ping-pong is lane-specific: it is a correctness hole where a coverage plan has
already promised a length, and a load-bearing mechanism where none has.
