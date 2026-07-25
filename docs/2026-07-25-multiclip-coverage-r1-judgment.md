# MULTI-CLIP COVERAGE -- r1 judgment

**Run:** `kibitz-runs/2026-07-25-multiclip-coverage/r1/`. Code baseline
`a1d810f1`; doc commit `2d2f7f90`. Panel: codex `gpt-5.6-sol` high (pin
verified) + agy `Gemini 3.6 Flash (High)` (pin verified), independent.
Claude is the grounded panelist and sole judge.

**Both seats: VERDICT no.** Neither says the requirement is wrong -- both say
the BRIEF was an unresolved question set. Fair.

## 0. THE FINDING THAT MATTERS MOST: the codebase already named this fix

`nodes/otr_silent_composite.py:244-266`, `_should_loop_fill`, verbatim:

> "`audio_driven_face` (HuMo) is ALSO exempt (2026-06-30 HuMo-improve plan):
> looping a talking/lip-synced face DESYNCS the mouth from its own audio (the
> loop replays an earlier mouth shape against later audio) -- WORSE than a
> static hold... **The real fix is phrase-chunking (render the beat's correct
> duration so it never underruns) -- tracked as a follow-up; this is the safe
> interim behavior.**"

So the build reached the operator's conclusion on 2026-06-30, wrote it down,
shipped a deliberate interim, and named the permanent fix: **phrase-chunking**.
This arc is not inventing a feature -- it is executing a fix the codebase has
been waiting for. Any design that contradicts that note needs a reason.

**It also means there are THREE silent coverage mechanisms today, not one:**
mirror/ping-pong in the engines (`wrapper_bridge.py:435`, used by
`eng_wan_ti2v.py:521` and `eng_ltx_8gb.py:426`), loop-fill in the composite
(`otr_silent_composite.py:244`), and held-last-frame as the floor. I knew about
one. codex found the third.

## 1. Corrections to my brief's premises -- all confirmed against the code

- **`ltx_av` is NOT safe from underrun.** I claimed it renders to target
  natively. It caps at `_LTX_AV_MAX_FRAMES` (`eng_ltx_av.py:58`, default 497,
  env-overridable) and clamps at `:950-953`. Long audio-conditioned beats
  underrun. codex caught it; my fact 10 was wrong.
- **"`ShotRow` is closed" is not an active guarantee.** `ShotRow` (`_Forbid`,
  `schemas.py:302`) has no `role` or `char_id`, yet production shot dicts carry
  both and are read all over `render_driver`. Only execution groups are
  validated at lock. So the typed schema is not enforced on real rows -- a
  correction with consequences for any "just add a field" plan.
- **The composite loop-fills short clips** (`otr_silent_composite.py:244`) --
  a coverage mechanism my brief did not mention at all.

## 2. Where both seats converge (adopted)

1. **CUT the ExecutionGroup / provider-consumer DAG expansion.** Both, firmly.
   Groups are per-role; clip sequencing is intra-beat. **This also retires my
   own earlier M9 for good** -- I proposed that DAG twice and was wrong twice.
2. **ONE `ShotRow` per beat.** Do not explode a beat into N shots or N groups.
   Downstream -- manifest, timeline, captions, credits, `obs_publish` -- assumes
   one identifier per beat.
3. **CHAIN capability must be an explicit per-adapter declaration.** Both.
   "Accepts a still" is not "guarantees first-frame continuity": codex grounds
   it at `motion_common.py:393` (`accepts_still` only controls minting) and in
   the Bug Bible (HuMo's reference is a soft identity hint, not a first-frame
   lock); Veo's `lastFrame` is interpolation, not chaining. Engines without
   strict first-frame support use JUMP CUT.
4. **The partitioner must be PURE and STATIC** -- fixed profile ceilings, never
   live VRAM, never mutable env. Both, and it follows from the fact that stills
   are minted before the render phase.
5. **Boomerang goes away on moving-video lanes.** Both. agy would delete the
   mirror outright; codex would leave `still_*` and full-duration procedural
   lanes alone. Judge: **keep the helper, forbid it on moving lanes** -- the
   `allow_mirror=False` seam landed at `a1d810f1` is exactly this shape, and
   deleting a pure helper that `still_*`/decorative lanes legitimately use
   would be a rip for its own sake.
6. **No operator boomerang mode in this build.** Both. It would preserve the
   behaviour the requirement replaces.

## 3. THE REAL SPLIT -- do audio-synced lanes get multi-clip at all?

- **agy: EXEMPT them.** Slicing a beat for `audio_driven_face` /
  `audio_conditioned_video` causes mouth-pose resets and visual popping at
  every sub-clip frame 0. Force a single continuous render or fail closed.
- **codex: INCLUDE them.** `ltx_av` caps at 497 frames, so long audio beats
  underrun regardless; exempting them just moves the failure.

Both are right about something, and neither answer is complete: agy is right
that an arbitrary cut mid-word is a visible artefact; codex is right that
exemption leaves long audio beats with no legal outcome.

**JUDGE CALL -- take the codebase's own answer, which neither seat proposed:
PHRASE-CHUNKING.** `otr_silent_composite.py:244-266` already names it. Cut
audio-synced beats at PHRASE / silence boundaries rather than at arbitrary
frame counts. A pose reset at a natural speech pause is where a cut is
invisible -- it is how dialogue has been cut on film for a century. That
satisfies agy's objection (no mid-word popping) and codex's (long audio beats
get a legal path), and it honours the 2026-06-30 design note instead of
contradicting it.

**Consequence:** audio lanes are chunked on a DIFFERENT axis (speech
structure) than scene lanes (frame budget). That is per-adapter policy, which
is what this build is for. Cut-point selection needs the beat's line timing,
which the ledger already carries (`render_driver._line_index`,
`_cumulative_beat_start`).

## 4. Second split -- where does multi-clip LIVE?

- **agy: entirely inside `render_shot`** -- render the segments, concatenate,
  return ONE clip per beat. Downstream sees nothing new.
- **codex: flatten clip rows into the manifest** with clip identity, because
  the SFX bed compiler places stems by `start_s` (`otr_master_audio_mux.py:170`)
  and reusing the parent beat start would stack every stem at once.

**JUDGE CALL: agy's containment for v1.** Decisive argument -- codex's SFX
objection only bites IF clips become manifest rows. Under containment the beat
still emits exactly one clip row with one start and one duration, so the SFX
compiler, captions, timeline and `obs_publish` are untouched by construction.
That neutralises codex's must-fixes 3, 10 and most of 8, and it is the smaller,
provable first slice. If per-clip observability is later needed, it is an
additive receipt under the beat's row, not a reshape of the manifest.

**Kept from codex regardless:** the terminal-frame seam must be ONE canonical
extractor, not per-engine (`schemas.py:216` already normalises canonical
clips), and clip/seam persistence must be transactional and fatal before
publication -- current clip persistence is best-effort
(`render_driver.py:3024-3032`), which contradicts the asset invariant.

**Also kept from codex (must-fix 12):** prepare the engine ONCE per beat and
render the beat's segments under that lease. `_render_one`
(`render_driver.py:2424-2458`) currently prepares, loads and tears down per
clip -- N clips would mean N model loads, which would make the feature cost
more than it delivers.

## 5. First vertical slice

codex proposes `ltx_8gb` scene beats (deterministic 9-frame minimum, 161-frame
cap, 8n+1 quantization, already ping-pong-fills underruns); agy proposes
`wan_ti2v` with JUMP CUT before CHAIN. **Judge: codex's `ltx_8gb`**, because
its frame contract is already discrete and deterministic -- exactly the pure
static contract requirement 4 demands -- whereas `wan_ti2v` is the engine whose
budget reads live VRAM, i.e. the hardest case, not the first one.

Acceptance for the slice: one beat exceeding the cap, at least two FORWARD-only
clips, exact master duration, canonical clip + seam assets on disk,
`RESULT SUCCESS`, `obs_publish OK`, and mechanical proof in the trace that no
reverse and no loop supplied any coverage.

## 6. Open for r2 (coding plan)

- The pure per-adapter frame contract's exact shape, and the phrase-chunk
  cut-point selector for audio lanes.
- The `strict_first_frame | soft_reference | none` continuity declaration and
  its per-engine inventory.
- Whether `ltx_8gb`'s `strength=1.0` init is genuinely CHAIN-capable -- codex
  flags it as needing a live first-frame seam check, not an assumption.
