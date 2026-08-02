# Prior maths vs current maths -- which answer is actually better?

**Operator, 2026-08-02:** "you unearthed some real maths solutions we solved
months ago. Do any of the previous maths solutions provide a better answer than
your current solutions? Let's not lose the lessons from the past -- let's have
you and codex compare."

**And the operator's fair caveat:** "of course I didn't have Fable 5 or Opus 5
months ago, and not even GPT-5.6, so maybe our current maths are more advanced."

That caveat is testable, and the answer is uncomfortable. **The prior ruling was
produced by `gpt-5.6-sol` at high effort on 2026-08-01 -- the same model, at the
same effort, one day ago** (`kibitz-runs/2026-08-01-fastwan-frame-cap/`,
`codex_model_selected.txt` = `gpt-5.6-sol`, `codex_reasoning_selected.txt` =
`high`). So the "older = weaker model" defence does not apply to the document
that matters most here. The genuinely older maths (2026-07-25 coverage
partitioner) is a separate question and is treated in section 4.

Codex: verify every claim below against the real files. Do not launch renders or
boot a server.

## 1. THE PRIOR SOLUTION (2026-08-01, gpt-5.6-sol high)

`docs/2026-08-01-fastwan-frame-cap-TWO-STRIKES.md` + its kibitz ruling. The
load-bearing findings:

* **P1 -- Authority separation.** "Declare the static tier cap authoritative for
  topology and render length; declare the dynamic model an admission assertion
  only. **Do not feed live VRAM into planning.**" `effective_frame_contract()` is
  pure by contract and must answer identically at plan time and render time.
* **P2 -- Scope error in the flat number.** The 6563 MiB bench figure is
  `peak_delta_mib`, NOT whole-machine usage; `VramPeakProbe` records machine-wide
  peak. "The 12,181-12,614 MiB figures therefore cannot be compared directly with
  the bench delta or the cost-model estimate."
* **P3 -- The window hypothesis is not proven.** `vae_temporal=16` proves the
  configured decode tile window only; it does not prove sampling, attention or
  conditioning are frame-count independent. "**Do not implement a window-aware
  cost model from the current evidence.**"
* **P4 -- The assert, done correctly.** Extract a pure
  `assert_frame_affordable(free_mib, frames, canvas, engine)` and invoke it after
  length selection **on both branches, using the same hoist-adjusted effective
  free**. It may return the length or raise; it must never resize a planned
  segment.
* **P5 -- A defensible number.** Under the current model at a declared
  13,000 MiB minimum effective free, **65 is the highest safe 4n+1 rung** -- and
  that is explicitly NOT an 8 GB qualification.
* **P6 -- No refit from bench data.** The standing ruling forbids refitting
  `FRAME_COST_MODEL` from bench data; instrument the real `prepare()` +
  `render_clip()` path first.
* **P7 -- Three cap surfaces** (`video.max_render_frames`, `render.frame_budget`,
  `launch.env.OTR_FASTWAN_8GB_MAX_FRAMES`) must move atomically; the env value
  outranks the ledger value in `_floor_length()`.

## 2. MY CURRENT SOLUTION (2026-08-02) -- and where it is WORSE

1. **"Measure the cost row with `--estimator-fit`, then re-enable enforcement."**
   This is precisely the "just fix the coefficients" move that **P6 says is not
   available today**. I re-derived a rejected approach.
2. **I was about to accept a flat cost row `(6600, 0)`** from the r2 panel's
   SHOULD-FIX. That number is a peak DELTA. **P2 already caught this exact scope
   error one day ago.** Folding it in would have written a bench delta into a
   field the admission gate reads as machine-wide overhead.
3. **My affordability assert broke `fastwan_8gb`** -- an engine that was green.
   The cause: I omitted the hoist correction. **P4 named that correction
   explicitly**, and `_floor_length()` already implements it
   (`eng_wan_ti2v.py:799-800`, `free_mb = free_mb + hoisted_vram_mb`). The prior
   ruling handed me the fix and I implemented half of it.
4. **I never adopted the authority separation (P1)**, which is structurally the
   better answer: it REMOVES the static-vs-dynamic contradiction instead of
   moving the knife edge to a different free-VRAM level.

Verified state today: `assert_frame_affordable` exists
(`motion_common.py:339`) and is **wired nowhere** -- the pure function P4 asked
for was extracted, then its call sites reverted. So P4 is half-done, in the half
that does nothing.

## 3. WHERE THE CURRENT WORK IS GENUINELY NEW (be fair to it)

* **The `--estimator-fit` harness** (upper envelope, slope clamped >= 0,
  `math.ceil` so the bound only rounds UP, partial-ladder receipt withheld,
  `resolution_basis` + `row_ready_mib` stamped). Prior **P-MUSTFIX-10 ASKED for
  exactly such a runner.** But it demanded one that "loads
  `workflows/otr_canonical.json` and drives the real `prepare()` +
  `render_clip()` path". Mine is built into `run_video_arm_bakeoff.py`, which is
  a BENCH runner under the section 0A carve-out and submits stock-node API
  graphs. **So it does not satisfy P-MUSTFIX-10, and by P6 its output may not be
  written into `FRAME_COST_MODEL` at all.** Codex: confirm or refute.
* **The zero-slope hole fix** in `compute_real_frame_budget` (reject non-finite /
  negative overhead / slope / margin; `budget_mb < overhead` raises
  unconditionally; zero slope legal only after overhead fits). Found
  independently by two r3 lanes, no prior equivalent. Believed sound.
* **The mirror deletion** on `wan_ti2v` under the operator's no-mirror ruling,
  and coverage planning replacing it. Postdates the prior ruling.

## 4. THE GENUINELY OLD MATHS (2026-07-25) -- still correct

`_ladder_partition` (`coverage_plan.py:174-197`) splits a total into exactly
`count` legal lengths, filling each toward the ceiling in order so the SHORT
segment lands LAST, "where a viewer expects a beat to end". Both r2 panels used
it to correct my table, independently, to `[49]*7 + [33]*3` for `humo_14B_169`.
**This July solution is sound, current, and was the thing that caught my error.**
No supersession needed. Codex: confirm the fill-toward-ceiling ordering is
deterministic and that no newer requirement (no-mirror, per-segment audio
slicing) invalidates it.

## 5. WHAT I WANT CODEX TO RULE ON

1. **Is the prior authority separation (P1) still the right answer today**, after
   the mirror deletion and coverage planning landed? Or has the ground shifted
   enough that a different shape is now better?
2. **Confirm P2's scope error** -- is 6563 MiB genuinely a peak delta that cannot
   be written into `FRAME_COST_MODEL` as overhead? If so, what measurement WOULD
   be admissible, and from which harness?
3. **Does my `--estimator-fit` harness satisfy P-MUSTFIX-10, or not?** If not,
   name the smallest change that makes a compliant instrumentation run possible
   without violating repo section 0A.
4. **Is P5's number (65 at 13,000 MiB effective free) still correct** given the
   hoist correction already in `_floor_length`? Prior MUST-FIX-1 warned the
   13,575 MiB figure came from an older WAN failure and may not describe FastWan
   today.
5. **Ranking:** for each of the four disagreements in section 2, say plainly
   which answer is better -- prior or current -- and why. Where the prior answer
   wins, say what must be undone.
6. **What else in the archive have I re-derived worse?** Name any other prior
   maths solution this session's work has ignored or contradicted.

## CONSTRAINTS

100% local, open source, offline-first. 16 GB RTX 5080, 14.5 GB real-world
ceiling. `wan_8gb`'s sampler recipe is FROZEN. The only workflow JSON is
`workflows/otr_canonical.json`; the bench carve-out is section 0A of
`CLAUDE.md` and is MEASUREMENT ONLY. Every second of audio gets original video --
no mirrors, no ping-pong. Fail loud, no fallbacks. **Do not launch renders or
boot a server.**
