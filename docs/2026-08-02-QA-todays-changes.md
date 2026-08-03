# QA: today's video-path changes -- what works, what doesn't, what we already knew

**Operator, 2026-08-02:** "have Sonnet 5 and agy flash 3.6 QA -- and perhaps it's
a bit of trial and error, marking what works, what doesn't, what DID work in the
past, without reinventing the wheel."

**That last clause is the assignment.** This repo has twice re-derived a worse
answer to a problem it had already solved: the fastwan cost-model refit (solved
2026-08-01, re-derived worse 2026-08-02) and the HuMo frame cap (justified by a
receipt that has never existed here). Before you flag anything, check whether the
archive already settled it.

**Do not boot a server, do not launch renders, do not edit code. READ-ONLY.**

## What changed today

| # | File | Change |
|---|---|---|
| 1 | `coverage_plan.py` | new `requires_coverage_execution`, replacing `is_multi_clip` as the router predicate |
| 2 | `render_driver.py` | new `_assert_beat_affordable()` admission boundary; router uses the new predicate; per-segment video seed |
| 3 | `motion_common.py` | new `has_measured_cost_row()` |
| 4 | `eng_humo.py` | `_LOG` defined (it was REFERENCED but undefined); `VramPeakProbe` added; `_HUMO_14B_SAFE_RENDER_FRAMES` 49 -> 97; base `HuMoEngine` now CAPPED (was uncapped at 177); `HuMo17BEngine` given its own explicit contract |
| 5 | `mouth_policy.py` | corrected a stale docstring claim about per-segment seeds |
| 6 | `tools/engine_matrix.py` | effective-canvas, multi-clip and evidence columns |

Suite at time of writing: 1073 passed on the affected surface, 0 failures.

## The reasoning behind the two contentious ones

**The cap (change 4).** External research (grade A) settled that orientation
cannot change allocation at equal pixel count: HuMo/Wan use square patch
`[1,2,2]`, a GLOBAL attention window `[-1,-1]`, and RoPE reshaped to `f*h*w`, so
480x832 and 832x480 both give a 1,560-token DiT grid per latent time. The old
split -- 49 landscape, 177 portrait, same checkpoint, same 399,360 pixels -- had
no basis, and its cited receipt (`docs/2026-06-27-humo-bakeoff`) is not in this
repo. 97 is the length HuMo was TRAINED at, which is a bound with a citation,
unlike either previous number. **It is a reasoned bound, not a measured ceiling,
and the code says so.**

**The admission boundary (change 2).** It refuses where a MEASURED cost row
exists and reports "unenforced" where none does, rather than judging an engine
against `_DEFAULT_FRAME_COST`. Taken literally, a prior ruling said a missing row
must fail qualification -- but only two engines have rows, so that would ground
the roster. The judgment call was that a borrowed row refuses good renders and
admits bad ones with equal confidence, so the honest move is to stop pretending
rather than to enforce someone else's number. **Challenge this if you disagree.**

## Known gap, already identified -- confirm the scope, do not re-find it

`_assert_beat_affordable` is called inside coverage execution only. The
single-clip path returns through `render_shot()` before reaching it, and
`ltx_audio_in` is not in `PLANNING_CAP_ENGINES` -- so **the engine with the
hottest measured peaks is not covered by the new guard.** This is known and
scheduled. What is wanted from you: confirm the exact scope (which renders are
and are not covered), and flag anything ELSE that path misses.

## The regression risk to hunt hardest

Base `HuMoEngine` went from `safe_render_frames = None` to a real cap. The
exact-fit branch in `render_clip` is gated on `if cap is not None`, so **code
paths that never ran for base `humo` now run.** One test already caught this
(a fake returning 4 frames for a 33-frame beat, which the short-render rule
correctly refuses). Search the whole tree -- `nodes/`, `scripts/`,
`config/profiles/*.json`, `workflows/otr_canonical.json` -- for anything else
gated on `safe_render_frames is None`, on the 177 ceiling, or on the removed 49.

## Where to check before proposing anything

* `docs/*.md`, especially `2026-07*` and `2026-08*`
* **`BUG_LOG.md` and `BUG_LOG_2026-06.md` in the repo ROOT** (~450 KB and ~90 KB).
  These were unread until today and contain BUG-LOCAL-265, the lesson that a VRAM
  reading taken without clearing pipeline residue measures the previous phase.
* `docs/PROD_BUG_LOG.md`, `docs/PRODUCTION_SPRINT_LESSONS.md`
* `kibitz-runs/**/*.md` -- prior panel rulings, several of which this session
  violated by not reading them
* `comfyui-custom-node-survival-guide/BUG_BIBLE.yaml`

## Report format

Ranked worst-first, at most 12 findings:

* **WHAT** -- one sentence
* **WHERE** -- `path:line`
* **STATUS** -- BROKEN NOW / RISKY / ALREADY SOLVED IN PAST / WORKS AS INTENDED
* **PRIOR ART** -- cite the doc or log if the archive covers it; "none found" if
  you searched and it does not
* **FIX** -- only for BROKEN/RISKY, and only if the archive does not already
  prescribe one

No style, naming or formatting notes. Anchor every finding to a real
`path:line`. **UNVERIFIED is a legitimate and useful answer** -- an honest
"I could not check this" beats a confident wrong claim, which is precisely the
failure mode being guarded against.

## CONSTRAINTS

100% local, offline-first. 16 GB RTX 5080, 14.5 GB real-world ceiling. Every
second of audio gets ORIGINAL video -- no mirrors, no ping-pong, no held frames.
`wan_8gb`'s sampler recipe is FROZEN. The only workflow JSON is
`workflows/otr_canonical.json`.
