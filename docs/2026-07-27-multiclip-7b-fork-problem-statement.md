# PROBLEM STATEMENT -- multi-clip 7b: the plan cannot see the environment

**For a kibitz arc (r2 -> r3 -> r4). Written before any round runs.**
HEAD `091f0966` on `v2.0-alpha`. Suite 6913 passed / 27 skipped / 1 xfailed.
Canonical workflow `5377914B` byte-identical.

This is not "review my plan". There is a specific ARCHITECTURE FORK with two
named options, a known cost on each side, and six existing tests that constrain
the answer. The arc's job is to settle it against the real code, not to
brainstorm.

---

## 1. THE DIVERGENCE, with line numbers

Two phases compute a frame ceiling for the same beat, independently.

**Phase 1 -- plan time.** `nodes/otr_shot_lock.py:1101` `_stamp_coverage_plan`:

```
contract = _fc.frame_contract_for(_vreg_local.get_engine(engine_id))   # :1153
plan = _cp.partition_beat(target, contract)                            # :1159
_cp.validate_coverage_plan(plan, contract)
...
shot["coverage_plan"] = plan.to_dict()
shot["jump_still_requests"] = [...]        # one still MINTED per jump segment
```

`frame_contract_for` returns the adapter's STATIC class attribute -- for
`ltx_video`, `FrameContract(min_frames=9, max_frames=169, quantum=8, ...)`
(`eng_ltx_video.py:396-403`). The partition, the segment count, and therefore
the NUMBER OF STILLS MINTED all follow from that literal.

**Phase 2 -- render time.** The engine computes its own length and reads the
environment while doing it:

| engine | function | reads |
|---|---|---|
| `ltx_video` | `_ltx_frame_length` (`eng_ltx_video.py:155-179`) | `OTR_LTX_MAX_FRAMES`, `OTR_LTX_MIN_DECODE_FRAMES` |
| `ltx_video` (loop) | `_ltx_loop_source_length` (`eng_ltx_video.py:196-213`) | `OTR_LTX_MAX_FRAMES`, `OTR_LTX_LOOP_MIN_DECODE_FRAMES` |
| `wan_ti2v` | `_floor_length` (`eng_wan_ti2v.py:363-402`) | `OTR_WAN_TI2V_MAX_FRAMES`, then `self.profile_max_render_frames()` |
| `ltx_8gb` | `_resolve_render_config` (`eng_ltx_8gb.py:257-258`) | `OTR_LTX_8GB_MAX_FRAMES`, clamped `[9, 16384]` |
| `ltx_audio_in` | module scope (`eng_ltx_av.py:59`) | `OTR_LTX_AV_MAX_FRAMES`, read at **IMPORT** |
| `humo_14B_169` | `render_clip` (`eng_humo.py:474-478`) | `OTR_HUMO_14B_SAFE_FRAMES` |

Both halves are internally consistent. Neither logs anything. That is exactly
why the divergence is silent: the plan minted N stills for N segments, and the
renderer produced a different number of frames per segment than the plan
promised, and nothing compares the two.

The full 15-variable audit table (read site, import-vs-render, clamping) is in
`docs/2026-07-26-chunk-7b-window-prompt.md`. Do not re-derive it. DO verify any
row you are about to act on.

---

## 2. THE OBVIOUS FIX WAS BUILT AND REVERTED

A blanket `assert_env_agrees` helper -- "env != declaration is terminal",
wired into all five frame-count env vars -- was built in this build and
reverted. It produced six failures, and NONE of them was a bug.

**THE SIX CONSTRAINING TESTS, by nodeid, with what each actually asserts.**
Any proposal must say what happens to each one. "Rewrite the test" is an
answer, but it must be argued, not assumed -- rewriting a test so it agrees
with your change is the "my test agreed with the bug" failure mode this build
hit in chunk 6 and again in chunk 7a QA.

1. `tests/test_look_qa_round5.py::TestLtxFrameCap::test_cap_env_override_respected`
   `OTR_LTX_MAX_FRAMES=57`; a 238-frame ask returns 57. An operator cap BELOW
   the declared 169 is honoured, not refused.

2. `tests/test_look_qa_round5.py::TestLtxFrameCap::test_non_8n1_cap_snaps_down_below_cap`
   `OTR_LTX_MAX_FRAMES=120`; returns 113 (`8n+1` below the cap). The env cap is
   not merely accepted, it is SNAPPED to the engine ladder.

3. `tests/test_look_qa_round5.py::TestLtxFrameCap::test_below_floor_env_clamps`
   `OTR_LTX_MAX_FRAMES=2`; returns `_LTX_MIN_FRAMES` (9). A nonsensically low
   cap clamps to the floor rather than refusing.

4. `tests/test_look_qa_round5.py::TestLtxFrameCap::test_cap_below_floor_wins`
   `OTR_LTX_MAX_FRAMES=57`, decode floor unset (default 169). A 30-frame ask
   rises only to 57, not to 169: the operator ceiling outranks the engine's own
   decode floor.

5. `tests/test_remaining_video_contracts.py::test_wan_ti2v_ceiling_precedence_and_no_change_when_unpinned`
   Pins the PRECEDENCE: env pin `49` outranks the profile-carried
   `max_render_frames=17`; unpinned + `max_render_frames=0` leaves 177
   unchanged; a malformed pin AND a malformed stamp fall back to the engine max
   without crashing.

6. `tests/test_wan_ti2v.py::test_floor_max_override_is_an_absolute_hard_cap`
   `OTR_WAN_TI2V_MAX_FRAMES=49`; `_floor_length(177) == 49`. Written FOR the
   2026-07-24 WAN 8GB launch contract, after an 8GB leg inherited the 177-frame
   engine max, asked for a whole 7-second beat, and died in the cost model.

These are production behaviour, not stale debt. `eng_ltx_8gb.py:43` lists
`OTR_LTX_8GB_MAX_FRAMES` in the module docstring as a normal override for
VRAM-constrained boxes.

**So the problem is NOT that env overrides exist. The problem is that the plan
cannot see them.**

---

## 3. OPTION A -- REFUSE

Env disagreeing with the declared literal is terminal, before any GPU work,
naming the variable and both numbers.

**Mechanics.** One shared helper called from every engine that reads a
frame-count env var; a suite-wide test that every such var is wired to it.

**What it costs.**

* The 8GB tier loses `OTR_LTX_8GB_MAX_FRAMES` and `OTR_WAN_TI2V_MAX_FRAMES`.
  An operator who wants a lower cap must edit the adapter -- a file the repo
  treats as a hardware-INDEPENDENT declaration. That inverts the reason the
  literal was made a literal in the first place.
* Test 6 above exists because the alternative already failed in production.
* It cannot reach `OTR_ACTIVE_PROFILE` at all. `motion_common.profile_max_render_frames()`
  (`motion_common.py:442-455`) is a SECOND, non-env-named lever on the same
  `wan_ti2v` ceiling, and it is the NORMAL 8GB path -- refusing it breaks that
  tier outright rather than protecting it.
* It detects the disagreement. It does not remove the second policy that
  creates it. This build has now hit "two policies over one state" five times
  (chunk 1a routing mirrors, chunk 4 still-spine inference, QA4 route ordering,
  QA6 segment length, QA7 the terminal-frame key), and every time the fix that
  held was collapsing to one call, not adding a comparison.

**What it buys.** It is simple, it is a literal reading of the operator's "it
either works or it fails", and it needs no schema change.

---

## 4. OPTION B -- RESOLVE ONCE AT PLAN TIME AND STAMP IT

The environment and the profile are read exactly ONCE, where the plan is built,
and the resolved ceiling is frozen into the coverage plan the same way
`render_frames` already is. Both phases then read the SAME number, so the
divergence is unconstructible rather than detected.

**Mechanics, roughly.**

* `resolve_frame_contract(engine, *, env, profile_max_frames) -> FrameContract`
  -- ONE authority, pure given its inputs, returning a contract whose
  `max_frames` is the resolved ceiling rather than the class literal.
* `_stamp_coverage_plan` calls it and stamps the result into the plan dict.
* The engine's own length helpers ask the SAME resolver instead of reading env
  themselves.
* `assert_coverage_plans` validates against the STAMPED contract.

**What it changes about MEANING.** A `FrameContract` stops being "the literal
in the adapter" and becomes "the literal, as resolved for this box, frozen at
plan time". `frame_contract.py`'s module docstring currently promises the
opposite in as many words: *"Pure, STATIC numbers: never live VRAM, never
mutable environment."* That promise either has to be rewritten honestly or the
resolved value has to live somewhere that is not a `FrameContract`.

**Schema delta.** `CoveragePlan.to_dict()` (`coverage_plan.py:100-111`) today
emits exactly `target_visible_frames`, `join_mode`, `segments[]`. Option B adds
a `contract` block, which means `from_dict` must round-trip it, the ledger
schema grows a field, and every fixture that hand-builds a plan dict is now
building an incomplete one.

---

## 5. FACTS THE FORK WRITE-UP DID NOT HAVE (verified this window)

These were checked against the real code today. They change the cost estimate
on both sides, so weigh them.

**5.1 -- the profile ceiling is ALREADY in scope at the stamping call site.**
`otr_shot_lock.py:1520` calls `build_execution_plan(beats, budget, creative,
policy, led)`, and that same `policy` dict is what `:1536` reads
`max_render_frames` out of when it stamps the ledger. `_stamp_coverage_plan` is
called from inside `build_execution_plan` (`otr_shot_lock.py:1326`) but is
NOT currently passed `policy`. So plumbing the profile ceiling to plan time is
one extra argument, not a new channel. Option B is cheaper than it looks.

**5.2 -- the profile ceiling reaches BOTH sides from ONE ledger field.**
ShotLock stamps `video.max_render_frames` (`otr_shot_lock.py:1536`);
`render_driver.py:3005` copies it into the render policy; `motion_common.prepare`
(`:431`) captures that into `self._active_profile`; `profile_max_render_frames()`
(`:442`) reads it. Same number, one source. A resolver called on both sides with
that number cannot disagree WITHIN one process.

**5.3 -- env is process-global, so plan and render already see the same env
INSIDE one run.** ShotLock and the render driver are nodes in the same ComfyUI
server process. The stamp therefore earns its keep specifically ACROSS
processes: a replayed ledger, `/otr/video_render_single`, the soak path, or a
server restarted with different env between plan and render. That is a narrower
window than the fork write-up implies, and it is worth saying out loud -- but it
is also the window in which a silent divergence is completely undetectable
today.

**5.4 -- `OTR_LTX_AV_MAX_FRAMES` is read at IMPORT (`eng_ltx_av.py:59`).** A
resolver that reads env per call and a constant that froze at import are two
different numbers whenever the env changes after import. `eng_ltx_av.py:1229-1234`
already declares the LITERAL 497 rather than the constant, and says why.

**5.5 -- THERE IS A THIRD INPUT NEITHER OPTION ADDRESSES: LIVE VRAM.**
`eng_wan_ti2v._floor_length` (`:363-402`) does not stop at the env/profile cap.
It ends in `compute_real_frame_budget(free_vram_mb(), target, w, h, name)` and
quantizes DOWN to what free VRAM affords. So the render length varies with
machine state that the plan cannot see and must not see -- stills are minted
long before. Today this is invisible because `render_clip` (`:536-543`)
ping-pong-extends the short render back up to the target. **7c deletes the
ping-pong.** After that, a VRAM-shortened segment is simply a short beat.

Partial mitigation already exists and is worth confirming rather than
rebuilding: `assemble_beat_segments` proves the exact decoded frame count for a
MULTI-segment beat, so a short segment there fails closed at assembly. A
SINGLE-segment beat takes the historical path (`render_beat_coverage`
early-returns) and is not proven. That asymmetry is checkable and should be
stated as a test, not assumed.

**A proposal that settles env-vs-contract and leaves the VRAM path silent has
solved the smaller half of one problem.** Say explicitly whether VRAM belongs
in 7b, in 7c with the ping-pong rip, or in its own chunk -- and why.

---

## 6. THE ANCHOR'S LEANING, AND THE TENSIONS IN IT

Stated so the arc can attack it rather than agree with it.

The lean is toward **B, with A's refusal relocated rather than discarded** --
i.e. one resolver, called by both phases, plus a stamped-versus-freshly-resolved
comparison at the second boundary. B alone removes the divergence inside one
process; the comparison catches the cross-process case from 5.3; and because
the knob moved the number on BOTH sides, no operator capability is lost and the
six tests in section 2 stay green **provided the resolver keeps the env read as
its implementation**.

Three tensions in that lean, named honestly:

**6.1 -- B may silently delete a check that exists on purpose.**
`assert_coverage_plans`'s docstring (`render_driver.py:3388-3405`) says it
re-validates "AGAINST THE LIVE CONTRACT, not just for internal arithmetic: an
adapter whose declared frame contract changed since the plan was made (a
version bump, a re-registered engine) must not silently execute a plan its
current contract would reject." If it starts validating against the STAMPED
contract, that check is gone unless it ALSO compares stamped to live. Any
proposal must say which of the two it does, or that it does both.

**6.2 -- the six tests survive only under one specific shape of B.**
They call `_floor_length` / `_ltx_frame_length` DIRECTLY, with env set and no
plan in sight. If those helpers stop reading env and read only a stamped plan,
all six break and B is no cheaper than A. If they delegate to a resolver that
reads env, all six pass untouched. That is not a detail -- it decides whether B
costs six test rewrites or zero.

**6.3 -- "one authority" can become "one authority plus a stamp nobody reads".**
Chunk 4's lesson was that two policies over one state is the defect. A stamped
contract that the render path does not actually consult is exactly that shape
with an extra field. Whatever lands must show the render path READING the
resolved number, not merely that it was written down.

---

## 7. WHAT THE ARC MUST ANSWER

Answer these specifically, with file:line. Do not restate the options.

1. **A or B or a third shape?** If a third, name it and say which of A's and
   B's costs it avoids.
2. **What happens to each of the six tests in section 2?** Green untouched,
   rewritten (and why that is not agreeing with the bug), or deleted.
3. **Does the stamped contract belong in `CoveragePlan`, or beside it?** A
   `FrameContract` whose `max_frames` moved with the box contradicts
   `frame_contract.py`'s own opening promise. Rewrite the promise, or put the
   resolved ceiling somewhere else and say where.
4. **What does `assert_coverage_plans` validate against after the change** --
   the stamped contract, a freshly resolved one, or both (6.1)?
5. **Where does live VRAM (5.5) belong**, given that 7c deletes the ping-pong
   that currently hides it?
6. **`OTR_ACTIVE_PROFILE`** -- confirm or refute that a resolver taking
   `profile_max_frames` closes it (5.2), and name the test that would prove it.
7. **`OTR_FORCE_ENGINE_MAP`** -- `resolve_final_shot_engines` runs
   `assert_coverage_plans` AFTER the override on both branches
   (`render_driver.py:3501` and `:3511`). VERIFY that this already re-validates
   the plan against the FORCED engine. If it does, the deliverable is a test
   that says so, not a second check.
8. **`OTR_LTX_LOOP_VIA_REVERSE`** is ON by default and returns `2N-1` frames
   (`eng_ltx_video.py:196-213`, `_boomerang_frames` at `:186`), so a 169-frame
   ask comes back as 193 with no env set at all -- `ltx_video` violates its own
   declared 169 ceiling TODAY. Is that 7b's (a contract that is currently
   false) or 7c's (a fallback to delete)? Pick one and justify the ordering.

---

## 8. ACCEPTANCE, whichever branch wins

* Full Windows suite green (baseline 6913 passed / 27 skipped / 1 xfailed) and
  Bug Bible green (17). Test venv
  `C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe`.
* Canonical workflow `5377914B` byte-identical, or the node/wiring/widget
  change lands IN `workflows/otr_canonical.json` in the SAME commit.
* `docs/ENGINE_MATRIX.md` regenerated via `tools/engine_matrix.py`; the
  `--check` drift gate passes. Never hand-edited.
* **Every fix mutation-proven** -- the test must be shown to FAIL for the right
  reason when the fix is reverted. A mutation run caught a vacuous assertion in
  7a that a green suite and two QA panels had both missed.
* Landed in green pushed slices, pathspec-only, `HEAD == origin` verified after
  each. Never one large red commit.
* UTF-8 no BOM, ASCII, SFW. THE LAW holds: an audit may improve a story, never
  fail one for length, language, style, visual vocabulary, or quality.

## 9. OUT OF SCOPE FOR THIS ARC

* **7d, the live render slice.** No GPU is available this window.
* Re-deriving the 15-variable env table. It is in
  `docs/2026-07-26-chunk-7b-window-prompt.md`. Verify rows you act on; do not
  rebuild the survey.
* Re-litigating chunk 7a. `supports_multi_clip` is deleted, all 31 engines
  carry a static `FrameContract`, multi-clip is universal, and only the CHAIN
  is still earned (via `continuity=strict_first_frame`). The operator ruled:
  *"There's no gate with opt in or opt out... It either works or it fails."*
* The full 7c fallback rip. Its list is in
  `docs/2026-07-27-next-window-prompt-nogpu.md`. Question 5 and question 8 above
  ask only where a BOUNDARY falls between 7b and 7c, not for the rip itself.

## 10. GROUND RULES FOR REVIEWERS

Cite `file:line` from the real tree at
`C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio`. A claim
with no line number will be dropped in judgment, and a claim whose line number
does not say what it is said to say will be dropped louder. Two adversarial
panels have found six real defects in already-green, already-mutation-proven
code in this build -- and the test stubs agreed with all six bugs. Prefer the
claim you can point at over the one that sounds right.
