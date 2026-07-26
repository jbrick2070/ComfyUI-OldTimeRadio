# r3 WIRING PLAN -- multi-clip 7b, after r2 convergence

HEAD `6bde4b36` on `v2.0-alpha`. Suite verified green THIS WINDOW:
**6913 passed / 27 skipped / 1 xfailed** in 117s. Bible ran. Baseline matches
the handoff exactly.

r2 ran both seats against
`docs/2026-07-27-multiclip-7b-fork-problem-statement.md`. This document is the
judged merge, and it CORRECTS that problem statement in two places. Review THIS
document, not the r2 input.

---

## 0. WHAT r2 SETTLED (do not relitigate)

Both seats independently, and the driver anchor agrees:

* **Option A is CUT.** It breaks six documented operator-knob tests, cannot
  reach `OTR_ACTIVE_PROFILE` at all, and enumerates inputs forever.
* **The six constraining tests stay GREEN UNTOUCHED**, via resolver delegation:
  the engine helpers keep reading env, but through one shared resolver rather
  than five private reads. Neither seat proposed rewriting them. Good.
* **The resolved ceiling gets an explicit schema before any code is written.**

## 1. CORRECTIONS TO THE r2 INPUT -- both found by the panel, both verified

**1.1 -- SECTION 5.5 OF THE PROBLEM STATEMENT WAS WRONG. Live VRAM does NOT
silently shorten a render.** `compute_real_frame_budget`
(`motion_common.py:321-363`) was REWRITTEN by S4 platform-portability on
2026-07-10 precisely to kill that behaviour; its docstring says so -- *"the
frame budget is the STATIC widget value -- geometry only ... NEVER a
VRAM-adaptive resize. The pre-S4 version silently shrank tight-VRAM clips
toward the floor"* -- and it now RAISES `MotionBudgetError` when the target
cannot fit. Tests pin the raise (`tests/test_wan_ti2v.py:229-252`,
`tests/test_remaining_video_contracts.py:173-189`). The driver anchor carried
the same misread and it is struck. **VRAM is fail-loud and out of scope.**

**1.2 -- THE BOOMERANG OVERSHOOT IS DELIBERATE AND PINNED, so it must NOT be
clamped.** One r2 seat proposed clamping `2*src-1` to the resolved ceiling.
`tests/test_ltx_boomerang.py:48-57` (`test_loop_source_length_no_freeze_shortfall`)
asserts `2*src-1 >= target` for exactly `target=169`, and its comment names the
bug that motivated it: *"the 169 -> half 85 -> snap 81 -> 161 < 169 FREEZE the
roundtable caught"*. Clamping down reintroduces that freeze. **REJECTED.** The
boomerang is 7c's to delete; 7b's deliverable is a guard test that DOCUMENTS
the current contract violation, not a fix that trades one bug for an older one.

## 2. THE FINDING NEITHER SEAT HAD

`render_driver.py:2952-2958`, inside `render_beat_coverage`'s segment loop:

```
got = int((clip or {}).get("frame_count") or 0)
if got and got != int(segment.render_frames):
    raise RenderError("shot %s segment %d rendered %d frame(s) but its plan "
                      "asked for %d. NO FALLBACK ...")
```

**The env-vs-contract divergence is ALREADY TERMINAL on the multi-segment
path.** It compares the OUTPUT to the plan, so it catches all fifteen env vars,
the profile ceiling, the boomerang, and the provider clamps in one predicate,
without enumerating any of them. An input-comparison design is permanently one
new env var behind; this is not.

Two gaps remain, and they are what 7b is actually for:

* **GAP 1 -- the SINGLE-segment path is not proven at all.**
  `render_driver.py:2861-2872`: `if plan is None or not plan.is_multi_clip:`
  early-returns to `render_shot` and nothing compares the returned
  `frame_count` to the plan. **Every beat in production today is
  single-segment**, so the proof that exists covers the path nobody is on.
* **GAP 2 -- the check fires AFTER the GPU work.** That, and only that, is what
  plan-time resolution buys. Price it that way.

**GAP 1 IS NOT A ONE-LINER, AND THIS IS THE r3 QUESTION.** On the single
path the request was built WITHOUT `segment_index`, so the engine is asked for
the BEAT's `target_frame_count` -- not the plan's `render_frames`. For any
`allow_tail_trim` plan those differ by `trim_tail`, so a naive equality check
fails every trimmed single-clip beat. That is the same `trim_tail`
computed-and-never-applied drift already on 7c's list. **r3 must decide:**
does 7b prove against what was actually ASKED (the beat target), or does it
wire the plan's `render_frames` into the single path -- which pulls 7c's
trim_tail item into 7b?

## 3. THE SHAPE (r2 survivors, merged)

**3.1 -- `frame_contract_for()` STAYS STATIC.** `tools/engine_matrix.py:145-152`
calls it from the live registry to generate `docs/ENGINE_MATRIX.md`, and the
`--check` drift gate runs in the suite. An env-aware `frame_contract_for` makes
the generated matrix machine-dependent and the gate non-deterministic.
VERIFIED. A NEW `resolved_frame_contract_for(engine, *, env, profile_max_frames)`
carries the resolution.

**3.2 -- the stamp lives BESIDE `coverage_plan`, not inside it.**
`shot["resolved_frame_contract"]`. Reusing `FrameContract` for "resolved for
this box" contradicts its own opening docstring (`frame_contract.py:1-12`) and
destabilises the generated matrix. Keys, exactly: `min_frames`, `max_frames`,
`quantum`, `discrete_frames`, `native_fps`, `allow_tail_trim`, `continuity`,
`engine_id`, and the ceiling's SOURCE (`literal` | `env:<VAR>` | `profile`).

**3.3 -- `assert_coverage_plans` keeps the live guard AND gains the comparison.**
`tests/test_multiclip_coverage_stamp.py:248-262`
(`test_render_boundary_rejects_a_plan_the_LIVE_contract_now_refuses`) pins the
existing behaviour: narrow the live contract after stamping and it must raise
`"cannot execute"`. Stamped-only validation deletes that test's reason to
exist. Validate BOTH.

**3.4 -- `OTR_FORCE_ENGINE_MAP` needs NO second check.** Verified:
`resolve_final_shot_engines` runs `assert_coverage_plans` after
`apply_engine_override` on the legacy path (`render_driver.py:3501`, `:3511`),
and the frozen path fails on routing-env drift instead of rewriting
(`render_driver.py:3324-3334`). Covered by
`tests/test_multiclip_coverage_stamp.py:289-316` and
`tests/test_route_freeze_wiring.py:324-335`. One seat proposed re-PARTITIONING
the plan against the forced engine at render time -- **REJECTED**: re-planning
after the stills are minted is the silent re-plan this build exists to remove.
Refusing is correct and already happens.

**3.5 -- `eng_ltx_av.py:57-60` is a real, fork-INDEPENDENT defect.**
`_LTX_AV_MAX_FRAMES = int(os.environ.get(...))` at module scope means a
malformed value is an import-time `ValueError` -- the engine does not fail
closed with a named message, it fails to exist, and `frame_contract_for` then
resolves the whole adapter to `SINGLE_ONLY` via its swallowed-import path. This
lands FIRST because it needs no fork decision.

## 4. SLICES -- each green and pushed on its own

| # | slice | depends on the fork? |
|---|---|---|
| 7b-1 | `eng_ltx_av` import-time env crash -> lazy safe parse, contract literal 497 untouched | NO |
| 7b-2 | GAP 1: the single-segment coverage proof (see the r3 question in section 2) | NO |
| 7b-3 | `resolved_frame_contract_for` + per-engine resolver specs; `frame_contract_for` untouched | yes |
| 7b-4 | plumb `policy` into `_stamp_coverage_plan`; stamp `shot["resolved_frame_contract"]` | yes |
| 7b-5 | `assert_coverage_plans`: keep the live guard, add stamped-vs-live | yes |
| 7b-6 | guard tests documenting the deferrals (boomerang -> 7c; force map already closed) | NO |

## 5. MUTATION TARGETS (name them now, prove them at the push)

Each must be shown to FAIL for the right reason when reverted:

1. resolver ignores the env cap -> the ceiling test must fail.
2. resolver ignores the profile cap -> the `OTR_ACTIVE_PROFILE` test must fail.
3. `assert_coverage_plans` reads the stamp only -> `test_render_boundary_rejects_a_plan_the_LIVE_contract_now_refuses` must fail.
4. `assert_coverage_plans` reads live only -> the new cross-process test must fail.
5. `frame_contract_for` made env-aware -> the `engine_matrix --check` drift gate must fail.
6. `OTR_LTX_AV_MAX_FRAMES` malformed -> import must NOT crash, and the new test must fail if the lazy parse is reverted.
7. the single-segment proof deleted -> a short single-clip beat must stop being caught.

## 6. WHAT r3 MUST ANSWER

1. **The GAP 1 question in section 2** -- prove against the ASK, or wire
   `render_frames` into the single path and pull trim_tail forward? Name the
   tests each choice moves.
2. Is `if got and got != ...` (`render_driver.py:2952`) a fail-OPEN hole? A
   clip reporting no `frame_count` skips the check. How many of the 31 engines
   actually return a real `frame_count` from `render_clip`? Name them, or name
   the test that would.
3. Exact signature and per-engine precedence table for
   `resolved_frame_contract_for` -- LTX caps/snaps per call
   (`eng_ltx_video.py:155-179`); WAN env outranks profile then clamps
   (`eng_wan_ti2v.py:386-402`); LTX-8GB allows env to 16384 against a static 161
   (`eng_ltx_8gb.py:257-258`); HuMo bare-`int()` (`eng_humo.py:475-478`).
   What does each do with a malformed value?
4. Does 7b-2 (GAP 1) belong BEFORE 7b-3..5, given it is the only slice that
   changes production behaviour for beats that actually run today?

## 7. GROUND RULES

Cite `file:line` from the real tree at
`C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio`. A claim
with no line number is dropped; a claim whose line number does not say what it
is said to say is dropped louder. Two r2 seats between them produced eight
verified findings and two that did not survive grounding -- one of which was the
driver's own. Prefer the claim you can point at.

THE LAW holds: an audit may improve a story, never fail one for length,
language, style, visual vocabulary, or quality. UTF-8 no BOM, ASCII, SFW.
$0 external spend.
