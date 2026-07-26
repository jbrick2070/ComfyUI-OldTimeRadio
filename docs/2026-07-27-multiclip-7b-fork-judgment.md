# JUDGMENT -- the multi-clip 7b architecture fork

**Decision: NEITHER A NOR B AS WRITTEN. Option C, and the ordering changed.**

Arc: r2 (agy + codex) -> r3 (agy + codex `gpt-5.6-sol`). Driver anchor written
BEFORE the r2 fan-out and preserved at
`kibitz-runs/2026-07-26-mc7bfork-agy/r2/driver_anchor.md`. Every claim below was
verified against the tree at HEAD by reading it; claims that did not survive are
recorded as such, including one of the driver's own.

---

## 1. THE DECISION

**Option A (REFUSE on env-vs-declaration) is CUT.** Both seats and the anchor
agree, for the same three reasons: it breaks six documented operator-knob tests
that exist because an 8GB leg already died in production without them; it
cannot reach `OTR_ACTIVE_PROFILE` at all; and it enumerates its inputs, so it is
permanently one new environment variable behind.

**Option B is ADOPTED IN PART, and demoted from "the fix" to "an
optimisation".** The reason is the finding in section 2: the divergence B was
written to prevent is already terminal on the multi-segment path. B's remaining
value is real but narrower than advertised -- it moves an existing refusal
earlier than the GPU work.

**Option C, what actually lands:**

1. Close the proof gap on the path production actually runs (single-segment).
2. Resolve env + profile ONCE, in a NEW `resolved_frame_contract_for`, leaving
   `frame_contract_for` static so the generated engine matrix stays
   machine-independent.
3. Stamp the resolved contract BESIDE `coverage_plan`, not inside it.
4. Keep the existing live-contract guard at the second boundary and ADD a
   stamped-versus-live comparison. Both, not either.

---

## 2. THE FINDING THAT REFRAMED THE FORK -- driver, not the panel

`render_driver.py:2952-2958`, inside `render_beat_coverage`'s segment loop:

```
got = int((clip or {}).get("frame_count") or 0)
if got and got != int(segment.render_frames):
    raise RenderError("shot %s segment %d rendered %d frame(s) but its plan "
                      "asked for %d. NO FALLBACK ...")
```

This compares the OUTPUT to the plan, so it already catches all fifteen env
vars, the profile ceiling, the boomerang, and the provider clamps -- in one
predicate, with nothing enumerated. It was landed by the 2026-07-26 QA panel
and neither r2 seat found it; the fork write-up did not know it existed.

Predicted in the anchor before the fan-out ("neither seat will find
`render_driver.py:2952-2958`") and confirmed. What remains:

* **GAP 1 -- the SINGLE-segment path is not proven at all.**
  `render_driver.py:2861-2872` early-returns to `render_shot`. Every beat in
  production today is single-segment, so the proof that exists covers the path
  nobody is on.
* **GAP 2 -- the check fires after the GPU work.** This, and only this, is
  what plan-time resolution buys.

## 3. THE FAIL-OPEN HOLE INSIDE THE PROOF -- agy r3, CONFIRMED

`if got and got != ...` skips the check entirely when a clip reports
`frame_count` 0 or absent. A fail-OPEN guard inside a fail-closed function, the
same shape as the four swallowed fail-closed sites chunk 1a's QA found. agy
raised it from the anchor's own open question and it is real. How many of the
31 adapters return a truthful `frame_count` from `render_clip` is UNKNOWN and
must be established, not assumed -- `eng_ltx_video.py:1219-1220` does; that is
one of thirty-one.

## 4. GAP 1 IS SIMPLER THAN BOTH THE DRIVER AND agy THOUGHT -- codex r3

The r3 plan's central open question ("does 7b prove against what was ASKED, or
wire `render_frames` into the single path and pull trim_tail forward?") **rests
on a false premise, and codex r3 killed it.** VERIFIED:

`build_request_from_shot` calls `frame_count = segment_render_frames(shot,
segment_index)` unconditionally (`render_driver.py:2213-2215`), and
`segment_render_frames` (`render_driver.py:800-837`) says in its own docstring:
*"When a plan exists it answers from the plan for EVERY index, segment 0
included"* -- the index-0 short-circuit was already removed by the chunk-6c
loop's end-to-end test. So on the single path the engine is ALREADY asked for
`plan.segments[0].render_frames`, not the beat target.

Consequences:
* The driver's trim_tail coupling **does not exist**. Struck.
* agy r3's fix ("wire `segment_index=0` into single-clip request generation")
  is **unnecessary** -- 0 is already the default and already reads the plan.
* GAP 1 collapses to: compare the canonical output frame count against
  `plan.segments[0].render_frames`. Trim application stays in 7c.

Note the seat asymmetry: the r2 codex seat ran on `gpt-5.5` and did not find
this; the r3 seat on `gpt-5.6-sol` did. See section 9.

## 4b. THE ORDERING -- the two seats disagreed; agy wins on codex's own standard

agy r3 said put the single-segment proof AFTER the resolver; codex r3 said keep
it first because it depends only on the existing stamped plan. agy's reasoning
is a concrete production consequence, verified:

On an 8GB box with `OTR_WAN_TI2V_MAX_FRAMES=49`, a 177-frame beat is stamped
today as a legal SINGLE-segment plan (`min=17 quantum=4`, so `(177-17) % 4 == 0`
-- legal under the static 177 ceiling). `_floor_length` then clamps the render
to 49 and the composite's ping-pong fills the rest. Landing the proof first
turns that lane into a hard refusal **with no remedy available**, because only
the resolver can make the plan say 49-and-multi-clip.

The deciding standard is codex's own: every slice green and pushed on its own.
A slice that cannot be made green on a supported box is not a green slice, it
is a known-fail. **Order: 7b-1 (done), the resolver, the stamp, THEN the
single-segment proof, THEN the boundary comparison.**

## 5. CLAIMS THAT DID NOT SURVIVE GROUNDING

**5.1 -- the DRIVER's own, and the most important one.** Problem-statement
section 5.5 and anchor bullet A1 both claimed live VRAM silently shortens a
render. **False.** `compute_real_frame_budget` (`motion_common.py:321-363`) was
rewritten by S4 platform-portability on 2026-07-10 to kill exactly that; its
docstring says *"NEVER a VRAM-adaptive resize. The pre-S4 version silently
shrank tight-VRAM clips toward the floor"*, and it now RAISES
`MotionBudgetError`. Pinned by `tests/test_wan_ti2v.py:229-252` and
`tests/test_remaining_video_contracts.py:173-189`. Caught by codex r2. VRAM is
fail-loud and out of scope. **The judge was wrong and the panel was right.**

**5.1b -- the DRIVER's second wrong claim.** The r3 plan's headline open
question (ASK vs plan on the single path, and a trim_tail coupling) was built
on a premise that is false in the code. See section 4. Two of the driver's own
load-bearing claims were refuted by the panel this arc; both times the panel
was right. That is the arc paying for itself, and it is the reason the standing
rule is to judge every claim against source INCLUDING one's own.

**5.2 -- "clamp the boomerang to the resolved ceiling" (agy r2). REJECTED.**
`tests/test_ltx_boomerang.py:48-57` pins `2*src-1 >= target` for exactly
`target=169`, and names the bug it exists for: *"the 169 -> half 85 -> snap 81
-> 161 < 169 FREEZE the roundtable caught"*. Clamping down trades a declared-
ceiling violation for a returning visible-freeze. Deferred to 7c with a
tripwire test instead (landed, section 7).

**5.3 -- "re-partition the plan against a forced engine at render time"
(agy r2). REJECTED**, and agy itself reversed this at r3. Re-planning after the
stills are minted is the silent re-plan this build exists to remove. Refusing
is correct and already happens.

**5.4 -- "add a force-map coverage check" -- NOT NEEDED, verified.**
`tests/test_multiclip_coverage_stamp.py:288-316`
(`test_the_legacy_path_validates_the_plan_against_the_FINAL_engine`) already
forces `ltx_video` with a narrowed contract and asserts `RenderError` after the
override, then asserts the shot really is on the forced engine. Question 7 of
the problem statement is answered: the boundary is closed, the test exists, and
a second check would be the duplicate-authority shape this build removes.

## 6. WHAT BOTH SEATS GOT RIGHT THAT THE DRIVER HAD NOT

* **`frame_contract_for` must stay static** (codex r2). `tools/engine_matrix.py:145-152`
  calls it from the live registry to generate `docs/ENGINE_MATRIX.md`, and the
  `--check` drift gate runs in the suite. An env-aware `frame_contract_for`
  makes a GENERATED, COMMITTED document machine-dependent. Verified. This alone
  rules out the most obvious implementation of B.
* **The stamp needs an explicit schema before any code** (both seats).
* **The six constraining tests stay green untouched** via resolver delegation
  (both seats, independently). Neither proposed rewriting them, which was the
  failure mode the problem statement was most worried about.
* **`eng_ltx_av`'s import-time env parse is a real, fork-independent defect**
  (both seats). Landed; see section 7.

## 7. WHAT LANDED THIS WINDOW

**`499541b6` -- 7b slice 1: a malformed env value may not take an adapter's
IMPORT down.** Four module-scope bare `int()`/`float()` parses in
`eng_ltx_av.py` became a safe parser with a LOUD warning and the declared
default. The failure mode was invisible rather than loud: a `ValueError` during
import meant the adapter never registered, and `frame_contract_for` answers
`SINGLE_ONLY` for an adapter it cannot reach (`frame_contract.py:243-247`), so
one typo silently removed an engine and reverted its lane to unbounded
single-clip. Fixed for all four constants, not just the one the panel named.
Mutation-proven: restoring the bare `int()` fails exactly the three
`OTR_LTX_AV_MAX_FRAMES` tests and leaves the other seven passing.
Suite 6913 -> 6923.

**7b slice 6 -- the 7c deferral tripwire.** `ltx_video` declares a 169 ceiling
and returns 193 frames for a 169 ask with no env set at all, by default. Two
tests in `tests/test_ltx_boomerang.py` pin that this is true TODAY, explain why
clamping is rejected (5.2), and instruct the 7c author to DELETE them in the
commit that removes the boomerang rather than relax them.

## 7b. FOUR BLOCKERS ON THE RESOLVER SLICE -- codex r3, ALL FOUR VERIFIED

Every one of these was checked against source this window. **The resolver slice
must not start until they are answered**, because each one silently defeats it.

**B1 -- the canonical workflow never wired `max_render_frames`.**
`OTR_VideoDirector.INPUT_TYPES` declares it last in `optional`
(`otr_video_director.py:302-318`), but node 87 in `workflows/otr_canonical.json`
has NO input descriptor for it -- its `inputs` array stops at `dtype_policy`.
`widgets_values` nonetheless carries a 15th trailing `0`: the default, sitting
there unbound. So the profile-ceiling channel the whole of Option B rests on
(director widget -> ledger -> `prepare` -> `profile_max_render_frames()`) is
**not operable from the canonical workflow today.** Per the operator's standing
rule, the node/wiring/widget change goes IN `otr_canonical.json` in the SAME
commit as the code, with the widget/link audit and `OTR_WorkflowValidator` run.
Unwired code is dead code, and this one is already dead.

**B2 -- ComfyUI will serve a STALE plan across a frame-cap env change.**
`OTR_ShotLock.IS_CHANGED` (`otr_shot_lock.py:1412-1422`) fingerprints only
`route_freeze.routing_env_snapshot()`, and that snapshot contains exactly two
variables -- `OTR_FORCE_ENGINE_MAP` and `OTR_ENABLE_HUMO_HOSTS`
(`route_freeze.py:50-51, 63-76`). Change `OTR_WAN_TI2V_MAX_FRAMES` and ShotLock
does not re-lock, so a cached coverage plan and its stamp survive the change
that was supposed to move them. `route_freeze.py:46-48` carries the standing
warning verbatim: *"Any new routing env var MUST be added here AND to the
snapshot, or it escapes the freeze ... an env read outside the snapshot is a
hole in the ledger."* The resolver's env set must be captured ONCE, used for
resolution, and fingerprinted into `IS_CHANGED`.

**B3 -- both plan boundaries currently SWALLOW what 7b intends to make
terminal.** `_stamp_coverage_plan` catches contract exceptions and emits no
plan at all (`otr_shot_lock.py:1150-1155`, `except Exception: return`), and
`assert_coverage_plans` catches live-contract exceptions and falls back to
arithmetic-only validation (`render_driver.py:3430-3438`, `contract = None`).
A resolver that raises would be absorbed at BOTH ends and the build would look
fine. This is chunk 1a's exact lesson -- *when you make something newly
terminal, grep every caller for a broad catch in the SAME change* -- and here
the two catches are already identified, so there is no excuse for repeating it.

**B4 -- the output-equality proof is only as good as `frame_count`, and for 13
of 31 engines that number is an ESTIMATE.** `eng_cloud_video.canonicalize`
computes `frame_count = int(round((asset.duration_s or 0.0) * (asset.fps or
0.0)))` (`eng_cloud_video.py:491`) -- duration times fps, not a counted frame.
`docs/ENGINE_MATRIX.md` puts the roster at 18 local and 13 provider engines.
So a provider-side clamp moves the duration AND the derived count together and
the proof cannot see it, while rounding alone can trip it spuriously. The tool
to fix this ALREADY EXISTS: `ffprobe_counted_frames` in `wan_shared.py`, landed
by chunk 6a for the assembly boundary. This is a wiring job, not a new
capability -- but it must be done before the single-segment proof is trusted,
or the proof is decorative for 13 lanes.

Related and also verified: codex r3's note that plan-time rejection prevents
**paid provider submission**, not merely GPU work. With 13 provider engines,
that materially raises the value of GAP 2 (moving detection to plan time) above
how the driver priced it.

## 8. STILL OPEN

**THE EXECUTION ORDER, settled:**

1. B1 -- wire `max_render_frames` into `workflows/otr_canonical.json` node 87.
   Nothing downstream works without it and it is the cheapest of the four.
2. B4 -- make `frame_count` a COUNTED number via the existing
   `wan_shared.ffprobe_counted_frames`, and close the `if got` fail-open
   (section 3) in the same change; they are one predicate.
3. The resolver -- `resolved_frame_contract_for`, `frame_contract_for` left
   static, WITH the complete per-engine precedence table written first
   (codex r3 MUST-FIX 5 lists the six families and their current behaviours),
   and B3's two swallowing catches fixed in the SAME change.
4. The stamp beside `coverage_plan`, plus B2's `IS_CHANGED` fingerprint.
5. The single-segment proof (now trivial, per section 4).
6. The boundary comparison: keep the live guard, add stamped-vs-live.

**Also open:** codex r3 MUST-FIX 4 -- the resolver needs a `request_template`
argument, because Veo coerces to 8 seconds for 1080p/4K or reference-image
requests (`eng_google_veo_video.py:244-273`), so a stamp resolved from env and
profile alone can still disagree with the adapter. Verify before adopting.

**Also open:** codex r3 MUST-FIX 9 -- a wording defect in the r3 plan, not in
the shipped code. Section 3.5 of that plan said "lazy safe parse"; slice 7b-1
deliberately shipped a NON-lazy, non-fail-closed safe parse and said so in its
docstring, because choosing a malformed-value policy is the fork's job, not a
crash fix's. The plan's wording was wrong; the code is deliberate. The
malformed-value policy decision moves to the resolver slice, where it belongs.

**Corrected mutation target (codex r3 SHOULD-FIX 2):** "make
`frame_contract_for` env-aware -> `engine_matrix --check` must fail" is
NONDETERMINISTIC as written -- with the env absent the matrix is unchanged and
the mutation survives. The mutation test must SET a non-default env and assert
the generated matrix does not move.

## 9. A PROCESS DEFECT WORTH MORE THAN ONE FINDING

The r2 codex seat ran on **`gpt-5.5`, not the `gpt-5.6-sol` of record.** Cause:
`kibitz/scripts/kibitz.py`'s `CODEX_MODEL_PREFERENCE` tuple read
`("gpt-5.5", "gpt-5-codex", "gpt-5")` while the live catalog already carried
`gpt-5.6-sol`, `-luna` and `-terra`. Every arc since that catalog shipped has
quietly run the older model, and the only evidence was one line in
`codex_model_selected.txt` -- which is precisely why GO_FORWARD's budget table
says to check it every arc.

Worse, the AUTO-PICK FALLBACK cannot be trusted to age out of this either: it
takes the highest `gpt-5*` slug by reverse sort, which would have selected
`gpt-5.6-terra` -- alphabetically last, not strongest.

Fixed at the root in `kibitz/scripts/kibitz.py` (preference tuple, with the
reasoning in a comment) and pinned belt-and-braces via `KIBITZ_CODEX_MODEL` for
r3, which confirms `gpt-5.6-sol`. **NOTE FOR THE OPERATOR: `kibitz/` is
UNTRACKED in this repo, so that fix is NOT in any commit and will not survive a
fresh clone.** It belongs upstream in the kibitz skill.
