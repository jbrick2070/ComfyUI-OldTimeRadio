# The FRAME_COST_MODEL refusal -- Codex consult, 2026-08-13

Ran because the plan's prescribed fix (delete the two rows) was a
no-op. Codex confirmed the no-op and REFUTED the split I proposed:
it argued both refusals must be gated, not just the per-frame one.
I took its answer. Shipped as 9eea64a4.

Driver: Claude (Cowork) wrote the anchor against the real Windows
files first and remains the judge; every claim below was ground-
checked before anything was folded in. Reviewer: Codex CLI, which
the operator's 2026-08-11 routing makes the consult of record for a
quandary in place of a full kibitz arc.

---

## Part 1 -- the anchor, as sent

# Codex consult -- the FRAME_COST_MODEL refusal, and why "delete the row" is a no-op

You are reviewing a REAL Windows repo at
`C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio`.
Read the actual files before answering. Do not take my summary on trust --
every line reference below is checkable and I want you to check it.

## The live failure this is meant to fix

Two production render legs died last night with `MotionBudgetError`:

```
MotionBudgetError: static frame budget 69 (snapped 69) exceeds the cost-model's
affordable 18 frames (free=9549 MB, margin=0.85)
```

Engines: `fastwan_8gb` and `wan_ti2v`. The project plan concluded, correctly,
that this is NOT a VRAM fault -- both engines would refuse the same beat on a
completely empty 16 GB card, because a 125-frame beat prices at
`7000 + 125*185 = 30,125 MB`, nearly twice the card.

The plan's prescribed fix, verbatim, is:

> **Delete or replace the two `FRAME_COST_MODEL` rows.** Two red legs turn green.
> The repo's own docs say empty is correct.

## Why I believe that prescription is WRONG AS WRITTEN

`nodes/_otr_video_engines/motion_common.py`:

```
264  FRAME_COST_MODEL = {
265      "wan_ti2v": (7000.0, 185.0),
266  }
269  _DEFAULT_FRAME_COST = (7000.0, 185.0)
319      overhead, per_frame = FRAME_COST_MODEL.get(engine_name, _DEFAULT_FRAME_COST)
```

`nodes/_otr_video_engines/eng_fastwan_8gb.py:296` adds the second row, and its
own comment at :291-294 says the row "is byte-identical to the default today
and buys only explicitness".

So `_cost_model_for()` falls back to the SAME tuple. Deleting both rows changes
nothing at runtime; both legs fail identically. I want you to confirm or refute
that specific claim by reading the code.

## The second thing I think I have found: TWO refusal paths, ONE authority

The repo already built the right abstraction and then only wired it to one of
the two call sites.

* `motion_common.py:367` -- `QUALIFIED_COST_ROWS = frozenset()` with a long
  docstring saying the wan_ti2v row is DISQUALIFIED in writing, that "an 'is
  there a row?' test admits a row whose own author disqualified it", and that
  "Empty is the correct value until a row is re-measured through the real
  `prepare()` + `render_clip()` lifecycle".
* `motion_common.py:370` -- `cost_row_may_refuse(engine_name)` returns
  membership in that frozenset.

Call site A -- the coverage-planned beat admission boundary,
`render_driver.py:3332` -- DOES consult it, and returns "admission NOT enforced"
for every engine. Inert, by design.

Call site B -- `eng_wan_ti2v.py:801`, inside `_floor_length()`, calls
`motion_common.compute_real_frame_budget(...)` DIRECTLY. That function's refusal
branch (`motion_common.py:472-480`) never consults `cost_row_may_refuse`. This
is the path that raised on both legs (the message wording "static frame budget
%d (snapped %d)" is unique to it).

`fastwan_8gb` inherits `_floor_length` from the wan_ti2v engine class.

## The candidate fix I am weighing

Gate the ENFORCEMENT inside `compute_real_frame_budget` on
`cost_row_may_refuse(engine_name)`, so both paths answer to one authority:

* keep the malformed-cost-model check (`motion_common.py:452-460`, non-finite or
  negative overhead/per_frame/margin) UNCONDITIONAL -- it is a configuration
  error, not a budget answer;
* make the fixed-overhead refusal (`:462`) and the per-frame refusal (`:472`)
  fire only when the engine's row is QUALIFIED;
* when unqualified: predict, log, return the geometry-snapped length, refuse
  nothing.

Seed rows stay in the table, because the `QUALIFIED_COST_ROWS` docstring
explicitly argues for keeping them ("a disqualification that is written down
teaches more than a deletion that leaves no trace").

## What I need from you -- answer each, grounded in the files

1. **Confirm or refute the no-op claim.** Does deleting the two
   `FRAME_COST_MODEL` rows change any runtime behaviour at all?
2. **Is call site B really ungated, and is it really what raised?** Trace
   `_floor_length` -> `compute_real_frame_budget` and check the exception text
   against `motion_common.py:475-480`.
3. **Is gating enforcement on `cost_row_may_refuse` the right root fix, or is it
   too wide?** With `QUALIFIED_COST_ROWS` empty, this makes
   `compute_real_frame_budget` unable to refuse for ANY engine. The stated fear
   in this repo is that an in-process CUDA OOM corrupts ComfyUI's caching
   allocator, so refusal-before-load is the only guard. Am I removing a real
   guard, or a guard that was already enforcing a number nobody stands behind?
   Is there a THIRD option better than both (e.g. leave the fixed-overhead
   refusal armed and gate only the per-frame one)?
4. **What breaks?** Find every test that asserts the CURRENT refusing behaviour.
   I already know of `tests/test_clip_fill.py` (several) and
   `tests/test_vram_admission_boundary.py:70-72`. Are there others -- including
   any test that asserts a lane preflight row, an evidence-manifest claim, or a
   doc-consistency check that would go stale?
   `scripts/build_video_evidence_manifest.py:276-286` writes prose about this.
5. **Is there any OTHER consumer** of `compute_real_frame_budget` /
   `assert_frame_affordable` / `FRAME_COST_MODEL` that would silently change
   meaning -- especially anything that reads the table's emptiness as a signal?

Be adversarial. If my framing is wrong, say so plainly and say what the real
defect is. Cite file:line for every claim.

---

## Part 2 -- Codex's review, verbatim

## Verdict

Your diagnosis is correct. The prescribed “delete the two rows” fix is a render-time no-op because both engines immediately fall back to the same `(7000.0, 185.0)` tuple. The real defect is that `compute_real_frame_budget()` enforces an explicitly disqualified model without consulting the qualification authority.

### 1. Deleting the rows

Deleting the rows does not change any render or admission calculation today:

- `wan_ti2v`’s row is `(7000.0, 185.0)`, and `_DEFAULT_FRAME_COST` is exactly the same tuple: [motion_common.py:264](/C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_video_engines/motion_common.py:264), [motion_common.py:267](/C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_video_engines/motion_common.py:267).
- `_cost_model_for()` uses `FRAME_COST_MODEL.get(engine_name, _DEFAULT_FRAME_COST)`: [motion_common.py:317](/C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_video_engines/motion_common.py:317).
- FastWan injects the same tuple, explicitly described as byte-identical to the default: [eng_fastwan_8gb.py:289](/C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_video_engines/eng_fastwan_8gb.py:289), [eng_fastwan_8gb.py:296](/C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_video_engines/eng_fastwan_8gb.py:296).

So both failed calls would compute the same overhead, slope, affordable frame count, and exception after deletion.

Deletion is observable only through direct table introspection. It would break the row-presence assertions in [test_fastwan_8gb.py:78](/C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/tests/test_fastwan_8gb.py:78) and [test_vram_admission_boundary.py:58](/C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/tests/test_vram_admission_boundary.py:58). No production consumer reads table emptiness or membership as a signal.

Therefore: “delete the rows” is not merely incomplete; as written, it cannot turn either red leg green.

### 2. Call site B is ungated and produced this message

The runtime trace is:

1. `FastWan8gbEngine` subclasses `WanTi2vEngine` and does not override `_floor_length()`: [eng_fastwan_8gb.py:120](/C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_video_engines/eng_fastwan_8gb.py:120).
2. On the non-`multi_clip` render path, `render_clip()` calls `_floor_length()`: [eng_wan_ti2v.py:1057](/C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_video_engines/eng_wan_ti2v.py:1057), [eng_wan_ti2v.py:1072](/C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_video_engines/eng_wan_ti2v.py:1072).
3. `_floor_length()` reads free VRAM, adds any hoist correction, and calls `compute_real_frame_budget()` directly: [eng_wan_ti2v.py:798](/C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_video_engines/eng_wan_ti2v.py:798), [eng_wan_ti2v.py:801](/C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_video_engines/eng_wan_ti2v.py:801).
4. There is no qualification check anywhere in that chain.
5. The exception text at [motion_common.py:475](/C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_video_engines/motion_common.py:475) is exactly:

   `static frame budget %d (snapped %d) exceeds the cost-model's affordable %d frames (free=%.0f MB, margin=%.2f)`

Call site A cannot have produced the live exception under this HEAD: it returns before calling `assert_frame_affordable()` whenever `cost_row_may_refuse()` is false: [render_driver.py:3332](/C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_video_engines/render_driver.py:3332). The registry is empty: [motion_common.py:367](/C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_video_engines/motion_common.py:367).

So yes: call site B is the live refusal path for both engines.

### 3. Correct root fix

Gate both budget refusals inside `compute_real_frame_budget()` on `cost_row_may_refuse(engine_name)`.

The right ordering is:

1. Geometry snap.
2. Preserve the current no-readable-VRAM return.
3. Resolve the cost model and margin.
4. Run the finite/non-negative validation regardless of qualification.
5. Calculate diagnostic prediction data.
6. If unqualified, log it explicitly as non-enforcing and return `snapped`.
7. Only if qualified, enforce both fixed overhead and per-frame affordability.

That matches the declared meaning of the authority: the question is whether a row “may REFUSE a render,” not whether a row exists: [motion_common.py:370](/C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_video_engines/motion_common.py:370).

Do not leave fixed-overhead refusal armed. That is not a better third option because:

- The overhead comes from the same disqualified or borrowed tuple as the slope.
- Missing engines receive that overhead through `_DEFAULT_FRAME_COST`.
- The evidence generator documents that the shipped overhead was derived from an absolute peak even though the guard compares against machine-free memory, double-charging the desktop baseline: [build_video_evidence_manifest.py:123](/C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/scripts/build_video_evidence_manifest.py:123).
- The qualification authority makes no distinction between “qualified slope” and “qualified overhead.” Inventing one now would create a second authority without evidence.

This does remove the only pre-load refusal on `_floor_length()` while `QUALIFIED_COST_ROWS` is empty. But it removes a number nobody has qualified, not a valid safety envelope. The repo already declares the intended state as “NO local lane is guarded”: [evidence/README.md:61](/C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/docs/evidence/README.md:61), [evidence/README.md:66](/C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/docs/evidence/README.md:66).

The actual OOM risk remains real and should be stated plainly. The legitimate restoration path is real-lifecycle measurement followed by qualification—not continued enforcement of the known-false seed.

Two hardening details should accompany the fix:

- `cost_row_may_refuse()` should require both qualification and an explicit table row. Currently a future name can enter `QUALIFIED_COST_ROWS` while absent from `FRAME_COST_MODEL`, silently “qualifying” `_DEFAULT_FRAME_COST`: [motion_common.py:319](/C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_video_engines/motion_common.py:319), [motion_common.py:383](/C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_video_engines/motion_common.py:383).
- Global `OTR_VIDEO_COST_*` overrides can replace the coefficients after an engine name has been qualified: [motion_common.py:320](/C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_video_engines/motion_common.py:320). Before any row is qualified, decide whether such overrides disarm qualification or require a separate explicit consent act.

### 4. What breaks or becomes stale

Current refusing assertions that must change:

- Three tests in `test_clip_fill.py`:
  - pressure refusal: [test_clip_fill.py:44](/C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/tests/test_clip_fill.py:44)
  - fixed-overhead/starvation refusal: [test_clip_fill.py:60](/C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/tests/test_clip_fill.py:60)
  - large-canvas per-frame refusal: [test_clip_fill.py:81](/C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/tests/test_clip_fill.py:81)
- The old “unpinned Wan must raise” regression: [test_remaining_video_contracts.py:193](/C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/tests/test_remaining_video_contracts.py:193).
- The direct assertion that a present-but-disqualified row still refuses: [test_vram_admission_boundary.py:69](/C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/tests/test_vram_admission_boundary.py:69).
- Both `_floor_length()` refusal cases: [test_wan_ti2v.py:234](/C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/tests/test_wan_ti2v.py:234).
- The hoist-correction test’s initial refusal: [test_wire_w3b_wan_ti2v_session.py:467](/C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/tests/test_wire_w3b_wan_ti2v_session.py:467).

The qualified-boundary refusal test should remain and should still pass: it explicitly inserts `wan_ti2v` into `QUALIFIED_COST_ROWS` before expecting a failure: [test_vram_admission_boundary.py:85](/C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/tests/test_vram_admission_boundary.py:85), [test_vram_admission_boundary.py:98](/C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/tests/test_vram_admission_boundary.py:98). The hoist test should similarly qualify the row so it continues testing future enforcing behavior.

Other stale surfaces:

- The evidence generator incorrectly says the static path fires despite being unqualified: [build_video_evidence_manifest.py:281](/C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/scripts/build_video_evidence_manifest.py:281).
- The generated manifest repeats that claim: [video_evidence_manifest.json:14](/C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/docs/evidence/video_evidence_manifest.json:14).
- Lane preflight itself will remain green. It checks `cost_row_may_refuse()` and merely requires a substantive unenforced explanation; it does not assert the static-path wording: [test_lane_preflight_matrix.py:790](/C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/tests/test_lane_preflight_matrix.py:790), [test_lane_preflight_matrix.py:1113](/C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/tests/test_lane_preflight_matrix.py:1113).
- The older production-log regression says unpinned Wan must raise and names `test_remaining_video_contracts.py`; that historical entry needs an append-only correction: [PROD_BUG_LOG.md:2734](/C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/docs/PROD_BUG_LOG.md:2734).
- `GO_FORWARD_PLAN.md`’s deletion prescription is wrong because it overlooks the fallback: [GO_FORWARD_PLAN.md:421](/C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/docs/GO_FORWARD_PLAN.md:421).
- Bug Bible 12.98 likewise prescribes deletion or table absence and therefore encodes the same incorrect fix: [BUG_BIBLE.yaml:7557](/C:/Users/jeffr/Documents/ComfyUI/comfyui-custom-node-survival-guide/BUG_BIBLE.yaml:7557), [BUG_BIBLE.yaml:7565](/C:/Users/jeffr/Documents/ComfyUI/comfyui-custom-node-survival-guide/BUG_BIBLE.yaml:7565). I found no 12.98-specific executable assertion in `tests/bug_bible_regression.py`; the entry appears only in the YAML and coverage index.

There is also no current test proving that malformed finite/negative values remain fatal while an unqualified, merely unaffordable prediction does not. That regression should be added explicitly.

### 5. Other consumers

The complete in-repo executable consumer set is:

- Direct production call: `_floor_length()` → `compute_real_frame_budget()`: [eng_wan_ti2v.py:801](/C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_video_engines/eng_wan_ti2v.py:801).
- Wrapper: `assert_frame_affordable()` → `compute_real_frame_budget()`: [motion_common.py:386](/C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_video_engines/motion_common.py:386).
- Its only production caller: `_assert_beat_affordable()`, already qualification-gated: [render_driver.py:3332](/C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_video_engines/render_driver.py:3332), [render_driver.py:3364](/C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_video_engines/render_driver.py:3364).
- `MotionEngineBase` exposes the function as a static method, but there is no additional production call through that alias: [motion_common.py:504](/C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_video_engines/motion_common.py:504).
- FastWan mutates the cost table at import: [eng_fastwan_8gb.py:296](/C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_video_engines/eng_fastwan_8gb.py:296).

No in-repo runtime consumer treats `FRAME_COST_MODEL` emptiness as a signal. Qualification is read exclusively from `QUALIFIED_COST_ROWS`; table presence and qualification are already designed as separate concepts.

So the root fix is: retain the seed rows, make `compute_real_frame_budget()` qualification-aware, gate both refusal branches, preserve configuration validation, and correct the tests/docs that currently claim the disqualified static path is supposed to fire.
