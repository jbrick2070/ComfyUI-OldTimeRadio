# PROBLEM STATEMENT -- the multi-clip honesty rule fails the honest beats

**Date:** 2026-08-06
**Driver:** Claude (Cowork), CODER window, rung 4.
**Status:** mechanism CONFIRMED by direct file read; found by a 5-surface bug hunt and
survived an adversarial refutation that attacked it on six axes.
**Gate:** full `kibitz-plugin:kibitz` four-round arc before code.

---

## 1. THE DEFECT

`grade_multiclip_honesty` exists to catch a lane that PADS a beat -- mirroring or
holding frames to fill a duration it did not really render -- while claiming real
multi-clip coverage. It compares two numbers:

```python
# nodes/_otr_video_engines/acceptance.py:177-178
native = row.get("native_frame_count")
if native is not None and int(native) != int(row.get("frame_count") or 0):
```

**Those two numbers are produced at different SCOPES.** `frame_count` is the whole
assembled BEAT; `native_frame_count` is one SEGMENT.

`render_beat_coverage` builds the assembled beat by copying the LAST segment's clip and
overwriting only four keys:

```python
# nodes/_otr_video_engines/render_driver.py:3483-3490
beat_clip = dict(clip or {})          # <- clip is the LAST segment
beat_clip["path"] = assembled
beat_clip["frame_count"] = int(plan.target_visible_frames)   # <- whole beat
beat_clip["segment_count"] = int(plan.segment_count)
beat_clip["join_mode"] = str(plan.join_mode)
```

`native_frame_count` and `extension_mode` are **never recomputed for the assembled
beat**, so they keep the last segment's values. `build_clip_manifest` then copies both
verbatim (`render_driver.py:4509-4510`).

**Result: the rule fires on exactly the beats that prove it was satisfied.**

## 2. FAILURE SCENARIO (concrete)

A `wan_ti2v` beat of 241 visible frames is planned as three native segments of 81
(`PLANNING_CAP_ENGINES` includes `wan_ti2v`; the shipping profile pins
`max_render_frames: 81`). Each segment renders 81 REAL frames and stamps
`native_frame_count=81`, `extension_mode="none"`. `assemble_beat_segments` decode-proves
241 frames.

The manifest row then says `frame_count=241`, `native_frame_count=81`. The grader emits:

> `multiclip_honesty <shot_id>: this beat delivered 241 frame(s) of which only 81 were
> rendered, while declaring extension_mode='none'`

and `scripts/grade_episode.py` exits 1 -- **FINDINGS on an episode where all 241 frames
are original rendered video.**

## 3. WHY IT IS REACHABLE ON THE SHIPPING PATH

**Corrected after r1 -- this section previously said "LIVE", which it does not prove.**
Source inspection establishes that the path EXECUTES. It does not establish that a
production failure occurred, and under the admission rule in `docs/PROD_BUG_LOG.md` this
stays a STATIC finding until a retained ledger, manifest, grader result and asset exist.

* `frame_contract.PLANNING_CAP_ENGINES` includes `wan_ti2v` specifically so its beats
  are split into affordable native segments.
* `config/profiles/otr_g4_wan_ti2v.json` pins `max_render_frames: 81`, and `wan_ti2v`
  declares strict-first-frame continuity, so it is chainable -- any beat longer than 81
  frames at 25 fps is planned multi-segment, which is most of them.
* **CORRECTED (Codex r1 M7):** an earlier draft claimed a durable campaign runner
  invokes `scripts/grade_episode.py` per leg. **It does not.**
  `scripts/otr_w45_campaign.py` never calls it; the only runner call is in the temporary
  `tmp/_w45_campaign.ps1`. The grader is a **manual post-run gate**. That materially
  lowers the severity: the verdict is wrong whenever a human runs it, and the durable
  manifest receipt is wrong always, but no campaign is silently failing on it.

## 4. THE SECOND, QUIETER HALF

Even with the grader set aside, **the durable `clip_manifest.json` row is itself
wrong.** It asserts that a 241-frame beat contains 81 natively rendered frames. Anything
that later trusts that receipt -- a coverage audit, a reuse detector, a future
qualification gate -- inherits the lie. The receipt exists precisely to be the evidence a
pad cannot forge, and today it misreports every honest multi-segment beat.

## 5. THE SHAPE OF THE FIX -- REWRITTEN AFTER r1

**The original proposal here was WRONG and the panel killed it.** It said
`native_frame_count = SUM(segment native counts)`. Three chained 81-frame renders do
**243** frames of work and deliver **241**, because two duplicated successor head frames
are dropped at the seams -- so summing fails the equality test for a second, different
reason. The driver anchor, Codex and Antigravity reached that independently.

The corrected shape:

1. **Accumulate per-segment receipts in the render loop.** Today `segment_row` omits
   `native_frame_count` and `extension_mode` entirely, so assembly has nothing to
   aggregate.
2. **Mint the beat receipt once at assembly, in DISTINCT counts.** Bug Bible **12.69**
   and `docs/PRODUCTION_SPRINT_LESSONS.md:540-562` already require requested / rendered /
   visible / trimmed to be kept apart rather than conflated. This defect IS that rule
   being violated; the fix restores it rather than inventing new vocabulary.
3. **The honest quantity is DELIVERED-NATIVE frames:** for each segment, intersect its
   native prefix with `[drop_head, drop_head + visible)` and sum the survivors. Padding
   entirely removed by trimming must not poison the beat; padding that survives must.
4. **Fix the RULE, not only the receipt.** `native == frame_count` is a single-segment
   proxy for "was any of this duration manufactured?".
5. **`extension_mode` aggregation must be deterministic:** `"none"` if all none; the
   shared mode if all non-none agree; `"mixed"` otherwise.
6. **A MISSING receipt must be a FINDING** inside the rule's scope. Today the check is
   guarded by `native is not None`, so a beat with no receipt and `extension_mode="none"`
   passes -- which contradicts the fail-closed model the receipt exists to serve.

**Open questions for the panel:**
1. Where do the per-segment receipts live at assembly time? The loop binds `clip` per
   segment; is there a retained list, or must one be accumulated?
2. Is summing correct when a segment is TRIMMED to fit the target? A trimmed tail means
   fewer visible frames than rendered -- does `native` describe frames RENDERED or
   frames DELIVERED? The rule's wording ("of which only N were rendered") implies
   delivered-vs-rendered, so the two may legitimately differ and the equality test may
   be the wrong test entirely.
3. Should a single-segment beat with a tail trim be covered too? The grader currently
   skips `segment_count <= 1`, so that case is silently unchecked.
4. Does any other consumer read `native_frame_count` expecting SEGMENT scope? A
   repo-wide grep found writers in the two WAN adapters and `wan_shared`, and readers in
   `render_driver` and `acceptance` only -- but that must be re-confirmed.

## 6. THE TEST SITUATION IS PART OF THE DEFECT

`tests/test_wire_w5_acceptance_grader.py` builds synthetic rows where the "clean"
multi-segment case has `native_frame_count == frame_count == the whole beat`. That is
the OPPOSITE of what production emits, so the suite is green while every real
multi-segment beat is misgraded. **Any fix must add a test built from the shape
`render_beat_coverage` actually produces**, not from a hand-written row that assumes the
conclusion.

## 7. RECEIPTS

* `code-complete + suite-green` -- achievable offline.
* **Live proof needs a multi-segment WAN beat**: one leg long enough to plan >1 segment,
  then `scripts/grade_episode.py` returning ACCEPTED on it, with the manifest row
  showing `native_frame_count == frame_count` and every frame accounted for.
