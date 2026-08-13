# OTR Go-Forward Plan

**Forward-only.** Open work, live bugs, standing operator rules, the budget ladder.
Completed work lives in `docs/HANDOFF_LOG.md` (newest at top) and every prior
revision of this file is in git. If a thing is DONE, it does not belong here.

## THE QUEUE -- OPERATOR-ORDERED 2026-08-07. WORK IT TOP TO BOTTOM.

This order is the operator's, set after the shield-scoping ship. It OVERRIDES
the section numbering below (which is historical) and the older "work by
priority, not by number" line. A window takes the topmost item that is not
blocked on the operator, finishes it green and pushed, then re-reads this list.

**ORDER SET BY THE OPERATOR 2026-08-07** and it now runs straight through
`ROADMAP.md` as ONE runway -- rows 1-9 live here, rows 10-13 are the roadmap's
later runway, reordered to match. A window works the topmost UNBLOCKED item.

| # | Item | Where | Kind | Blocked on |
|---:|---|---|---|---|
**CLOSED, do not re-open:** row 1 (model-slug curation chunk B, `262dfa8f` + `22012263`),
row 8 (multi-GPU upscale stage), row 9-C3 (Macbeth safety probe -- all four live cells
passed, `macbeth_probe` removed from both profiles). Detail lives in `docs/HANDOFF_LOG.md`.

| # | Item | Where | Kind | Blocked on |
|---:|---|---|---|---|

Then, in `ROADMAP.md`: **10** lean-mean/dead-code -> **11** RunPod + AMD/Mac
platform tests -> **12** install path -> **13** product docs + v2 release.

**Items 3, 4, 6 and 7 are blocked on the operator.** (Item 5 left this list on
2026-08-09: H3 is no longer a pending ruling, it is a sprint series whose first
sprint the operator will name -- see 0-QUINQUE.) A coder window that
reaches one without an answer skips to the next unblocked item rather than
guessing.

**READ THIS BEFORE HUNTING FOR WORK (state as of 2026-08-09 late).** With item 1
closed, **THE QUEUE HAS NO UNBLOCKED CODING ROW LEFT** -- every remaining row is
waiting on the operator, and item 9's remainder is held behind Lemmy besides. A
coder window that reaches this line has not run out of work; it has run out of
QUEUE. The unblocked work lives in **STILL OPEN, SMALL, UNSCHEDULED** below and
in the 6-STATUS follow-ups. Take the topmost item there instead of forcing a
blocked row, and do not read the queue's silence as "nothing to do".

**GREP OR MEASURE BEFORE REPEATING A BLOCKER OUT OF THIS FILE.** This file
once asserted for a day that item 1 was blocked on a missing
`OPENROUTER_API_KEY`; the key was in the User env all along and a window
repeated the claim to the operator as fact. It has cost two windows. The
same defect points the other way too: a row describing finished work as
pending costs the next window a re-grounding pass. **When you close an
item, close its ROW in the same push.**

### VIDEO LANE QUEUE (queue item 5) -- ONE LANE OPEN AT A TIME

**ALL 21 PACKETS ARE CLOSED (2026-08-12).** Lanes 0-21 are built, receipted
and pushed; the preflight matrix is 29 engines x 7 gates with ZERO non-green
cells and ZERO `EXPECTED_RED` entries. Per-lane detail lives in
`docs/evidence/lane_receipts/` and the closed diagnoses in `docs/HANDOFF_LOG.md`
-- **not here**.

**WHAT IS ACTUALLY LEFT IS THE RENDER GATE**, and it is the only forward item in
this section: the operator's 45-word render of EVERY visual path. Status below.

**THE WHOLE CHEAP SHELF IS GREEN** -- four visualizers, four still families,
8/8 with INERT G2 rows and declared continuity. All four still families refuse
a missing still, the ffmpeg gate is at preflight on the shared base, and the
synthesised dark floor is kept as a control with no occupant (**L22**), asserted
both ways.

**Lanes 19-21 are NEW ENGINE work, not repairs.** Lanes 10-18 fixed lanes that
already existed; 19 and 20 ADD the two MiniMax H3 adapters (one shared
implementation module, two registrations, two public ids) and 21 is a
standalone `h3_low_mime` runner that is NOT registered this build. Read
`docs/2026-08-09-TRANSPLANT-PLAN-per-lane.md` section 1 for the full H3 spec
before starting -- it is the largest surface in the campaign and the corpus
explicitly put it LAST so it inherits every lesson. Nothing on this shelf's
lessons transfers automatically: H3 has a real frame grid (124..362 step 17), a
24->25 fps conversion, a Sage gate and a mouth-policy carve-out. Baselines to
detect drift against: Bug Bible **20 passed / 24 skipped / 3 xfailed** at
**272** entries (**still exact 2026-08-13**); full suite **10363 passed / 110
skipped / 1 xfailed, NOTHING DESELECTED**; `scripts/build_variants.py --check`
**50 variants, 0 failures**. **THE OLD BASELINES ON THIS LINE WERE STALE and
cost a re-grounding pass** -- they read 9963/109/1 and 46 variants, which was
lane 10's era, while lanes 11-22 and the title-card work had moved the suite
~400 tests since. If your run does not match, check the DELTA against
`git stash` + `pytest --collect-only` before assuming a regression: that is how
the 2026-08-13 window proved its +7 was its own seven new tests and the rest was
drift already on `origin`. Box was left CLEAN (no resident server, port 8000
clear, VRAM 657 MiB).

**What is already done FOR you, and what is not:**
* **Lanes 16-18 (the remaining still lanes): your G3 rows are GREEN** (lane 10
  put `continuity=` on the shared `_CheapFamilyBase`) **and your ffmpeg
  PREFLIGHT gate is already inherited** (lane 15 put it on the same base). Your
  G2 rows are still yours, and so is the `_require_still` call -- lane 15 took
  the refusal for `still_motion` ALONE and left the base default False.
* **Lanes 16-18 (the remaining still lanes), on G2: READ LESSON L19 FIRST.**
  Lane 15 already took the INERT answer for `still_motion` and its receipt
  records the one wrinkle these lanes share: the still builders pass dims
  through `even_dim()`, a yuv420p mod-2 CODEC snap, which is not a native
  canvas and is a no-op at every canvas in play. All four
  visualizers closed G2 by declaring the profile canvas channel INERT, never by
  declaring a canvas -- four for four. Your lanes render through ffmpeg from a
  supplied still, so check whether YOUR path has a size of its own before
  copying either answer; the reasoning transfers only if the premise does.
  Historical note on why that default exists -- lane 11
  drafted `render_canvas = (1472, 832)` on a measured-looking argument and it
  was WRONG: that number is the default of `OTR_VIDEO_LANDSCAPE_CANVAS`, an
  operator lever, and a declaration is applied LAST and would silently disable
  it for one lane. The right close for a procedural lane with no native canvas
  is `PROFILE_CANVAS_DOCUMENTED_DEAD` with the mechanism written out. Copy the
  REASONING, not the shape -- if your lane does have a canvas-dependent
  property, declaring may be right for you.

**RUN `scripts/build_variants.py --check` BEFORE STARTING A LANE.** Lane 7
inherited five red variants from lane 5 and had to distinguish them from its
own. A red at the start of a lane belongs to whoever caused it.

**REVIEW ROUTING -- OPERATOR DIRECTIVE 2026-08-11 (supersedes the 2026-08-04
full-kibitz gate while it stands).** Operator, handing the build off while away:
"skip the kibitz ... do a coding post QA with Sonnet ... using Codex CLI for
your quandaries and Sonnet QA for post QA, no full r1-r3 is needed." So, per
coding item:
* **NO full `kibitz-plugin:kibitz` r1-r4 arc.** Do not open one, and do not
  report a scoped tail as an arc.
* **Codex CLI is the consult of record for a QUANDARY** -- a genuine fork, a
  defect whose model you doubt, a third failed fix. Use it instead of guessing
  or instead of stopping to ask.
* **Sonnet 5 runs the post-coding QA** on the finished diff, before the push.
  Lane 9 proves it earns its keep: it found a stray table cell carrying stale
  contradictory text and a superseded comment block, both real, both fixed
  pre-push.
* **KEEP CODING.** The operator is away and explicitly asked that the window
  not stop. Take routine judgment calls yourself, record them in the receipt,
  and only escalate what is genuinely irreversible or outside the queue.
The two-strikes floor from `CLAUDE.md` still stands underneath this: if a bug
survives two of your fixes, consult before the third swing -- that consult is
Codex CLI now, not a four-round panel.

**WHAT DROPPING THE PANEL DOES NOT DROP (operator, 2026-08-11, explicit):
"of course Bug Bible to be run at every turn, check for BOM, that always
stands."** Routing reviews to Codex + Sonnet removes ONE gate and nothing else.
Every turn, unchanged:
* **Bug Bible regression EVERY TURN**, not just at wrap-up -- `cd` to
  `C:\Users\jeffr\Documents\ComfyUI\comfyui-custom-node-survival-guide` and run
  the RELATIVE path `tests\bug_bible_regression.py` (an absolute forward-slash
  path fails to collect). Baseline **20 passed / 24 skipped / 3 xfailed at 272
  entries**. Sync to the Bible repo's origin/main first; never pin a stale copy.
* **BOM check on every touched file, always** -- UTF-8, NO BOM. First three
  bytes must not be `EF BB BF`. Never write Python source with `Set-Content` /
  `Out-File` (they inject BOM or mojibake); use
  `[System.IO.File]::WriteAllText(..., UTF8)` or the file tools.
* Plus the standing per-change set: full suite, `build_variants.py --check`,
  AST parse on touched `.py`, dead-ref grep, and HEAD == origin after the push.

**Panel history:** lane 7 ran an r1 panel (it found lesson L12 before the
fix shipped; artifacts `kibitz-runs/2026-08-11-lane07-ltx-audio-in/r1/`).
No lane has had a full four-round arc, and lanes 0-6 have had no panel on
their diffs at all -- still the operator's call. Live routing is the
REVIEW ROUTING block above, not this history.

Operator build law, reaffirmed 2026-08-10: **one lane is open at a time; close
its QA before touching the next.** A lane may take several commits when
measurement must be separated from a root fix, but no other lane starts between
them. A lane's registration, public id, alias, node-87 strings,
profile/variant, `ENGINE_MATRIX.md` row and canonical-workflow delta land
ATOMICALLY with that lane -- there is no later global naming sweep.

The per-lane loop is in `docs/2026-08-09-TRANSPLANT-PLAN-per-lane.md`; the gate
each lane must flip green is a row of `tests/test_lane_preflight_matrix.py`.
Every open defect below is bound to a lane by an `EXPECTED_RED` entry in that
suite, so the work list is executable rather than prose: run the suite, read
your lane's rows.

Status values: `DONE` (pushed green, preflight row green, smoke receipted),
`OPEN` (the one lane in progress), `PARKED` (attempted, backed out, note
written), `TODO`.

| # | Lane (public / internal) | Owns | Status |
|---:|---|---|---|
| 0-9 | **CLOSED -- 9 packets, 9 engines, 16 live legs.** Detail lives in `docs/evidence/lane_receipts/lane0*.md` and `docs/HANDOFF_LOG.md`, never here | scaffolding, 3 WAN, 4 HuMo, 3 LTX | **DONE** through `77fa4dad` |
| 2b | boot-contract enforcement TIMING | plumb `boot_contract` into the frozen director policy so the check fires at ShotLock preflight instead of inside the render phase; keep the render-time check as defence in depth | TODO |
| 5a | cost-row seeds | **PARTLY DONE** -- the three HuMo NET figures are in the manifest as `otr_side_legs` and seed rows (11,911 / 12,664 / 13,321 MB). `wan_i2v` is recorded `seeds_cost_row: false` (an nvidia-smi sample, a lower bound, not a probe max). `wan_ti2v` + `fastwan` peaks were MEASURED and dropped by `render_driver._clip_summary`; **the passthrough patch is in the 2026-08-11 handoff reply and should ride lane 7's commit**, then ONE re-smoke each recovers both -- no measurement campaign. | TODO |
| 5b | `wan_ti2v` retention (S7) | instrument the post-close boundary, collect telemetry on a live chained leg, THEN pick a release branch from what it names. A measurement campaign, not a code change -- inventing a release without the telemetry is what S7 forbids. **S7.1 also adopts the free-units instrument (operator 2026-08-11): record `free_vram_mb()` at render start and its MINIMUM during the window; that difference IS the demand in the units admission compares against, with no baseline arithmetic to get wrong.** **THE TELEMETRY THIS ROW WAS WAITING FOR ARRIVED 2026-08-13, unprompted, on a live chained render-gate leg -- and it cost that leg.** Measured: `wan_ti2v VRAM render-phase peak 12870 MB / post 7942 MB`. Nearly 8 GB stayed resident AFTER the render phase closed. The `fastwan_8gb` leg then died on the next shot: `MotionBudgetError: static frame budget 69 (snapped 69) exceeds the cost-model's affordable 18 frames (free=9549 MB, margin=0.85)` -- the beat needed 69 frames, only 9.5 GB of 16 was free because of the retention above, and the engine REFUSED rather than silently resizing (correct fail-closed behaviour, not the defect). So this is not a `fastwan_8gb` fault and **not** the still-spine defect -- that repair held, there is no missing-still signature anywhere in the leg. It is cross-engine retention: engine A does not release before engine B's cost model prices its work. S7 says pick the release branch FROM the telemetry rather than inventing one; the telemetry now exists and has a consequence attached. **A SECOND SAMPLE LOOKED LIKE A CONTROL AND WAS NOT -- corrected 30 minutes after it was written, and the correction is the finding.** `wan_i2v` was first measured `peak 14106 MB / post 2504 MB`, which reads as a clean sibling release, and this row briefly concluded the retention was `wan_ti2v`-specific. The SAME engine on its NEXT render in the SAME leg measured `peak 16074 MB / post 8517 MB`. Every sample tonight:

| engine | peak MB | post MB |
|---|---|---|
| `wan_i2v` | 14106 | **2504** |
| `wan_i2v` | 16074 | **8517** |
| `wan_ti2v` | 7985 .. 13041 (7 samples) | 6737 .. 8113 |

So `wan_i2v` retains too, and **retention is NOT engine-specific** -- `wan_ti2v` is merely consistent about it while `wan_i2v` released once and held once. Do not build an engine-specific fix on the single clean sample; that is exactly the trap this row fell into for half an hour. Note also `wan_i2v` peaking **16074 MB**, over the 14.5 GiB working ceiling and at the 16 GB card's limit, which is its own open question. What the samples DO still rule out is "add a release between engines" is the WRONG SHAPE: an engine-boundary reclaim already exists (`render_driver.py:3893`, the CS-3 inter-beat reclaim) and it FIRES -- its steps are `unload_llm`, `_unload_bark`, `gc.collect`, `soft_empty_cache` and the cuda flushes, every one aimed at OTR's OUT-OF-BAND caches, which is what it was built for. `soft_empty_cache` frees cached blocks; it does not EVICT a resident ComfyUI-managed model, and no step in that reclaim does. So the question row 5b should answer is: **what decides whether a finished render gives its VRAM back**, given the SAME engine did both within one leg. The candidate branch is a targeted `free_memory(required, device)` before a DIFFERENT engine prices its work -- explicitly NOT `unload_all_models`, which V-4/V-5 forbid and which was measured freeing 0 MB. Nine samples exist now; **get more before choosing, and do not conclude from one** | TODO -- 9 samples, no conclusion yet |
| 9b | `ltx_video` HEADROOM | the f169 marker leg peaked at 15,916 MB ABSOLUTE -- over the 14.5 GiB working ceiling, 97.6% of this 16 GB card -- while its NET 13,313 MB is comfortably under. No diet contract has ever been tried on this adapter, and lane 7b proved the `reserve_vram_gb` half of that lever is INERT on the LTX-AV adapter (its own in-process 4.0 GB reserve dominates), so whether `--disable-pinned-memory` alone buys anything HERE is genuinely unknown. A measurement, not a code change | TODO |
| 10 | `mesh_stage` | S8b-16 hy3d graph gate, dead profile-canvas channel, continuity declaration, V-1 self-probe | **DONE** -- 4/4 red gates green, live smoke PASS (50 frames at the declared 1472x832, magic-byte proved). Receipt: `docs/evidence/lane_receipts/lane10-mesh_stage.md`. Its G3 fix also closed the FOUR still lanes' G3 rows (shared `_CheapFamilyBase`) |
| 11 | `viz_green` | profile/canvas contract, ffmpeg gates, continuity declaration | **DONE** -- 7/7 green, two live smoke legs. Its G2 closed by declaring the profile canvas channel INERT, **not** by declaring a canvas: a declaration would overrule `OTR_VIDEO_LANDSCAPE_CANVAS` on a lane with no native canvas (lesson L19, found by the Codex consult). Receipt: `docs/evidence/lane_receipts/lane11-viz_green.md` |
| 12 | `viz_camera` | same visualizer checks, this lane only | **DONE** -- 7/7 green, live smoke. Same two answers as lane 11 (channel INERT + `continuity=`), with the L19 premise re-checked on this engine's own render path. Receipt: `docs/evidence/lane_receipts/lane12-viz_camera.md` |
| 13 | `viz_mxc_cpu` | profile/canvas, dependencies, continuity | **DONE** -- 7/7 green, live smoke. Same two answers again, premise re-derived on its own painter. Still HOLDS the "declares NOTHING" canvas control (it declared nothing, so the control did not move). Receipt: `docs/evidence/lane_receipts/lane13-viz_mxc_cpu.md` |
| 14 | `viz_mxc_mandala` | S8b-16 pycairo half, profile/canvas, continuity | **DONE** -- 7/7 green, live smoke. The pycairo NAMED refusal was ALREADY in place (verified, not rebuilt, and already covered by a forced-ImportError test). Same two answers as 11-13, premise re-checked hardest here. **ALL FOUR VISUALIZERS NOW CLOSED.** Receipt: `docs/evidence/lane_receipts/lane14-viz_mxc_mandala.md` |
| 15 | `still_motion` | G7.4/S8b-15 `still_plan` authority, S8b-12 ffmpeg gate + missing-still refusal | **DONE** -- 7/7 green, two live smoke legs (a render AND the refusal firing). Closed the **black-beat defect** (`_require_still`, scoped to this lane) and put the ffmpeg gate at PREFLIGHT on the shared base, which **all four still lanes inherit**. Receipt: `docs/evidence/lane_receipts/lane15-still_motion.md` |
| 16 | `still_pan` | the now-proven still-lane rules, this lane only | **DONE** -- 7/7 green, two live smoke legs (render + refusal). Took the `_require_still` call lane 15 left open, on its own evidence. Still HOLDS the "declares NOTHING" canvas control (it declared nothing). Receipt: `docs/evidence/lane_receipts/lane16-still_pan.md` |
| 17 | `still_flat` | same checklist independently | **DONE** -- 7/7 green, two live smoke legs. Took the refusal too, so **ALL FOUR still families now refuse a missing still**. Premise re-checked on a THIRD builder (`ffmpeg_still_static_cmd`, fit+pad). The dark-floor branch is KEPT as a control with no occupant (lesson **L22** -- lane 16's "delete it" instruction was REVISED), with both halves asserted. Receipt: `docs/evidence/lane_receipts/lane17-still_flat.md` |
| 18 | `still_word` | preserve its existing missing-still refusal, add the ffmpeg + single-authority contract | **DONE** -- 7/7 green, two live smoke legs. Two of its three items were ALREADY closed (Sprint B's refusal; lane 15's inherited ffmpeg gate) and were verified rather than rebuilt -- the lane's deliverable is the acceptance check, since nothing had asserted the gate on this lane before. **THE WHOLE CHEAP SHELF IS NOW GREEN.** Receipt: `docs/evidence/lane_receipts/lane18-still_word.md` |
| 19 | `h3_low_video` / `minimax_h3_video` | the shared H3 implementation with this FIRST adapter only, 124..362 model / 129..377 canvas math, 24->25 delivery, continuity, Sage-free boot, V-1 self-probe | **DONE** -- 7/7 green on arrival, live smoke PASS (129 canvas frames at the declared 864x480, exactly 5.160 s, ZERO audio streams, 6,315 MB absolute cold). G2 INVERTED as predicted and the canvas was DERIVED from the `low` token's own meaning. **The smoke found TWO shipped production bugs the suite could not see** -- the `h3` contract was missing the `--reserve-vram 12` that every passing H3 leg held (**L24**), and three shared modules reached siblings by an absolute `nodes.` import that works in tests and raises on a server, one of them silently making five lanes motion-EXEMPT (**L23**). Receipt: `docs/evidence/lane_receipts/lane19-h3_low_video.md` |
| 20 | `h3_low_audio_in` / `minimax_h3_audio_in` | the second adapter, mouth policy carve-out, soft-reference/JUMP, seed-43 workhorse profile | **DONE** -- 7/7 green on arrival, live smoke PASS on the FIRST attempt (129 canvas frames at 864x480, exactly 5.160 s, ZERO audio streams on the lane that LOADS an audio VAE, 6,678 MB absolute cold, 239.4 s). A SECOND registration out of lane 19's module, split into `_MiniMaxH3Base` + two subclasses with lane 19's behaviour held byte-for-byte. Continuity is `soft_reference` and is NOT inherited: the ref2va node has no `first_frame` input at all. The mouth-policy equality test became a membership test in the same commit -- without it every H3 character beat raises `MouthPolicyError` at PLAN time, which the test asserts as a consequence rather than a mapping. The handoff's one unverified item (the `COMFY_AUTOGROW_V3` sockets) is SETTLED: a V3 node's `FUNCTION` is `EXECUTE_NORMALIZED`, a plain passthrough, so in-process they take plain dicts. Receipt: `docs/evidence/lane_receipts/lane20-h3_low_audio_in.md` |
| 21 | standalone `h3_low_mime` runner | G5.2 keeps-audio exemption, clip/stem receipts, durable output path, solo-runner QA. NOT registered this build | **DONE** -- `scripts/otr_h3_mime_runner.py`, live run PASS: 864x480, model f90 = **3.750 s exactly at the model's own 24 fps**, and **`nb_streams=2` -- one video plus ONE audio stream**, the inversion this lane exists for. Lossless FLAC score stem from the same decode, plus a ducked voice-over review copy with the picture COPIED not re-encoded; both originals preserved. Registers nothing, and the test asserts that from both sides. **Three live API-boundary failures before the pass** (input dir, DynamicCombo serialization, the `/history` key that says `images` for an mp4) -- all three invisible to a CPU test, recorded as **L27**. Receipt: `docs/evidence/lane_receipts/lane21-h3_low_mime.md`. **One operator ruling owed:** CLAUDE.md 0A says "no other runner" and should be amended to name this script |
| 22 | all-row + episode gate | every preflight row green, every expected-red removed, every solo-smoke receipt present, then ONE end-to-end episode | **THREE OF FOUR MET 2026-08-12.** The preflight matrix is **29 engines x 7 gates with ZERO non-green cells and ZERO `EXPECTED_RED` entries** -- the table is empty for the first time since it was written. All **21 lane receipts** are present in `docs/evidence/lane_receipts/`. The last expected-red (`google_omni_video` G3) was closed here rather than left: it was the one lane the 21-lane transplant deliberately did not own, and row 22's own contract is what brought it in. Its comment had reasoned "CONTINUITY none" since the lane was written while the call inherited the default -- L3 exactly, and one keyword to fix. **REMAINING: the render gate, and the operator SUPERSEDED it** -- "we need a 45 word render of every visual path" (2026-08-12) replaces "ONE end-to-end episode". That sweep is the last thing in the queue |

### WHAT IS ACTUALLY LEFT -- READ THIS FIRST (2026-08-13 morning)

Rows 1 and 2 of the operator order are CLOSED. Row 3 (the render gate) is
part-run. This is the live state.

**LANES STILL TO RUN (5 of 21):**

| lane | boot needed | note |
|---|---|---|
| `ltx_audio_in` | `LTX` token **+ `OTR_HEADLESS_DISABLE_PINNED=1`** | in flight |
| `humo`, `humo_14b_169`, `humo_1_7b`, `humo_1_7b_169` | `HUMO` token + reserve 2.921 + disable pinned | never run in this gate |
| `minimax_h3_video`, `minimax_h3_audio_in` | no token + reserve 12 + disable pinned, sage off | never run in this gate |

**LANES THAT FAILED AND NEED A RETRY, with the reason and what must change:**

| lane | failed on | retry needs |
|---|---|---|
| `fastwan_8gb` | `MotionBudgetError` -- the **disqualified `FRAME_COST_MODEL` row** | **CODE FIX SHIPPED 2026-08-13 -- just re-run, no further change.** And the retry note that stood here was WRONG: "delete/replace the row" is a **proven no-op**, because `_cost_model_for` falls back to `_DEFAULT_FRAME_COST`, the byte-identical `(7000.0, 185.0)`. The real defect was that `compute_real_frame_budget` never asked `cost_row_may_refuse`; it does now. Both legs' exact failing calls return their full frame counts |
| `wan_ti2v` | same disqualified row | same -- fixed by the same commit |
| `wan_i2v` | page-thrash, 89.7-106.3 s/it vs a documented 29 s/it pathology; peaked 16,074 MB and retained 8,517 MB | genuinely unknown whether this lane is viable on this box. Never completed a leg. Needs the VRAM question (row 5b) answered first, or a diet contract tried |

**LANES THAT PASSED (6 of 21 this run + the cheap shelf):** `still_pan`,
`ltx_8gb`, `ltx_video` this run; the four visualizers, four still families and
`mesh_stage` from earlier sweeps.

**THE THREE CODE ITEMS OWED, none built, in value order:**

1. ~~**Delete or replace the two `FRAME_COST_MODEL` rows.**~~ **DONE
   2026-08-13 -- and NOT by deleting them, because deletion is a PROVEN
   NO-OP.** `_cost_model_for` falls back to `_DEFAULT_FRAME_COST`, the
   byte-identical `(7000.0, 185.0)`; `eng_fastwan_8gb.py:293` says so about its
   own row in as many words. Both legs refuse identically with the table empty,
   so the prescribed fix would have shipped green and changed nothing. The real
   defect: `compute_real_frame_budget` is the SECOND call site of that
   disqualified row and the only one that never asked `cost_row_may_refuse` --
   `render_driver._assert_beat_affordable` has asked since it was written, so
   the row was inert on the reviewed path and live one call away, through
   `eng_wan_ti2v._floor_length`. **Fix: both refusals now answer to that one
   authority**, validation stays unconditional, and the predicate additionally
   requires the row to EXIST so a typo cannot promote the shared fallback into
   a guard. Both legs' exact failing calls now return their full frame counts.
   The lab rows (`wan_ti2v` 6,910.8 + 25.874/frame; `fastwan_8gb` 7,317.9 +
   6.900/frame) were NOT adopted -- the standing ruling still forbids a bench
   substituting for the real lifecycle, and no row is qualified. Bible **12.98**
   amended: its "DELETE the row" prescription is what this disproved.
   **The overhead half was gated too, and that was a correction mid-build:** it
   looks like the defensible term, but the binding NET-NOT-ABSOLUTE provenance
   rule records that it came from an ABSOLUTE peak while the comparison runs
   against FREE bytes, double-charging the desktop baseline. Caught by the
   Codex consult, not by me.
2. **The in-decode halt for the writer runaway.** Fires on an unclosed-string
   token count (primary) and a repeating window (secondary), NEVER on
   `target_words`. Must raise a REROLLABLE capacity phase or it silently
   becomes the writer veto the directive forbids. Touches every pass P0-P5, so
   it earns a review first. The runaway text is now captured -- see the
   cadence-lock section.
   **THE REVIEW IS DONE (2026-08-13) and the design is settled:
   `docs/2026-08-13-codex-consults/indecode-halt-codex-review.md`.** It is
   file-grounded and it changed four things about the shape I took in. Read it
   before writing a line; the summary is not a substitute.
   * **NOT in `invoke_structured_slot`** -- that function has no tokenizer, no
     model and no `generate()` call. It goes in `_build_truncating_generate_fn`,
     installed unconditionally whenever `schema_model is not None`, which is the
     one route P0/P1/P2/P3/P5 all traverse. (There is no P4 authoring pass.)
   * **NOT through the existing `if stop:` path**, and construction failure must
     be LOUD -- the `except Exception: stop-strings disabled` fallback there is
     right for an optional quality stop and wrong for a liveness contract.
   * **NOT an exception raised from inside the criterion.** `generate()` returns
     NORMALLY on a criteria hit (verified against the installed transformers
     5.10.4). Latch `hit`/`reason` on the criterion, let generate return, decode
     once, then raise. Classify in this order: `guard.hit` -> capacity ceiling
     -> `ended_with_eos` -> another stop. Never infer degeneracy from a short
     generation; the optional substring stop also returns early.
   * **NOT `phase="output_limit"`** -- that would make the diagnostics lie. New
     `GenerationDegeneracyError`, phase `decode_degeneracy`, added to the shared
     caught family with the rerollable predicate widened to accept both.
     `_otr_scifi_codex.py:1879` summarises every rerollable capacity error as
     "model output ended at the provider capacity limit", which would be false
     for a guard halt.
   * **Threshold: 2,048 open-string tokens** provisionally (~3 min instead of
     21 at the observed ~11 tok/s), calibrated later as
     `max(2048, 4 x maximum healthy per-string token count)` measured with the
     production tokenizer over accepted artifacts. Honest caveat recorded in the
     review: while the schema permits unbounded strings, NO finite bound can
     promise it never cuts a legitimate one.
   * **n-gram ships as TELEMETRY ONLY at first.** An independent hard halt would
     reject intentional refrains, parallel rhetoric and repeated schema keys.
   * **One claim in the review I have NOT verified and the next window must:**
     it asserts "the outer candidate loop is not currently unbounded", which
     contradicts this plan's premise that the loop is unbounded and correct.
     Ground that before building on either statement.
   * Its lexer note is worth heeding: scan the DECODED fragment character by
     character with `clean_up_tokenization_spaces=False`, decoding only unseen
     ids. Do not assume a quote gets its own token id.
   * **TRANSPORT PARITY IS ONLY HALF DONE.** `6e9ed140` fixed the PRE-CALL leg:
     `_otr_model_loader` now raises the phase-carrying
     `PromptContextOverflowError` at both `GenerationContextOverflowError` sites
     instead of a bare, phase-less `ModelLoaderError`, so an identical capacity
     failure is rerollable on both transports. **The POST-GENERATION leg is
     still divergent** and the guard must close it: that module tests
     `len(generated_ids) >= effective_max_new_tokens` where the writer tests
     `==`, it does not compute `ended_with_eos`, and it attaches no raw
     evidence. So a runaway on that transport is still worse-diagnosed than the
     same runaway on the writer's.
3. **`wan_ti2v` VRAM retention (row 5b).** 15 samples, post clustering ~8.1 GB
   regardless of peak. NOT engine-specific (`wan_i2v` did both). No conclusion
   drawn deliberately.

**LEMMY** is untouched and deferred by the operator.

### OPERATOR ORDER FOR THE NEXT WINDOW (set 2026-08-12 late, supersedes below)

Work these three in THIS order. The operator set it after watching a live leg.

| # | Item | Where | State |
|---:|---|---|---|
| ~~**1**~~ | ~~**BUG BIBLE -- TWO PROMOTIONS OWED**~~ | survival-guide `80fc358` | **CLOSED 2026-08-12.** Landed as **07.31** (scroll span vs canvas allocation) and **07.32** (legibility through a lighten-only blend), both `verdict: promoted`, both index rows appended, README count moved at all three call sites. The required field set was READ off `test_every_parsed_entry_has_the_documented_fields` -- `id, phase, area, symptom, cause, fix, verify, tags`, every one truthy (`legacy_id` is documented but NOT required, which is what the backed-out attempt is likeliest to have tripped on). Filed **phase 7 / area video**, not the phase-12 slot the last five OTR promotions took: these are concrete compositing and scroll geometry. Bible regression reproduced the documented baseline EXACTLY, 20 passed / 24 skipped / 3 xfailed |
| ~~**2**~~ | ~~**TITLE-CARD LEGIBILITY**~~ | `dc21e4df` + `d75a8866` | **CLOSED 2026-08-13, PROVEN ON A PUBLISHED EPISODE.** Built in the spec's six-step order, suppression stayed step 4. Live leg `signal_lost_wigners_whisper`: **628 TITLE events** on ASS layer 1, BOTH cards (open 0.00-10.00s, close 123.16-131.12s), Matrix decode intact, the block cursor rendering from its `\p1` drawing, and the apostrophe surviving `_ass_escape` on a real title. Measured on the FINAL artifact: core-vs-outline **13.66:1** opening / **12.60:1** closing against a gate of 4.5:1, where the pre-fix control measures **1.01:1** core-vs-scene. Then the operator eyeballed it -- "hard to see from a distance but much improved" -- and `d75a8866` took the outline 3 -> 6 with a shadow (+55% dark mass within 5px), verified by re-burning the surviving intermediate rather than re-rendering |
| **3** | Resume the 45-word render gate | the section below | 9 of 21 legs passed; needs THREE contract-grouped boots |
| **4** | LEMMY | the Lemmy sections further down | operator hands it over once 1-3 are done |

**Item 1 is the whole reason the sweep is stopped.** The hero title is phosphor
green drawn into the procgen CRT frame and composited `screen` + `green_only`,
which can ONLY lighten -- measured 1.13:1 over a lit monitor, and a black
outline there is a mathematical no-op (`screen(A, 0) = A`). Direction is chosen
and driver-verified: emit the title through `OTR_CaptionBurn`, which already
runs AFTER the blend and whose ASS styles already carry `OutlineColour` /
`BorderStyle` / `Outline` / `Shadow`. **The Matrix decode animation is KEPT.**
Everything needed is in the spec -- rasterisation site, the card schedule
(`_resolve_card_windows` already yields BOTH the open and close windows, so
"start and end" needs no new timing work), the canonical chain, the risky half,
and a measured acceptance test. Do not re-derive it.

**DO NOT CONFLATE THE TWO CREDIT SURFACES.** The end-credits SCROLL is a
different thing and is already FIXED (`f70df546`). Conflating them cost the
operator an evening believing the title was done.

### THE 45-WORD RENDER GATE -- item 2 (live status 2026-08-12 late, sweep STOPPED)

**IT NEEDS THREE BOOTS, NOT ONE. This was wrong in the plan until now and would
have burned ~5 GPU-hours.** Legs are grouped by the `launch.boot_contract` in
their own profile (nested under `launch`, NOT top-level -- a top-level read
returns `None` for every profile):

| boot | flags | legs |
|---|---|---|
| A -- unclamped | none | `ltx_8gb`, `fastwan_8gb`, `ltx_video`, `wan_i2v`, `wan_ti2v` |
| B -- `humo_diet` | `--reserve-vram 2.921 --disable-pinned-memory` | `humo` x4 **+ `ltx_audio_in`** |
| C -- `h3` | `--reserve-vram 12 --disable-pinned-memory`, sage off | `minimax_h3_video`, `minimax_h3_audio_in` |

**THE TABLE ABOVE IS INCOMPLETE, corrected 2026-08-13 while running it.** The
`boot_contract` governs the VRAM CLAMPS. It does NOT enable the heavy engines --
that is the launcher's SECOND ARGUMENT, and it defaults to enabling NOTHING:
`LTX` sets `OTR_ENABLE_LTX_VIDEO` + `OTR_ENABLE_LTX_AV`, `WAN` sets both Wan
engines, `HUMO` sets HuMo. So boot A above is not one boot: its five legs need
at least two different tokens, and booting all five with no token leaves the
heavy engines OFF and the lanes fall to the floor. **Group by
`(boot_contract, launcher token)`, not by contract alone.** `ltx_8gb`,
`fastwan_8gb` and the H3 pair carry NO enable flag and so ride any boot.
Confirmed grouping:

| boot | token | clamp | legs |
|---|---|---|---|
| 1 | `WAN` | none | `wan_i2v`, `wan_ti2v`, `fastwan_8gb`, `ltx_8gb` |
| 2 | `LTX` | `--disable-pinned-memory` (ltx_av_diet) | `ltx_video`, `ltx_audio_in` |
| 3 | `HUMO` | `--reserve-vram 2.921 --disable-pinned-memory` | `humo` x4 |
| 4 | (none) | `--reserve-vram 12 --disable-pinned-memory` | the two H3 lanes |

**THE TOKEN AND THE ENV ARE TWO SEPARATE THINGS AND YOU NEED BOTH.** Walked into
this 2026-08-13 despite quoting L24 an hour earlier. The launcher's SECOND
ARGUMENT (`LTX` / `WAN` / `HUMO`) enables the ENGINES. The profile's own
`launch.env` satisfies the CONTRACT. Booting the LTX lane with the right token
and no env gets you a server where `ltx_video` runs happily and `ltx_audio_in`
is heading for a render-phase refusal an hour later, because `ltx_av_diet`
requires `OTR_HEADLESS_DISABLE_PINNED=1` and nothing had set it. Read the env
out of the profile and export it BEFORE `Start-Process`; do not infer it from
this table, which lists the clamp but cannot show you the token. Verify it
landed by grepping the boot log for the flag on the command line -- the
launcher's own `[launch]` echoes go to the detached console, NOT to the log,
so their absence proves nothing either way.

**AND THE PROFILE FILENAME IS NOT THE ENGINE ID.** `config/profiles/otr_w45_fastwan.json`
registers the engine **`fastwan_8gb`**; `--only fastwan` is refused. The refusal
is clean and loud ("names engines that are not registered local engines"), so it
costs a relaunch rather than a leg -- but derive `--only` names from a prior
results file's `engine` field, never from the profile filenames.

B covers `ltx_audio_in` too: its `ltx_av_diet` contract sets
`reserve_vram_gb: None`, which means "does not constrain that knob", so HuMo's
2.921 satisfies both while the shared `disable_pinned_memory` is what that lane
actually needs. **Take each boot's env from the profile's own `launch.env`
block** rather than typing it -- that is what stops a contract being written
down but not applied (lesson L24).

**The HuMo contract check fires INSIDE THE RENDER PHASE, not at preflight**
(lane 2b, still TODO). So a HuMo leg on the wrong boot spends ~80 min on writer
+ audio and only then refuses. That is the ~5 hours grouping avoids.

**9 of 21 legs now PASS.** 4 still lanes + 4 visualizers + **`mesh_stage`**,
which is the one that matters: it died twice (2.6 min in CastLock on a rolled
bank, then 18.5 min on the still-spine beat-id split) and now PASSES at 21.6 min
with `delivered: ['mesh_stage']` and stills keyed `music_opening_001`. That is
the live proof the beat-id deletion worked.

**Remaining: 12** -- the 5 in boot A, the 5 in boot B, the 2 in boot C.

**RUN 6 RESULTS SO FAR (2026-08-13): 3 PASS, 3 FAIL.** `still_pan` PASS (carries
the title card), `ltx_8gb` PASS 18.7 min, **`ltx_video` PASS 68.1 min -- a lane
that had NEVER completed a 45-word leg and was in the never-run list**, 10 beats
with zero runaways and zero rerolls, the cleanest leg of the sweep.
`fastwan_8gb` and `wan_ti2v` FAIL on the disqualified cost-model row (see the
correction above -- NOT a VRAM fault). `wan_i2v` FAIL on page-thrash. Remaining:
`ltx_audio_in` running, then HuMo x4 and the H3 pair.

**RUN 6 (2026-08-13, boot 1 = WAN token, unclamped).** `ltx_8gb` **PASS**
(18.7 min) -- one of this morning's three failures cleared, and it had died in
the fable2 writer, so that is the post-fix data point the judging pass asked
for. Its reuse audit is worth reading rather than skimming: 8 advisory findings
including a `mirror/ping-pong -- 8-frame reflection about frame 104`, reported
as ADVISORY while the leg still says PASS. `fastwan_8gb` **FAIL** (44.6 min) on
the cross-engine retention above, NOT on the still-spine defect it failed on
this morning -- that repair held. `still_pan` also re-proved PASS earlier in the
night and is the leg that carries the title card.

### THE TWO MotionBudgetError LEGS WERE NOT A VRAM PROBLEM (corrected 05:30)

**PARTLY CORRECTED 2026-08-13 by running the arithmetic instead of reading it.**
The empty-card claim below is TRUE for `wan_ti2v` (125 frames at 832x480 prices
above any free level this card can reach) and **FALSE for `fastwan_8gb`**: its
beat was **69** frames, not 125, which at `free=16000` prices 7000 + 69*185*
(480/832 area scale) inside the budget and returns 69. So `fastwan_8gb` refused
because free VRAM really was low -- the retention WAS a contributor on that leg,
and only the `wan_ti2v` leg refuses on an empty card. The conclusion the section
draws is unchanged (the row is disqualified and had to stop refusing); the
supporting sentence was over-general, and both legs are green on the fix either
way.

**`wan_ti2v` and `fastwan_8gb` would have failed ON AN EMPTY CARD.** Both died
with `MotionBudgetError`, both reported a low `free=` figure (9,389 and 9,549
MB), and I attributed both to the `wan_ti2v` retention documented above. That
was wrong, and the arithmetic settles it:

```
FRAME_COST_MODEL = {'wan_ti2v': (7000.0, 185.0), 'fastwan_8gb': (7000.0, 185.0)}

  free  9389 MB -> affordable   5 frames   (needed 125)  REFUSES
  free 14500 MB -> affordable  28 frames   (needed 125)  REFUSES
  free 16000 MB -> affordable  35 frames   (needed 125)  REFUSES   <- empty card
```

A 125-frame beat prices at `7000 + 125*185 = 30,125 MB`, nearly TWICE this card.
No free-VRAM level on this hardware satisfies it. The retention is real
telemetry and it was never the cause of these two failures.

**AND THE REPO ALREADY KNEW.** The doc block at `motion_common.py:341-360`,
directly beneath the table, disqualifies that exact row in writing: it "refused
a real production leg -- static frame budget 173 ... affordable 24 frames
(free=13481 MB) -- on an engine that had already shipped", it is "wrong in BOTH
directions", `_planned_length` "already stopped consulting it for exactly this
reason", it "refuses EVERY segment length the coverage planner produces,
including 93 frames at 14,500 MB free", and -- explicitly -- **"Empty is the
correct value until a row is re-measured through the real prepare() +
render_clip() lifecycle".**

The table still holds exactly two engines. They are exactly the two that failed
tonight. Nothing else is in it.

**THE LAB ALREADY MEASURED THE RIGHT ROWS, and the shipped ones are off by an
order of magnitude.** `vram-recipe-lab/docs/ENVELOPE_LADDERS.md`, "Job B --
WAN/FastWan cost-row ladder", fitted from a real frame ladder (25/65/93/129/177):

| engine | SHIPPED row | LAB-MEASURED row | per-frame error |
|---|---|---|---|
| `wan_ti2v` | 7000 + **185**/frame | 6,910.8 + **25.874**/frame | **7x too steep** |
| `fastwan_8gb` | 7000 + **185**/frame | 7,317.9 + **6.900**/frame | **27x too steep** |

The OVERHEAD is nearly right in both; the SLOPE is the defect.

**AND THIS REFINES THE CORRECTION ABOVE RATHER THAN CONFIRMING IT.** "Retention
was irrelevant" is true only against the BROKEN row. Recomputed with the lab row,
a 125-frame `wan_ti2v` beat needs `6910.8 + 125*25.874 = 10,145 MB`:

* at tonight's `free=9389` (x0.85 margin = 7,980 usable) it STILL refuses;
* at `free=14500` (12,325 usable) it FITS.

So with a correct cost row the ~8 GB retention becomes decisive after all. Both
are real contributors and dismissing either is wrong.

**A JUDGEMENT THE OPERATOR OWNS.** The standing ruling admits a row only when
"re-measured through the real `prepare()` + `render_clip()` lifecycle -- which
the standing ruling requires and no bench may substitute." The lab IS a bench.
So adopting these numbers verbatim is exactly what that sentence forbids, even
though they are self-evidently closer to truth than 185/frame. The choices are:
delete the rows (restores the documented-correct empty table, no measurement
needed), adopt the lab rows (needs the operator to rule the bench admissible),
or re-measure through the real lifecycle.

**So the fix is a DELETION, not a measurement campaign** -- and the standing
ruling already names the bar for putting a row back: re-measured through the
real lifecycle, which no bench may substitute. Until then an empty table means
the geometry plan stands and no row may refuse. That is a small, high-value
change and it is NOT built: it turns two red legs green and it deserves the
review a refusal-path change earns.

**WHAT THIS DOES NOT EXCUSE.** `wan_ti2v` really does retain ~8 GB (15 samples,
post clustering at 8.1 GB regardless of a 7,985-13,065 MB peak, which looks like
cached model weights rather than a leak) and `wan_i2v` really did thrash. Those
stay open as row 5b. They are simply not what killed these two legs.

**`wan_i2v` PAGE-THRASHES ON THIS BOX -- new, and it had never been run before.**
Killed at 108.6 min after ONE beat, measured at **89.7 - 106.3 s/it** on a 20-step
segment. The plan's own documented thrash pathology is **29 s/it** (ltx 12.5 GB +
humo 7 GB co-resident), so this is THREE TIMES worse than the case that was bad
enough to write down. At ~33 min per segment and multiple 1-2 segment beats, the
leg needed 5-11 hours; it was interrupted rather than allowed to run.

The cause is visible in its own telemetry: an earlier `wan_i2v` render in the
SAME leg peaked at **16,074 MB** -- over the 14.5 GiB working ceiling and at the
16 GB card's limit -- and left **8,517 MB** resident. The engine is paging
against its own residue. **So retention has TWO symptoms, not one:** it kills
OTHER engines' cost models (that is what took `fastwan_8gb`) and it degrades the
RETAINING engine into a state where it cannot finish. Same root, two failures.

`wan_i2v` is listed as never-run in every prior sweep, so there is no baseline
claiming it ever completed a 45-word leg on this hardware. Do not assume this is
a regression.

**AND `/interrupt` CUTS A SAMPLER BUT NOT AN LLM DECODE.** It was POSTed twice
tonight: against the writer runaway it did nothing (the transformers
`generate()` call never polls it, so `queue_running` stayed 1 until the server
was restarted), and against this thrashing render it landed within ~2 minutes
(ComfyUI's sampler checks between steps). That distinction matters for the
`scripts/otr_api.py:820` interrupt fix -- it will cancel a stuck RENDER and will
NOT cancel a stuck WRITER, which is the case that motivated it.

**A WRITER RUNAWAY HIT THIS BATCH TOO**, on `fastwan_8gb`'s P1 (not P3) with a
sober BBC eclipse story: 15,297 output tokens after a 1,087-token prompt, on a
`DramaticQuestionV4` whose entire schema is THREE short strings. It self-healed
on the next rung at temperature 0.320. So the runaway is neither P3-specific nor
comic-premise-specific -- two occurrences in one night, different passes,
different sources. See the runaway section below; the heartbeat and the evidence
log both landed tonight so the next one is fully observable.

**A `mesh_stage`-class PASS is weaker evidence than it looks.** That lane leaves
NO per-shot clips, so `coverage` reads `not measured` and the frame-level
no-reuse audit audits zero clips -- yet the runner still prints PASS. It fails
closed when the registry is unreadable but stays quiet when there are simply no
clips. Queued fix: say so out loud in the verdict.

**Bank pinned `scifi_news`** (`--source-bank`): the gate's question is "does this
engine render", the bank is noise on top of it, and rolled banks already ate two
legs. `scifi_news` specifically because its pack closes the cast-coverage
asymmetry that killed `mesh_stage`, and it does not route to the fable2 writer
that killed `ltx_8gb`.

**Results files are overwritten per run** -- snapshot `tmp/_w45_results.json`
before every launch. Runs 1-4 are preserved as
`tmp/_w45_results_run{1,2,3,4}_20260812.json`.

Operator acceptance gate: a 45-word render of EVERY visual path. Runner
`scripts/otr_w45_campaign.py`, 21 legs -- 19 incumbent lanes on one boot, then
`minimax_h3_video` + `minimax_h3_audio_in` on a SECOND boot with
`--reserve-vram 12` (they would starve `wan_i2v` and the HuMo tiers on a shared
server). Results in `tmp/_w45_results.json`; run 1 preserved at
`tmp/_w45_results_run1_20260812.json`.

**8 PASS, 3 FAIL, 10 NEVER RUN.** The sweep was STOPPED deliberately, mid-run,
to fix a defect that would otherwise have consumed five of the remaining legs.

| verdict | legs |
|---------|------|
| PASS | still_flat, still_pan, still_motion, still_word, viz_camera, viz_green, viz_mxc_cpu, viz_mxc_mandala |
| FAIL | mesh_stage, ltx_8gb, fastwan_8gb |
| never run | ltx_video, ltx_audio_in, humo x4, wan_i2v, wan_ti2v, + the H3 pair |

**NOT ONE OF THE THREE FAILURES WAS A VIDEO DEFECT.** That is the headline of
the whole sweep so far:

- `mesh_stage` -- died in `OTR_CastLock`; the freeze cascade stamped
  `needs_full_rerun` because a cast member (MARIA) owned no line. Rolled the
  shakespeare bank. OPEN -- see the cast-coverage section below.
- `ltx_8gb` -- died in the fable2 WRITER, markup ladder exhausted. That server
  carried fix A but NOT fix B, so it is evidence that carrying the rejected
  draft ALONE was not enough -- the post-fix data point the judging pass asked
  for.
- `fastwan_8gb` -- the only genuine render failure, and it is the still-spine
  (PBUG-20260811-02, reproduced). Candidate repair under QA.

**RESTART FROM `ltx_video`.** Reset + boot per CLAUDE.md sections 4 and 5, then
run `--only` over the ten remaining incumbent legs, then reboot with
`OTR_HEADLESS_RESERVE_VRAM_GB=12` and `OTR_HEADLESS_DISABLE_PINNED=1` for
`--only minimax_h3_video,minimax_h3_audio_in`.

**THE SWEEP CAN NOW PIN ITS BANK** (`--source-bank`, shipped `f9b51675`).
Rolling stays the DEFAULT because it is how both writer defects were found. Pin
when the question is "does this engine render"; roll when it is "does the
pipeline hold across banks". Two legs were spent proving something about the
writer before this existed.

**Diagnosis is far cheaper than it was.** `scripts/otr_api.py`
`describe_execution_error` names the failing NODE, the exception type and the
tail of the traceback instead of truncating the history repr at 500 characters
mid-frame -- that truncation cost the diagnosis of both writer failures. And a
`scifi_news_pro` leg now persists its seed / frame card / stance BEFORE the
passes that can die on them, so a dead leg is reproducible; three earlier deaths
were not.

### THE WRITER CAN LOOP FOREVER ON A CAPACITY RUNAWAY -- OPEN (found 2026-08-13)

Found on a live leg while running the render gate, not by review. Two separate
things, both real, neither patched -- the second is a five-minute fix and the
first is a contract decision that deserves daylight.

**1. A decode can run 21 minutes with nothing watching it.** CORRECTED 2026-08-13
against the actual leg log (`comfyui_8000.prev2.log:255-291`) after an Opus
design pass; the first version of this entry, pushed in `2ddea5a2`, got three
things wrong and recommended a fix that would not have fired. What really
happened on the UCLA-marmot leg:

```
22:58:42  P3 attempt 1/3  base call        t=0.720
23:19:50    OUTPUT_TRUNCATED  14191 tok after a 2193-tok prompt   <- 1268 s
23:19:50  P3 attempt 2/3  structural retry t=0.320
23:41:26    OUTPUT_TRUNCATED  14191 tok again                     <- 1296 s
23:41:26  P3 attempt 3/3  typed repair     t=0.100
23:42:34    failed: draft.cast_coverage, missing announcer        <- 68 s
23:42:34  cycle 1 exhausted (PostValidationError) -> cycle 2
```

* **TWO rungs ran away, not three.** Rung 3 returned a COMPLETE parseable draft
  in 68 seconds. The ladder self-healed out of the capacity failure; the runaway
  cost 42 minutes but was never terminal by itself.
* **Cycle 2 opened on `PostValidationError`, NOT on the capacity phase.** The
  capacity error is consumed INSIDE the ladder (`_otr_structured_call.py:1075`)
  and only reaches the cycle loop if it is the LAST rung's error. **So "bound
  capacity cycles at `_otr_scifi_codex.py:2226`" -- what this entry previously
  called the obvious answer -- provably would not have fired on this leg.** Keep
  it as a cheap backstop; it is not the fix.
* **P3 is the RADIO SCORE draft** (`RadioScoreDraftV4`, `_otr_scifi_codex.py:426`),
  not prose -- prose is P5 (`ScriptTextDraftV4`, `:612`). This matters: P3's
  ARRAYS are bounded by schema (scenes <=3, shots <=2, beats <=4, cues <=3) while
  its STRING fields have no max, so under constrained decode a 14,191-token run is
  provably stuck inside ONE JSON string. That is a structural fact and it is the
  seam a directive-safe guard can use.
* The leg did not run to the six-hour wall -- the operator killed it at 23:44.
  "Cannot terminate on its own" remains true as a property; it was not observed.

**The loop is still unbounded and is still CORRECT for validation errors** --
that is how `scifi_news` recovers from its announcer-coverage weakness
(`016ad146`, 0-for-4 to reliable), and no fix may weaken it.

**PRIOR ART: this is PBUG-20260729-02 again** (`docs/PROD_BUG_LOG.md:2816`),
same signature (14,697 tokens, ~24 min) on P5 instead of P3, already PARKED with
an operator ruling on the books: *"the writer should never veto, the writers
should keep on passing in a loop to agents to clean up the ledger."* That ruling
also explicitly forbids capping the output budget to the word target.

**THE REAL FAULTS, and where the time goes.** Everything on that leg except the
two runaway decodes took seconds:

* **D1 -- no in-decode guard.** `OTR_LedgerScriptWriter.py:996` calls
  `model.generate()` with no `streamer` and no `stopping_criteria` (the criteria
  block at `:977` is behind `if stop:`, and `invoke_structured_slot` never passes
  `stop`). Nothing observed the decode for 21 minutes. **The streamer half of
  this is now fixed** -- see the heartbeat work -- so a runaway is at least
  VISIBLE; the halting half is not.
* **D2 -- the outer loop cannot be bounded by phase**, per the correction above.
* **D3 -- no cancellation reaches an in-flight decode.** `_otr_structured_call.py`
  has ZERO interrupt polls, so cancellation granularity is one whole ladder (43
  min here). Client-side, `scripts/otr_api.py:820` returns TIMEOUT and **never
  POSTs `/interrupt`** -- that is the documented ghost, and it is a five-minute fix.

### THE RUNAWAY TEXT, CAPTURED AT LAST (2026-08-13 04:20) -- IT IS A CADENCE LOCK

The evidence log shipped in `f912af64` fired on its first live runaway, and it
answers the question every earlier analysis could only infer:

```
RUNAWAY EVIDENCE (55577 chars, 13828 tokens, ended_with_eos=False)
HEAD: {"title": "Echoes of Error", "premise": "A brilliant engineer discovers a
      flaw in her company's cutting-edge AI system, which could have
      catastrophic consequences if exposed too early..."
TAIL: Welcome, dear listener, to Echoes of Error. Let the echoes inspire us. Let
      the truth unite us. Let the future be ours to shape, as one, echo by echo.
      This is Echoes of Error. Let the echoes echo. Let the truth prevail. Let
      the future be ours to forge, hand in hand, echo by echo. Welcome, dear
      listener, to Echoes of Error. Let the echoes resound. Let the truth
      triumph. Let the future be ours to build, side...
```

**It is an anaphoric peroration loop** -- the model entered an inspirational
closing cadence and could not leave it. The structure is MIXED, which is the
part that matters for the fix:
* **verbatim anchors** -- "Welcome, dear listener, to Echoes of Error" and
  "echo by echo" recur exactly;
* **slot-substituted variation** -- inspire/echo/resound, unite/prevail/triumph,
  shape/forge/build.

`repetition_penalty=1.03` was ACTIVE and did not stop it, because the varying
half is what the penalty is designed to encourage. `min_p=0.05` was active too.

**THIS REFUTES THE REGISTER-COLLISION EXPLANATION for this instance.** The
premise is earnest -- an engineer, an AI flaw, catastrophic consequences. There
is no comic source, no OnlyFans, no tonal bind to hedge around. The Fable
analysis that produced tonight's pack-prose fixes may still be right about the
MARMOT leg; it is not what happened here. Treat the pack changes as a plausible
contributing fix, NOT as the cure.

**DESIGN CONSEQUENCE, and it confirms Fable's ordering for a better reason than
Fable had:** lead the halt with the **open-string token counter**, keep n-gram
detection as secondary. A window WOULD catch this one on its verbatim anchors,
so n-gram is not useless -- but the anchors are separated by ~25 tokens of
variation, so it fires late and only when the loop happens to contain repeats.
The counter is indifferent to the loop's shape: 13,828 tokens inside ONE
unclosed JSON string is pathological whatever the prose is doing, and it would
have halted this at ~2,000 instead of 13,828.

**Directive-safe signals** (none read `target_words`): non-closure of the
currently-open JSON string; n-gram self-similarity; deviation from the pass's own
healthy history (the very next leg ran P3 base calls in 60/75/65 s on the same box
-- the runaway was 20x that); tok/s and wall clock. Note `min_p=0.05` and
`repetition_penalty=1.03` were ALREADY active and did not stop it.

Recommended shape, three separate commits: an in-decode degeneracy + interrupt
criterion; the client `/interrupt` fix; and the honesty fixes in D-below. A
capacity-cycle bound ships as a backstop only, never as the headline.

**The variable was the NEWS ITEM, not the bank and not the code.** Diffed at the
time: only two commits touched the writer path since the last green `still_pan`,
and the relevant one (`98fb258f`, num_characters cap 6 -> 10) is arithmetically
IDENTICAL at the `num_characters=2` these profiles request. `--source-bank` pins
the BANK; the item still rolls. A re-roll on the same bank drew a ScienceDaily
physics story and rendered clean. Do not read a single bad leg as a regression
without checking which item it drew.

**2. `OUTPUT_TRUNCATED` gives advice that is wrong exactly when it fires.** The
guard says "Give this pass a slot whose window fits prompt+artifact". A
`ProviderCapacityMessages` pass sets `_otr_reserve_remaining_output_capacity`, so
`requested_tokens = context_cap` BY DESIGN and `effective < requested` is true the
moment the prompt is non-empty. There is no bigger slot -- the pass already has
every token there is. The message should distinguish the reserve-remaining case
and say "the model did not stop", because as written it sends the reader hunting
a config defect that does not exist. It cost this session about twenty minutes.
Branch on `reserve_remaining` at `OTR_LedgerScriptWriter.py:1052`.

**3. The 14k-token evidence is captured and then thrown away.** The capacity
raise attaches `raw_completion` at `OTR_LedgerScriptWriter.py:1089` and NOTHING
reads it -- the leg log prints `raw head: <empty>`. So at the one moment the
runaway text exists in memory, it is discarded, and the next reader has to
reproduce a 21-minute decode to see what the model was actually saying. Log a
head+tail of it.

**4. The two local transports disagree.** `_otr_model_loader.py:1305` raises a
bare `ModelLoaderError` with NO phase for the same condition the writer raises as
a phase-carrying `PromptContextOverflowError`, so an identical runaway on that
transport is not rerollable at all. It also tests `>=` where the writer tests
`==`.

### THE STILL-SPINE REPAIR -- ROOT CAUSE FIXED AND PROVED LIVE (2026-08-12 late)

**CLOSED as a repair, still OPEN as a PBUG.** Three swings; the third one
deleted the thing instead of adjusting it, and `mesh_stage` -- the leg that died
on it -- now renders.

`3446af3f` retargeted link 255 so the image producer reads the POST-AUDIO
ledger, and that killed both branches which mint `b000_music_open`
(`derive_opening_music_beat` returns None below the 2 s head gap; the fallback
branch's condition literally means "this is a PRE-AUDIO ledger"). Meanwhile
`render_driver._canonical_visual_beat_id` still rewrote the CONSUMER's lookup TO
that id. The still was on disk, in `required_scene_targets`, with a hash --
under the other name. The repair did not fail to fire; it fired and moved the
mismatch from the closing beat to the opening one.

Shipped, each its own green chunk:
* `4e49ee4b` -- `_still_spine_row_for_mesh` chose a row by ABSENCE: `""` was in
  the key set, so a music beat probing with an empty `char_id` matched any mesh
  row. PREREQUISITE, not cleanup -- until it landed, a same-id mesh test could
  not go red and the mesh half silently served a plausible wrong image.
* `c9c8e5c0` -- **the root cause**: `_canonical_visual_beat_id` DELETED, both
  call sites on the shot's own beat id. `_OPENING_MUSIC_SUFFIX` SURVIVES -- it
  is a classifier with three callers (`:1407`, `:2821`, `:4509`), not a
  translation. Do not reintroduce the function under any name.
* `a2a85bcc` -- the post-audio join fails LOUD at ShotLock and stays fail-soft at
  `SignalLostVideoRenderer`. `strict` is caller-scoped on purpose: the free
  function has two live callers with opposite criticality, and a global contract
  is wrong for one of them whichever way it is written.

**STILL OPEN and unchanged:** PBUG-20260811-02 cannot CLOSE, because its
acceptance leg (60-second opening AND closing cues) **is not configurable
today** -- `CUE_DURATIONS` is 12/8/4, both fiction assemblers omit
`target_duration_s`, `EpisodeAssembler` only TRUNCATES, and the canonical
runner's `--set` is whitelisted. With `_MUSIC_MAX_CHUNK_DUR_S = 22.0`, a 12 s
opening is ONE chunk, so `_002`/`_003` may be unreachable in production at all.
Operator ruled 2026-08-12: ship the fix, leave the PBUG open, do not build the
plumbing yet. **So the multi-chunk path remains UNPROVEN and no green sweep may
be worded as proving it.**

Also open, found by the panel, not yet built: `OTRShotLock.IS_CHANGED`
fingerprints only routing env while `lock()` reads a mutable durable ledger
(Bible 06.01) -- and `tests/test_route_freeze_wiring.py:278` deliberately pins
ShotLock and VideoDirector to the SAME fingerprint, so "always re-execute" is a
contract change needing its own justification, not a test edit in passing.

### SUPERSEDED -- the pre-fix candidate write-up (kept for the record)

`PBUG-20260811-02` moved from "root cause NOT ESTABLISHED" to established, on a
live reproduction (`fastwan_8gb`), and a candidate repair is committed
(`3446af3f`, `ae76fb3f`). **It is NOT proven and must not be called fixed.**

- **The general fault:** the image producer planned stills from the PRE-AUDIO
  ledger, so it planned against ids the finished episode does not use --
  `EpisodeAssembler` mints one row PER CHUNK (`music_{cue}_{NNN}`). Fixed by
  retargeting canonical link 255 to ShotLock's post-audio ledger. **This is the
  fix that scales;** a reservation can only ever name `_001`.
- **The local fault:** BOTH the opening and closing reservations suppressed
  themselves on a ROLE test, so a pre-audio sentinel under an authored id killed
  the reservation meant to cover it. Both are unconditional now, since `_add`
  already dedupes by beat id. **Hardening -- NOT the multi-chunk solution.**
- **STILL OPEN, flagged independently by two cross-checks:**
  - `OTR_ShotLock.overlay_audio_timing` swallows every exception and returns the
    PRE-AUDIO ledger, and a `_same_frozen_episode` mismatch does the same,
    warning rather than raising. **So the fix's premise can silently fail** and a
    multi-chunk `_002`/`_003` would die exactly as before. Bible **12.57**
    already prescribes the answer: reject mismatches. Untestable today --
    `overlay_audio_timing` short-circuits under `OTR_TEST_MODE=1`.
  - `ShotLock.IS_CHANGED` covers routing environment state but NOT the ledger it
    reads from disk, so image planning depends on a hidden disk read outside the
    dependency signature (Bible 06.01).
- **ACCEPTANCE TEST before this closes:** a canonical `fastwan_8gb` leg with
  **60-second opening AND closing cues**. `_MUSIC_MAX_CHUNK_DUR_S = 22.0`, so a
  60 s cue becomes THREE 20 s chunks; the original short cue never exercises the
  chunked path, which is the half the reservations cannot reach.

### THE FABLE2 WRITER -- TWO FIXES SHIPPED, THE THIRD SHELVED ON EVIDENCE

**`scifi_news_pro` is the ONLY bank routed to this writer**
(`banks.json: scifi_news_pro_multipass`; a grep for the whole-play grammar
across `nodes/story_packs/` returns exactly one file). Every other bank's Python
owns the speaker labels. **So the cross-bank writer gate CANNOT qualify this
writer** -- five green banks proved the pipeline, not `_otr_scifi_fable2.py`.
Coverage is repeated `scifi_news_pro` runs across distinct sources. Three axes
vary per run: the news item, one of 14 frame cards, one of 6 stances
(`OTR_FABLE2_SEED` pins the card/stance deal, not the news item and not the
sampling).

**SHIPPED** (r1 arc: Fable cold -> driver -> Codex -> Antigravity -> judged;
artifacts in `kibitz-runs/2026-08-12-writer-genre-slippage/`):

- `2572b493` -- **the repair turn now carries the rejected draft.** It never did:
  the retry ordered the model to keep the same wording about a text it had never
  been shown, so every attempt after the first was a COLD REGENERATION -- which
  is why four attempts produced four DIFFERENT malformed shapes instead of
  converging. Temperature decays only when the draft actually rode along.
- `45d1d3f8` -- **the one-shot format example was DEAD CODE.** The parameter and
  both use sites existed; nothing ever passed one, and the pack's `examples` was
  empty. Now a gardening-programme example, deliberately a different domain
  because it arrives as the model's own assistant turn, validated against the
  real parser.
- `8a7a4d62` -- fix B silently ate part of fix A's budget: the guard counted only
  prompt plus draft, never the example B injects on every call.
- `61ae356c` -- required ledger saves REFUSE instead of continuing silently.

**SHELVED, deliberately -- the candidate-retirement ladder (lesson 35).** A
judging pass found that the ALTERNATE PRODUCER SLOT every lane assumed -- Fable
proposed it, the driver accepted it, Codex and Antigravity both wrote MUST-FIX
items for it -- **does not exist**: `_ALLOWED_SLOTS = ("creative", "technical")`
and `repair_slot_fn` is never passed in this module. Building it means a new
widget plus canonical JSON wiring, which is its own project. All three dead legs
also predate fixes A and B, so the failure mode may already be largely closed.
**Gate the build on post-fix evidence that legs still exhaust.**

**CUT, do not re-open:** upstream address-shaped roster names (both mechanical
lanes rejected it independently -- `CastShape.name` is the ledger join key into
casting, credits, portraits and voices); prefix or fuzzy matching of abbreviated
speaker cues; loosening the parser.

### CAST COVERAGE -- OPEN, and it is a defect class rather than a lane

A cast member with NO line hard-fails the freeze cascade for EVERY bank, but
**only 2 of 10 packs tell their writer** (`scifi_news`, `scifi_news_pro`). Four
narrative packs share an identical six-stage structure and are all silent --
including `folger_scene_adaptation`, the shakespeare pack `mesh_stage` died on.

The obvious seam is WRONG: `outline_phase_system` plans ONE phase and cannot
enforce an episode-level invariant. Codex placed the check after the Stage-2
loop (`_otr_outline.py:1862`), BEFORE any Stage-3 call, and refuted the driver's
premise that a retry path already exists there -- `generate_outline()` completes
every Stage-2 and Stage-3 call and then only RAISES.

**Codex's cross-bank findings, each a claim to verify before building:**

1. **The legacy line composer has fix A's defect class** -- every attempt in
   `compose_line_draft` reuses unchanged `messages` and never sends the rejected
   line or the reason back. **If true, this is the same cold-regeneration bug in
   the writer five banks share.** Highest-value open item in this document.
2. `_otr_scifi_codex.py` has **NO `led.save()` at all**, so a P3/P5 death loses
   every accepted-stage receipt.
3. `scifi_news` ALREADY closes the coverage asymmetry through a fresh-candidate
   loop. It is the model to copy and must NOT be modified.
4. A deterministic sorted round-robin already assigns story speakers on the
   multi-cast fallback, which would violate "no deterministic Python deciding
   story" and could satisfy any coverage check without a model choosing.
5. Unchecked `led.save()` remain at `OTR_LedgerScriptWriter.py` lines 4774, 4868,
   4913, 5077, 5759, 5896, 5945 and 5990. Line 4245 is a diagnostic stamp: warn,
   then preserve the original halt.
6. Pack `examples` are POPULATED in three of the four packs and INERT by design
   (`_otr_story_pack.py:155-159`). Do not "fix" them.

### CLOSED LANE DIAGNOSES -- MOVED OUT 2026-08-12

The full lane 10 / 15 / 19 diagnosis write-ups (313 lines) lived here after
they were acted on and closed. Per this file's own rule -- *if a thing is
DONE, it does not belong here* -- they now live in `docs/HANDOFF_LOG.md`
under "CLOSED LANE DIAGNOSES", alongside the per-lane receipts in
`docs/evidence/lane_receipts/`. Nothing was deleted.

### CURRENT BASELINE -- carry forward, detect drift

| Thing | Value as of 2026-08-12 (story-writer + still-spine wrap) |
|---|---|
| Branch / HEAD | `v2.0-alpha`, == `origin/v2.0-alpha` (measured 2026-08-12 at the story-writer wrap, `ae76fb3f`) |
| **GROUNDING RULE (learned 2026-08-09)** | **`kibitz-runs/` IS GITIGNORED (`.gitignore:251`).** Two days of audit work lived in `kibitz-runs/2026-08-07-slugfest/` -- 71 slugs across 11 lists -- and was invisible to every doc search AND every `git log --all` search. The operator had to remember it existed. **Before grounding any item that smells previously-investigated, list `kibitz-runs/` by hand.** |
| Suite | **10340 passed / 110 skipped / 1 xfailed, exit 0** (measured 2026-08-13 at `d75a8866`, NOTHING deselected). Was 10309; the title-card build added 31 across three test files. **A caution learned building them:** all seven of the first tests passed while wrapped multi-line titles were silently wrong, because every fixture title fit one line -- including the "hostile" long-word case, which is one unbreakable word and never wraps. A green suite over a too-narrow fixture set is the failure mode here, not a flaky test. Prior note kept: **10309 passed / 110 skipped / 1 xfailed** (measured 2026-08-12 LATE at `f70df546`, NOTHING deselected). Was 10281 at `ae76fb3f`; this session added 28 across the mesh-identity, post-audio-join and credits-scroll work. Prior note kept: **10281 passed / 110 skipped / 1 xfailed** (measured 2026-08-12 at `ae76fb3f`, NOTHING deselected). Was 9963 at the lane-10 wrap; this session added ~318 tests across the writer, ledger, still-spine and campaign work. **The xfail count went 1 -> 2 -> 1**: PBUG-20260812-02 opened a STRICT xfail that did its job and forced its own deletion the same day when the field was fixed. Deselecting to get a green number hides real failures |
| Bug Bible | **20 passed / 24 skipped / 3 xfailed** at survival-guide `80fc358` (**275** entries, index **388** rows). 2026-08-12 promoted **07.31** + **07.32**, the two the previous window owed. **Run the gate with NO `--pack-dir`** -- that is the invocation the 20/24/3 baseline describes, and the 24 skips ARE the pack-dependent tests. Passing `--pack-dir` at OTR turns 20 of them red on pack code untouched by any Bible edit, which reads like a regression and is not one. Also pre-existing: `tools/reload_bug_bible.py` exits 1 at HEAD on 29 stale legacy_id/xref complaints; it is not the gate. Earlier: **12.97**, a model field name that collides with its base class (from PBUG-20260812-02, admitted on a live leg). It checked the others against the index FIRST and promoted no duplicate -- **PBUG-20260811-02's class is already 12.57**, whose own rule (resolve the durable owner, prove same-run identity, REJECT mismatches) also condemns the warn-and-continue fallback OTR still has open. NEVER re-scrape indexed history |
| Variants | `build_variants.py --check` **46 variants / 0 failures** on THIS box. **THAT COUNT IS WORKSTATION-DEPENDENT and 46 is not the repo's number** (lane 11, 2026-08-11): `git ls-files` counts **45** tracked variants, and the 46th on disk is another window's untracked `otr_upscale_ltx_probe.json`. The gate globs the DIRECTORY, so its headline silently counts files the repo has never seen -- the same defect class as the sbcov crash above. Compare `--check`'s 0 FAILURES, not its count. **RUN IT BEFORE STARTING A LANE** -- it had been RED since lane 5 and lane 7 had to separate inherited drift from its own; a red at the start of a lane belongs to whoever caused it |
| Canonical workflow | **TOUCHED 2026-08-12** -- link 255 retargeted so `OTR_MetaBriefImagePromptGen` reads `OTR_ShotLock`'s POST-AUDIO `patched_ledger_json` instead of the pre-audio freeze cascade (`[255,62,1,89,0] -> [255,90,0,89,0]`). Validated: 23 nodes / 56 links unchanged, acyclic, referential integrity clean, `validate_canonical_workflow` OK, 50 variants REGENERATED (`--check` 0 failures) and 4 hand-kept `.env.json` master_hash re-stamped. **The diff is ONE line** -- a first attempt round-tripped the JSON and reformatted all 3506 lines; that was reverted and redone as a surgical string edit |
| Cloud profiles | `macbeth_probe` gate REMOVED from both; `openrouter_model_pins` + `audio_cache` remain |

A window that reads a different suite number has inherited drift -- find out
why before building on it.

### WHEN THE SUITE IS RED, CHECK `git status` FIRST

Standing gotcha, learned 2026-08-11: three "failures" were a concurrent
window's uncommitted edit, not a defect. Check for another window's
in-flight work before diagnosing -- and never sweep their files into your
commit. Deselecting tests to get a green number hides real failures.

### LEMMY COCKNEY -- NOT BLOCKED (corrected 2026-08-10), Phase 1 shipped, 2-4 open

**THE "ACTIVE, SECOND WINDOW -- do not collide" HEADER WAS STALE and is
withdrawn.** Measured 2026-08-10: **no window is holding any Lemmy file** --
`git status` shows nothing dirty under `cast_pools.py`, `cast_lock.py`,
`_otr_dialogue_policy.py`, `_otr_line_composer.py` or `_otr_compose_exchange.py`.
That window shipped Phase 1 in `bec0ca79` and left. The only window still
holding anything is the VIDEO one (`eng_wan_i2v.py`), which touches nothing here.
Lemmy is CPU-only work and available.

**AND r1's OWN `final.md` IS NOW STALE ON ITS STEP 1.** It opens the order of
work with *"Reconcile D-1: `accent: 'neutral'` contradicts a description saying
broad friendly Cockney"*. **That is FIXED** -- `config/cast_pools.py:317` reads
`"accent": "cockney"`, landed by Phase 1 AFTER r1 was written. Anyone resuming
from `kibitz-runs/2026-08-08-lemmy-cockney/r1/final.md` must skip its step 1.
D-4 (governing the writer) is also partly addressed by Phase 1's
`dialogue_orthography`, `speech_signature` and `nodes/_otr_dialogue_policy.py`.

**LEMMY r2 RAN 2026-08-10 (both lanes) -- see `kibitz-runs/2026-08-10-lemmy-cockney/r2/`.
r3 NOT run, deliberately: r2 hit a PLAN-LEVEL blocker that is now operator row 14.**

**SHIPPED from r2's cleanest finding:** the qualification receipt is honest now.
`approved_native_routes` had listed bark as APPROVED via the bare string
`canonical_bark_preset_v1` -- no artifact, no hash, no test lines, no operator
verdict -- while r1 states plainly that no voice on any engine is audition-proven.
A field that reads as evidence and is not, sitting inside the policy meant to
prevent exactly that. `approved_native_routes` is now EMPTY, `canonical_route`
keeps bark as a ROUTING fact with `qualification_receipt: None`, and
`QUALIFICATION_RECEIPT_REQUIRED_FIELDS` + `is_qualified_route()` define what a
real receipt must contain (artifact + sha256, both audition lines verbatim, seed,
engine/impl version, identity, settings, **operator_verdict**, date). Fail-closed:
a bare string, a missing field, or a present-but-empty field is UNQUALIFIED.
34 new tests, mutation-checked -- restoring the bare string turns the guard red.
**Zero behaviour risk: `LEMMY_VOICE_POLICY` has no production consumer** (Phase 1
shipped it defined-but-unwired), so this could land without a re-baseline.
**This gives the audition (D-3) a target to fill in rather than an approval to
explain away.**

**WHAT IS ACTUALLY LEFT, verified at the line 2026-08-10 -- D-2, the
partial-rollout state both r1 lanes called the worst possible one.** The
per-character voice pin still does NOT reach pre-locked LEMMY:
`_otr_casting.py:1815-1837` builds `cast_voice_slots` for every row, but
`timbre`, `role` and `age_band` all come from the ensemble slot, and a pre-locked
row has none -- the code comment says so outright ("empty for the pre-locked
announcer / LEMMY rows which have no ensemble slot"). `voice_cast_decision` is
likewise built only for open characters and only under `hybrid_voice_fit_enabled()`.
So he is hard-pinned to bark via `lemmy_row()` and cast on **gender alone** on the
other six char-voice engines.
**D-2 IS OPEN ON SIX ENGINES ONLY (scoped 2026-08-11).** The `indextts2` half is
closed by the Branch A qualified route, which pins him directly and never reads
`cast_voice_slots`. On the OTHER SIX char-voice engines nothing changed: a
pre-locked LEMMY row still has no ensemble slot and is still cast on GENDER
ALONE. **A pin there still owes a DECLARED re-baseline** -- Branch A did not,
measured (unclaimed rows are byte-identical against a no-policy baseline at both
`allow_voice_reuse` settings), but that result does not transfer.

### OPERATOR DECISION -- six committed variants whose SOURCE PROFILES are untracked (raised 2026-08-11, lane 11)

**`scripts/build_variants.py --check` CRASHES ON A FRESH CLONE.** Not fails --
crashes. Found while closing lane 11, verified rather than reasoned, and it is
the gate the build law requires green before EVERY lane.

`workflows/variants/otr_sbcov_{1..6}.json` + their `.launch.md` recipes are
**COMMITTED**. Their six source profiles `config/profiles/otr_sbcov_{1..6}.json`
are **UNTRACKED** -- `git log --all` on them is empty, they are NOT gitignored,
they are dated 2026-07-20 and marked `status: draft`. So the repo ships six
generated artifacts whose sources it has never tracked.

`--check` regenerates every committed variant from its profile id. A missing
profile raises `ProfileError`; the loop catches only `EmitRefused`. Measured:

| | MRO |
|---|---|
| `EmitRefused` | `EmitRefused -> RuntimeError -> Exception` |
| `ProfileError` | `ProfileError -> ValueError -> Exception` |
| caught by `except EmitRefused`? | **False** |

It reports "46 variants, 0 failures" on this box **only because those six
untracked files happen to exist here.**

**THE PROVENANCE SETTLES IT, and it was found by the Codex consult reading a
file I had not opened.** `tmp/_gen_profiles.py` (2026-07-20) is the generator
that wrote all six, and its own docstring says:

> "These are throwaway smoke config (like a temp probe script) -- written into
> config/profiles/ because load_profile() only reads that dir, and DELETED
> after the sweep. **NOT committed.**"

So the profiles are behaving exactly as designed. **What leaked is the OTHER
half: the twelve generated `otr_sbcov_*` variant + recipe artifacts got
committed and should not have been.** The question is therefore not "adopt six
sources?" but:

**Your call:**
1. **DELETE the twelve leaked `workflows/variants/otr_sbcov_{1..6}.json` +
   `.launch.md` artifacts** (recommended -- it matches the stated intent, and
   they are unreproducible on any box without the throwaway profiles), OR
2. **COMMIT the six profiles** if the sbcov sweep is in fact still live work
   and those variants are meant to ship.

A coder window must not pick: deleting committed artifacts changes what ships.
**Independently of the choice, `--check` should catch `ProfileError` and report
a named failure instead of crashing** -- a plain robustness bug any window may
fix once you have decided.

**Not blocking lane 11 any more.** The lane closed 7/7 on its own merits by
declaring its profile canvas channel INERT rather than reconciling it, which
makes G2.3 skip the profile comparison entirely -- so `otr_sbcov_4.json` is
irrelevant to it. Detail: `docs/evidence/lane_receipts/lane11-viz_green.md`.

**Related, same defect class, found the same way:** the "46 variants" figure
quoted as a baseline in this file is WORKSTATION-DEPENDENT. `git ls-files`
counts **45** tracked variants; the 46th on disk is another window's untracked
`otr_upscale_ltx_probe.json`. `--check` globs the directory, so its headline
count silently includes files the repo has never seen.

### WAITING ON THE OPERATOR -- the whole list, in one place (2026-08-09)

Nothing below is blocked on a coder. Each row needs YOU. Collected here because
they were scattered across nine sections and no window could answer "what is
waiting on me?" without reading the whole file.

| # | What | What kind of answer | Where the detail is |
|---:|---|---|---|
| ~~7~~ | ~~**The 8 swept survival-guide guards**~~ **DECIDED 2026-08-10: KEEP THEM.** The operator delegated the call (*"between you and Sonnet you can ensure we keep or remove those tests"*) after the original question was found to carry a false premise. Two corrections of record: `656c36e` touched TWO files, not eight (`BUG_BIBLE.yaml` + `tests/bug_bible_regression.py`), so the 8 `test_otr_*` items are FUNCTIONS in one file; and **none of them is video-related** -- they guard positioned-media timeline ownership, explicit word delivery, outer-word-fit fail-closed, protected-suffix surface, cast-role identity, rename transactions, canonical ledger text metrics, and P5 text transport. New video-lane tests would replace NONE of them, so deleting cost coverage and bought nothing. A follow-up commit in that repo should document what rode along in `656c36e` | -- | -- |
| ~~8~~ | ~~**`otr_upscaled_dir()` is DEAD**~~ **ANSWERED 2026-08-10: DELETE. Executed -- helper, `__all__` entry, contract-test reference and a dangling history mention all removed.** | -- | -- |
| ~~9~~ | ~~**`meta.perfect_run_spacesaver`**~~ **ANSWERED 2026-08-10, and the answer REOPENED IT AS A FEATURE, not a cleanup.** Operator: *"once people get the workflow going wouldn't it be nice not to store all the little files on their drive and just save the last otr/obs episode ... if it doesn't work let's rip it out and design a new one, or keep it actually and make it work as intended"*. So: the flag is NOT to be quietly dropped -- it either becomes real or is replaced by a designed successor. Scoped as a coder item below | -- | -- |
| 10 | **The 28 portrait conflicts are unreviewed.** `FATHER BROWN` shipped female; `Clara` gendered male with "her" in her own prose | READ THE LIST, rule case by case. Do not total it. `ROSALIND` is NOT one -- Ganymede keeps her female voice by your earlier ruling | sprint item 8 |
| ~~11~~ | ~~**`story_orchestrator.py` cast-merge ruling**~~ **ANSWERED 2026-08-10: operator wants AN LLM PASS to clean this up**, not a test pinning today's behaviour and not a mechanical re-baseline. Scoped as a coder item below | -- | -- |
| 12 | **v2.1 candidate:** a low-footprint LOCAL model steeped in OTR diction | A CONSCIOUS RE-OPEN. It collides with your 2026-08-04 "story quality is done" directive, so only you can reopen it | STILL OPEN item 7 |
| 13 | **The Nano Banana still model may be RETIRED UPSTREAM (new 2026-08-09, and it may affect already-shipped stills).** The `Nano Banana 2 (Gemini 3.1 Flash Image)` selector resolves to `gemini-3.1-flash-image-preview` and is sent to a **Vertex proxy** (`cloud_media_invoke.py:510`), NOT the catalog endpoint that was measured -- so catalog presence does not prove that route works. Codex reports a public Google shutdown of 2026-06-25 with `gemini-3.1-flash-image` as replacement; **UNVERIFIED from this box** and the id was still catalog-listed on 2026-08-09 | **ONE RENDER SETTLES IT** -- push one still through the Nano Banana lane; it either renders or the proxy rejects the id. Then: repoint the selector at the stable twin (already shipped, confirmed live), or leave it. It is a MODEL SWAP on the stills path, so it is recipe-adjacent and yours, not a coder's | `docs/2026-08-09-BUILD-SPEC-slug-provenance-non-video.md` Â§0A |

| 14 | **LEMMY PHASE 2 CANNOT BE BUILT AS SPECIFIED -- a scope call, surfaced by the r2 panel 2026-08-10 and confirmed at the line.** The plan's central behaviour is *"if an engine cannot meet the Cockney floor, suppress the cameo on that engine rather than silently substitute"*. **That is not expressible in the current graph.** Lemmy is selected UPSTREAM in the writer (`OTR_LedgerScriptWriter.py:4412-4426`); the voice engine is chosen LATER by nodes 80/81, and `BatchCharacterVoices` exposes ONE engine for the entire character bus. So by the time anything knows the engine, the cameo is already written into the script | **PICK ONE:** (a) downgrade the requirement to a FAIL-CLOSED CastLock error (no new surface, but a bad combination stops the render instead of degrading); (b) add ONE upstream engine-policy authority feeding writer + CastLock + renderer so the writer can decline the cameo before authoring -- this needs a NEW NODE/WIDGET SURFACE and `otr_canonical.json` wiring, which r1 explicitly ruled out of scope, so it is a deliberate scope change only you can authorise; or (c) accept the cameo on any engine and drop the floor | `kibitz-runs/2026-08-10-lemmy-cockney/r2/` |
| 15 | **Should LEMMY be able to appear in `scifi_news` again?** It was his FIRST lane and it worked under the legacy cast picker; the lane later became CONTENT-OWNED, so the writer's cameo picker never runs and the ledger records no decision at all. Not a careless break -- a capability lost to an architectural change (PBUG-20260811-03, root cause established) | A PRODUCT call, then a coder task. YES -> the lane runner offers the cameo when it builds its cast. NO -> it stamps an explicit declined-policy so the ledger says so. Either way the silence ends | `docs/PROD_BUG_LOG.md` PBUG-20260811-03; OPEN BUGS trio above |

**Two that are NOT waiting on you, despite reading that way:** H3 no longer owes
a dropdown ruling (it became a sprint series), and queue item 1 is closed.

### LEMMY BRANCH A -- IN PROGRESS 2026-08-10, foundation landed

**Authoritative plan:** `docs/2026-08-10-OPEN-PLAN-lemmy-cross-engine.md`
(rewritten by a second window from my open plan; my rebuttal at
`docs/2026-08-10-REBUTTAL-lemmy-identity-premise.md` was accepted and folded in).

**G0 IS CLOSED -- the operator APPROVED on the record**
(`docs/2026-08-10-G0-RIGHTS-DECISION-CARD-lemmy.md`, decided
2026-08-10T20:37:17Z, against a same-day snapshot of the Gemini API Additional
Terms and the Prohibited Use Policy, both quoted in the card). Scope: use of
self-generated Google TTS output (voice Algenib) as a clone reference for LOCAL
engines. Tier left UNDETERMINED and marked for the evidence packet -- it governs
what Google may do with our data, not our rights to the output.

**THE PREMISE CORRECTION THAT SURVIVED REVIEW, and must not be re-inherited:**
Lemmy was never redrawn per episode. 1,633 ledgers, 186 LEMMY rows: 151 carry
`voice_ref_id=None` (the bark-preset path, expected) and **33 of the remaining 35
are the SAME reference**, `vz_donor_marshal_indian`. The second window explained
the mechanism -- all 33 had `meta.episode_seed=None`, so CastLock derived an
identical selector seed every time. **He was ACCIDENTALLY PINNED.** The fix is an
explicit qualified re-pin, NOT a rewrite of the generic selector. A current
40-seed sweep selects 14 refs, so an unpinned future route WOULD vary.
The defect is a **floor-EVIDENCE failure** (the incumbent cannot prove the
configured floor) -- NOT, as I originally wrote, proof from the `_indian` name
or the `warm` timbre tag that it violates accent/vocal weight. Neither is
supported by the bank metadata and that overreach was correctly rejected.

**SHIPPED so far (both independent of which voice wins the audition):**
* **Plan 5.1 -- `nodes/_otr_voice_route.py`.** `validate_qualified_voice_route`
  makes a route PROVE itself: file exists, bytes hash to the receipt, the
  ENGINE TRIPLE agrees (route == active scalar == bank entry), rights not
  expired/revoked, closed status vocabularies, contract version supported.
  Fail-closed on every unknown. Returns reasons, not a bare bool. The legacy
  `is_qualified_route` REMAINS as a compatibility helper and may never authorize
  a selected route -- a test pins that it says yes where the new one says no.
  65 tests.
* **Plan 5.3 (first half) -- route identity in the cache key.** `route_id`,
  `route_contract_version`, `qualification_record_id`, `weight_revision` added to
  `ResolvedVoiceRequest` AND to `IN_KEY_FIELDS`; partition invariant verified
  intact. Legacy rows keep byte-identical keys via empty/zero defaults, so
  nothing untouched gets re-baselined.
  **`REQUEST_SCHEMA_VERSION` 2 -> 3**, deliberately: adding a key field changes
  every request's cache_key, and the bump routes that invalidation through the
  designed `needs_rerender` slim-migration path instead of silent key drift.
  Measured at bump time: **ZERO cached entries on this box**, so the practical
  cost was nil. `CACHE_SCHEMA_VERSION` unchanged -- the sidecar shape did not
  change, only what the key is computed over.

* **Plan 5.2 -- CastLock re-pin ordering. SHIPPED 2026-08-10 (`e791344b`).**
  `lock()` now runs the plan's six ordered steps: the revision is stamped before
  any route resolution; bank + engine metadata resolve ONCE for both cast
  policies (preserve_ledger still passes `bank_entries=None`, so its
  `char_voice_engine` stamp stays `auto` and does not start pinning a concrete
  engine); rows match on normalized name OR char_id against a new
  `LEMMY_VOICE_POLICY["character_key"]`, because Lemmy is positional `c02` and a
  matcher keyed to a literal `lemmy` char_id would claim nobody; the route is
  proved ahead of both the hybrid voice-fit and the generic selector; only the
  claimed row changes; and a selected route that fails qualification raises
  `VoiceRouteError` with NO fallback. New `_resolve_policy_claim` /
  `_apply_policy_claim` on CastLock, and `select_policy_route` /
  `resolve_policy_route_claim` / `cast_row_matches_policy` in
  `nodes/_otr_voice_route.py`. 32 tests.
  **IT SHIPPED INERT and that is the point:** `approved_native_routes` is still
  `{}`, so nothing is selected, the voice bank is not even consulted on the
  dormant path, and no current render changes. It activates the day G1 Test A
  puts an operator receipt in that dict. Sonnet QA pass was clean on all nine
  contract items; its one finding (the claim resolver not being handed the bank
  `lock()` had already loaded) is fixed in the same commit.

* **Plan 5.3 second half -- reference resolution, receipts, fingerprint.
  SHIPPED 2026-08-10 (`fdc016ef`).** `resolve_and_verify_reference` (the 5.1 seam
  that had never actually been built) now runs right after `cast_lookup`, and a
  proved local route RENDERS ITS OWN BYTES -- without that override the route
  proved one file while the generic resolver handed the adapter another.
  `build_resolved_request` carries the four route fields into the cache key.
  `IS_CHANGED` stopped returning the constant `"static"` for local legs, which
  had asserted that a local voice render can never change: swap the reference WAV
  under a pin and ComfyUI would serve the previous render while the ledger
  claimed the new route. It is now a fingerprint over route identity, runtime and
  the actual reference bytes -- with a ledger carrying NO routes still returning
  the literal string `"static"`, never a network call, and NaN (fail-open) on an
  unreadable expected local file. Per-line receipts now also land on the LOCAL
  path for policy-route lines (`voice_route_id` + `sample_rate`), scoped to those
  lines because stamping every local line would reload-and-resave the whole
  ledger on every ordinary leg -- which `test_end_to_end_google_tts_cache_off_
  byte_identity` forbids, correctly. 36 tests.
  **Sonnet QA caught a real defect and it is fixed in the same commit:** the
  "receipt must persist" rule first fired on a bare degraded COUNT plus "was any
  route in this render a policy route", so an unrelated line's failed stamp would
  have thrown away good, fully evidenced route audio. `_persist_ledger_stamps`
  now reports WHICH line_ids failed and the raise compares that set against the
  route's own lines.

**BRANCH A IS DONE -- shipped, G1 passed, proven in a live six-bank sweep.**
Detail lives in `docs/HANDOFF_LOG.md` and
`docs/2026-08-11-FINDING-lane-cast-contract-divergence.md`; it is not repeated
here, because this file is forward-only.

**WHAT REMAINS ON THE LEMMY SPRINT.** Remediation plan `.gemini/antigravity/brain/c494e2df-.../implementation_plan.md`,
triaged chunk-by-chunk 2026-08-11:

* **Chunk A1 -- DONE 2026-08-12.** `OTR_INDEXTTS2_EMO_ALPHA` was read at
  generate time while the cache key captured `profile.default_params` at
  request-build time, so an env override changed the RENDER without changing the
  KEY, and `IS_CHANGED` carried no alpha term either. Closed with the NUMERIC
  sibling of the mechanism that already existed for this exact class:
  `identity_params` folds an env-selected MODEL into the key for Google TTS, and
  the new `render_time_params` folds env-resolved numeric knobs into
  `quantized_params`. The engine resolves the value through the SAME function
  its forward calls, so key and render cannot disagree, and per-render env
  pickup is preserved. Every engine without such a knob is byte-identical and
  still answers `IS_CHANGED == "static"`; only `indextts2` now fingerprints.
  Tests: `tests/test_lemmy_emo_alpha_cache_key.py`.
* **Chunk C items 2/3/5.** SceneSequencer integration coverage at 22050/44100 --
  every existing fixture starts AT the 48000 bus, so nothing proves the real
  resample path -- plus the rate-assumption sweep. Item 1 is done.
* **Chunks A2 -> A3 -> A4.** v2 identity + replay bridge. **Operator ruled
  2026-08-11: AUTO-PROMOTE on a clean replay** -- if A4 reproduces all six frozen
  clip hashes, A2/A3 rewrite the receipt's identity fields with no second
  sign-off. `tts_emo_alpha` follows the existing loose numeric convention; no new
  null-shape walker.
* **Chunk B -- DONE 2026-08-12, and it was NOT the tidy-up it was filed as.**
  The three cloning adapters' private `_resolve_ref` tried exactly ONE candidate
  (`<comfy_base>/models/<ref>`) and otherwise returned a cwd-relative
  `os.path.abspath`, while the voice node's own `_resolve_ref_to_disk` knew
  about three more places -- including the `C:\ComfyUI-Models` root from the
  Comfy Desktop 1.0.4 migration. **On this box BOTH Lemmy reference WAVs live
  ONLY under that migrated root**: the qualified Branch-A reference
  `lemmy_algenib_cockney_v1.wav` and the historic incumbent
  `vz_donor_marshal_indian.wav` are both ABSENT from the historical location and
  PRESENT under the migrated one. So the node's existence check confirmed the
  file and the adapter that had to OPEN it resolved to a path that is not there.
  Now one shared `resolve_voice_ref_path` in `_otr_audio_engines/base.py`, with
  the historical candidate still tried FIRST so the fix can only turn a miss
  into a hit. Tests: `tests/test_lemmy_voice_ref_resolver_breadth.py`.
  **NOT logged as a PBUG:** it is grounded in the filesystem and a faithful
  replay of the old resolver, not in a captured live failure, and the admission
  rule wants the bug to have failed in a live run. A forced-Lemmy live render
  would settle it either way and is the natural next Lemmy step.
* **Chunk E.** Release/OBS audit: does swapping Lemmy's voice count as an
  EDITORIAL RECAST for an audience that already heard the old one? Operator only.
* **Branch B stays unbuilt.** It existed only for a G1 failure, and G1 passed.


### GOOGLE SLUGS -- TWO OPERATOR CALLS OWED (the build shipped `4bc760c8`)

Verifier: `python scripts/verify_google_slugs.py` (exit 0/2/3/4/5; never
writes a date back). Measurement: `docs/2026-08-10-MEASUREMENT-google-catalog-slug-provenance.md`.

**TWO THINGS NEED THE OPERATOR, and neither is a code defect:**

1. **`gemini-2.0-flash` and `gemini-2.0-flash-lite` are GONE from Google's
   catalog.** Found by the new verifier on its first live run (exit 2, a complete
   52-id listing). They are still shipped in `GOOGLE_API_LEGACY_TEXT_MODELS`, so
   selecting either would hit a model Google no longer lists. **Left in place and
   left loud** -- removing a model from the operator's choices is his call, and
   making the gap impossible to hide is what the tooling is for.
2. **`gemini-3.1-flash-image-preview` is still the one named preview exception.**
   It IS in the catalog, and that settles nothing: runtime sends it to a Vertex
   proxy, not the endpoint fetched, so presence proves the id is listed, never
   that the route works. It also undercuts the reported 2026-06-25 shutdown --
   the id is listed 46 days later. Cheapest settlement is unchanged: one still
   through the Nano Banana lane. It either renders or the proxy rejects it.

Run it any time: `python scripts/verify_google_slugs.py` (exit 0/2/3/4/5; it
never writes a date back).

### FOLLOW-UP CHIPS OWED -- coder work, no operator input needed

Consolidated 2026-08-09 from four tombstones. **Verified against the tree, and
two chips that were listed as owed are already DONE** -- the upscale-model SHA
is pinned (`eng_spandrel_esrgan.py:79`, a real 64-hex digest, not the empty
bootstrap) and the `visual_style_receipt["attempts"]` thread landed in
`e16e9a63`. Both are struck below rather than deleted, so nobody re-derives them.

1. **Stale metadata retention across legs** (SF#1, Codex r4 MF-1). A downstream
   reader consuming an obsolete field is a different defect class from "helper
   unwired". Needs its own arc.
2. **Audio-cache corruption-warning coverage. REWRITTEN 2026-08-09 from the
   kibitz r1 judgment -- THE ORIGINAL CHIP TEXT WAS WRONG TWICE and is not a safe
   basis to build from.** It claimed `_write_audio_atomic` had no coverage; in
   fact missing payload, hash mismatch, cache-key mismatch and partial
   atomic-write failure are ALREADY covered at
   `tests/test_audio_cache_wiring.py:117-163,205-219` -- a grep for
   `degraded_write|_write_audio_atomic` misses them because they use other names.
   And there are **nine** warning exits, not "a warning":
   `_otr_audio_cache.py:333,343,348,351,355,361,364,372,382`.
   **Real deliverable: warning/COUNT coverage layered on existing behavioural
   coverage.** "ONE bounded warning" (`:172-175`) is the load-bearing half of the
   promise, and a test asserting mere presence would pass if the code logged six.
   **THE TAXONOMY IS NOW DECIDED (Fable design ruling 2026-08-09) AND THE
   SIDECAR HALF IS SHIPPED.** Absent -> silent (the definitional miss every cold
   cache hits). Schema drift -> silent (a DESIGNED invalidation; the record is
   intact, the reader's target moved). **Present-but-unparseable -> ONE bounded
   warning**, added in `get()`'s `except`, which is where it is FORCED to live:
   `load()` reaches the sidecar only through `get()`, and by then the failure
   class has already collapsed into `None`.
   **Why it is corruption and not an ordinary miss:** `put()` publishes the
   sidecar LAST via `os.replace` precisely so its presence IS the commit signal,
   so a garbled one means the commit marker itself is damaged -- the strictest
   corruption this cache can exhibit, and the only one that was silent while
   nine lesser ones warned. It is also a CLOUD cache, so each silent instance
   re-bills the provider with no trace. BOM contamination from a stray
   PowerShell write lands in exactly this branch, which this project has met
   before. The docstring was RIGHT; the code was wrong.
   **Remaining here:** the payload-level warning/COUNT coverage described above.
   **Two existing tests cannot reach the branch they are named for:**
   `sample_rate`/`channels` at `:137-150` are IDENTITY fields
   (`_otr_resolved_request.py:75-84`), so changing one changes the cache KEY and
   `get()` returns an ordinary miss before reaching `:348/351`. **BUG-12.87
   again** -- the second live instance found on 2026-08-09. Correct technique:
   mutate the persisted SIDECAR with the request/key held fixed.
3. **A dying cache-enabled line reports `cache=off`** (SF#1, Fable gate).
   `cache_status` initialises `"off"` at `_otr_voice_node_common.py:829`;
   `generate_voice` sits inside the try at `:831-850` and the enabled-path P-OBS
   tail is emitted from its `finally` at `:935-939`, so a raise mid-generation
   reaches the tail with the initial value intact.
   **Correction (kibitz r1):** the earlier text here implied the cache-OFF path
   also emits `cache=off`. It does NOT -- cache-off emits the original BARE line
   with **no `cache=` token at all** (`:790-796`), pinned by the byte-identity
   test at `tests/test_audio_cache_wiring.py:592-598`.
   **Fix, panel-preferred over inventing a token:** set `cache_status = "miss"`
   immediately after an enabled lookup returns `None`, BEFORE generation -- true
   at that moment, existing vocabulary, and it cannot touch cache-off bytes.
   Then extend the two-node partial-exception test (`:681-748`) to require
   exactly one dying-line P-OBS carrying `cache=miss`.
   Receipts: `kibitz-runs/2026-08-09-audio-cache-chips/r1/judgment.md`
   (GITIGNORED; **r1 ONLY -- a scoped receipt, never to be called an arc**).
4. **`SpandrelEsrgan._resolve_model` robustness pair.** An unreadable NON-winning
   candidate aborts the whole search instead of skipping; the winning file is
   stat'ed twice with a TOCTOU window. **THIRD touch of this logic, so the
   two-strikes rule makes a full kibitz panel MANDATORY before any code.**
5. ~~Upscale-model SHA pin~~ **DONE** -- `_model_sha256` is pinned.
6. ~~`visual_style_receipt["attempts"]` always reports 1~~ **DONE** (`e16e9a63`),
   and the shared `on_attempt_complete` contract three callers depend on is now
   pinned against the real ladder.

### OPERATOR-DIRECTED WORK, SCOPED 2026-08-10 (answers to the decision round)

**A. THE SPACE-SAVER BECOMES REAL (or gets a designed successor).**
Operator's words: *"wouldn't it be nice not to store all the little files on
their drive and just save the last otr/obs episode ... if it doesn't work let's
rip it out and design a new one, or keep it actually and make it work as
intended."*

`perfect_run_spacesaver` is currently a DEPRECATED no-op checkbox that still
stamps `ledger.meta`. It is NOT to be silently dropped.

* **The checkbox itself must survive regardless of the design.** `widgets_values`
  is POSITIONAL (BUG-LOCAL-097) -- removing a widget shifts every saved value in
  every saved graph. Any successor either reuses this slot or appends a new one.
* **THIS FEATURE DELETES USER FILES, so it needs a real spec before code.** The
  output-tree contract exists because obs/ must hold exactly ONE mp4 per episode
  and intermediates were deliberately moved under per-episode subdirs. A
  space-saver interacts with that contract directly.
* **WHAT IT SWEEPS -- operator, 2026-08-10: "it deletes images and video clips,
  not the ledger."** So the target set is the per-episode visual intermediates
  (stills/frames and the per-beat/segment clips), and the LEDGER IS EXPLICITLY
  PRESERVED. That is the safe shape: the ledger is the replay and audit record,
  so an episode stays reproducible and inspectable after the sweep even though
  its intermediate media are gone. Keep the published `otr/obs/` deliverable.
* Open design questions that remain: is AUDIO in or out (the operator named
  images and clips, not audio -- and the content-addressed audio cache is a
  separate store with its own lifetime, so do not sweep it by accident); does it
  run only on a SUCCESSFUL publish; opt-in per render or a global setting; and
  what a re-render does when it needs a swept clip.
* **Hard constraint:** it may never delete anything before `obs_publish OK`, and
  it must never touch another episode's tree.

**B. AN LLM PASS TO RECONCILE MERGED CAST ROWS.**
Operator, on the merge that can take a character's VOICE from one row and its
GENDER from another (`story_orchestrator.py:416-421`, each field folded under its
own independent guard): *"there should be an LLM pass to clean this stuff up."*

* This is CORRECTNESS work, not story-quality chasing -- a character's
  gender/voice contradicting itself is explicitly carved out of the 2026-08-04
  "story quality is done" directive.
* **The ledger-completeness rule governs any new pass:** enumerate every field it
  writes, give each exactly one owner, and prove it on a LIVE leg. A pass that
  reconciles cast rows touches casting, which is downstream of everything.
* Note the collision: changing merged output MOVES the gender roll, which breaks
  replay parity and needs a declared re-baseline. The LLM pass must be designed
  with that in mind rather than discovering it late.
* Held while LEMMY is active -- it exercises casting.

### CARRY-FORWARD -- how to run a bank-specific qualification (from the closed announcer-intro work)

PBUG-20260807-01 is FIXED and LIVE-PROVEN (5/5 legs, four banks, three model
families). Receipts in `docs/PROD_BUG_LOG.md`. Three things worth not
re-learning:

1. **Always override the bank per leg and ASSERT `meta.source_bank`.** The
   canonical graph is pinned to `scifi_news`, which returns BEFORE the writer's
   close/intro block -- so an un-overridden leg is green and proves nothing.
2. **Assert on `meta.source_bank`, never `meta.source_meta.kind`.** `kind` is the
   FETCH MECHANISM (`media_archive_rss`, `original_llm`), a different vocabulary.
   A verifier that read `kind` false-flagged a passing leg.
3. **Dry-run first.** Catalog model ids carry size suffixes (`... (12.0 GB)`);
   the plain id is a hard `ValueError`. The `--dry-run` `applied:` line is the
   only proof an override actually landed.

### STANDING POLICY -- provider model slugs (from the closed curation chunks)

Queue item 1 is CLOSED (chunks A and B both shipped; see the tombstone in THE
QUEUE). What survives as a RULE:

**SUPERSEDED AND HARDENED 2026-08-10 -- the operator calls them EVERGREEN
slugs.** "Remove all dead slugs and only keep dynamic ones -- latest -- that
won't die." It is now an ENFORCED policy, not a preference:
`EVERGREEN_EXEMPTIONS` in `nodes/_otr_slug_provenance.py` fails the build on any
concrete slug that has neither a pointer nor a written, measured reason none
exists. Applied fully where possible (the Google text lane is three pointers and
nothing else, and gained `gemini-pro-latest`, which Google publishes and this
pack simply was not offering). Exempt where measured impossible: Google ships NO
`-latest` for image, speech or music, ElevenLabs has no pointer convention, and
Comfy's partner catalog is a pinned list whose pointer support is unverified.

**AND BOTH OPENROUTER SLOTS NOW DEFAULT TO `openrouter/auto`** (operator, same
day). The argument that carried it is his and it is a good one: a
`~vendor/model-latest` pointer dies with its vendor, a router does not --
`~latest` is evergreen at the VERSION level, a router is evergreen at the VENDOR
level. Two routers are offered (`openrouter/auto`, `openrouter/auto-beta`) and
exactly two are eligible: `bodybuilder`, `fusion` and `pareto-code` declare no
`supported_parameters` at all, so they cannot serve a schema-constrained writer
pass. **The standing objection is recorded in the code and was overruled, not
forgotten:** a router picks by criteria this pack does not control, so the writer
model can differ between episodes -- which the 2026-08-04 "story quality is DONE"
directive had settled. `meta["resolved_models"]` makes every run auditable after
the fact, which is what makes it survivable.

* **Prefer `~family-latest` pointers.** Carry a concrete id only where a version
  genuinely matters, and DATE it.
* **NEVER carry a `:free` / promo slug.** A `:free` id is a price promise baked
  into an identifier, and promises expire while identifiers do not. That is what
  killed `tencent/hy3:free`.
* **Do NOT add an auto-router.** `openrouter/auto` and `auto-beta` are real and
  listed, but `auto-beta` routes by what the community spent most on over a
  trailing week -- a config that resolves differently week to week undercuts the
  resolved-model replay stamp. Good for exploration, wrong for reproducible
  receipts.
* **The technical default stays `deepseek/deepseek-v4-pro`.** The only DeepSeek
  pointer is the FLASH tier, so aliasing it is a capability DROP, not a swap.

**The honest limit, do not report it later as a failure:** curation removes our
OFFER. It cannot remove a dead slug from a STALE PER-MACHINE cache under
`OTR_OPENROUTER_FULL_CATALOG=1` -- `models/openrouter_models.json` is untracked
and gitignored. Running the refresh script clears it.

### BIBLE PROMOTION -- standing contract + what is pending

**A window MAY promote a SINGLE genuinely-uncovered entry** directly under the
Three-File Contract, because `otr_coverage_index.yaml` makes checking coverage
cheap. **The BULK fan-out over the backlog is still the operator's** (waiting-on-
operator row 6). Operator's reason for the change, 2026-08-07: *"we keep hitting
the same bugs so we need to update the bible and test regularly."*

Live at survival-guide `656c36e`: **263 entries, index 371 rows**, 20 passed /
24 skipped / 3 xfailed. Recent promotions: **BUG-12.87** ("a gate reports success
from its own error path" -- the tell being that another component has quietly
reimplemented the same check with a comment saying the shared one cannot be
trusted) and **BUG-12.86** (a receipt keyed on a producer string the producer
never emits, so it reads empty forever).

**Pending operator decision (waiting-on-operator row 7):** `656c36e` also swept
in 8 pre-existing uncommitted `test_otr_*` guards, staged by a whole-file
`git add` without checking `git status` in that repo first. Nothing is broken and
the suite is green, but the commit message does not describe them.

### STILL OPEN, SMALL, UNSCHEDULED

0. **THREE FINDINGS FROM THE UPSCALE-FINGERPRINT CHUNK (2026-08-08), all
   deliberately NOT swept into `088dabc8` / `7c26ec86`:**
   * ~~**`scripts/validate_canonical_workflow.py` CAN EXIT 0 WITHOUT
     VALIDATING.**~~ **FIXED `5fdf93f1` (2026-08-09), and it was WORSE than
     recorded here.** The contract had never run ON ANY BOX: the package dir
     is `ComfyUI-OldTimeRadio` (HYPHEN) and ComfyUI loads it BY PATH, so the
     old `importlib.import_module(name.replace("-","_"))` was PERMANENTLY
     unsatisfiable -- it always took the skip, always returned `[]`, always
     printed OK. The item-8 receipt "clean (23 nodes, 56 links)" was the skip
     path. Fixed at root with `spec_from_file_location` on `__init__.py` (the
     technique `otr_macbeth_probe.py` already used to route AROUND this
     script), and an unrunnable contract is now a PROBLEM. 5 tests in
     `tests/test_validate_canonical_workflow_fails_closed.py`, including a
     control that fail-closed has not become fail-always.
     **Carry-forward lesson:** the workaround shipped and the shared gate
     stayed broken for weeks. When you route around a gate, fix the gate.
   * **`otr_upscaled_dir()` (`nodes/_otr_paths.py:383`) is DEAD.** Its only
     producer was `OTR_RTXUpscale`; references now are the definition, the
     `__all__` entry, and the output-tree contract test that iterates every
     helper. Documented in place in `7c26ec86`. Removing a public path
     helper is an operator call, not a comments-only change -- item 10
     (lean-mean/dead-code) material.
   * **`SpandrelEsrgan._resolve_model` robustness pair** (Sonnet QA on
     `088dabc8`, deferred on purpose): an unreadable NON-WINNING candidate
     aborts the whole search instead of skipping to the next, and the
     winning file is stat'ed twice with a TOCTOU window that can turn
     ordinary absence into a bare NaN. **This is the same absence-vs-fault
     logic already reworked twice, so per the two-strikes rule the third
     attempt MUST get a full kibitz panel BEFORE any code.**
   **Banked so it is never re-derived:** `Path.is_file()` on Python 3.12.11
   does NOT swallow every `OSError` -- pathlib filters through
   `_ignore_error()`, which whitelists only ENOENT/ENOTDIR/EBADF/ELOOP and
   RE-RAISES `PermissionError` and `OSError(EIO)`. Measured on the venv
   interpreter and independently confirmed against its `pathlib.py` source.
   Both r4 review lanes asserted the opposite and a compensating
   classification pass was written, proven unreachable, and deleted.

1. **`load_all_ledger_fixtures` / `_looks_like_l3_ledger` (`tests/_helpers.py:26-118`)
   is DEAD test infrastructure** -- no callers anywhere, and none of the 5 JSON
   fixtures match its `l3-` prefix filter. Delete-or-revive, separate small item.
2. **The parked worktree rip must be RE-GROUNDED.**
   `.claude/worktrees/awesome-brahmagupta-a509b4` holds the uncommitted removal
   of `news_coda_spoken_reduction` / `finalize_news_coda_surface`, and its target
   lines now live INSIDE `_compose_and_stamp_announcer_close`. **ARC** if reopened.
3. **`BUG_BIBLE.yaml` does not `yaml.safe_load`** -- `ScannerError` at line 834
   col 217, an unquoted inline JSON fragment whose colon breaks the scan.
   Pre-existing at survival-guide `3759ae5`; the contract test counts `^- id:`
   with a regex, so nothing notices. The README calls it machine-readable.
   Separate repo, separate item.
4. **`docs/known-failures.md` + the conftest KNOWN-FAIL-GUARD** exit **2**, not
   1, on a new failure, and swallow the traceback in `-q` runs. Not a defect,
   but it has cost mis-diagnoses -- know it before reading a red line.
5. **`reasoning.default_enabled` is captured and never read.**
   `_otr_openrouter_backend.py:918` slims it into every cached catalog row, but
   only `mandatory` and `supported_efforts` are ever consulted (`:324`, `:330`).
   Found 2026-08-07 while grounding the slug curation; PRE-EXISTING, not caused
   there. Sibling to the freshly promoted BUG-12.86 class (a field that reads
   as though it informs a decision and informs nothing). Per the admission rule
   a static observation does NOT create a PBUG -- delete-or-consume, small item.
6-STATUS. **PARTLY CLOSED 2026-08-09 (`ab76f6bc`).** `nodes/_otr_slug_provenance.py`
   + `tests/test_slug_provenance.py` now require an ENTRY for every shipped
   concrete slug in the comfy / google_api / elevenlabs / comfy_image /
   google_image lanes -- either a verified date or an explicit `UNVERIFIED`
   marker naming the authority that could settle it. **Dates were NOT invented:**
   21 of 35 are honestly UNVERIFIED and the suite prints that backlog out loud.
   The 6 comfy dates are carried forward from the 2026-08-07 OpenRouter check,
   recorded as a SIGNAL (OpenRouter presence is not proof Comfy serves a slug).
   Also `tests/test_saved_workflow_model_values_resolve.py`: a saved graph's
   model widgets must exist in the LIVE dropdown -- the class-level fix for the
   `value_not_in_list` that made `otr_story_only.json` unrunnable, which the
   slugfest audit had PREDICTED two days earlier.
   **STILL OPEN from the audit, deliberately not in that commit:**
   * **VIDEO lanes** (`eng_cloud_video`, `eng_google_omni_video`,
     `eng_google_veo_video`) -- excluded because a concurrent window owns those
     files and a guard would red-flag its in-flight edits. Add them once it lands.
   * **LOCAL checkpoint filenames** (`z_image_turbo_bf16.safetensors`,
     `flux-2-klein-4b-Q4_K_M.gguf`, `stable_audio_3_small_music.safetensors`,
     ...) -- same staleness class, authority is local disk.
   * **FOUR `preview`-marked slugs** (`gemini-3.1-flash-tts-preview`,
     `gemini-2.5-flash-preview-tts`, `gemini-2.5-pro-preview-tts`,
     `lyria-3-clip-preview`). "preview" in an identifier is a LIFECYCLE PROMISE
     baked into an id -- the same class as `:free`, which is what killed
     `tencent/hy3:free`. Decide whether to ban, warn, or date them.
   Full inventory: `kibitz-runs/2026-08-07-slugfest/antigravity_slug_audit.md`
   (GITIGNORED -- read it there, it is not in git).

6. **THE OTHER PINNED SLUG LISTS -- same defect class, NOT yet curated
   (operator asked 2026-08-07: "do we need to review the comfy llm engine slugs
   too?"). Answer: yes, and it is wider than that one list.** Chunk A dated and
   guarded the OpenRouter ids only. The same "concrete version pin, no date,
   nothing able to notice it went stale" pattern is live in at least:
   `_otr_comfy_backend.COMFY_LLM_MODELS` (6 slugs, curated 2026-07-04, **no disk
   cache and no refresh script -- the catalog IS a constant**);
   `_otr_google_api/models.py` (`GOOGLE_API_LEGACY_/STABLE_/STATIC_TEXT_MODELS`);
   and the audio/image engine lists (`eng_cloud_elevenlabs._SUPPORTED_MODELS`,
   `eng_google_tts`, `eng_google_lyria`, `eng_cloud_image._NANO_MODELS` /
   `_SEEDREAM_MODELS` / `_KREA_MODELS` / `_LUMA_PHOTON_MODELS`,
   `eng_google_image.SUPPORTED_MODELS`).
   **SIGNAL-CHECKED 2026-08-07 -- and the news is better than first stated:
   ALL SIX `COMFY_LLM_MODELS` slugs are LIVE.** Checked against the live
   OpenRouter catalog: `google/gemini-3.5-flash`, `deepseek/deepseek-v3.2`,
   `mistralai/mistral-large-2512`, `x-ai/grok-4.20`, `openai/gpt-5.5`,
   `anthropic/claude-opus-4.7` all present. **No dead ids** -- unlike the
   OpenRouter dropdown, which carried a genuinely dead one. This list is
   VERSION-BEHIND, not broken, and that distinction should not be lost again.
   *Authority caveat:* Comfy Cloud's partner catalog is what decides whether
   Comfy SERVES a slug; OpenRouter presence is a signal, not proof. An
   authoritative pass still owes a check against Comfy's own catalog.
   ~~**The stale PREMISE is the real defect, not the version lag.**~~
   **RESOLVED 2026-08-09 (`5fdf93f1`) -- AND THIS PLAN'S READING WAS THE WRONG
   ONE. Do not re-open it; the comment is now load-bearing documentation.**
   This file had recorded the `_otr_comfy_backend.py:84-91` block ("Reasoning
   models ... are DELIBERATELY EXCLUDED") as STALE, reasoning that essentially
   every frontier SKU now advertises reasoning -- including `deepseek-v3.2` and
   `x-ai/grok-4.20` sitting inside that very list -- so the comment described an
   exclusion the list no longer performed. **That conflates two different
   things.** The rule excludes reasoning-BRANDED SKUs (the `-pro` / `-thinking` /
   `sonar-reasoning` tiers) because they EMPIRICALLY BREAK STRUCTURED JSON; it
   never claimed the survivors cannot reason. The list still performs exactly
   that, which is why `deepseek-v3.2` sits there labelled NON-reasoning while
   `deepseek-*-pro` does not.
   **The re-confirmation was accidental, and therefore worth trusting:** an
   unrelated OpenRouter roundtable run on 2026-08-09 put `deepseek/deepseek-v4-pro`
   on a 3-model panel and it returned EMPTY CONTENT with `finish_reason=length`,
   having spent its whole token budget on hidden reasoning -- the exact failure
   the block was written about, on the exact SKU pattern it names, thirteen
   months of model churn later.
   **Carry-forward lesson:** "this comment sounds dated" is not evidence. The
   cheap half of this item was never the comment; re-dating the slugs is the
   whole remaining mechanical half.
   **Why it is NOT a copy of chunk A:** each lane has a DIFFERENT source of
   truth. Comfy Credits slugs are Comfy Cloud's partner catalog, Google's are
   Google's, ElevenLabs' are theirs -- none is verifiable against OpenRouter's
   `/api/v1/models`, so each needs its own verification path. The reusable part
   is the POLICY and the guard shape (`OPENROUTER_VERIFIED_ON_BY_ID` +
   `tests/test_openrouter_slug_curation.py`): every concrete id carries a
   verified-on date, and a test fails when one is added undated.
7. **v2.1 CANDIDATE (operator, 2026-08-07): a low-footprint LOCAL model added
   to the mix.** Operator: *"if there's a cool local model we can add to the mix
   without much footprint change ... but that's later, maybe 2.1."* The two
   ideas raised were a model steeped in public-domain OTR diction, and one
   tuned for Early Modern English on the Shakespeare lane. **Grounding worth
   keeping so it is not re-derived:** the Shakespeare lane would benefit LEAST
   -- THE ADAPTATION DESIGN makes it a VERBATIM compiler and the ownership table
   rules `exchange_compose` NOT RUN there ("there is no dialogue to author"), so
   a period-tuned model would only touch announcer/bridge/stage-setting prose,
   never a line Shakespeare wrote. The `original` and `media_archive` lanes DO
   generate their text and are where such a model would actually land. Note the
   collision to settle first: this is adjacent to the 2026-08-04 "story quality
   is done, stop chasing it" directive, so reopening it is a conscious operator
   call, not a drift. Offline-first also means a FINE-TUNE, not a download.

## ON DECK -- WHAT REMAINS OF CONTINUITY CORRECTNESS

### 0. VIDEO MATRIX PATTERN -- FOUR ROUNDS, FOUR NOs, DID NOT CONVERGE

**Do not build from the current plan.** Arc r1-r4 complete (r4 single-lane --
Antigravity out on provider quota). Judgments:
`kibitz-runs/2026-08-06-2026-08-06-matrix-pattern/` (**LOCAL ONLY, gitignored**).
Spec: `docs/2026-08-06-SPEC-subsystem-matrix-pattern.md` (`11e893f6`), superseded
in detail by the r3 `final.md` and the r4 judgment.

**Why it did not converge, and it is not a design problem:** most of what remains
is ABSENT HUMAN-OWNED DATA. Someone must WRITE a one-line `doc_purpose` for each
of ~32 registered engines and DECIDE the total `family -> display_group`
taxonomy. No further review round produces that content. The frozen
`(engine_id, field)` grandfather set is likewise un-enumerated, so two builders
could grandfather different holes and both pass.

**What survived all four rounds (this part IS sound):** templated prose fragments
whose numbers are placeholders resolved from the live registry; extending
`AdapterDescriptor` rather than minting a second descriptor; a generator-side
validator that keeps the registry pydantic-free; init order import -> audit ->
rows; the two-unit cost split (VRAM MB vs provider USD); precedence stated as a
HUMAN rule; and a mandatory `python tools/engine_matrix.py` regeneration step
because `--check` never rewrites the doc.

**Corrections banked:** `str.format_map` CANNOT resolve dotted flat keys
(`KeyError: 'reference'`, proven) so the placeholder grammar must be explicit;
`provider_side` migration is **behaviour-affecting, not STATIC** -- it changes
dispatch at `render_driver.py:1653-1668` and there is a THIRD classifier at
`scripts/otr_w45_campaign.py:82-108,120-132` where a provider id without a
`cloud_`/`google_` prefix would enter the LOCAL campaign.

**Next step is NOT another round.** Author the content, enumerate the grandfather
set, plan the `provider_side` migration with parity tests, write the `CLAUDE.md`
rule text -- then re-enter at **r3**, per the standing rule that a plan-level
gap drops back rather than being patched from inside r4.

### 0-QUINQUE. MINIMAX H3 -- A SPRINT SERIES ON THE VIDEO PATHS (operator, 2026-08-09)

**THE RULING IS IN, AND IT DISSOLVES THE OLD QUESTION.** This section used to
say the next step was an operator ruling on "does H3 belong in the video
dropdown given the 4 s floor vs the sub-4 s beats". That framing is RETIRED.
The operator's 2026-08-09 direction: H3 is **"a series of sprints all to refine
the video paths"** -- scope TBA.

**What that changes for a window picking this up:**
* It is NOT a yes/no dropdown admission any more, so do not go looking for a
  verdict to record. There is nothing blocked on the operator here.
* The unit of work is a SPRINT against the video paths, not a one-shot chunk.
  Expect several, each with its own kibitz gate, each landing green and pushed
  on its own.
* The 4 s floor is now an INPUT to that refinement -- a constraint the video
  paths have to accommodate or explicitly route around -- rather than a
  disqualifier that settles admission.

**Scope is TBA and that is deliberate.** Do NOT invent the sprint list. When the
operator names the first sprint, write it into THE QUEUE at the top of this file
as its own row, and leave this section as the standing context.

**Grounding that survives the reframing:**
* Problem statement `docs/2026-08-03-PROBLEM-STATEMENT-minimax-h3.md` is
  UNTRACKED and another window's working file -- never stage, edit or delete it
  from a different window. Read it; do not touch it.
* The matrix-pattern spec already names MiniMax as a churn driver
  (`docs/2026-08-06-SPEC-subsystem-matrix-pattern.md` section 5), so a video-path
  sprint will likely collide with the un-converged matrix work (section 0). Read
  that section's "what survived all four rounds" before designing anything.
* The recipes are NOT on the table (standing directive). A video-path sprint
  refines PATHS -- routing, canvas negotiation, admission, extension -- never
  the shipped render recipe.

### 1. The reference A/B still owes a verdict (the one real open item)

The reference path is PROVEN WIRED, not proven EFFECTIVE. Live leg
`signal_lost_lute_strings_fools_tongue_20260805_021040` shows three `scene_character`
rows stamped `portrait_anchor_mode='reference_latent'`, and the two rows sharing char
`c03` share one anchor -- so the engine declared the capability, the portrait row
resolved, the file was on disk, and the anchor entered the cache key.

**What nobody has answered: does `z_image_turbo_nvfp4` actually ATTEND to the prepended
reference, or does it accept and ignore it?** The architecture takes it with no missing
weights (header probe: `cap_pad_token` and `x_pad_token` present, `siglip_embedder`
absent), but graph shape cannot prove three faces became one. That needs:
- a control arm with **`OTR_PORTRAIT_REFERENCE=0`**, on its own fresh server boot --
  env vars cannot reach a resident ComfyUI process, and `OTRImageGenDispatcher` has no
  `IS_CHANGED` to notice the flag, so the arms MUST NOT share a boot;
- the control asserts `portrait_anchor_mode == 'seed'`, NOT `''`. The seed pin is still
  enabled in that arm. Only setting BOTH `OTR_PORTRAIT_REFERENCE=0` and
  `OTR_PORTRAIT_IDENTITY_SEED=0` yields `''`;
- an operator eyeball on the two arms, which is the actual verdict.

If the reference turns out to be a no-op, **Track 2 Step 8 (flux2_klein)** is the built
answer -- klein is genuinely reference-trained and its weights are on disk. It is
deliberately NOT built yet. Switching to it is a Director widget pick, not code.

### 2-PRE. OPERATOR CALLS ALREADY MADE -- do not re-open, do not re-panel

**BANANA ROUTE: VISUALS ONLY. The spoken script is NOT touched
(operator ruling 2026-08-06).** Operator, asked whether the filter should reach
spoken lines as well as image prompts: *"No. Just visuals. I do not want people
discussing the Cavendish versus the other variety."*

So the substitution happens on the STILL/VIDEO PROMPT and nowhere else. The
announcer still says "he drew his revolver" over a shot of a man holding a
banana -- which the problem statement flagged as either the joke or the thing
that breaks it. It is the joke. **The dialogue ledger, the writer, and the
adaptation lanes are all out of scope**, which also keeps this clear of the
closed story-quality directive and of the fidelity lanes' invents-nothing rule
at the TEXT level.

This closes the second half of section 7 of
`docs/2026-08-06-PROBLEM-STATEMENT-banana-route.md` (committed `9c686886`).

**THE DEFAULT AND THE REACH ARE ALSO RULED NOW (2026-08-06, `ec9da848`) --
that question is CLOSED, do not reopen it.** Global default **ON**, with
`shakespeare` + `public_domain` defaulting **OFF** via the copied `_LEMMY`
exclusion idiom, plus `OTR_BANANA_INCLUDE_FIDELITY_BANKS` as the operator's
force-on override. **NO node widget and NO `workflows/otr_canonical.json`
change.** So `Is this a dagger which I see before me` stays a dagger on the
fidelity lanes unless the operator flips the override. Two env switches
(`OTR_BANANA_STILLS`, `OTR_BANANA_VIDEO`), one per funnel. The whole contract is
`docs/2026-08-06-BUILD-SPEC-banana-route.md` at `ec9da848`; SHIPPED -- see
section 0-QUATER above (`bc8a1bde`).

### 2. Operator calls nobody can make for you

- **ARIEL and PUCK.** The curated supplement ships 10 entries and deliberately omits
  these two: Folger's stage directions use "he" for both, but neither has a roster fact
  and both are editorial. They stay on the roll, so the corpus gate asserts 40 of 42.
  Say the word and they become 42.
- **Tier floor 2 or 3.** Shipped at 2: it removes every 100% voice pin while 5 of 24
  (engine x gender x timbre) combos still honour the requested timbre. Floor 3 also
  removes them but leaves only 2 of 24 honouring timbre -- it buys spread by deleting the
  dimension. `OTR_CAST_MIN_TIER_POOL=3` makes it a one-leg A/B, and the floor is folded
  into the cast seed so the two settings can never both claim policy '3'.
- **`num_characters` is still 2.** Every published adaptation ran 2, so a 7-speaker scene
  loses five people. Correct gender for two survivors is still a truncated scene. This
  collides with the count-match invariant at `OTR_LedgerScriptWriter.py:4119` and is its
  own piece of work, not a tail of this one.

### 3. Standing facts worth not re-deriving

`slot.gender` is NOT a voice field. It feeds the description LLM
(`_otr_casting.py:777`), the outline prompt (`OTR_LedgerScriptWriter.py:4144`), the
dialogue cast block (`_otr_line_composer.py:446`) and the image prompt's gender anchor
(`otr_meta_brief_image_prompt.py:78-90`). **The gender fix therefore changes scripts and
portraits.** That is a downstream consequence of a correctness fix, not a violation of
the closed story-quality directive -- exactly as "Malvolio speaks with a woman's voice"
is a bug while rewriting his dialogue is not.

Do NOT feed pinned genders into `prior_genders`, and do NOT re-call
`_plan_gender_distribution` with a reduced count. Measured: `(1, ['male'])` returns
female 400/400, and the shuffle's stream consumption varies with count (getrandbits
0, 0, 3, 3, 9, 11 for counts 0..5). The shipped design overrides in place and leaves the
allocator untouched.

**Source-grounding sprint, the one piece left:** chunk 3b-ii -- the supply line
that feeds grounding into the writer -- is BUILT-BUT-UNWIRED and PARKED under the
story-quality directive. The delivery mechanism exists and nothing calls it. A
contributor may pick it up; chunk detail under THE CODING SPRINT item 1.

## THE CODING SPRINT (operator directive 2026-08-04; re-sized by the r1-r4 arc)

Item 1 is the structural work and consumes most of a session; items 2-3 are
small and share one campaign. **Items 8 AND 9 are DONE**
-- item 9 SHIPPED 2026-08-06 as `e499b7fc`; detail in `docs/HANDOFF_LOG.md`.** The live open work is sections 0
(video matrix pattern, did NOT converge), 0-BIS (no-mirror, CODE-READY),
0-QUATER's deferred shield-scoping chunk (own kibitz arc), and the 0-QUINQUE
MiniMax ruling.
Work by priority, not by number -- the numbering is historical.

**RENDERS HAVE RESUMED** (2026-08-05). The 08-04 "no render runs this session"
line is spent -- it governed that session only, and the 08-05 handoff opens on a
live-proof obligation. Reset per `CLAUDE.md` section 4 before any leg: selective
CIM kill by CommandLine, never a blanket python kill (it severs the MCP tooling).

Everything below was verified against the real files on 2026-08-04, is
non-GPU, and is provable by the suite alone. Work them in order; each ends
green and pushed on its own.

### HARD GATE FOR THIS SPRINT -- NO CODE SHIPS WITHOUT A FULL KIBITZ (operator directive 2026-08-04)

Operator: "for the next coding sprint be sure any coding work has a full
`/kibitz-plugin:kibitz` review." **This is a gate, not a suggestion.** Now also
written into `CLAUDE.md` above the two-strikes rule, because that older rule says
a first-try root fix does not need a panel and would otherwise win the conflict.

* **FULL = the default four-round arc.** r1 arc -> r2 coding -> r3 wiring -> r4
  convergence; 8 external calls (two reviewers x four rounds). Not a scoped tail,
  not one round, not a continuation receipt. If a round genuinely cannot run, say
  which one and why -- a partial campaign is NEVER reported as a full arc.
* **Invoke `kibitz-plugin:kibitz` by name.** `anthropic-skills:kibitz` is the
  older duplicate; it is not what was asked for.
* **Panel = Codex + Antigravity.** Claude drives from Cowork, so the driver's own
  family is excluded -- do not launch a second `claude -p` lane against yourself.
* **Use the ComfyUI profile.** `.kibitz/comfyui.local.md` is already in the repo
  (written 2026-07-11). Regenerate via `kibitz/scripts/comfyui_profile.py` if the
  tree has moved past it. Node contract, widget/`widgets_values` drift and
  `IS_CHANGED` are exactly the defect classes this sprint can produce.
* **Anchor first, judge last.** Write `driver_anchor.md` from the REAL Windows
  files (Desktop Commander -- never the lagging Linux mount) BEFORE the fan-out,
  then ground every panel claim and discard the misreads. A panel run on a false
  premise is worse than no panel.
* **When it binds:** the campaign runs on the item's plan BEFORE the code, and its
  MUST-FIX list is answered BEFORE the commit. The panel does not replace the
  suite and the suite does not replace the panel -- both, every item.
* **Batching (amended by the r1 panel, 2026-08-04):** item 1 gets its OWN
  campaign covering its 2-3 chunks -- it is the structural change and precisely
  what a panel is for. Items 2 + 3 SHARE ONE campaign over their combined diff:
  every item still passes through a full arc (the directive is satisfied), but a
  fixture-isolation chunk does not buy 8 external calls of wall-clock on its own.
  This batching is the gate-compliant reading, not an exemption.
* **Artifacts:** `kibitz-runs/<date>-<topic>/r<N>/` with the per-round
  `input.md`, `driver_anchor.md`, reviewer files, `judgment.md`, `final.md`.
  UTF-8, no BOM. (r4 fixed a conflict here -- this matches the standing
  re-ground gate's location and the tool's actual output; the earlier
  `docs/<date>-<topic>/kibitz/` line is withdrawn.)

Kibitz is CPU/API only and costs the operator nothing, so it never competes with
work for the GPU -- but it does cost wall-clock. Four campaigns is real time on
top of the ~2h40m of coding below. If the session runs short, finish FEWER items
fully reviewed rather than more items unreviewed.

### 1. THE PUBLIC_DOMAIN LANE IS TOLD TO CARRY WORDS IT IS NEVER SHOWN (the session's main work -- r1 panel re-scoped 2026-08-04)

The headline defect, and the one that manufactured "Arkham, Massachusetts" over
H. G. Wells. The pack orders the model to carry the author's language:

* `nodes/story_packs/public_domain/faithful_radio_adaptation.json:13`
  (`exchange_system`) -- "Where the source gives these characters words, CARRY
  THEM. Keep their diction, their rhythm, their argument."

And `nodes/_otr_compose_exchange.py` (994 lines) has **ZERO** references to
`source_text`, `full_text`, `source_meta` or `excerpt` -- verified by grep.
**The instruction is bound to an absent document.** A model told to carry words
it cannot see will invent words and believe it complied.

**SCOPE RULING (r1, grounded against `docs/2026-08-03-fidelity-pass-ownership.md`
line 25): this item is PUBLIC_DOMAIN ONLY.** The ownership table rules
`exchange_compose` **NOT RUN** on the Shakespeare verbatim lane ("It exists to
author dialogue. There is no dialogue to author."), so enhancing the composer
for Shakespeare invests in a pass the verbatim executor removes. Shakespeare
gets exactly ONE change this sprint: its dangling comma (item 3). The keystone
"compile source speech, do not generate it" (THE ADAPTATION DESIGN) binds the
VERBATIM lane; `public_domain` is the operator-ruled FUZZY PROSE lane, where
grounding the generative composer is the correct move, not a contradiction.

Three legs, ALL required -- the panel killed the raw-injection shape:

**(a) A BOUNDED source window over the COMPLETE canonical body -- never the
payload's `full_text`, which is itself truncated.** r2 correction of this
plan's own premise: `canonicalize_public_domain_text(..., max_chars=12000)`
(`_otr_public_domain_sources.py:337-343`) truncates at 12,000 CHARS, and
`payload_from_manifest_unit` stores THAT as `full_text` -- while the corpus
runs **916 words (`cradle_protocol`) to 25,200 words (`beckoning_fair_one`)**
across 65 units. So "the material already arrives, it needs passing" is false
for large sources: the payload carries a prefix. The selector reads the
complete canonical body from the SOURCE layer, separated from the interpreter
excerpt. Hash discipline: exactly ONE of 65 units ships a provenance sidecar
(`time_machine__arrival.provenance.json`), and its `body_sha256` covers
normalized RAW bytes, not the canonicalized body -- two NON-interchangeable
fields. Derive a `canonical_body_sha256` at fetch/selection time, bind
selection + receipts to it, and do NOT call it authenticated provenance. Do
NOT migrate the 65 closed manifests for it (`_SOURCE_KEYS`/`_UNIT_KEYS` closed
at `:48-63`); carry it in `source_meta` and snapshots. Coordinate system (r3):
refactor `canonicalize_public_domain_text` into an UNCAPPED normalization
owner plus a separate 12,000-char legacy payload projection; spans are
half-open Unicode char offsets (`start_char`/`end_char`) into the uncapped
string; `canonical_body_sha256 = sha256(canonical_body.encode("utf-8"))`;
stamp normalization + selector versions. Transport (r3): `SourceFetchResult`
exposes only payload/source_meta/source_rights and `_resolve_inputs` collapses
to a three-tuple, and the snapshot envelope is the SEVEN-KEY payload
(`_otr_source_snapshot.py:48-50`) whose `full_text` is the truncated prefix --
so extend the PUBLIC-DOMAIN snapshot with the CANONICAL BODY as the SOLE
replay authority (r4 cut the "or exact selected text" alternative -- selected
text cannot recreate pre-outline grounding or select windows for a NEWLY
generated outline), under a versioned body/hash/normalization contract. A
legacy seven-key snapshot FAILS with a typed grounding-version error -- but
ONLY when the snapshot's bank is `public_domain`/adaptation (r4, both lanes
converged): the seven-key envelope is the UNIVERSAL loader, and an
unconditional rejection would break every other bank's existing snapshots and
bake-off replays. Keep the full document OUT of meta/ledger (`source_meta` is
copied into durable metadata at `:3548`). Budget: capacity
is EVERY backend, not GGUF alone -- the fitting seam
(`_otr_generation_budget.py:132`) spans GGUF (`estimate_prompt_tokens`,
estimator, `_otr_gguf_backend.py:1264-1273`), OpenRouter, Google and Comfy --
so select the window against the COMPLETE assembled message (system seam,
cast, prior lines, contracts, source block, output reservation), reserve
conservatively with stated margin, and refuse `prompt_no_room`
deterministically BEFORE provider execution; receipts distinguish
estimated_prompt_tokens / requested_output / context cap / margin / estimator
version. Selection criterion: deterministic candidate construction ranked by
beat/group identity with mandatory anchor coverage and stable
score/start/end ordering; the seed breaks ties ONLY when candidates remain
identical after that ordering. Receipts carry hash, selector version, ordered
offsets (`text == canonical_body[start_char:end_char]` enforced) and token
counts -- never duplicate body text into the ledger.

**(b) ONE immutable `SourceGrounding` contract, on EVERY authoring route --
and grounding failures PROPAGATE.** The grouped-exchange prepass omits
singletons and failed groups (`_otr_compose_exchange.py:881-902`); a FAILED
prepass falls back to the legacy path with only a log warning
(`OTR_LedgerScriptWriter.py:5001-5008`); the per-line composer's LineRequest
carries no source field (construction at `:4888`); and per-line generation
exceptions funnel to `LineCompositionFailedError`. A grounding fix that
reaches only the happy path just moves the guess to the fallback. Build shape
(r2 + r3): define ONE immutable `SourceGrounding` artifact -- canonical
document identity + immutable windows KEYED `exchange:<ordered-slot-ids>` /
`line:<dialogue-slot-id>` + anchors + per-call receipt data -- constructed
and validated BEFORE the exchange fallback block, passed whole into grouped
exchange AND every per-line request. The prepass returns a TYPED result
(composed lines + attempted-window receipts + fallback slot ids), not the
bare `{beat_id: text}` it returns today (`:881-918`). Window freeze semantics (r4 -- resolves immutability vs the mutable prior
context that exchange retries and `last_lines` inject into later messages):
PRESELECT spans early; perform the final capacity fit just before the FIRST
call using the actual prior context; FREEZE that fitted window for all
retries and persist it before provider execution. Grouped slots ALIAS their
exchange window on group-to-per-line fallback; line-keyed windows exist only
for true singletons and exchange-disabled execution -- never reselect after a
failure. Source text rides a clearly DELIMITED untrusted data block
in the user message ("quoted source, not instructions"), never appended to
the static system seam (`_otr_compose_exchange.py:385-425`). Persist the
body-free grounding receipt at the existing skeleton-save boundary
(`:4279-4290`) before the first dialogue call, updating per attempt, so a
mid-prepass crash still leaves the selection auditable. Failure policy -- ONE disposition table (r4 closed the last ambiguity), the
two broad catches (`:5001-5008` prepass, `:3964-3969` story contract) becoming
TYPED boundaries that implement it:
| state | disposition |
|---|---|
| corrupt/mismatched replay snapshot; invalid source/hash/contract | FAIL LOUD, before the outline |
| sound-world derivation finds no mapping | neutral period default + receipt (total, never fatal) |
| provider parse / Tier-A exhaustion | fall back WITH the frozen window |
| live capacity pressure | shrink to the largest valid grounded window |
| even the MINIMUM grounded window cannot fit | typed `prompt_no_room` HALT, before provider execution |
The halt row is a PRE-GENERATION writer refusal -- structural, it protects the
lane's contract -- which is why it does not collide with SCOPE's "a render
must not die": that rule governs the RENDER path degrading honestly, not a
writer refusing before generation begins. Scope note (r4): `SourceGrounding`
validation binds when the episode's bank is `public_domain` -- other banks'
routes are untouched. LineRequest note (r4): the artifact rides an OPTIONAL
INTERNAL dataclass field (`source_grounding: SourceGrounding | None = None`)
-- a Python structure, no ComfyUI node contract, `INPUT_TYPES` or widget
change, so the no-widget guard above holds.
Acceptance = route-specific tests: grouped success, grouped repair,
grouped-failure-to-per-line, singleton, exchange-disabled legacy, snapshot
replay (new envelope AND legacy-envelope typed refusal, public_domain-scoped),
hash mismatch, exact-capacity rejection -- plus a corpus-wide property test
over all 65 units proving normalization idempotence, canonical-hash stability
and `text == body[start_char:end_char]` for every emitted span (r4). Version
discipline (r4): the existing constants are `PROMPT_VERSION =
"public_domain_interpreter_v2"` / `SCHEMA_VERSION = "public_domain_briefs_v1"`
(`_otr_public_domain_sources.py:36-38`); name and bump every changed one, and
give SourceDocument / SourceOverview / SourceGrounding / normalization /
selector / snapshot their own explicit versions.

**(c) World anchors, DERIVED FIRST -- and the sound world gets ONE owner that
feeds every surface.** Prefer deriving a typed grounding sidecar from EXISTING
metadata + the selected spans. New manifest fields are a LAST resort:
`_SOURCE_KEYS`/`_UNIT_KEYS` are closed frozensets
(`_otr_public_domain_sources.py:48-63`, same for `_SCENE_KEYS`), so new fields
mean a schema version + migration across all 65 units. AND the competing frame
must actually be disabled, not outvoted: the adaptation `sound_world` is a
content-blind draw (`OTR_LedgerScriptWriter.py:3962`, palettes at
`_otr_style_catalog.py:442-463` -- grate/mantel/teacup over whatever source
rolled it). r2 sharpened the shape: the catalog renders the drawn sound world
into `contract.grammar` SEPARATELY from the `contract.sound_world` stamp and
the canon derivation, so a stamp-only fix leaves the prompt grammar still
carrying the contradictory palette. ONE source-aware derivation function must
feed the stamp, the grammar and canon for `style_pool_class == "adaptation"`
(arc_shape gate at `:4325` is the shipped precedent), with an explicit neutral
period default when no mapping exists -- and it runs BEFORE the grammar is
built (or the grammar re-renders from the final contract), or the prompt
grammar keeps the contradictory palette while the stamp looks fixed (r3, both
lanes independently). DECIDE whether derivation failure is fatal: today's
broad catch silently disables the whole story contract. Reconcile with the
EXISTING anchors owner: `meta["specificity_anchors"]`
(`OTR_LedgerScriptWriter.py:4259-4266`) already derives and injects an anchor
projection -- the new source anchors REPLACE it or deterministically merge
into it, never run beside it as a second independent voice. Do NOT delete the
adaptation styles -- operator-authored 2026-07-14; fix the DRAW and the
plumbing, not the styles.

**Two receipts, named now so neither is overstated later:**
* `code-complete + suite-green` -- the most a session without the live leg can claim.
* `production-qualified` -- only after a canonical `public_domain` leg passes a
  rubric: no unsupported foreign place/character/object; the source's setting
  and principal event retained; provenance receipt complete; `obs_publish OK`;
  asset on disk.

**Two rules from the 08-03 craft brief, both hard-won, both easy to violate:**
1. **Never name the feared failure.** Writing "no Arkham" into a prompt IMPLANTS
   Arkham. Forbid by CATEGORY, never by example.
2. **Every fidelity instruction must be PAIRED with the material it binds to.**
   An unpaired "carry the words" is the bug, not the fix.

**Size honesty (r1) and CHUNK ORDER (r3 -- the naive order was CYCLIC):** this
is THE SESSION, not 90 minutes. r3 caught a dependency cycle in the obvious
build order: the sound world feeds `contract.grammar`, the grammar is consumed
by the OUTLINE (`OTR_LedgerScriptWriter.py:3948-3963` -> `:4129`), and beats do
not exist until the outline returns -- so a sound world derived from
beat-keyed windows is impossible. Build in THIS order, one green pushed chunk
each:
**CHUNKS 1 AND 2 ARE DONE AND PROVEN ON RENDERS. Chunk 3 (the grounding supply line) is PARKED under the story-quality directive -- the Source-grounding note in section 3 above is authoritative; a contributor may pick it up.**

**Carried into chunk 3 from the chunk-2 QA (do not lose):** snapshot replay
has no whole-body carrier, so an adaptation lane replaying a frozen source
falls back to the drawn palette and a live run and its replay produce
different sound worlds. The tempting fix -- rebuild the document from the
snapshot's `full_text` -- is WRONG and was rejected: that field is the
truncated projection, so it would mint a document whose total-coverage
guarantee describes a prefix. The correct fix is the snapshot-envelope
extension already specified in 1(a) below.

1. **Uncapped `SourceDocument` + a pre-outline `SourceOverview`** (r4): split
   the normalization owner, then derive deterministic COVERING windows with
   exact-span evidence for cast, setting, principal turns and ending. This is
   what grounds the PRE-OUTLINE authors -- the interpreter today reads the
   CAPPED payload (`_otr_public_domain_sources.py:520-543`, running at
   `OTR_LedgerScriptWriter.py:3748-3757`) before contract (`:3948`) and
   outline (`:4129`); beat-keyed grounding alone arrives too late for them.
   Transport (r4): ONE transient typed field --
   `SourceFetchResult.source_document` -> typed normalized result ->
   `resolved["source_document"]` -- MECHANICALLY excluded from meta/ledger
   serialization; snapshot replay reconstructs the same type.
2. **Contract / grammar / outline from the overview's document-level
   anchors**: the one derivation function runs BEFORE grammar build (or
   grammar re-renders from the final contract), feeding stamp + grammar +
   canon. Pre-outline derivation uses DOCUMENT-level anchors only -- selected
   spans do not exist yet (r4 wording fix).
3. **Beat-keyed window selector + `SourceGrounding` threading + typed failure
   boundaries** (post-outline, when beats exist). The route matrix must NAME
   the announcer routes -- intro / rewrite / outro authoring at
   `OTR_LedgerScriptWriter.py:5104-5116`, `:5272-5285`, `:5357-5409`
   (verify-at-build) -- and decide per route: grounded, or constrained to
   already-grounded accepted fields.
No node signature, widget, link or schema change is intended anywhere in this
item -- the canonical JSON stays byte-identical through the sprint; if any
chunk turns out to need an INPUT_TYPES change, section-0 same-commit rules
apply and the plan must say so first. The bench items were conditional filler
and are now unreachable; that is fine.

**Ceiling to be honest about:** this can be built and unit-tested here, but its
real proof is a render. Renders HAVE RESUMED (2026-08-05), so the
`production-qualified` leg is runnable whenever a render window is free; until
it runs, claim only `code-complete + suite-green`.

### 2. THE NON-COMMERCIAL NOTICE REACHES NO HUMAN SURFACE (~30 min)

Fully scoped, ledger-clean, and the smallest real win on the board.
`nodes/OTR_LedgerScriptWriter.py:3590` stamps `meta["noncommercial_notice"]` (via
`_otr_provenance.noncommercial_notice`, `:124`) and logs it. **Nothing renders
it.** `nodes/otr_credits_roll.py:516` reads only `credits_source_line`.

Add a sibling printed-credits item beside that block -- `:516-518` is the exact
three-line shape to copy -- plus an integration test. The ledger field already
exists and already has an owner, so this adds a CONSUMER, not a field: no
ownership question to answer. Fires on Folger sources.

**Acceptance (r1 + r2 + r3 + r4):** `meta.noncommercial_notice` present -> ONE
rendered credits item; absent when empty, exact text, exactly ONCE. The
existing source item renders as `>> SOURCE: ...` (`otr_credits_roll.py:510-518`)
-- state the notice's literal prefix the same way, and the notice renders even
when a malformed legacy ledger lacks `credits_source_line`; ADJACENCY (source
line immediately followed by the notice, each its own `intercept` entry)
applies when both exist. No new wrapping helper: the existing intercept renderer already
measures and wraps every entry through `_wrap`
(`otr_credits_roll.py:1131-1135`). Test the ORDERED flow list (do not convert
`col3_flow` to a dict -- duplicate `"intercept"` keys collapse). Integration
fixture proves the Folger wording survives flow construction unchanged.
Legibility on canvas is eyeballed on the next permitted render, not claimed
from the test.

### 3. THE TEST-ORDERING POLLUTION (~30 min)

`tests/test_public_domain_sources.py` pollutes
`tests/test_public_domain_interpreter.py::test_empty_cast_is_rejected_and_retried_to_failure`.
Confirmed 2026-08-04: fails when the two run adjacently, passes **11/11** when
the interpreter file runs alone, invisible in full-suite order. Pre-existing --
already proven by stashing and reproducing at the prior commit.

Worth the half hour because it costs a real signal: any targeted run touching
those two files reports a red line that has to be re-diagnosed as benign every
time. **Build shape (r2 -- the r1 "cleanup fixture" idea was WRONG and is
withdrawn):** the mechanism is MODULE-IDENTITY breakage, not leaked state.
`test_module_import_is_lazy` (`tests/test_public_domain_sources.py:223-233`)
calls `importlib.reload(pd)` twice, which REPLACES the module's class objects,
while the interpreter test file imported exception classes at collection time
-- so `except OldClass` no longer matches instances raised by the reloaded
module. No cleanup fixture can restore class identity. Fix (r3-refined): run
the lazy-import assertion in a SUBPROCESS -- `sys.executable`, repo-root
`cwd`, `check=True`, fresh import with the read guard installed -- and pin
BOTH test-order permutations as regressions. The private-module-name
alternative is CUT: it risks exercising fallback import paths instead of the
production `nodes._otr_public_domain_sources` package identity.

### 4. FINISH ITEM 7 -- the leak is closed and proven; testability is not (~2h)

The spoken-citation defect SHIPPED and is live-proven (`3943dd38`, `0957e169`,
`104c3f78`; receipt in `PROD_BUG_LOG.md` PBUG-20260805-04). Seven legs, six lanes,
zero leaked lines, and the corpus audit held at 69 findings across eight new
episodes. Licensed sources are now credit-only -- the announcer names neither the
licence nor the licensor, because Folger publishes the edition and Shakespeare
wrote the play.

What it still owes, all specified in `kibitz-runs/2026-08-05-item7-citation/r4/final.md`:

* **B4 -- extract the coda helper** so tests exercise the production reader.
  **TRAP:** do NOT extract `OTR_LedgerScriptWriter.py:5463-5588` verbatim.
  `news_meta` is defined inside that range and read by the caller below it --
  extracting as written is a `NameError` on every episode. Both review lanes
  caught this independently. Keep `news_meta` in the caller.
* **B6 -- bump `CURRENT_SCHEMA_VERSION`** (`nodes/_otr_ledger.py:58`). The audit
  must REQUIRE `spoken_coda_source` on post-fix ledgers while tolerating its
  absence on the 1,587 legacy ones; without a version boundary a dropped receipt
  is indistinguishable from history. `LEGACY_SCHEMA_VERSIONS` in
  `scripts/audit_spoken_citations.py` is already written to expect the bump.
* **Writer-level routing tests** (depend on B4): both fidelity banks x
  {non-empty, empty} provenance, plus an owned/non-empty case with
  `_style_grammar_on == False`. Assert the coda is PRESENT, not merely that the
  URL is absent. Control is `media_archive`, NEVER `scifi_news` -- that lane
  dispatches to `scifi_news_circuit` and returns before this block.
* **Bug Bible coverage** -- mandatory per `CLAUDE.md`, not a judgment call. The
  rule to promote: **a fix applied to a function with no callers is not a fix.**
  The 2026-08-04 attempt at this same defect edited `spoken_coda_line()`, which
  had zero readers, and 30 episodes leaked after it "landed".

Also parked and owed a merge: another session's worktree
`.claude/worktrees/awesome-brahmagupta-a509b4` holds the uncommitted deletion of
the dead `news_coda_spoken_reduction` receipt chain and `finalize_news_coda_surface`
(no callers tree-wide, no producer for its two trigger flags). It stood down so it
would not collide with B4. Re-ground it against the new helper boundary, then merge.

### 5. 1,090 CAST ROWS CLAIM A NON-COMMERCIAL MODEL IS COMMERCIALLY CLEAN

`eng_indextts2.py:55` says `commercial_clean = False` (bilibili non-commercial);
all 40 bank rows say `true`; `cast_lock.py` trusts the bank row. The row flag is
the CLIP's licence and the engine flag is the MODEL's -- genuinely different
facts, both already in the right layers. **Stamp the JOIN. Do NOT edit the 40
bank rows** (`otr_dl_indextts2_refs.py:11-17` documents them as clip provenance;
the ingest mints three rows across three engines from one PD clip).

**Must heal ATOMICALLY or it creates the defect it fixes:** the stamp
(`cast_lock.py:742`), the `gated` counter (`:575/:614/:661/:670`) AND the three
report strings (`:578/:618/:673`) -- otherwise the report prints `clean=True`
beside a ledger saying `False`. Resolve ONE profile by `(role, engine)` --
role-scoped, never engine-name-scoped. **Enforcement stays OFF.**
Prospective-only for the 1,090 frozen ledgers.

### 6. A TERMINAL FREEZE GATE THAT HAS NEVER READ A POPULATED FIELD

`find_scene_coherence_issues` reads `lines[].scene_id`; the `scifi_news` lane
writes `beats[].scene_id`. 55 ledgers assert the check, 0 carry the field, 55
pass. Nothing in `nodes/` writes `lines[].scene_id` on ANY lane -- the check
never had a producer.

Join per line: `beat_id` -> beat -> `scene_id` -> declared scene. Add a **VACUITY
refusal** (an armed gate that examined zero linkages FAILS -- that is how this
survived 55 episodes). **Split request from verdict:** keep a
configuration-derived `scene_coherence_required` and write
`{required, checked, verdict, issues}` into `report.info` -- `run_gap_audit` is
READ-ONLY (`_otr_ledger_freeze.py:664-698`), so the gate must not mutate the
ledger; the phase wrappers already persist the report. Measure OFFLINE over the
published corpus first, then arm in ONE change -- no intermediate flag-off ship.
Replace the stale hard-coded bank list at `tests/test_scene_guard_v4.py:89-99`
with registry-derived coverage (it omits `scifi_news`, the one bank that enables
the flag).

**The vacuity class is now proven twice** -- this gate, and the freeze test at
`test_g9_sfw_ship_stop.py` that filtered on a retired code prefix (fixed
`4506b1ed`). Any NEW armed gate ships with a vacuity assertion.

### 7. CHARACTER GENDER IS ROLLED ON PROSE LANES -- Scrooge shipped female (spec REWRITE owed; r2 and r3 both returned NO)

Live 2026-08-05: `EBENEZER SCROOGE` = female, `JACOB MARLEY` = other,
`HENRY HARTWICK OGLETHORPE` = female. Meanwhile MACBETH, BANQUO, PROSPERO and
MIRANDA are all correct.

**The split IS the diagnosis, and it means the render code is not broken.**
Shakespeare ships 14 provenance sidecars carrying `characters` with genders; the
prose lane has ONE tracked sidecar and its `characters` key is `None`. The pin
chain already exists and is lane-neutral (`_otr_roster_gender.py`, 12.6 KB, on
disk). Shakespeare is right because the DATA is there. Prose is rolled because it
is not. **This is a vendor-time data gap with a working consumer** -- the exact
inverse of the Item 7 bug, where the value existed and nothing read it.

Spec: `docs/2026-08-05-character-gender-ladder-SPEC.md` (Fable, driver-grounded).
A four-tier TOTAL ladder -- roster -> pronouns in the source text ->
character-in-work web lookup -> name-frequency percentage -- stamping `gender`
(always populated, never `unknown`, because a voice must still be cast) plus
`gender_source` and a confidence. Operator rulings baked in: Shakespeare's KNOWN
rows are untouchable, the announcer stays randomly male/female by design, and the
invented lanes (`original`, `scifi_news`, `scifi_news_pro`, `media_archive`) keep
rolling -- their characters do not exist, so a name search there risks matching a
real person.

**Codex r2 verdict: NO. Eleven must-fixes, at `kibitz-runs/2026-08-05-gender-ladder/r2/codex.md`.**
The diagnosis survived; the mechanism did not. The three that matter:

1. **The web search would silently do nothing.** The spec passes a tools/plugins
   argument to `OpenRouterBackend.generate`, which swallows unknown kwargs through
   `**_ignored` -- no error, no search, a confident answer from a model that never
   looked. That is the same silent-no-op class as the defect above.
2. **"LLM extraction over the FULL unit text" cannot run.** `beckoning_fair_one.txt`
   is 143,176 bytes and 58 of 65 source files exceed 12,000 bytes, against a
   32,768 estimated-token per-call cap.
3. **Blanket surname aliases are identity-unsafe.** Two rows sharing a surname
   with the same gender currently produce a confident pin rather than an
   abstention.

**r3 RAN 2026-08-06 and it is STILL NO** -- Codex NO with 7 must-fixes, agy
yes-with-fixes with 9. That is three NOs across two rounds, and r3's findings invalidate
the CODING PLAN rather than its line numbers, so the standing re-ground rule applies:
**the next step is a SPEC REWRITE folding r2 + r3, then r3 again. Not r4.**

Both lanes independently found (a) a manifest sequencing deadlock -- the stamper is
specced to run per-unit inside the vendor fetch loop, but the manifest is written only
AFTER the loop, so it can never see the unit it was called for; and (b)
`RosterGenderVerdict` has no `gender_source` / `gender_confidence` fields, so the
ladder's whole output cannot be carried without changing every verdict-construction
path. Codex also confirmed the r2 finding that the OpenRouter backend still has no
web/plugin parameter, so the web-search tier would silently do nothing.

Judgment: `kibitz-runs/2026-08-06-2026-08-06-gender-ladder-r3/r3/judgment.md` (LOCAL
ONLY). **Trap: commit `496d9d57` inserted ~90 lines near the top of
`nodes/_otr_roster_gender.py`, so every cite into that file in the r3 review has
SHIFTED. Re-pin before acting.** The rewrite should also CONSUME the
`normalize_gender` boundary item 8 installed there rather than adding a second
normalization path.

**Found while grounding, and it reopens an operator ruling:** 32 of 85 Shakespeare
roster rows are `unknown` TODAY -- 38% of the lane assumed solved. Comedy of Errors
ships 7 characters, every one unknown. The narrower ruling that fits the evidence:
Shakespeare's KNOWN rows stay untouchable, but tiers 3-4 may fill only its
`unknown` rows. That fixes 32 rows without ever second-guessing a parsed
dramatis personae. **Operator decision, not a driver call.**

### Bench leftovers (relocated)

The old conditional bench block is gone (unreachable once item 1 grew, and two
of its three items already live in NEXT CODING QUEUE item 6). The remaining
one: **the three works that refuse to vendor** (`ghost_ship` gid 11045,
`purple_cloud` 11229, `beleaguered_city` 11521 --
`scripts/otr_vendor_public_domain_library.py:303/341/542` against the parser
at `:594-686`) **needs one Gutenberg fetch, so it is operator-opt-in only** --
not schedulable inside an offline sprint.

**Do NOT start the Shakespeare verbatim executor in this session.** It is a
multi-session structural change gated on the ownership table
(`docs/2026-08-03-fidelity-pass-ownership.md`) with four overwrite paths to close
first, and starting it half-way is worse than not starting it.

## NEXT CODING QUEUE -- confirmed against the tree 2026-08-04 (non-GPU, suite-provable)

Verified by grep/read this session, after the operator parked SFX and asked what
other coding work is real. Same rules as ON DECK: in order, one green pushed
chunk at a time, full `kibitz-plugin:kibitz` per item while the sprint gate
stands.

1. **Style/identity campaign, items 1-4 (one campaign, ~1 day).** Highest
   leverage: fixes the credits style line for all six banks uniformly.
   Re-verified: `run_story_brief_reflection` is `_otr_story_brief.py:446`;
   `_build_left` is `video_engine.py:1442`; the treatment line renders as
   padded `Style    :` at `video_engine.py:1762` (an earlier "no longer greps"
   note here was a driver grep miss, corrected by the r2 panel -- BOTH
   consumers are live and both move in the repoint). Sharper finding on
   item 4: `ending_template` is NOT "computed and never read" -- the catalog
   computes it (`_otr_style_catalog.py:906`) AND the composer reads it
   (`_otr_line_composer.py:809-810`); what is missing is the THREAD between
   them -- no call site passes it into a LineRequest. That is a DECISION GATE
   inside the campaign, not a confirmed build step: wire the thread or rip the
   dead ends, decided with the panel at build. Same for the ghost-name fork:
   **default = scrub briefs after cast lock** (conservative -- no unlocked name
   reaches the listener); the operator may overrule to propagate pitch names.
   `style_seed_env` confirmed validator-only (`capability_profiles.py:116`).
   Item 5 (the 120-key `meta` rip) stays a gated block of its own, NOT part of
   this campaign.
2. **The P0 repair rung tells the truth (2 items, ~0.5 day).**
   `repair_literal_source_metadata` (`_otr_scifi_source_repair.py`, called from
   `_otr_scifi_codex.py`): (a) emit a receipt per pruned span/evidence-row/fact
   -- silent pruning violates the plan's own Invariant 3; (b) give the repairer
   the `allowed_source_fields` allowlist or prune per row, so one bad rehome
   stops poisoning the whole artifact. Suite-provable with fixtures. The HTML
   block-join separator stays an OPERATOR decision (coordinate-system change)
   and is NOT part of this chunk.
3. **Script-parse repair: fix the SPEC, then code it (~0.5 day docs + panel,
   then 1-2 days code).** The claim that increments 1-5 are code-ready is
   STALE: `docs/2026-08-03-script-parse-repair-CODE-READY.md` itself says r3
   returned seven must-fixes that invalidate the call/trace design and
   everything after its STATUS block is a draft. The next chunk is the spec
   correction folding those seven in (a kibitz arc IS the vehicle); only then do
   increments 1-5 become codeable.
4. **Passage-lane craft scoring + the stichomythia floor (~1 day).**
   `_otr_passage_selector.py` has no scoring functions -- the Fable craft
   criteria (French-scene boundaries, entrance/exit starts, couplet ends,
   continuation-word penalties) are all unbuilt. Score, keep top-K, seeded hash
   within the class. Same chunk: the per-beat word floor excludes stichomythia,
   so a merge rule or floor exemption in `_otr_episode_budget`. Pure Python +
   tests. **Sequencing (r1): runs AFTER the ON DECK sprint lands and re-grounds
   against whatever item 1 changed** -- both touch the fidelity selection
   surfaces.
5. **Cast-list parser: the two weak plays (~0.5 day).** Midsummer 1/12 and
   Comedy of Errors 1/7 gendered -- mechanicals/servants in shapes
   `_otr_character_roster.py` does not read yet. Vendored texts are on disk;
   offline; suite-provable against the sidecars.
6. **Small-items batch (~0.5 day, ONE campaign over the batch, one commit each):**
   * `OTRImageGenDispatcher` (`otr_image_gen_dispatcher.py:1412`) has no
     `IS_CHANGED` while depending on external file existence -- confirmed by
     grep, none in the file. Decide the CONTRACT before coding (r2): either
     fingerprint EVERY actual external dependency or deliberately force
     re-runs for this side-effectful node -- a partial path fingerprint still
     serves stale results.
   * Rotated server logs have no retention policy.
   * The `provider_side` three-part-rule regression (picked AND forced
     `cloud_kling_avatar`).
   * The shared `row_is_active(...)` evaluator over captured state -- confirmed
     absent from the tree -- closing the four env-read sites named in OPEN BUGS.

## PARKED -- D2 (renders have resumed; run when a render window is free for fail-hunting soak legs)

Reset per AGENTS.md section 4, boot headless, run **320-word `public_domain` or
`shakespeare` still legs until one fails** (~1 in 6). Three legs on 08-04 all
published, which at that rate is a ~58% chance of zero -- neither confirmed nor
cleared.

**Either outcome is valid.** A publish is a clean leg; a **fail-closed with
complete evidence is the PROOF D1 WORKS** and is the outcome you want. When it
fails the server log names the branch itself -- arm, token, index, canonical
`prompt_hash`, repr-escaped excerpt -- plus a compact JSON `MISSING_TARGET`
record emitted BEFORE the raise (the canonical runner truncates the exception at
500 chars, `scripts/otr_api.py:749`). The log survives reboot;
`scripts/_otr_rotate_log.ps1` rotates instead of truncating. D3 then fixes THAT
branch at its root and `PROD_BUG_LOG.md` gets a mechanism, not a guess.

**Do NOT:** weaken the completion gate, revive the portrait-init fallback, or
rebuild the withdrawn "give the collapse guard a still owner" fix -- the 08-04
postmortem disproved that chain (70 whiffs and 69 cast-time deferrals across 11
passes that ALL published).

Record: `docs/2026-08-04-POSTMORTEM-still-unmaterialized-320w.md`,
`docs/2026-08-04-D1-SHIPPED-still-skip-evidence.md`.

## AFTER THIS SPRINT -- the standing block order

One coder window at a time; every chunk = focused tests + full suite + Bug Bible
+ commit AND push + `HEAD == origin/v2.0-alpha`.

```text
  -> WAN 8-GB low-VRAM launch contract  (CODE-COMPLETE; blocked on ONE operator
                                         decision -- see OPEN BUGS)
  -> [r3+r4] Randomizer A
  -> [r3+r4] dynamic_story           (wiring only -- rev-5 DESIGN stays FINAL)
  -> re-observe the PARKED story bugs on the next real render legs;
     batch-triage whatever is left
  -> THEN, and only then, ROADMAP.md -- OFF THIS PLAN
       (its order after the SFX park: product expansion -> LEAN-MEAN ->
        RunPod -> release)
```

**LEAN-MEAN IS NOT IN THIS QUEUE and must not be re-added.** Operator direction
2026-07-29 moved FRONT and TAIL both to the Lean-mean campaign section of
`ROADMAP.md`, with their chunk chains, the W2 migration-first mandate, the
ENGINE_MATRIX W6 sub-step and the full `r2 -> r3 -> r4` pin carried over intact.
It runs after the randomizer and `dynamic_story` (the SFX step that used to sit
between them is PARKED). A window that wants to rip dead code is on the wrong
document.

**Block detail:**

1. **Randomizer Rolls Design A** -- `docs/2026-07-12-randomizer-rolls-r2-coding-plan.md`.
   NOT gated any more: extensibility landed and absorbed its `_otr_lane_specs`
   authority, so this shrinks to `_otr_bank_roll` + eligibility. Its r3 brief must
   carry two deltas -- the absorbed authority, and that the bank list is a LIVE
   registry read (`list_bank_ids()` can return a CLIENT bank; eligibility must
   treat one as an ordinary peer) rather than a six-row literal. 1-2 d + 1 GPU day.
2. **`dynamic_story` visual direction** -- rev-5 FINAL; roster-agnostic; re-derive
   IDs at build. After the randomizer. The "do not rerun the design panels" rule
   and the r3+r4 requirement are NOT in conflict: the rule protects the DESIGN,
   r3 asks whether that design still wires to the code that exists today, and the
   roster, the routing authority and the writer tail have all moved since rev-5.
   5-9 coder-days + 2-4 GPU days.
3. **SFX campaign -- SUPERSEDED 2026-08-06. It is not parked any more; it is
   being RIPPED.** The 2026-08-04 park (operator doubt + an 8-15 coder-day lift)
   became a deletion when the operator ruled *"I do really want to rip out SFX
   100%, that's my aim."* **The live work is section 0-TER of this document**;
   this entry survives only so a reader who remembers "parked" finds out here
   that it expired.
   Nothing spends against a REVIVAL, and now nothing preserves one either: the
   Timeline Cue Ledger and generated-SFX designs are slated for retirement with
   the code. What the rip does NOT touch is the b-roll role tombstones (they
   still fail loud on stale ledgers) or the `[ENV|SFX|MUSIC:]` text sanitizers
   (defence against a model hallucinating a tag, not an SFX feature).

Open judgment question (render-window, not a coder slot): the LOCAL mistral/gemma
writer matrix. The Sonnet arm of the creative-writer question is answered
(`docs/2026-07-17-model-bakeoff-scoreboard.md`); the local roster comparison
never ran.

### STANDING RE-GROUND GATE -- r3/r4 before ANY block above (operator 2026-07-24)

Every remaining block was planned against a tree that no longer exists. Since
those docs were written the LLM vetoes were ripped, THE LAW landed, six banks were
renamed onto new packs, word-fit ceilings were retired, the whole extensibility
build shipped, and the suite grew past 8,000. Line cites, seam names and file
inventories are the first things to rot, and every one of these blocks is a rip or
a rewire that acts on exactly those.

- **Default entry point is `r3` (wiring).** These plans already have an r1 and an
  r2 on record, so the cheap re-ground is wiring against CURRENT code, then `r4`.
- **Drop to `r2` when r3 finds the CODING PLAN wrong**, not just the line numbers.
  Stale cites are an r3 fix. A seam that no longer exists, an authority that moved,
  a precondition another build already satisfied or destroyed -- that invalidates
  the coding plan itself, and patching an r2 from inside an r3 produces a plan
  nobody reviewed.
- **If in doubt, start at r2.** A wasted r2 costs one panel round; executing a
  stale coding plan costs a day of rips against the wrong file list.
- **No block executes without an r4 convergence at current HEAD.** Record the run
  under `kibitz-runs/<date>-<block>-r<N>/` and cite it in the block entry.

## SCOPE FOR v2.0 (operator -- read before picking up fidelity work)

**Two banks, not three.** `shakespeare` is VERBATIM and gets the executor.
`public_domain` stays PROSE and is explicitly allowed to be FUZZY -- operator: "the
LLM's job to try to do book prose but not perfect", and "I'm fine if it can pick up
real dialogue great, if not that's OK". Best-effort, never verbatim, never gated.

**But fuzzy WORDING is not the same as melding two stories, and that distinction is
the actual requirement.** Operator: "public domain does need to be updated so it
doesn't try to meld two different radio drama things and tries to keep true to the
source." That names the delivered defect exactly -- an H.G. Wells chapter performed
as taking place in "Arkham, Massachusetts", Lovecraft's fictional town, with the
time machine shrunk to a pocket watch. Two authors fused into one episode. The
contract for this bank:

* FREE with the wording. Invent the speech; carry the source's own quoted lines
  where they exist (the Wells chapter has real ones -- "Story!" cried the Editor).
* NOT free with the WORLD. Its place, people, period and events are the source's.
  No relocating, no importing a second work's setting or characters, no genre
  transposition.

Two changes serve that, and neither is a gate:
1. **PROMPT-ONLY** -- the pack asks for the source's world and its own quoted
   speech, and forbids importing anything the source does not contain.
2. **Stop the content-blind rolls contaminating it.** A catalog sound world ("a
   fire in the grate, a mantel clock, a teacup") was imposed on Wells' Richmond
   parlour, and `arc_shape` stamped "heist" on a man demonstrating a time machine.
   Foreign frames arriving from a dice roll are melding by another route.

**`public_domain_plays` is DEFERRED TO v2.1**
(`docs/2026-08-03-public-domain-plays-PLAN.md`, research complete, nothing built).
That avoids a third bank row, which is never one line: it would force a pack
directory, a registered fetcher, an executable pipeline, family-policy coverage and
updates to exact-roster contract tests.

**Two hard operator rules that override the repo's written ethos here:**
* **The word count is a REQUEST, not a gate.** No refusals, no hard gates, no
  shunts. Shipped: `select_passage` returns its closest performable passage rather
  than raising (`a4bc7917`).
* **No "dread py assertion workflow killers."** A render must not die. The
  reconciliation: fail loud in AUTHORING-TIME TOOLS (the fetcher, manifest
  validation -- things a human runs and reads), but in the RENDER PATH degrade to
  the best available result and write an honest machine-readable receipt into the
  ledger saying what degraded and why. The ledger tells the truth; the episode
  still ships. No hazard / under-construction flags -- `runnable` was checked and is
  NOT one; all six real banks are runnable and its only job is making the
  "+ Add Your Own" signpost fail loud on selection.

**An extra LLM VERIFIER pass is sanctioned** ("is this accurate to the story, are
these characters really in the scene?") -- but as a RE-SELECT, never an abort: if it
rejects a window, take the next-best candidate, bounded, then ship the best one with
a receipt. **Deferred until specced (r1):** it currently has no stated input, output
or artifact position. If built, it reviews the FINAL accepted script, stays
advisory/non-terminal under THE LAW, and records evidence-backed findings; until
that spec exists, do not quietly add another model pass.

**Casting must be smart about voices and gender.** The dramatis personae section is
the ROSTER and its descriptions carry gender; it is parsed at VENDOR time into the
provenance sidecar so the render path never infers (`d8752d69`). A manifest-approved
roster also replaces the refuted "speaker appears twice" heading rule -- that rule
would have deleted BEATRICE's single speech from `much_ado__act3_scene1`, the scene
named for her, where that one speech IS the payoff.

**Still open on casting:**
* **Midsummer 1/12 and Comedy of Errors 1/7 gendered** -- mechanicals and servants
  in cast-list shapes not yet read, recorded `unknown` rather than guessed. Operator
  is open to a vendor-time LLM/web lookup as the final tier for stragglers, under
  its own `gender_source` so it stays auditable. (Corpus elsewhere: AYL 6/6, Lear
  9/9, R&J 3/3, Much Ado 4/4, Tempest 3/3, Twelfth Night 5/6, Macbeth 7/8.)
* **Voice-pool capacity may belong in window ELIGIBILITY, not just casting.** The
  Bark pool is 6 male / 4 female against a 6-character ceiling, and Macbeth 1.3
  needs five distinct voices.
* **Disguise ruling (settled):** ROSALIND-as-Ganymede and VIOLA-as-Cesario keep
  FEMALE voices -- the source prefix says who speaks and the irony depends on the
  audience hearing her; the announcer states the disguise from a manifest field.

**Visual style: randomized is fine** (operator). The earlier objection to
`archival_documentary` over a Folger comedy was about TRUTHFULNESS, not variety --
the credits claimed a story scaffold the episode did not have. With the words
genuinely the play's own and the strip naming the real source, a randomly drawn look
is artistic range. (`visual_style_policy` was RIPPED on 2026-08-04 accordingly.)

## THE PASSAGE LANE (operator ruling; built, with craft criteria still unbuilt)

> "For shakespeare I'm open to a version that is very strict and finds, based on
> word count and random choice, hones in on a specific part of a play to get real
> specific dialogue, no paraphrasing."

A play episode is a contiguous WINDOW of consecutive speeches, carried verbatim,
chosen to fit the word budget, the cast ceiling and the beat topology. Built as
`nodes/_otr_passage_selector.py` (`a82460ec`), 24 tests, proven on all 14 vendored
scenes: every selected line is verbatim from its source file.

**The number that governs everything:** a passage is performed against VOICED
BEATS, and beats step with the ACT TOPOLOGY, not the word count --
`voiced_beat_count()` in `_otr_episode_budget` is the one owner. 30-120 target words
buy THREE beats, 150-200 six, 300-1200 fourteen. At 120 words a passage is a
two-or-three speech fragment; at 300 it is an eleven-to-thirteen speech exchange.
**The fidelity floor should be 300, not the operator's initial 120** -- three beats
cannot hold a change of mind, and every manifest already recommends 300. A long
speech spans consecutive beats in the same voice (`ceil(words/80)`,
`BEAT_WORD_HARD_MAX`); without that the lane silently loses Lear's love test,
Prospero's history and Juliet's balcony speeches.

**Craft criteria for selection, from the Fable review -- NOT yet implemented:** keep
windows inside one French scene (never cross an `[Enter ...]` that adds a speaker);
prefer starts on an entrance or a question, penalise openings on continuation words
(And/But/Nay/'Tis) and speeches under 4 words (Folger prints shared verse lines
separately, so those start mid-breath); prefer ends on an exit, a scene end or a
rhymed couplet, avoid ending on a question or a trailing dash. Score, keep the top
K, then apply the seeded hash within that class. Showcase example: Romeo and Juliet
2.2 lines 257-318, `[Enter Juliet above again.]` "Hist, Romeo, hist!" through
"Sleep dwell upon thine eyes... [He exits.]" -- 14 speeches, ~250 words, entrance
start, couplet-and-exit end, a complete arc that maps 1:1 onto the 14-beat topology.

**Prose is a different lane and the review was blunt about it.** Wells' chapter is
~70% narration; a characters-only performance discards the book's actual asset. The
faithful prose lane should be a NARRATOR/READER role speaking the author's own
sentences, abridged by CUTTING ONLY with every dropped span logged in provenance --
"abridged verbatim", which is also the period-correct radio form. Defer the
paraphrased variant until a dialogue-poor source genuinely needs it: as specced it is
indistinguishable to a listener from the existing original lane with borrowed names,
and it reopens the failure class just closed. If built, the announcer must say
"freely adapted from".

**Also flagged, unfixed:** the per-beat word FLOOR (20 at three acts) excludes
stichomythia -- "Nothing, my lord." / "Nothing?" / "Nothing." cannot be three beats --
so rapid exchanges need a merge rule or a floor exemption; `[aside]` and `[within]`
are machine-readable delivery hints worth carrying into per-beat metadata; and the
Wells manifest synopsis says the traveller "returns with a strange machine" when in
the real chapter he returns limping and the machine never appears.

## THE ADAPTATION DESIGN (hardened, NOT yet built)

Plan of record: `kibitz-runs/2026-08-03-adaptation-fidelity/r2/final.md`.

**The keystone correction: compile source speech, do not generate it.** A ledger row
that merely POINTS at a source segment proves structure, not meaning --
`PRODUCTION_SPRINT_LESSONS.md` lesson 11 documents that exact failure class.
Source-owned text must be materialized deterministically from an authenticated
segmented artifact and verified against it. "Summarize into X words" then means
SELECTING WHICH REAL SEGMENTS FIT THE BUDGET, not paraphrasing -- which also removes
the VRAM hazard, since no model sits in the source-speech path.

Settled by arithmetic: an episode cannot exceed **1,520 words** (19 voiced beats at
act_count 7, `BEAT_WORD_HARD_MAX` 80), so full-scene performance is impossible
without redesigning beat topology. Build target is the 300-word unit.

**NEXT, IN ORDER:**

1. **The segmented source artifact** (schema, spans, hashes,
   `body[start:end] == segment.text`, omission receipts) and the pass-to-field
   ownership table -- **nothing else codes until that table exists.**
2. **Cast from the selected cut.** Real scenes carry 3-12 speakers against a
   6-character ceiling (`_otr_casting.py` 1-6, `OutlineRequest` rejects >6), so which
   speakers appear must follow from the cut that fits the word budget. Coupled hard
   to the capacity guard: at act_count 1 there are exactly THREE voiced beats, so a
   4-person cast is a mathematically guaranteed `CastVoiceCoverageError` -- the
   failure that killed `scifi_news` in the six-bank run. `compute_episode_budget`
   must also receive the TRUE locked cast.
3. **Loosen the count-match invariant** (`OTR_LedgerScriptWriter.py:4061-4067` hard-
   raises on any locked != requested) and change the pack text that tells the model
   to drop figures.
4. **Extend `_otr_provenance.py`** -- do not add a second attribution owner -- and
   bind its output to the verified body hash.
5. **Schema migration** to retire `cast_hints`; still required by the validators and
   by `public_domain_manifest_schema.json`, so manifests and tests migrate in the
   same change. (`visual_style_policy`, the other half of this item, was ripped
   2026-08-04.)

**KNOWN AND NOT FIXED:** `canonicalize_shakespeare_text` truncates at 12,000 chars
and the interpreter sees only the first 5,000, so a 3,445-word scene reaches the
brief as ~880 words, silently. Belongs with the artifact work, where each beat is
fed its own segment rather than a blind prefix.

## STYLE / IDENTITY DECISION WORK (one campaign, next CODER window)

Grounded by the 2026-08-03 four-agent forensics; every line has a file:line in the
session traces.

1. **"Invent one and tag it"**: add a derived style/genre field to
   `run_story_brief_reflection` (`_otr_story_brief.py:446` -- proven content-loyal on
   both specimens), stamp beside `story_brief`, repoint the treatment `Style:` line
   (`video_engine.py:1762`) and the HUD (`video_engine.py:1336` -> `_build_left`
   `:1466`) at it. Highest-leverage item here: it fixes the credits line for all six
   banks uniformly.
2. **Rename `meta.style` -> `meta.story_scaffold`** (operator: too many metas; the
   field is neither scifi nor a description). Consumers move in ONE atomic change:
   writer stamps, credits `_story_style_receipt`, `visual_plan.style`,
   `video_engine.py:1336`, tests -- AND the ledger validators (r3):
   `_otr_ledger_consistency.py` pins the field in its matrix
   (`MatrixRow("style", ...)` at `:68`, `:177`) and `_otr_ledger_cleanup.py`
   reads it too; missing them fails ledger validation on the first episode.
3. **Ghost-name reconciliation fork**: pitch cast never reaches `lock_cast` (names
   are a pure pool draw; `source_character_names` deliberately None for invention
   lanes). Decide: scrub briefs after cast lock, or propagate pitch names. Evidence:
   Evelyn/Leonard as offscreen lore; Fogbound Rails bio still opens "Lizzie Gray".
4. **Dead fields found**: `ending_template` computed but zero LineRequest call sites
   pass it; `seed_policy.style_seed_env` validated but unconsumed; `dramatic_state`
   derived PRE-dialogue goes stale in the treatment.
5. **`meta` is a 120-key drawer** -- the cleanup the operator keeps asking for. Scope
   as its own rip with the ledger law (every field one owner).

## OPEN OPERATOR QUESTIONS (flagged, awaiting a ruling)

* **Does `media_archive` want the catalog premise at all**, or the same
  scaffold-off treatment as `original`? Found by the five-bank beat test: a
  `pirate_radio_resistance_drama` premise was drawn over a film-reel standoff
  seeded by a real Library of Congress item on 'Midnight' (1939) -- the operator
  caught it on screen. Second specimen of the content-blind-draw class. The
  scaffold-off rule so far was stated only for `original`.
* **Rename the un-namespaced `OTR_WAN_*` knobs?** `eng_wan_i2v`'s six frozen knobs
  are `OTR_WAN_STEPS` / `_CFG` / `_SHIFT` / `_SAMPLER` / `_SCHEDULER` / `_NEGATIVE`
  -- no `I2V` namespace, unlike every sibling. Default if unruled: leave them. The
  freeze already removed the power that made the missing namespace dangerous (they
  are consent-act-only now and cannot bind a production leg), and a rename would
  silently break operator muscle memory for a sweep.
* **`style_tail_policy`'s closed enum cannot express a SHIPPED path.**
  `VALID_STYLE_TAIL_POLICIES` has `full` and `minimal_clean`, but
  `build_radio_host_prompt`'s `ltx_radio_mouth` branch
  (`otr_meta_brief_image_prompt.py:394-401`) RETURNS EARLY with
  `"%s, warm dramatic lighting"`, skipping both `finish_visual_prompt(...,
  era_profile="still")` and the `image_grade_tail` append -- deliberately, per the
  2026-07-02 operator look direction. The `ltx_audio_in` bookend row nonetheless
  declares `style_tail_policy="full"`. Adding an enum token is an operator call:
  either add a third token for "canonical warm, no era tail, no grade tail", or
  ratify that the `ltx_radio_face` path is EXEMPT from the plan's style-tail
  authority. Default if unruled: the exemption, because it changes no behaviour.
* **After profile retirement, who owns a tier's native render ceiling?** The full
  statement of this one is in the WAN 8-GB row under OPEN BUGS -- it is the single
  blocker on a code-complete block.
* **`check_compatibility`: ratify the inert constant, or schedule the rip?** See
  Open risks.

## OPEN BUGS / DEFECTS (live, not yet closed)

MECHANICAL defects survive story-engine churn; STORY-QUALITY judgments do not. That
split is why the two eyeball-era entries at the end are PARKED rather than live.

**EVERY LINE CITE IN THIS SECTION IS SUSPECT.** Each one checked during the
2026-07-27 triage had moved: `_is_cloud_video_engine` is `render_driver.py:1599` not
`1274-1295`; the "NO FALLBACK to text-only" refusal is `:2148` not `1801-1817`;
`_use_i2v` is `eng_ltx_video.py:583` not `559-572`. The defects are mostly still
real; their coordinates are not. **Re-pin a row's cite when you touch it.**
Path note (verified 2026-08-04): engine adapters live under
`nodes/_otr_video_engines/` (and `_otr_audio_engines/`, `_otr_image_engines/`)
-- bare `eng_*.py` cites in these rows are shorthand for those paths.

### The 2026-08-11 bank-sweep trio (LEMMY sprint, all three OPEN)

Found by a six-bank live render sweep, not by tests. Full detail lives in
`docs/PROD_BUG_LOG.md` and `docs/2026-08-11-FINDING-lane-cast-contract-divergence.md`;
these rows exist so a window working THIS list actually sees them.

* **PBUG-20260811-03 -- `scifi_news` lost the LEMMY cameo it was built for.**
  ROOT CAUSE ESTABLISHED: `scifi_news` is a CONTENT-OWNED lane
  (`delivery_mode_for_meta(meta) == CONTENT_OWNED`, measured off the sweep's own
  ledger; `original` is `legacy`). Content-owned runners build their own cast and
  never run the writer's seeded picker, and `lock_cast()` is what applies the
  cameo -- so it cannot fire there. The empty `cast_contract` is the same
  deliberate decision: that block stamps `meta.episode_seed` and withholds
  `cast_seed`, because claiming one on a lane-owned cast detonated CastLock's
  replay before (`num_characters must be 1-6, got 0`).
  **THE OBVIOUS FIX IS THE WRONG ONE** -- routing content-owned lanes back
  through `lock_cast()` is precisely what that comment warns against. The repair
  belongs in the lane runner. **Operator row 15.**
  *Worst of the three by exposure:* nothing fails and nothing logs, so every
  `scifi_news` episode since the redesign has shipped with no cast contract.
* **PBUG-20260811-01 -- forcing the cameo kills the `scifi_fable2` writer on
  `scifi_news_pro`.** `pass 'script' failed after 4 attempt(s): markup ladder
  exhausted; BAD_LINE`. Reproduced at 30 AND 90 target words, so NOT a word
  squeeze; with the cameo on its natural roll the writer passes cleanly. Root
  cause not established.
* **PBUG-20260811-02 -- `scifi_news_pro` dies at node 92 with no materialized
  still for beat `music_closing_001`** (`still-spine handoff missing materialized
  scene still ... engine still_flat`), on the same profile where five other banks
  produced one. Seen ONCE. Re-run before treating the cause as understood.

### The P0 / source-span cluster (2026-07-30)

- **`full_text` reaches the span coordinate system carrying HTML BLOCK JOINS WITH
  NO SEPARATOR, and on the live evidence this is the DOMINANT P0 failure cause.**
  Measured in the campaign logs: `'...Field of Martian PolygonsNASA/JPL-'`,
  `'...and the School ofEngine'`, `'...what you're doing.Let's s'`, `'...(AMR).The
  resea'`. The RSS adapter strips tags without inserting whitespace, so two elements
  fuse into one token. `_normalize_span_source_text` collapses whitespace RUNS but
  cannot insert a space that was never there, so the model quotes the sentences a
  reader sees and they are not byte-exact in the stored text -- exactly the
  "non-literal source span" rejection that killed 12 of the 15 P0 legs.
  **Deliberately NOT fixed by A-3:** inserting separators is a WIDER change to the
  coordinate system `source_digest` pins -- an operator decision, and it belongs in
  the source adapter rather than the codex normalizer. Owed: which adapter builds
  `full_text`, whether a separator can be inserted at admission without breaking any
  accepted ledger, and a fixture from these four strings.
- **The deterministic P0 rung PRUNES SILENTLY, which violates the plan's own
  Invariant 3.** `repair_literal_source_metadata` drops an unsupported span, then its
  evidence row, then the fact -- and emits no receipt. An accepted P0 index simply
  has fewer facts than the model wrote, and nothing says which were dropped or why.
  Under "fail loud, not fatal" the degrade is the right direction and the silence is
  not.
- **The deterministic P0 rung is ALL-OR-NOTHING across an artifact, and can poison
  its own good work.** It is handed `a0_payload` (all seven keys) while
  `_validate_fact_index` restricts spans to `allowed_source_fields` (the projection).
  A quote rehomed into a field the projection omitted makes `post_validator` reject
  the WHOLE repaired artifact -- "cites source field ... outside the supplied P0
  evidence" -- so one unlucky rehome discards every correct prune in the same pass.
  Either give the repairer the allowlist or prune per row.
- **Nothing measures whether a pruned P0 index is ACCEPTED** (recorded, no action
  owed yet). No live leg has ever run with the deterministic rung reachable (it became
  reachable at `47c554fa`, after the campaign stopped), and the rejection logs carry
  only a truncated `raw head` plus no source payload, so the question cannot be
  answered offline. A-1's instrumentation is what makes the next campaign able to
  answer it.
- **`scifi_news` P0 convergence defect** -- both 120w and 320w legs fail in P0 after
  two attempts on non-literal fact source spans; provider/model convergence, extends
  BUG-11.35. NOT a word/length gate. Blocks the last 120w receipt and the
  `scifi_news` live reverify (PBUGs 20260712-22/23/24/25, fixed in tree, reverify
  still owed).
- **`scifi_news_pro` provider capacity** -- `requested_output=2800` vs provider cap
  `512`; the whole-artifact retry contracts LANDED @ `314dd481` are the base; the
  residual fix is now unblocked. Related independent items: the P9 8K
  structured-capacity follow-up + the GGUF structured-enforcement NEWBUG. Do not
  raise the minimum word target as a capacity workaround.

### The 8 GB / profile cluster

- **WAN 8-GB low-VRAM launch contract -- CODE-COMPLETE and PROOF-INCOMPLETE. It is
  not a coding item; the one thing blocking it is an OPERATOR DECISION.**

  **Already BUILT and WIRED end to end** (verified hop by hop): `otr_8gb_wan.json`
  `video.max_render_frames=17` -> `capability_profiles` optional-key validator ->
  `_otr_workflow_apply.py:532` flatten -> `workflows/variants/otr_8gb_wan.json`
  node-87 widget slot 14 = 17 -> `otr_video_director.py:423` policy stamp ->
  `otr_shot_lock.py:1722` `ledger.video.max_render_frames` ->
  `render_driver.py:3328` per-adapter policy -> `motion_common.profile_max_render_frames()`
  -> `eng_wan_ti2v._floor_length` hard cap (`:730`) and `_planned_length` refusal
  (`:785`), with `render_driver.py:3845` refusing on drift. Landed `f914f0a4`, dead
  node-87 widget repaired `7f4644a1` + `8f41af27`, WAN deliberately excluded from
  `frame_contract.PLANNING_CAP_ENGINES` by `b23fc035`, recipe frozen `71753cb4` /
  `8424f369`, whole-beat single-UNET-load `439ce8c7`. Regression net:
  `tests/test_remaining_video_contracts.py:16-194` (nine hop-by-hop tests) plus
  `tests/test_multiclip_effective_contract.py:216,234`.

  **THE ONE OPERATOR DECISION (the actual blocker).** The ceiling reaches a leg ONLY
  through a variant workflow or a hand-set widget: `otr_canonical.json` node 87 ships
  `max_render_frames=0`, so a plain canonical WAN run is UNPINNED and inherits
  `_TI2V_MAX_FRAMES = 177` -- exactly the 2026-07-23 failure shape. The obvious patch
  (pin 17 in the canonical) is WRONG: the canonical serves every tier, and 17 is the
  8-GB tier's number, so pinning it would cap LTX/HuMo 16-GB legs too. The channel
  that carries 17 today is `config/profiles/*.json`, which is on the RETIREMENT list
  -- so writing new behaviour onto it is forbidden. **Decision needed: after profile
  retirement, who owns a tier's native render ceiling?** The shape that fits the
  per-adapter-ownership doctrine is that `eng_wan_ti2v` DECLARES its own tier ceiling
  (a capability-row field), the widget becomes an operator OVERRIDE with 0 meaning
  "use the adapter's own contract", and the profile channel stops mattering. That is
  a real design change with a live-behaviour blast radius on any card with headroom
  (the VRAM predictor currently gets to ask for more than 17 and often can), so it is
  NOT being written on assumption. Ratify the shape first.

  **Also open, all PROOF obligations rather than build work:** the 18-engine GPU
  campaign is engine COVERAGE, not an 8-GB qualification; and a render on a PHYSICAL
  8 GB card is still owed -- the four-arm bench PREQUALIFIED on a 16 GB card told to
  reserve 8 GiB, which is not the same claim.

  **One untested edge, cheap to close whenever this reopens:** WAN is out of
  `PLANNING_CAP_ENGINES`, so a tier ceiling and a multi-clip plan CAN contradict by
  design, and `_planned_length` hard-refuses mid-episode when they do -- but no test
  asserts a 17-frame tier survives a multi-segment beat. `:216`/`:234` in
  `test_multiclip_effective_contract.py` pin the topology, not that outcome.

- **THE 8 GB PROFILE FAMILY CANNOT RUN ITS OWN WRITER** (found 2026-07-27;
  LIVE-REPRODUCED TWICE on two different banks, then confirmed by a two-strikes
  kibitz panel -- codex `gpt-5.6-sol` high and agy independently reached the same
  diagnosis). `config/profiles/otr_8gb_ltx.json` pairs a 12B GGUF writer
  (`gemma-4-12b-it-Q4_K_M`, 6.63 GB of weights) with `llm.gguf_n_ctx: 2048` under a
  declared `vram_ceiling_gb: 6.8`. The pipeline's own smallest prompt needs **2064
  input tokens** and P0 reserves 2800 output (`_P0_BASE_OUTPUT_TOKENS`), so the leg
  dies in `OTR_LedgerScriptWriter` before any render. Live preflight, verbatim:
  `Needed=8.13 GB (weights=6.63, kv=1.40 @ n_ctx=2048)`. **ctx is the SYMPTOM; the
  writer MODEL is the cause** -- 4096 puts it near 9.4-9.5 GB, OOM on the very card
  the tier exists for. Every 2048-ctx profile (`otr_8gb_ltx`, `otr_8gb_wan`,
  `8gb_lite`, `cpu_floor`, `otr_amd8_rocm`, `otr_cloud_lanes`) is `status=draft` and
  every one pairs 2048 with the 12B; the only `status=shipping` profile is
  `16gb_full` (4096 + Mistral-Nemo). **NOT a one-line profile edit:** the GGUF
  registry ships exactly two rows (`unsloth/gemma-4-12b-it-GGUF`,
  `unsloth/Qwen3-8B-GGUF`); `google/gemma-2-2b-it` is in the TRANSFORMERS catalog, a
  different lane -- agy proposed it and was wrong, recorded so nobody re-derives it.
  **Largely mooted by profile retirement:** with no profile passed, the canonical
  JSON's own `gguf_n_ctx=4096` / Q8_0 binds and the leg runs. **Fix the profiles or
  finish retiring them; do not leave both.**

- **A2 -- HELD pending the profile retire-now vs retire-later scope. The profile's
  `llm` section silently overrides the canonical JSON, and the applied-overrides echo
  HIDES it.** Held because its entire subject is `apply_profile_to_workflow` and the
  printed echo -- a channel directed to be retired, so building on it now may be work
  on something scheduled for deletion. The fix SHAPE is correct and ready when the
  scope is settled. The profile's `llm.*` values win over the widgets the operator set
  in `otr_canonical.json` (which ships `creative`/`technical` = `google/gemma-4-12b-it`,
  `gguf_n_ctx=4096`, `gguf_quant=Q8_0`, `llm_vram_ceiling_gb=14.5`), while
  `scripts/otr_api.py:817` flattens only `role_overrides` / `slot_overrides` /
  `features` + two `seed_policy` keys for the printed summary -- so the run reports
  "16 overrides" while ALSO having replaced the entire LLM configuration. **Causal
  chain corrected** (triage 2026-07-27, codex; grounded): the override does NOT come
  from the validator's `OTR_ACTIVE_PROFILE` export -- it happens at submission,
  `scripts/otr_canonical_api_run.py:157` -> `apply_profile_to_workflow`; and the real
  applier (`nodes/_otr_workflow_apply.py:492-540`) ALREADY flattens `llm`; only the
  printed echo is stale. **Fix: generate the echo FROM the applier's flattened map.**
  Adding `llm` to the echo by hand leaves the next drift intact.

- **The `ltx_8gb` render-length ceiling has TWO owners that only agree by
  coincidence** (found 2026-07-27, B6 panel, two lenses independently). The coverage
  PLANNER reads `config/profiles/otr_8gb_ltx.json` `video.max_render_frames`, and
  `ltx_8gb` is the sole member of `PLANNING_CAP_ENGINES`. The ADAPTER's own
  pre-render refusal reads `OTR_LTX_8GB_MAX_FRAMES`. Today both land on 161 (profile
  unpinned, env unset), so nothing breaks. But `workflows/variants/otr_8gb_ltx.env.json`
  ships `OTR_LTX_8GB_MAX_FRAMES=97` and NOTHING currently reads that file. The day a
  launcher honours it without also pinning the profile, the planner emits a 98-161
  frame segment and the adapter refuses it MID-EPISODE -- after the stills are minted
  and, on a multi-segment beat, after the 6.34 GiB checkpoint is hoisted.
  **Deliberately NOT fixed in B6:** pinning the profile to 97 changes how a 237-frame
  beat partitions, which is a production planning decision, not a cleanup. The preset
  carries a `_ceiling_note` saying do not export it alone. Compare WAN, which B3
  wired correctly: `otr_8gb_wan.json` sets BOTH `launch.env.OTR_WAN_TI2V_MAX_FRAMES`
  and `video.max_render_frames`.

### Coverage, canvas and clip-contract

- **The route lock is ONE NODE TOO LATE for the image phase** (found 2026-07-25, node
  order confirmed against the canonical JSON: `87 VideoDirector -> 88 ImageDirector ->
  89 MetaBrief -> 90 ShotLock -> 91 ImageGenDispatcher -> 92 VideoRenderBatch`).
  `resolve_final_shot_engines` runs at node 92, but stills are minted at 91 and image
  PROMPTS at 89. The landed fix closed the spine-validation gap; the image phase still
  relies on its own MIRROR (`otr_meta_brief_image_prompt._effective_prompt_engine_for_role`,
  whose docstring says it "mirrors the image dispatcher's effective-engine seam").
  **Chunk 1 of the coverage block is the fix.** Note node 89 precedes node 90, so
  hoisting to ShotLock still does not put MetaBrief downstream of the authority --
  that needs a VideoDirector-time freeze and is NOT in scope. (This is also the
  "image-phase still ownership" item from the campaign queue.)
- **THREE silent coverage mechanisms exist, not one** (found 2026-07-25).
  Mechanism 1, the engine mirror/ping-pong (`wrapper_bridge.extend_frames_to_target`),
  is GONE from `eng_ltx_8gb` -- pinned behaviourally by a test that detonates the
  helper and renders successfully. It REMAINS in `eng_wan_ti2v`, deliberately and
  permanently: WAN renders a short native clip on purpose and fills the beat with it,
  which is the shipped 8GB tier contract `PBUG-20260723-02` protects. **Still open:**
  composite loop-fill (`otr_silent_composite._should_loop_fill`, which also SUPPRESSES
  its own underrun warning once it activates) and held-last-frame. For `ltx_8gb` the
  composite path is now de facto unreachable -- the adapter returns exactly the
  requested count or raises -- but not structurally impossible:
  `encode_frames_to_silent_mp4` reports the size of the array it piped into ffmpeg
  rather than re-probing what ffmpeg wrote, so an encode-side drop could still
  under-report. PRE-EXISTING; close it when the assembly boundary is next opened.
- **`_should_loop_fill` names the permanent fix and it is now being built**
  (`otr_silent_composite.py:244-266`): *"The real fix is phrase-chunking (render the
  beat's correct duration so it never underruns) -- tracked as a follow-up."* The
  coverage block IS that follow-up.
- **THE 7d-PREFLIGHT THAT "PROVED THE GPU" RAN AT THE WRONG CANVAS** (found
  2026-07-27, B5 panel; verified -- and it corrects a claim this file once made).
  `render_single` and both HTTP entry points use the older ledger-free `build_request`,
  which never reaches the canvas seam and defaults to `OTR_VIDEO_RENDER_CANVAS`
  (832x480). So the "GPU IS PROVEN" leg (`ltx_8gb`, 25 frames, 3004 MB) exercised
  832x480, not the production canvas. `render_single` parity is explicitly deferred by
  the O1 judgment; what must NOT happen is another "proof" through that harness being
  read as a production proof. (The 512x288 canvas itself HAS since rendered live --
  bench arm D, three cells -- but through a DIRECT-NODE graph, which proves the canvas
  and the recipe, not the seam.)
- **The ShotLock WRITE-side canvas validation is still owed** (O1 judgment item 1).
  `otr_shot_lock.py` stamps `video.canonical_canvas` unvalidated from a possibly-empty
  policy. B5 made this non-load-bearing for the render (the engine declares its own
  canvas now), so it is no longer urgent -- the drift guard in
  `tests/test_ltx_8gb_canonical_canvas.py` covers the disagreement that matters. Close
  it when the general canvas resolver lands.
- **Odd-canvas evenness is validated at the ENCODER, not where the canvas is chosen.**
  The stride defect itself is closed (`b1f2ee86`): `ffmpeg_silent_mp4_cmd` declares the
  REAL width/height and `encode_frames_to_silent_mp4` REFUSES an odd canvas by name,
  because yuv420p subsamples chroma 2x2 and cannot represent an odd dimension. Still
  true and NOT fixed: neither `WanInitImageMixin._dims()` nor the `Canvas` schema
  validates evenness, so an odd canvas is caught late rather than at the choice. No
  live producer builds one today (832x480, 512x288, 1472x832 are all even).
- **`CanonicalClip.frame_count` -- "the integer timing authority" -- HALF CLOSED**
  (`58e288af` + `40780b82`, count closed @ `48e3c6fb` without paying a decode). Every
  module that writes a clip now ffprobes it, and a roster gate in
  `tests/test_terminal_frame.py` fails by name for any module that writes a clip
  without proving it. What this proves and what it does not: it proves the muxer wrote
  what it was piped, which is the right question for a clip written by ONE ffmpeg
  pass; it does NOT prove decodability, which is why `assemble_beat_segments` still
  decode-counts every ASSEMBLED beat and must keep doing so. **Re-verify before
  acting:** this row's "still self-declared elsewhere" pointers (the four `viz_*` and
  four `still_*` engines) were both closed afterwards, so the remaining open surface
  may be empty.
- **KNOWN LIMIT of the widened roster gate**, recorded so it is not rediscovered as a
  surprise: the codec flag is matched as a STRING CONSTANT, so a flag assembled at
  runtime (an f-string, `"-c:%s" % stream`) or the stream-index spelling `-c:0` is
  invisible to the sweep. Nothing in the tree does that today; an encoder that ever
  needs to must be pinned in `_ENTRY_POINT_PROOFS` by hand, which the inventory test
  makes a visible decision. Separately, ONE mutant survives the round by construction:
  deleting the self-proving membership assertion is catchable only by a meta-test of
  that assertion.
- **`ltx_av` underruns long beats** (found 2026-07-25, codex; confirmed). It caps at
  `_LTX_AV_MAX_FRAMES` (`eng_ltx_av.py:58`, default 497, env-overridable) and clamps
  at `:950-953`. It is NOT "renders to target natively" as three earlier docs claimed.
- **Ping-pong on a capped HuMo beat played lip sync BACKWARDS** -- FIXED in code @
  `a1d810f1`, but the finding is STATIC (no live artifact), so it is NOT a PBUG row. A
  capped-14B leg would reproduce it. Kept here so the live proof is not forgotten.
- **`docs/ENGINE_MATRIX.md` reports the DECLARED contract only** (found 2026-07-27).
  Correct today and consistent with its own stated design (every number read from the
  live registry). But the moment a profile pins an `ltx_8gb` ceiling, the matrix keeps
  printing `9-161 step 8` for a tier whose real window is narrower, and the `--check`
  drift gate cannot notice because it diffs the registry, which the effective contract
  never touches. Owed at the prequalification step, not before.

### Routing, env-capture and the credits card

- **`wants_talking_prompt()` escapes any routing freeze.** It calls
  `_recipe_config(self._recipe())` and `_recipe()` (`eng_ltx_av.py:402-432`) re-reads
  `OTR_LTX_AV_RECIPE` / `OTR_LTX_AV_SHARP` / the UNET name on EVERY call by documented
  design ("Read fresh every call"). So a `required="when_engine_talking"` row evaluated
  through the hook re-reads the environment after capture. S0b-core needs ONE shared
  `row_is_active(...)` evaluator over captured state, with the talking result inside
  `ltx_resolved`.
- **`provider_side` is a THREE-part rule, not an attribute.** `_is_cloud_video_engine`
  accepts a `cloud_` id prefix OR the attribute OR `node_key.startswith("cloud_")`.
  `cloud_kling_avatar` has no `provider_side` attribute and is caught by the id prefix
  alone, so an `engine_facts` builder using a bare `getattr` would classify it local
  and let the radio-host redirect send a cloud avatar to local LTX. Needs a regression
  on picked AND forced `cloud_kling_avatar`.
- **Four env-read sites missing from the S0b inventory:** `eng_ltx_video.py:541-564`
  (`OTR_ENABLE_LTX_I2V`), `render_driver.py:1176-1203` and
  `otr_meta_brief_image_prompt.py:297-300` (`OTR_ENABLE_HUMO_HOSTS`), and
  `eng_ltx_av.py:352-353,403-432` (recipe/UNET re-read outside `assert_usable`).
- **The credits card needs a SMALL-CANVAS VARIANT, and the ladder is not it.** At
  512x288 (the ltx_8gb tier) col1 is 65px past its footer even with every ledger row
  this policy may drop already dropped; at 640x360 it is 12px over. Both are drawn
  anyway (a terminal node never destroys a finished episode) and LOGGED at ERROR
  naming the canvas -- the old behaviour was drawn, clipped by PIL, silent. At 288
  lines the three-column console is already a polite fiction: col3's scrolling
  transcript is as unreadable as anything col1 clips. This is a DESIGN job -- a card
  laid out for a small canvas -- not more ladder heroics.

### Test-harness and tooling

- **The B7 forbidden sweep cannot see an UNTRACKED file, so a new test file passes the
  gate and fails one commit later.** `tests/test_b7_forbidden_sweep.py` builds its
  input from `git diff s29-clean-slate-gate -- *.py`, which covers tracked files only.
  A new test file added and gated in the same session is green; the moment it is
  committed it enters the diff, and a forbidden runtime identifier in it turns HEAD red
  with nothing else changed. Cost one red HEAD. **Not fixed, because the fix is a
  judgment call:** sweeping the working tree instead of the diff would widen the gate
  to every untouched file in the repo. Cheap mitigation until then -- re-run the full
  suite once after the FIRST commit of any new test file.
- **NOTED, not a defect: two `scripts/` bake-off runners now abort a whole sweep on a
  count mismatch.** `scripts/run_ltx_av_q_bakeoff.py:453` and
  `scripts/run_humo_bakeoff.py:660` call the encoder inside per-leg loops with no
  try/except and DISCARD its return value (both set `result["frame_count"]` from
  `int(frames.shape[0])` independently). A disagreement that was previously invisible
  there is now fatal to the run. That is the correct direction -- a lying count is not
  a leg worth finishing -- but a sweep operator should know it before an overnight run.
- **LATENT, not reachable today: the fewest-segments partitioner can accept a
  disproportionate trim on a WIDE discrete menu.** WIRE-W1 makes `partition_beat` take
  the lowest segment count that covers, including via a permitted tail trim. On a
  ladder that is always the right trade; on a DISCRETE menu whose largest entry dwarfs
  its smallest it need not be -- covering 1019 frames from a `(10, 999)` menu, two
  segments give `[999, 999]` and discard 979 frames where three give `[999, 10, 10]`
  exactly. **A bound was written, MEASURED and REVERTED, and the measurement is the
  point:** rejecting a trim of a whole smallest-clip turned `[12, 12]` into
  `[12, 4, 4]` on a `min=4 max=12 quantum=8` ladder -- a third render and a third model
  load to recover four frames -- across 4,885 cases in the sweep grid. The widest
  shipped menus are Veo's `(100, 150, 200)` and Pixverse's `(125, 200)`, whose worst
  real trim is 25 frames. Revisit only if an adapter declares a menu with an extreme
  ratio; the reasoning is recorded in `coverage_plan.partition_beat` so the next reader
  does not re-derive the bound and re-ship the regression.

### PARKED -- unverified at HEAD, re-observe on the next real render legs
(The 2026-07-24 "after SFX" checkpoint is VOID -- SFX is parked. The re-observe
now rides whatever real render legs come next, D2 included.)

Both were eyeball observations against a story engine that has since had its LLM
vetoes ripped, THE LAW imposed, six banks renamed onto new packs, word-fit ceilings
retired, the repair-first plan landed, and a ledger cleanup pass added. Neither has a
reproduction at current HEAD, and under the standing rule a finding with no
reproduction is not a row. **Do NOT schedule coder time against either.** They are
settled by the operator eyeballing a real render leg after SFX: still there -> re-admit
as a FRESH dated row with that leg as evidence; gone -> the LAW-era work already fixed
it, tombstone it.

- **Announcer framing defect** (`docs/2026-07-11-announcer-framing-defect.md`).
  Episodes START a story instead of admitting you into one; the announcer takes debate
  turns instead of framing. Operator eyeball 2026-07-11. If it survives re-observation
  the fix is still seam + score contract + fail-closed validator, never Python
  authorship.
- **Name-splice defect #2.** v4-campaign Phase 0 record in HANDOFF_LOG; its timebox
  predates THE LAW.

### Carried administrative rows

- **PBUG-20260710-07** -- root fix shipped; stays ROOT-OPEN in the log until ratified
  at the next operator fan-out (green codex leg `c1f3891f` is the retire candidate).
- **Phase-2 de-naming** (module filenames, `meta[]` ledger keys, wire-schema `.v4`
  literals) -- DEFERRED, operator-flagged, from the keep-6 rename.

## KNOWN OPEN -- do not rediscover these

* The VRAM admission guard covers coverage-executed beats only; the single-clip path
  returns via `render_shot()` first, and `ltx_audio_in` is not in
  `PLANNING_CAP_ENGINES` -- so the hottest-peaking engine is unguarded.
* `FRAME_COST_MODEL` is keyed by engine NAME while recipe/quant/LoRA/reserve are
  env-configurable; a measured row needs a calibration IDENTITY.
* Four adapters still cite missing receipts (`ltx_audio_in`, `mesh_stage`, `viz_green`,
  `viz_mxc_mandala`).
* The HuMo lip-sync onset fix is SPECIFIED but unbuilt, blocked on M1 classification
  (`BUG_BIBLE.yaml:2343`: audio leads the lips by 100-200 ms with the face static for
  the first 3-6 frames). Pre-roll + equal trim is algebraically a NO-OP if the lag is
  constant rather than onset-only, so the classification must come first: early-only ->
  pre-roll fix; constant -> advance the 25 Hz conditioning features; growing -> a
  rate/timestamp bug, not a pad. Run a matched no-LoRA control -- Kijai reports the
  lightx2v distill is not fully HuMo-compatible, so the defect may be ours.
* Cap authority is not yet collapsed to one (`video.max_render_frames` should be sole;
  env twins must be absent-or-equal).
* `otr_w45_campaign.py` runs SIX engines while claiming all local ones, and its
  acceptance would not reject a mirror. Fix before trusting a campaign result.
* The reuse detector cannot separate a deliberately quiet shot from a duplicated frame;
  it is ADVISORY in `otr_w45_campaign.py` until that is solved. The engine-layer and
  composite guards (`MirrorExtensionForbidden`, `ClipUnderrunsItsBeat`) are terminal
  and unaffected.
* `humo_1.7B` and `ltx_8gb` are marked CUDA-only with no fp8, no fp4 and no stated
  reason. Unexamined, not proven.
* M2's raw rows sit in swept `tmp/` with no pinned digest or config manifest.
* `docs/2026-08-02-IDEA-hardware-compatibility-matrix.md` -- captured, not scoped.
  Includes the Mac research: Metal has no `Float8_e4m3fn`, ComfyUI+MPS video is
  impractical (82 min for a 2-second clip), Draw Things and MLX are ~100x faster and DO
  support LTX-2.3 with joint audio, and the `viz_*`/`still_*` lanes need no GPU at all.
* Writer scaffolding repair increments 1-5 -- the spec needs its r3 CORRECTION
  before any code (NEXT CODING QUEUE item 3; the "code-ready" title is stale);
  the reuse detector to the panel; section 0A carve-out ruling before M2 numbers
  move caps; Wan 2.2 I2V checkpoint download + `wan_i2v` re-run; the
  `OTR_CastLock` freeze cascade (`wan_ti2v`).

## MODEL & CREDIT BUDGET (operator, 2026-07-24 -- read this EVERY window)

Every window states, in its first reply, which rung of this ladder it is on and why.
Pick the cheapest tool that can win; escalate only when the cheaper rung cannot decide.
Both pools reset weekly -- front-load heavy coder windows and big Codex spends early in
the credit week; late-week, drop to the $0 rungs instead of grinding a paid pool dry.

| Rung | Model / tool | Cost | Use for | Never for |
|---:|---|---|---|---|

Production (in-pipeline, all $0/local, offline-first): writers = Mistral-Nemo (ctx cap
16384) + `gemma-4-12b` (saved runtime-qualified local default); stills/video-init =
`z_image_turbo` (the Qwen-Image ENGINE is removed -- keep Qwen3/Qwen2.5 LLM support and
Z-Image's `CLIPLoader(type="qwen_image")` encoder, unrelated). Cloud writers stay
opt-in bake-off arms, never the default.

Per-window mapping: RENDER windows = local production models + the Codex-app monitor,
Claude only to launch and wrap. CODER windows = Claude codes, rung-1 Qwen triages every
failure first, and the full kibitz panel is now mandatory per item (see the ON DECK
gate). PLANNER = Claude + the local panel.

## THE LAW (operator, 2026-07-22 -- supersedes anything that disagrees)

> **AN AUDIT MAY IMPROVE A STORY. IT MAY NEVER FAIL ONE FOR LENGTH, LANGUAGE, STYLE,
> VISUAL VOCABULARY, OR QUALITY.**

The sole terminal spoken-prose policy is the shared whole-word safety authority.
Structural JSON/schema/IDs/roster/source-proof/rights/graph/markup/nonempty/
provider-integrity failures remain fail-closed because they protect a usable ledger
rather than judge prose. Across all six banks, requested word length, actual word
count, drift, one-breath estimates, visual/world vocabulary, noun/POS heuristics,
casing/title/honorific style, craft, and quality are guidance or telemetry only -- they
may never reject, reroll, retire, replace, or block an episode. Same-story LLM cleanup
is allowed.

**SUPERSEDED IN PART (operator directive 2026-08-03, `CLAUDE.md`):** the
"whole-word safety authority" above is NO LONGER TERMINAL for episode content
-- no profanity or violence filtering on the generation path, and the source's
own language is carried as written on the adaptation lanes. The paragraph
above is kept as the 07-22 record; its STRUCTURAL half (schema / IDs / roster
/ source-proof / rights / graph / markup / nonempty / provider-integrity
fail-closed) still stands in full. The runtime filters that survived the 08-03
rip are inventoried and queued for removal as ON DECK item 5.

## Standing operator directives (hard)

* **The recipes are not on the table.** "We spent a lot of time perfecting the recipes
  to look good and we can't lose that." No VRAM, speed or cap finding justifies a recipe
  change; measurement runs the SHIPPED recipe unchanged. This specifically forbids
  reading "peak falls as frames rise" as a reason to raise the 97 trained-length cap,
  and makes the deferred no-LoRA HuMo control a recipe change rather than a control.
* **Per-segment rendering is BY DESIGN** -- "each audio clip takes its own journey, to
  keep VRAM low." Never classify an assembled beat as one render.
* **One coder window in the code at a time**, serialized through this file. Two windows
  editing the same file -- especially the workflow JSON -- is how it gets corrupted.
* The remaining hard rules (root-cause fixes, no content guardrails on generated
  episodes, no word-count chasing, the full-kibitz gate, the ledger-completeness rule
  for any ripped LLM pass, git policy) live in `CLAUDE.md` and are not duplicated here.

## Window packing (credit discipline -- one line starts any window)

Starting any window costs the same boot context, so BATCH chunks per window and never
open one for a single small item. Every window starts by pasting its one-line kickoff --
the `otr-handoff` skill reads this file + git and states the current step. No manual
context handoff, ever. This planner window keeps GO_FORWARD + HANDOFF_LOG current;
coder windows never write plans.

| Window | Scope | Rung | Gate | Size |
|---|---|---|---|---|
| **CODER (open slot)** | The ON DECK queue at the top of this file, in order, one green pushed chunk at a time, each with its full kibitz | Claude codes + judges; kibitz = codex + agy | none -- work it now | ~2h40m + panels |
| RENDER | RESUMED 2026-08-05 (the 08-04 "no runs" line was that session only). Queue: D2 first, then the WAN/8-GB proof obligations. Reset per CLAUDE.md section 4 before every leg | local production + Codex-app monitor | operator asks for a live leg | GPU days |
| PLANNER | Bug Bible operator fan-out + the `check_compatibility` fork; plan upkeep | rungs 2-4 | parallel with any coder window | docs |

**NEVER boot a window by letter.** Boot by the ON DECK section, always:

> resume the OTR build -- you are a CODER window. Read GO_FORWARD "ON DECK" and execute
> THAT queue only, in its stated order, one green pushed chunk at a time, each with its
> full `kibitz-plugin:kibitz` review per the HARD GATE. State your MODEL & CREDIT BUDGET
> rung first.

### If the window is a REMOTE / cloud Cowork session -- READ THIS FIRST

Learned the hard way 2026-07-26. A Cowork session running IN THE CLOUD is not the same
box as the repo, and two of CLAUDE.md's assumptions do not hold:

- **Read/Write/Edit hit the CONTAINER, not the Windows files.** In a remote window every
  read, edit and write goes through Desktop Commander against
  `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio`, and so does git,
  the venv python and pytest. Everything else in CLAUDE.md holds.
- **There is a LAGGING Linux snapshot at `/mnt/user-data/uploads/`.** Never read the repo
  through it, and say so explicitly to every subagent -- a prior session's agents
  reported phantom corruption from that mount.
- **The bridge can drop mid-edit.** If the remote-device tools vanish, STOP -- do not
  retry in a loop and do not leave a half-applied edit. Report what is on disk,
  uncommitted, and wait. Nothing was lost when this happened because the last green chunk
  was already pushed, which is the actual argument for pushing every chunk.
- **The 60s MCP call ceiling.** The full suite takes minutes, so launch it detached
  writing to a `tmp/` log + a `.done` marker, then poll. PowerShell `*>` redirection
  writes UTF-16, so read results with `Select-String`.

## Parallel lane -- no coder slot required

- **Bug Bible operator fan-out** -- 9+ closed candidates + the duplicate-legacy_id
  cleanup waiting on one fan-out session.
- **Render-window fillers:** cpu-tier smoke (needs the google image lane or stills) +
  nv50 re-soak -- the two open portability remainders; release QA validation time, not
  coding.
- **SFX: RETIRED and RIPPED (operator ruling 2026-08-06, "rip out SFX 100%";
  executed `9eb6ede1` per `docs/2026-08-06-BUILD-SPEC-rip-sfx.md`).** The five
  bed engines are deregistered and barred via `RETIRED_ENGINE_IDS`, the bed
  compiler and mux mix branch are deleted, and
  `tests/test_rip_sfx_bed_guard.py` trips on any surface creeping back.
  Reviving SFX is a NEW design against the post-rip tree; the old design docs
  in `ROADMAP.md` are the historical record only.

## Bug Bible promotion field -- pending actions only

| Record | Pending action |
|---|---|
| `PBUG-20260712-22/23/24/25` | Live reverify -- blocked by the `scifi_news` P0 convergence defect, then fan-out |
| `PBUG-20260712-18/19/26` + `PBUG-20260713-15..18` + `-20` | Awaiting the next operator Bible fan-out (overlap check + approval) |
| `PBUG-20260713-19` | Live requalification pending (promoted BUG-05.11) |
| duplicate-id cleanup | Same fan-out: BUG-11.54 legacy_id -> `PBUG-20260713-21`; verify the acronym-union rule's legacy_id (both Bible rows cite `-10`; see the log's renumber note) |
| historical `PBUG-20260711-18` | Keep as a standing context/cap engineering risk; never eligible from static evidence |
| `PBUG-20260710-07` | Ratify retirement at the next fan-out (green codex leg `c1f3891f`) |

The active production-fix owner updates `docs/PROD_BUG_LOG.md`; the approval queue is
`docs/BUG_BIBLE_PROMOTION_QUEUE.md`; no plan review or invented fixture creates a row.

## Validation and handoff law

- **Current whole-tree receipt (2026-08-07 @ `2fc81f72`):** full Windows suite
  **9081 passed / 111 skipped / 1 xfailed** (~4:19); Bug Bible **17 passed**. Prior
  receipts live in `docs/HANDOFF_LOG.md` -- this file keeps only the current one.
- **Standing acceptance receipt:**
  `python scripts/audit_voice_gender_consistency.py --root "C:\Users\jeffr\Documents\ComfyUI\output\otr"`
  -- expect exit 0 over 1,595 ledgers. Exit 2 means the scan did not FINISH and its
  verdict is not a pass.
- Every code chunk: focused tests, full Windows suite, Bug Bible, AST/JSON/BOM/zero-byte
  checks, commit, push, verify `HEAD == origin/v2.0-alpha`.
- Every node/widget/link/schema change edits `workflows/otr_canonical.json` in the same
  commit and runs `OTR_WorkflowValidator`, JSON round-trip, strict link/input, live
  widget-vector, and generated-variant audits.
- Reset selectively before every headless run; never blanket-kill Python. Every run loads
  the canonical workflow and writes directly to canonical episode/OBS paths. Asset
  existence, not resident VRAM, proves completion.
- One coder edits code or `workflows/otr_canonical.json` at a time; read-only audits and
  documentation may run in parallel. HANDOFF_LOG + this file are the only tracking
  surfaces.
- **Count the suite on a clean tree:** `build_variants.py --all` also emits variants for
  any UNTRACKED profile on disk, and some profile checks are parametrized over the
  variants present, so another window's scratch profiles can inflate the number by a
  dozen tests that would not reproduce on a fresh clone.

## Open risks

- **NO CLIENT BANK HAS EVER RUN LIVE.** Every extensibility wave is proven by the suite
  and by contract tests, and the first real client bundle is still an unproven path end
  to end (fetch -> interpret -> writer -> cleanup -> tail -> publish). Treat the first
  live client-bank leg as a qualification, not a formality. Deferred power-user tiers
  (client own-runner + staging, dependency manifest, standalone story_rules) are
  explicitly OUT of v1 and are a NEW block if the operator ever wants them.
- **CLIENT-AUTHORED PYTHON executes in-process** (wave 3). The posture that must hold in
  every future change: `--activate` is the consent act; the seam fails LOUD
  (`UserBankExecutionError`) and never substitutes; client code never touches the
  canonical ledger; owner IDENTITY is verified so a bank can only run its OWN bundle; the
  shipped fetcher/interpreter registries are never widened to admit a client id. Do not
  relax any of these for convenience.
- **The client-facing surface is LIVE TEXT, not just docs:** the `custom_source_bank`
  row's `guide_ref` is raised to the operator by `require_runnable_bank`, and the
  `source_bank` tooltip repeats it. Any future change to the activation path (folder
  name, CLI verb, restart behaviour) must update `nodes/story_packs/banks.json`, that
  tooltip and `docs/EXTENDING_OTR.md` together, or the product will confidently instruct
  clients to do the wrong thing.
- **`check_compatibility` is RESERVED, not wired** -- operator/planner decision flagged,
  with a 2-of-2 recommendation to RIP (codex and Fable independently, Claude grounded
  both). The argument that decided it: the "it reserves the name" benefit is FALSE --
  `BUNDLE_ENTRY_ATTRS` constrains what OTR-side code may request from
  `bundle_entry_point()`, it reserves nothing against clients, and activation provably
  ignores whatever a client puts under that name (`tests/test_otr_check_cli.py:335`
  asserts a bundle whose `check_compatibility` is a plain integer activates). The only
  artifact that reserves the name is the `EXTENDING_OTR.md` paragraph, which exists
  either way. Case AGAINST: churn on landed green code for zero behaviour change, and the
  plan of record already names the future consumer (randomizer eligibility). Blast radius
  if ripped: ~5 code sites, 2 test files, 3 docs; no workflow JSON, no routing, no
  source-payload consumer. **Not a coder chunk** -- either ratify the inert constant or
  schedule the rip as a planner chunk. Proposed doctrine line: a name published to
  clients before its consumer exists lives in the client-facing DOC as "reserved, no
  contract, ignored if defined" and nowhere in executable code, because code that names
  an interface is read as enforcing it.
- **The ledger-cleanup pass runs on EVERY bank, not just client banks** (`3d97a130`). It
  is a no-op on a complete ledger and costs no LLM call there, but two shipped-lane
  behaviours did change and are worth watching on the next live legs: (a) unsafe spoken
  language on a `content_owned_readonly` bank is now REPAIRED at the writer tail instead
  of reaching G9 untouched, so a leg that used to die at freeze may now ship a sanitized
  line; (b) a blank `meta.episode_title` is now filled at the tail instead of exploding
  later in `otr_credits_roll`. Both are the intended direction under THE LAW; neither has
  a live receipt yet.
- No code lands mid-sweep of an active qualification campaign (the 420-rung
  uniform-code-confound lesson).
- `dynamic_story` touches the writer, the visual-style authority and the canonical
  workflow; it re-derives the live JSON at build. It is the only claimant on those
  surfaces.
- Generated-SFX R4 stays local/ignored evidence of a RETIRED campaign (the
  2026-08-06 rip, `9eb6ede1`); no R4.1 refit exists to run, and reviving SFX
  is a new design against the post-rip tree.
- Lean-mean front/tail drift: the constraint holds wherever it runs -- the tail's SW-1
  re-survey is mandatory against the then-current writer, and the two campaigns never
  share a window.

## Tombstones -- the only three a window might wrongly revive

Full list in `docs/HANDOFF_LOG.md` + `docs/PROD_BUG_LOG.md`. These three are
here because each has been re-proposed at least once:

* **The 20 fabricated-fixture `public_domain` episodes and the fixture itself** --
  operator ruling 2026-08-04: dropped and deleted, **never raise again**.
* **v4 improvement campaign banks #2-#5** -- PARKED, superseded by the keep-6
  rename + THE LAW. Revive only by operator decision
  (`docs/2026-07-17-v4-campaign/final.md`).
* **LEAN-MEAN** -- lives in `ROADMAP.md`, not this file. A window that wants to
  rip dead code is on the wrong document.

## Pointers

- `CLAUDE.md` -- hard operator rules; wins over this file wherever they disagree
- `ROADMAP.md` (later runway: product expansion -> lean-mean -> RunPod -> release; SFX RETIRED + ripped 2026-08-06)
- `docs/HANDOFF_LOG.md` (all completed-work history, newest at top)
- `docs/PRODUCTION_SPRINT_LESSONS.md` (incl. lesson 11 pointer-not-proof; 24 lost-anchor; 25 bank-teardown)
- `docs/SOURCE_BANK_PREFLIGHT.md` -- add-a-bank gate + the Teardown protocol
- `docs/PROD_BUG_LOG.md` / `docs/BUG_BIBLE_PROMOTION_QUEUE.md`
- `docs/2026-08-04-POSTMORTEM-still-unmaterialized-320w.md` / `docs/2026-08-04-D1-SHIPPED-still-skip-evidence.md`
- `docs/2026-08-03-fidelity-pass-ownership.md` (the ownership table the verbatim executor is gated on)
- `docs/2026-08-03-script-parse-repair-CODE-READY.md` (writer scaffolding repair increments 1-5)
- `docs/2026-08-03-public-domain-plays-PLAN.md` (v2.1, researched, nothing built)
- `docs/2026-07-31-four-arm-clamped-video-bench-SPEC.md` (the isolated-bench carve-out)
- `docs/2026-07-24-independent-source-banks-v1-plan.md` / `docs/EXTENDING_OTR.md`
- `docs/2026-07-12-randomizer-rolls-r2-coding-plan.md` / `docs/2026-07-12-dynamic-story-visual-scope.md`
- `docs/2026-07-11-announcer-framing-defect.md` (PARKED) / `docs/2026-07-11-timeline-cue-ledger.md`
- `docs/2026-07-17-model-bakeoff-scoreboard.md` (writer-model verdict)
- `workflows/otr_canonical.json` (the workflow source of truth)



---

## PARKED (operator ruling 2026-08-12): wire character casting to the VOICE REFERENCE BANK

**Status: PARKED, not rejected.** Operator: *"park it on go forward."* Raised
after the operator observed we should have far more voices than the writer is
being offered. He was right, by a wide margin.

### The finding, measured live

| pool | count | what it serves |
|------|-------|----------------|
| Bark `VOICE_PROFILES` (`config/cast_pools.py`) | **10** (6M/4F) | what the writer's casting menu offers |
| Kokoro presets on disk | 4 | ANNOUNCER only, a separate namespace |
| **Voice reference bank** (`config/voice_reference_bank.json`) | **204 declared, 153 resolvable on disk** (97M/106F/1N) | IndexTTS2 / Dia / Chatterbox cloning |

`_otr_voice_bank.default_char_engine()` returns **`indextts2`**, promoted to the
shipped character-voice default on 2026-06-04. It is a zero-shot CLONING engine:
`requires_voice_ref = True`, `voice_ref_kind = "wav_path"`. Every reference clip
is a distinct voice.

So the writer casts from **10 Bark presets** while the engine that actually
speaks the characters draws from a **153-voice reference bank**.
`_otr_casting.py` states it outright -- *"Open-character voices are always drawn
from the Bark pool (VOICE_PROFILES in config/cast_pools.py), so the tts_model is
Bark by construction"* -- and `_assert_unique_bark_voices` enforces uniqueness
across those 10.

### Why this is parked rather than done

`MAX_SPEAKING_CAST = 10` was set from the Bark pool and is therefore a Bark
artifact, not a real ceiling. But raising the constant alone achieves NOTHING:
`_deal_voice_menu` builds the menu from `VOICE_PROFILES` and refuses with
*"voice stock capacity 10 < cast size N"*. The actual work is pointing the
casting menu at the reference bank when the character engine is a cloning
engine.

**That is why it is parked and not done unilaterally.** It changes what
`voice_preset` and `tts_model` MEAN on every cast row, and cast rows are ledger
JOIN KEYS -- `cast[].name` / `char_id` / `voice_preset` / `voice_ref_id` /
`voice_engine`, joined from `lines[].speaker` and `beats[].char_id`. Under the
operator's hard rule (*the writer must OBEY the ledger for downstream content; a
hole in the ledger is a broken render*), a change of that shape needs every
field's owner enumerated BEFORE the call is moved, not after.

### What the work is, when it is taken up

1. Enumerate every consumer of a cast row's voice fields -- casting, TTS
   dispatch, per-beat audio slicing, credits, portraits, captions, `obs_publish`
   -- and name the new owner of each field. Exactly one owner each.
2. Make the casting menu engine-aware: Bark presets when the character engine is
   Bark, reference-bank entries when it is a cloning engine. Gender and
   `commercial_clean` already exist on bank rows.
3. Replace `_assert_unique_bark_voices` with an engine-agnostic
   one-voice-per-character invariant. The rule itself is right and must survive:
   two characters sharing a voice is a correctness defect.
4. Derive `MAX_SPEAKING_CAST` from the ACTIVE engine's pool instead of a
   constant. `tests/test_cast_size_is_a_request.py` already asserts the constant
   matches the live stock, so it will report the drift rather than hide it.
5. Prove on `scifi_news_pro` (the only bank on the fable2 writer) with a cast
   larger than 10 and complete speaker-to-`char_id` equality in the ledger.

Related and already shipped: `num_characters` is now a REQUEST rather than a cap
(operator directive, all banks) -- see `tests/test_cast_size_is_a_request.py`.

---

## LEMMY IS PARKED AND GATED, 2026-08-12 (operator ruling -- CORRECTED)

The operator has a second window that originally coded Lemmy. When Lemmy is
taken up, it goes THERE -- it holds the design context for chunks A2-A4, and the
operator wants the Lemmy fixes well documented.

**But it is NOT handed off yet.** Operator, 2026-08-12: *"I'm not handing Lemmy
until your story fixes are done, all sweeps."* So this is a QUEUE, not a
parallel split -- which is also the stricter reading of CLAUDE.md's one-coder-
window rule, and avoids the concurrent-git hazard described at the end of this
section entirely.

**THE GATE, both parts:**
1. the story-writer fixes are done (see "THE FABLE2 WRITER" above), and
2. **all sweeps are done** -- the 21-lane 45-word render gate, both boots.

Until both are met, the Lemmy window stays parked and this window is the only
one in the code. The work list below is kept ready so the handoff costs nothing
when the gate opens.

### WINDOW A -- LEMMY (the window that coded it)

Owns: `nodes/scene_sequencer.py`, `nodes/_otr_audio_engines/**`,
`nodes/_otr_voice_bank.py`, `config/voice_reference_bank.json`, and the Lemmy
tests.

Work list, already triaged in "WHAT REMAINS ON THE LEMMY SPRINT" above:

* **Chunk C items 2/3/5** -- SceneSequencer integration coverage at 22050/44100.
  Every existing fixture starts AT the 48000 bus, so nothing proves the real
  resample path. Item 1 is done.
* **Chunks A2 -> A3 -> A4** -- v2 identity + replay bridge. Operator ruled
  2026-08-11: **AUTO-PROMOTE on a clean replay** -- if A4 reproduces all six
  frozen clip hashes, A2/A3 rewrite the receipt's identity fields with no second
  sign-off.
* **A forced-Lemmy live render** to settle chunk B. Chunk B (the shared
  `resolve_voice_ref_path`) is written and tested but was deliberately NOT logged
  as a PBUG, because it is grounded in the filesystem rather than in a captured
  live failure. A forced-Lemmy render settles it either way and is the natural
  next Lemmy step.
* **Chunk E is OPERATOR ONLY** -- whether swapping Lemmy's voice counts as an
  editorial recast for an audience that already heard the old one.
* Branch B stays unbuilt; it existed for a G1 failure and G1 passed.

**Operator directive 2026-08-12: the Lemmy fixes must be WELL DOCUMENTED.**

### WINDOW B -- STORY WRITER + THE 21-LANE SWEEP (this window)

Owns: `nodes/_otr_scifi_fable2.py`, `nodes/_otr_fable2_markup.py`,
`nodes/_otr_outline.py`, `nodes/story_packs/**`, `nodes/_otr_ledger.py`,
`scripts/otr_api.py`, `scripts/otr_w45_campaign.py`,
`scripts/otr_writer_bank_gate.py`, and their tests.

### THE SHARED HAZARD -- git, not files

Both windows push to `v2.0-alpha`. This project has been burned twice by
concurrent windows: a staged deletion swept into another window's commit, and an
amendment lost in the gap between staging and committing.

**So: stage, commit and push ATOMICALLY in one call. Never leave anything staged
while the other window is live. Never `git add .` -- add by name.** Verify
`HEAD == origin` after each push.
