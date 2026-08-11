# RESUME HERE -- video lane build, window handoff 2026-08-11

Read this, then `docs/GO_FORWARD_PLAN.md` item 5 (the VIDEO LANE QUEUE table).
Everything else you need is linked from those two.

## Where it stands

**8 of 21 lane packets confirmed working and pushed, covering 8 distinct
engines across 9 live legs. Lane 9 is OPEN with two measurement legs due.
Nothing else is in flight; the working tree carries only other windows'
files.**

| # | Lane | State |
|---|---|---|
| 0 | scaffolding | DONE `49adc824` |
| 1 | `wan22_high_i2v` (`wan_i2v`) | DONE `b303afa3` |
| 2 | `humo14_high_audio_in_wide` (`humo_14B_169`) | DONE `e19dd473` |
| 3 | `humo17_high_audio_in_portrait` + `_wide` | DONE `d226bea5` |
| 4 | `humo14_high_audio_in_portrait` (`humo`) | DONE `b53ca2f1` |
| 5 | `wan22_high_video` (`wan_ti2v`) | DONE `d0536e72` |
| 6 | `wan22_high_fast` (`fastwan_8gb`) | DONE `930e3bda` |
| 7 | `ltx23_low_audio_in` (`ltx_audio_in`) | DONE `57665ee8` -- live 1024x576 f193 |
| 7b | `ltx_audio_in` headroom | DONE `310437ae` -- MARGINAL, 115 MB, diet boot shipped |
| 8 | `ltx098_low_video` (`ltx_8gb`) | DONE `c6a99764` -- live 512x288 f161 |
| 9 | `ltx23_high_video` (`ltx_video`) | **OPEN -- two measurement legs due** |
| 10-21 | mesh, 4 viz, 4 still, H3 trio | NOT STARTED |
| 22 | 30-word end-to-end episode gate | NOT RUN |

## THREE THINGS LANE 7 CHANGED FOR EVERY LANE AFTER IT

**1. Run `scripts/build_variants.py --check` BEFORE starting a lane.** Lane 7
inherited five RED variants from lane 5 -- they still carried `wan_8gb (16:9)`
in node 87, because lane 5 regenerated only the variant whose profile it edited
-- and had to separate them from its own drift. A red at the start of a lane
belongs to whoever caused it. It is 46 variants / 0 failures right now.

**2. A solo smoke means something different now.** `render_single` -- the path
EVERY lane smoke runs through -- never consulted `declared_render_canvas`. It
derived the canvas from `render_aspect` instead, so lanes 1-6 all validated the
aspect default rather than their own declaration. Invisible for six lanes
because all six declared exactly what that path already produced. Fixed;
pinned by
`test_ltx_8gb_canonical_canvas.py::test_render_single_takes_the_DECLARATION_not_the_aspect_default`.

**3. A lane's VRAM peak now reaches disk.** The `_clip_summary` passthrough
(relayed from the concurrent window, landed in lane 7's commit) means a smoke
report carries `vram_peak_mb` / `recipe` / `quant` / `render_canvas` /
`native_frame_count` / `extension_mode`. **One re-smoke each recovers the
`wan_ti2v` and `fastwan` peaks** that were measured and dropped -- no
measurement campaign, and it unblocks queue row 5a.

## QUEUE ROW 7b -- RESOLVED 2026-08-11 (`310437ae`)

Operator ruled: prove the lever, do not wave through 0.24%. Done. Both legs
cold, same recipe/canvas/frames/still, only the boot differing:

* `default` -- 14,465 MB absolute / 11,952 MB net -- margin 35 MB
* `ltx_av_diet` -- **14,385 MB** absolute / 11,872 MB net -- margin **115 MB**

Clears, but UNDER the 0.3 GiB threshold, so decision rule 2 applied: the diet
contract ships and the lane is flagged **MARGINAL** in the manifest with both
numbers. Output is BYTE-IDENTICAL between boots, so the diet is free. It bought
only 80 MB because `reserve_vram_gb` is INERT on this lane -- the adapter's own
in-process 4.0 GB reserve dominates any boot value -- so HuMo's ~1.9 GiB does
NOT transfer and the lever is nearly exhausted here.

"Confirmed working" = built, 7/7 preflight gates green, a live render smoked and
PROBED (canvas, exact frame count, silence, no trim), full suite green, pushed.

## Lane 9 is OPEN: `ltx23_high_video` (`ltx_video`) -- MEASUREMENT, not gates

Its preflight rows were already green before the lane started, so there is no
gate to flip. What landed in `7afe40e5`, all of it prerequisite:

- the canvas moved to **1024x576** by operator ruling. Its HQ two-stage path
  had the SAME fixed-x2 illegal-stage-A defect as lane 7 and the corpus never
  flagged it -- 832x480 halves to 416x240 and `240 % 32 == 16`. It hid because
  the OLD landscape default (1472x832) is also /64, so the path was legal until
  the lane was reconfigured. Lesson L13.
- a PREQUALIFICATION consent act (`OTR_LTX_VIDEO_PREQUALIFICATION`): this was
  the last frozen LTX adapter with no sanctioned way to measure, and its own
  env refusal correctly blocks moving the decode floor. Default off, LOUD, and
  its clips stamp `+prequalification[...]`.
- the L4 receipt fields it never produced. Without them a measurement has
  nowhere to stamp departures and no peak to report.

**STILL DUE -- two live legs:**

1. the **decode band at 1024x576** under the consent act, to answer S8b-14 at
   its ROOT. The 169 floor is canvas-dependent and was measured at 1472x832, so
   a 50-frame beat renders 169 and discards 119 -- ~3.4x the GPU work. The
   floor is deliberately UNMOVED and the trim ratio is logged loud until this
   leg answers it. Do NOT bundle a guess into the canvas fix.
2. a **single-render VRAM leg** for the low/high marker, which the corpus
   forbids shipping as a guess -- its only datapoint is a chained diagnostic
   that cannot decide it.

NOTE: the operator's ruling cited a lab 1024x576x193 warm row (7.36 GiB). That
row is recorded under `ltx_audio_in`, NOT this lane. The geometry argument
stands on its own, but this lane still has NO measurement at its new canvas.

Then lane 10 `mesh_stage` -- the most defective lane left, 4 red gates.

## The per-lane loop (do not skip step 1)

1. **Read `docs/LANE_BUILD_LESSONS.md` top to bottom** and check the lane
   against every entry. This is not ceremony -- lane 2 found HuMo carrying the
   identical defect that killed wan_i2v by doing exactly this, before writing
   any code.
2. Run `tests/test_lane_preflight_matrix.py`. The lane's RED rows are its work
   list. Quick matrix dump:
   `python -c "import sys;sys.path[:0]=['.','tests'];import test_lane_preflight_matrix as M;[print(n.ljust(28), ' '.join({'pass':'ok','exempt':'n/a','expected_red':'RED*','unexpected_pass':'??','RED':'RED'}[M.evaluate(g,n)[0]].ljust(4) for g in M.GATES)) for n in M.ENGINE_NAMES]"`
3. Code. 4. Wire (profile, boot lane, node-87 strings GENERATED never typed).
5. Regress: AST, dead-ref grep, full suite, Bug Bible.
6. Preflight row flips GREEN -- red row means no smoke, no commit.
7. Smoke the lane ALONE on the boot IT declares.
8. Commit AND push. 9. Append what bit you to the lessons ledger.

## Things that will bite you, learned the hard way

- **The strict unexpected-pass gate is your friend.** When you fix a lane, its
  `EXPECTED_RED` entry in `tests/test_lane_preflight_matrix.py` must be
  DELETED in the same commit or the suite fails and tells you to. That fired
  correctly on every lane so far.
- **Renames break tests that hardcode the aspect suffix or a bare engine id.**
  Grep for `(16:9)` near naming tables and for the internal id used as an
  expected widget VALUE. Derive from `_aspect_suffix(internal)`; assert that a
  saved value RESOLVES rather than that it is spelled a particular way.
- **`git add` BY NAME.** The tree carries other windows' dirty files
  (`tmp/*.ps1`, `docs/2026-08-10-OPEN-PLAN-lemmy-cross-engine.md`, the
  `config/profiles/otr_sbcov_*.json` set, `kibitz/`, `uv.lock`). None of them
  are yours.
- **Reset the GPU before every smoke**, selectively by CommandLine -- never a
  blanket python kill, which also takes the Claude MCP extension pythons.
  Helper: the scratchpad `boot_lane.ps1 -Lane WAN|LTX|HUMO -Contract default|diet`.
- **Commit messages: use `git commit -F <file>`.** A backtick in a `-m` string
  gets command-substituted by bash and the commit dies mid-message.
- **Pushes**: one attempt, then hand the operator a PowerShell block. A DNS
  blip on the Wi-Fi path cost one push tonight; the work was already committed
  so nothing was at risk.

## Naming: SETTLED 2026-08-11

`wan22_high_i2v` is correct and stands. The spec's `wan21` was a single
mistyped version number that every downstream document inherited -- the naming
itself was never in doubt. Spec and transplant plan corrected; no code moved,
because the build had used the right name from the start. The retired spelling
keeps a legacy-alias row so a paste from a stale copy of any reviewed doc still
resolves instead of erroring.

## ~~One thing the operator still owes a decision on~~ -- SETTLED (`f2470e31`)

The cold-peaks-vs-corpus-headline question is answered by the NET ruling:
derive cost rows from **NET** peaks (each leg's absolute minus its OWN pre-queue
baseline), never from absolute device totals, and seed a row ONLY from a true
`VramPeakProbe` maximum -- an `nvidia-smi` sample is a LOWER BOUND and would
make the row under-predict, admitting renders that then OOM. Every leg in
`docs/evidence/video_evidence_manifest.json` now records both surfaces.

Nothing is machine-qualified yet. `QUALIFIED_COST_ROWS` is still empty and the
manifest says "admission NOT enforced" per lane, in words. Deliberate
(standing default Q3), not an omission -- and `ltx_audio_in` additionally reads
**MARGINAL** there, with both boot numbers.

## Open rows that are NOT lanes

- **2b** -- move the boot-contract check from `assert_usable` into the ShotLock
  preflight. Needs `boot_contract` plumbed into the frozen director policy.
- **5b** -- S7 WAN retention. Instrument the post-close boundary, collect
  telemetry on a live chained leg, THEN pick a release branch from what it
  names. A measurement campaign, not a code change.
- **5a re-smoke (cheap, unblocks a queue row)** -- `_clip_summary` used to drop
  the telemetry, so `wan_ti2v` and `fastwan` smoked with no peak on disk. The
  passthrough landed in lane 7, so ONE re-smoke each recovers both. No
  measurement campaign.
- **8 GB re-measure** -- `otr_8gb_wan` went 17 -> 81 frames because 17 was
  narrowing the planner into 0.68 s segments. 81 has NOT been proved to fit on
  real 8 GB hardware. If it does not, the answer is a measured ceiling, not a
  return to 17.
- `docs/2026-08-10-FINAL-QA-video-build-corpus.md` still carries its ORIGINAL
  header verdict ("NOT IMPLEMENTATION-READY") while the master spec says that
  pass re-ran and cleared lane 1. Its 21-lane plan is what this build follows.
  One edit would stop the next window stopping at the wrong gate.
