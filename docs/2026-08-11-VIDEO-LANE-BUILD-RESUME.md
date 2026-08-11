# RESUME HERE -- video lane build, window handoff 2026-08-11

Read this, then `docs/GO_FORWARD_PLAN.md` item 5 (the VIDEO LANE QUEUE table).
Everything else you need is linked from those two.

## Where it stands

**5 of 21 lane packets confirmed working and pushed. Lane 6 is IN PROGRESS in
the working tree.**

| # | Lane | State |
|---|---|---|
| 0 | scaffolding | DONE `49adc824` |
| 1 | `wan22_high_i2v` (`wan_i2v`) | DONE `b303afa3` |
| 2 | `humo14_high_audio_in_wide` (`humo_14B_169`) | DONE `e19dd473` |
| 3 | `humo17_high_audio_in_portrait` + `_wide` | DONE `d226bea5` |
| 4 | `humo14_high_audio_in_portrait` (`humo`) | DONE `b53ca2f1` |
| 5 | `wan22_high_video` (`wan_ti2v`) | DONE `d0536e72` |
| 6 | `wan22_high_fast` (`fastwan_8gb`) | **IN THE WORKING TREE -- see below** |
| 7-21 | LTX trio, mesh, 4 viz, 4 still, H3 trio | NOT STARTED |
| 22 | 30-word end-to-end episode gate | NOT RUN |

"Confirmed working" = built, 7/7 preflight gates green, a live render smoked and
PROBED (canvas, exact frame count, silence, no trim), full suite green, pushed.

## Lane 6, exactly where I left it

Uncommitted in the working tree, and the naming half is DONE and verified live:

- `nodes/_otr_shared/public_engines.py` -- `fastwan_8gb` removed from
  `_PUBLIC_ENGINES`, `wan22_high_fast` added, label rewritten to sell
  throughput. It was an IDENTITY row (public id == internal id) so it needs NO
  alias row: a bare internal id already passes through `resolve_engine_id`
  step 3. Verified: `fastwan_8gb`, `fastwan_8gb (16:9)` and `wan22_high_fast`
  all resolve; the old id appears in NO menu option; menu still 27 rows.
- `tests/test_public_engines.py` -- `_TIER` table updated to match.
- `docs/ENGINE_MATRIX.md` + the three `otr_*_fastwan` variants regenerated.

**Still owed for lane 6:** the full suite was running when this window ended
(log: the scratchpad `lane6.log`), then a solo smoke at f81 on the `default`
boot, a receipt at `docs/evidence/lane_receipts/lane06-wan22_high_fast.md`, the
queue row, and the commit.

If the suite came back red, expect the same shape lanes 1-5 all hit: a test
asserting the OLD public id or the bare internal id. Fix it at the assertion,
never by reverting the rename -- and see L8/lane-3 in
`docs/LANE_BUILD_LESSONS.md` for the two patterns.

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

## Two things the operator still owes a decision on

1. **`wan22_high_i2v` vs the spec's `wan21_high_i2v`.** The lane loads a Wan
   **2.2** weight and the repo had already corrected this exact mislabel once.
   I shipped `wan22` and registered the spec's string as a legacy alias, so both
   resolve. One line to swap either way. Lesson L8.
2. **The cold peaks run higher than the corpus headline** -- 14,604 MB (14B
   wide f97), 15,261 MB (1.7B portrait f129), 13,800 MB (14B portrait f97), all
   absolute and COLD against a 13.06 GiB warm figure. Not a contradiction:
   different cache state, and these are device totals including the ~1.9 GB idle
   baseline. Lane 5's admission work should use THIS surface, not the lab's.

Nothing is machine-qualified yet. `QUALIFIED_COST_ROWS` is still empty and
`docs/evidence/video_evidence_manifest.json` says "admission NOT enforced" per
lane, in words. That is deliberate (standing default Q3), not an omission.

## Open rows that are NOT lanes

- **2b** -- move the boot-contract check from `assert_usable` into the ShotLock
  preflight. Needs `boot_contract` plumbed into the frozen director policy.
- **5b** -- S7 WAN retention. Instrument the post-close boundary, collect
  telemetry on a live chained leg, THEN pick a release branch from what it
  names. A measurement campaign, not a code change.
- **8 GB re-measure** -- `otr_8gb_wan` went 17 -> 81 frames because 17 was
  narrowing the planner into 0.68 s segments. 81 has NOT been proved to fit on
  real 8 GB hardware. If it does not, the answer is a measured ceiling, not a
  return to 17.
- `docs/2026-08-10-FINAL-QA-video-build-corpus.md` still carries its ORIGINAL
  header verdict ("NOT IMPLEMENTATION-READY") while the master spec says that
  pass re-ran and cleared lane 1. Its 21-lane plan is what this build follows.
  One edit would stop the next window stopping at the wrong gate.
