# RESUME HERE -- video lane build, window handoff 2026-08-11

Read this, then `docs/GO_FORWARD_PLAN.md` item 5 (the VIDEO LANE QUEUE table).
Everything else you need is linked from those two.

## Where it stands

**6 of 21 lane packets confirmed working and pushed. Nothing is in flight; the
working tree carries only other windows' files.**

| # | Lane | State |
|---|---|---|
| 0 | scaffolding | DONE `49adc824` |
| 1 | `wan22_high_i2v` (`wan_i2v`) | DONE `b303afa3` |
| 2 | `humo14_high_audio_in_wide` (`humo_14B_169`) | DONE `e19dd473` |
| 3 | `humo17_high_audio_in_portrait` + `_wide` | DONE `d226bea5` |
| 4 | `humo14_high_audio_in_portrait` (`humo`) | DONE `b53ca2f1` |
| 5 | `wan22_high_video` (`wan_ti2v`) | DONE `d0536e72` |
| 6 | `wan22_high_fast` (`fastwan_8gb`) | DONE `930e3bda` |
| 7 | `ltx23_low_audio_in` (`ltx_audio_in`) | DONE -- 7/7 green, live 1024x576 f193 smoke |
| 7b | `ltx_audio_in` headroom | **OPERATOR DECISION -- read below** |
| 8-21 | LTX pair, mesh, 4 viz, 4 still, H3 trio | NOT STARTED |
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

## QUEUE ROW 7b -- the one thing only the operator can settle

Lane 7's live COLD peak, stating both surfaces per the `f2470e31` ruling:

* **ABSOLUTE 14,465 MB** against a 14,500 MB ceiling -- **35 MB, 0.24%**.
* **NET 11,952 MB** (minus its own 2,513 MB pre-queue baseline) -- unremarkable,
  sits with the HuMo figures (11,911 / 12,664 / 13,321), seed-eligible.

It passes. On the will-it-fit question 0.24% is not headroom. A smaller canvas
is NOT available: the ia2v two-stage recipe needs /64 on both axes and 1024x576
is the smallest exact-16:9 rung that qualifies. The lever that IS available
without touching a recipe is a diet-style boot contract -- the mechanism lane 2
built and proved for HuMo. This lane declares none and smoked on stock
`default`. Decide: ship as-is, or spend one leg proving a diet contract.

Detail: `docs/evidence/lane_receipts/lane07-ltx23_low_audio_in.md`.

"Confirmed working" = built, 7/7 preflight gates green, a live render smoked and
PROBED (canvas, exact frame count, silence, no trim), full suite green, pushed.

## Lane 7 is next: `ltx23_low_audio_in` (`ltx_audio_in`)

Its work list, from the corpus and its RED preflight rows:

- **S8b-9, the one that can DELETE the lane from the dropdown.**
  `eng_ltx_av.py:177` is a bare module-scope `float()` on
  `OTR_LTX_AV_RESERVE_VRAM_GB` -- the one env read the guarded `_env_num` was
  never applied to. A malformed value raises at import, the guarded import in
  `_otr_video_engines/__init__.py` swallows it, and the lane vanishes with
  nothing in the log (reproduced once: registry 27 -> 26).
  `tests/test_ltx_av_env_import_safety.py` claims to cover every module-scope
  env read and omits exactly this one.
- **S8b-10**, the ia2v stage-A base latent: `:819` halves the canvas to
  416x240 and `240 % 32 == 16`. `assert_ltx_dims` only checks the full canvas,
  and `tests/test_ltx_av_ia2v_canonical.py:62-63` PINS the illegal value.
- **S3**, the HQ lane: declare `render_canvas = (1024, 576)` plus a named
  profile supplying 1024x576 and 193 frames (measured 7.36 GiB warm / 585.3 s).
  Do NOT touch the graph -- OTR's LTX graph is ahead of the lab's.
- The missing `ContractEnvConflict` refusal that `eng_ltx_video.py` already has.
- Public id + alias MOVE, profile/variant, matrix row, solo smoke.

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

## One thing the operator still owes a decision on

1. **The cold peaks run higher than the corpus headline** -- 14,604 MB (14B
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
