# Retro bug hunt -- the CODE lanes 0-6 shipped

**This is a BUG HUNT, not a re-plan.** Operator ruling 2026-08-11: the plan
behind this build was hardened over a full four-round arc and that ruling
stands. But lanes 0-6 shipped **code that no panel ever saw**. A reviewed plan
does not make an unreviewed diff safe -- different artifacts.

## Exactly what to review

The **PUSHED cumulative diff** for lanes 0 through 6:

    git diff 49adc824^..930e3bda

74 files, ~7,166 insertions. The code surface is smaller and is where to spend
your attention -- 18 files, ~1,078 insertions:

    git diff 49adc824^..930e3bda -- 'nodes/**/*.py' 'scripts/*.py' 'scripts/*.cmd' 'config/profiles/*.json'

**Review the PUSHED commits, not the working tree.** Another window is live in
the tree right now and there is uncommitted lane 8/9 work in it that is NOT
yours to review. Use the commit range.

The lanes and their commits:

| Lane | Commit | What it shipped |
|---|---|---|
| 0 | `49adc824` | lessons ledger, preflight matrix suite, evidence manifest |
| 1 | `b303afa3` | `wan_i2v` weight resolution + canvas declaration |
| 2 | `e19dd473` | `humo_14B_169`, the boot-contract mechanism + its launcher hooks |
| 3 | `d226bea5` | `humo_1.7B` + `humo_1.7B_169` |
| 4 | `b53ca2f1` | `humo` -- closes the HuMo family |
| 5 | `d0536e72` | `wan_ti2v` -- a live profile bug + the first naming MOVE |
| 6 | `930e3bda` | `fastwan_8gb` -- public surface + a live throughput proof |

## What you are hunting

**Every one of these lanes already has a green preflight row and a live solo
smoke that rendered and was probed.** So do not re-run the checks that already
passed. Hunt for **what tests do not catch**:

* wrong-but-plausible logic -- code that is obviously reasonable and subtly
  incorrect;
* silent-failure paths -- anything that can swallow, default, or fall back
  without saying so, especially inside `except`;
* resource leaks -- unreleased handles, unrestored globals, patchers that never
  detach, probes that never stop on the exception path;
* assumptions that hold on THIS box and nowhere else -- hardcoded roots, a
  path separator, a model that happens to be installed, an env var that happens
  to be exported, a GPU that happens to be NVIDIA;
* error handling that hides the cause, or that raises the wrong type so a
  caller's `except` catches something it should not;
* **receipts that could record a falsehood** -- a stamp written from intent
  rather than from what actually ran, a field that defaults to a value that
  reads as a measurement, a number whose surface is not what its label says.

## OUT OF SCOPE -- do not spend a word on these

These are RULED. If you surface one it will be discarded, and saying so costs
you a finding slot:

* naming: the `low`/`high` convention, any specific public id, `wan22` vs
  `wan21`, whether a lane should have an alias;
* the low/high MARKER assigned to any lane;
* the build order, the 21-lane plan, or which lane owns which defect;
* warn-versus-refuse policy where a ruling already exists;
* the NET-not-absolute cost-row surface;
* anything in the spec's standing defaults (Q1 H3 commit granularity, Q2 the
  multi-clip mouth warning, Q3 shipping the WAN TI2V envelope disqualified);
* prose, comment style, docstring length, or test naming.

## What a good finding looks like

Name the file and line, state the concrete failure -- inputs or conditions,
then the wrong outcome -- and say why the existing tests do not catch it. A
finding that cannot name the condition under which it goes wrong is a
hypothesis, and it will be labelled one.

If a finding **contradicts a green preflight row**, say so explicitly and
loudly. That is a finding about the GATE, not just the lane, and it means every
later lane is being graded by a weaker instrument than anyone thinks.

## Context you need

* `docs/LANE_BUILD_LESSONS.md` -- the ledger. L1-L12 are the defect classes
  this build has already paid for; a recurrence of one of those in lanes 0-6 is
  a genuine finding.
* `docs/VIDEO_LANE_PREFLIGHT.md` -- what the seven gates actually check, so you
  can reason about what they do NOT check.
* `CLAUDE.md` -- hard operator rules (root-cause fixes only, no content
  guardrails on generated episodes, the ledger-completeness rule).
* The box: Windows, RTX 5080 Laptop 16 GB, sm_120, torch 2.10 + CUDA 13,
  SageAttention + SDPA. **Flash Attention 2 does not exist on this platform.**
