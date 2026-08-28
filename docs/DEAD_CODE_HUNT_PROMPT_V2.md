# Dead code, stale claims and cruft -- hunt v2

**Hand to each reviewer INDEPENDENTLY. Do not show them each other's answers,
and do not tell them what a previous round found -- two blind samples are worth
more than one confirmed sample.**

You are auditing a real, shipping ComfyUI custom-node pack on Windows. Find
code that can be DELETED or CORRECTED without changing behaviour, and be honest
about what you could not verify.

    REPO:   C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
    BRANCH: v2.0-alpha
    PYTHON: C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe  (set PYTHONUTF8=1)
    TESTS:  python -m pytest -q -p no:cacheprovider tests   (~12,400 tests, ~7 min)
    CORPUS: C:\Users\jeffr\Documents\ComfyUI\output\otr\episodes\*\audio\*_ledger.json
            (~2000 frozen production ledgers -- REAL EVIDENCE, use them)

**READ-ONLY. Do not edit files, do not run renders, do not start a server** --
a GPU render queue is often active. CPU greps, AST parses, corpus scans and
reading are all fine and encouraged. PowerShell chains with `;` not `&&`.

## The six things to hunt, in value order

**1. STALE CLAIMS -- comments and docstrings that LIE about wiring.** Highest
value in this codebase and the easiest to miss, because a confident comment
reads as documentation rather than as a claim. A banner saying "used by the
writer's cast-lock path" when nothing calls it does active harm: it sends the
next auditor, and the next window, down a false trail. **Hunt these directly:**
grep for "used by", "called from", "consumed by", "routes through", "feeds",
then verify each claim against a real call graph. Report the ones that are
FALSE. A wrong comment is a defect even when the code beside it is alive.

**2. UNWIRED FIXES -- code that fixes a real bug and was never connected.**
This is NOT ordinary dead code and it is the most valuable thing you can find,
in either direction. Someone wrote a fix, wired it into a call site, and that
call site was later deleted -- taking the fix out of service silently. For each
one, answer three questions and keep them separate: (a) what defect did it fix,
(b) **is that defect still reachable today, or did something else supersede
it**, and (c) is there evidence of the defect in the ~2000 real ledgers? A fix
for a defect that can no longer occur is deletable; a fix for one that still
occurs is a BUG REPORT, not a cleanup item. Say which you found.

**3. Unreachable / unreferenced code.** Functions, classes, constants, whole
modules with no live caller. See the reachability rules below before you
conclude anything is dead.

**4. Retired mechanisms that left debris.** Env flags nobody sets, config keys
nothing reads, engines registered but not routable, tests asserting behaviour
that no longer exists, receipt fields no consumer reads.

**5. Duplicate implementations of one idea.** Two helpers computing the same
thing; a local reimplementation of something already in `_otr_shared/`. Note
the deliberate-duplication exception below before flagging any.

**6. Inert controls -- and propose REMOVING them, with the migration.** A
widget, env var or parameter that is accepted, displayed, and then not used to
control the thing it names. These are dishonest controls: they promise the
operator a knob that does nothing. Do not settle for "document it as inert" --
work out what deleting it takes (see the `widgets_values` note below) and put
that in the finding. If the honest fix is to make the control WORK rather than
remove it, say that instead and say what it should control.

## Reachability -- how to not be wrong

This is where audits of this repo go wrong, so spend your care here.

* **THINGS ARE RESOLVED BY STRING AT RUNTIME.** Engine ids live in
  `workflows/*.json` `widgets_values`, node names in `NODE_CLASS_MAPPINGS`,
  registry lookups, `getattr` dispatch, and per-class method resolution.
  **Grep the workflow JSONs and the registries for a symbol's string name
  before calling it dead**, not just for Python references.
* **CHECK UP A LEVEL. A blocking dependency is a claim, not a verdict.** If B
  references X, go read B. Two things are true often enough to check every
  time: **B may itself be dead** (a chain of orphans that only call each other
  looks perfectly referenced from the inside), and **B may not really depend on
  X** (it may import and never call, call only in an unreachable branch, or
  keep its own duplicate). *An import is not a use.*
* **A test-only symbol is not automatically dead.** It may be a parked
  capability, a safety tool, or an unwired fix. Say WHICH; do not assume.
* **Prefer AST over grep** for the final call. A grep hit inside a comment,
  a docstring or a string literal is not a reference.

## What is NOT cruft here -- do not flag these, you will be wrong

Each of these has cost someone real time to re-learn:

* **Long explanatory comments are the house style and are load-bearing.** This
  codebase deliberately records WHY, including operator rulings and past
  production failures. Flag a comment only when it is factually WRONG about the
  code beside it (that is category 1) -- never merely for being long.
* **Tombstones are intentional.** "X was removed on DATE because Y" exists so
  nobody re-adds X. Do not propose deleting a tombstone.
* **Video engine lane duplication is a RULING** (operator, 2026-08-23): each
  video lane owns its own prompt composer and helpers ON PURPOSE so lanes can
  diverge. Do not propose consolidating `nodes/_otr_video_engines/eng_*.py`
  helpers. Duplication *within* one lane is still fair game.
* **`nodes/_otr_video_engines/acceptance.py` must import nothing but
  `__future__`** -- a ratified structural rule with its own test, so the grader
  CANNOT consult live routing state. Its literal string comparisons are
  deliberate; do not "clean them up" into shared-constant imports.
* **Per-node try/except in `__init__.py` is partial-install resilience.**
* **`widgets_values` IS POSITIONAL -- and that is a REASON TO DO THE WORK, not
  a reason to keep an inert widget** (operator ruling 2026-08-28: *"why not
  delete an inert widget and just make the adjustments so it's ok -- that's
  being lazy not to remove an inert widget"*). He is right: an inert control
  LIES to whoever opens the graph, promising a knob that does nothing, and
  leaving it costs the user more than the migration costs us.
  **So: propose the deletion, and propose it WITH its migration.** Removing a
  widget shifts every later value in every saved `widgets_values` array
  (BUG-LOCAL-097), so the finding must name: the widget's INDEX in its node's
  `INPUT_TYPES` order, every workflow JSON in the repo that carries that node
  (`workflows/otr_canonical.json` and everything under `workflows/variants/`),
  and the re-index each of those arrays needs. All of it lands in ONE change,
  followed by the audit this project already requires -- widget count vs
  `widgets_values` length, per node.
  The only genuinely unrecoverable case is a workflow saved OUTSIDE this repo,
  which nobody can migrate. Say so where it applies, but do not let it veto a
  fix: the time to remove a dishonest control is BEFORE a v2 release, not
  after.
* **`docs/2026-*/` and `kibitz-runs/` are gitignored** working notes, not
  shipped code. `.comfyignore` excludes `scripts/` from the published package
  -- **not shipping is not the same as being dead.**

## Before you propose a deletion

1. **Check `docs/LEAN_MEAN_CLEANUP.md`.** It is the live cleanup-plan authority
   and classifies many modules already (REMOVE-SAFE / REMOVE-AFTER-MIGRATION /
   RE-GROUND). If your target has a row, quote it -- a RE-GROUND row is a gate,
   not a green light.
2. **Check `docs/OTR_STANDING_RULINGS.md` and `docs/PROD_BUG_LOG.md`.** If a
   standing operator ruling protects it, that outranks your finding and you
   must say so loudly.
3. **Find the atomic-edit hazard.** Deleting a module breaks any file importing
   it -- and if that file is a SHARED test covering several unrelated things,
   its whole collection fails and takes the unrelated coverage down silently.
   For each target, name every file that must change in the SAME commit.

## Output format

Ranked, most confident first. For each finding:

    ### <short title>
    CATEGORY: stale-claim | unwired-fix | unreachable | debris | duplicate | inert-control
    CONFIDENCE: CONFIRMED | LIKELY | UNVERIFIED
    WHERE: path/to/file.py:123-145
    WHAT: one sentence
    EVIDENCE: the greps/AST scans/corpus queries you ran and what they returned
    SUPERSEDED-BY: (unwired-fix only) what prevents the defect now, or "nothing"
    CORPUS: (unwired-fix only) occurrences in the ~2000 ledgers, with the query
    ATOMIC-WITH: every file that must change in the same commit
    RISK: what breaks if you are wrong, and the fast way to check
    PAYOFF: ~N lines, mechanical | needs-a-decision

End with **"WHAT I COULD NOT CHECK"**. That section is genuinely valuable --
do not pad it and do not omit it.

Twelve findings you can defend beat forty you cannot. One confirmed stale claim
that would have misled the next reader is worth more than three tiny deletions.
