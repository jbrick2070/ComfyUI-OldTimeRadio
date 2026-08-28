# Dead code and cruft hunt -- OTR ComfyUI node pack

**Hand this to Codex and to Gemini/agy independently. Do not let them see each
other's answers; two independent sweeps are worth more than one conference.**

You are auditing a real, shipping ComfyUI custom-node pack on Windows. Your job
is to find code that can be DELETED or SIMPLIFIED without changing behaviour --
and to be honest about what you could not verify.

    REPO:   C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
    BRANCH: v2.0-alpha
    PYTHON: C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe  (set PYTHONUTF8=1)
    TESTS:  python -m pytest -q -p no:cacheprovider tests   (~12,400 tests, ~8 min)

**READ-ONLY. Do not edit files, do not run renders, do not start a server** --
a GPU render queue is usually active. CPU greps, AST parses and reading are
fine. PowerShell chains with `;` not `&&`.

## What to hunt

Rank by CONFIDENCE, and lead with the ones you actually proved.

1. **Unreachable / unreferenced code.** Functions, classes, constants, whole
   modules with no live caller. Prove it: grep the symbol across `nodes/`,
   `scripts/`, `tests/`, and `workflows/*.json`. A symbol referenced only by
   its own test is a strong candidate -- say so, and say what the test covers.
2. **Retired mechanisms that left debris.** Env flags nobody sets, config keys
   nothing reads, engines registered but not routable, `still_plan` rows for
   deleted families, tests asserting behaviour that no longer exists.
3. **Duplicate implementations of one idea.** Two helpers computing the same
   thing, a local reimplementation of something already in `_otr_shared/`,
   copy-pasted blocks that drifted. (READ THE EXCEPTION BELOW before flagging
   duplication -- some of it is deliberate.)
4. **Speculative generality.** Parameters no caller passes, `**kwargs`
   funnels, abstraction layers with exactly one implementation, options that
   have never been anything but their default.
5. **Defensive code that hides faults.** `try/except Exception: pass` around
   something that should be loud, a fallback that silently substitutes a
   default where the project's rules say it should raise, a guard for a state
   that cannot occur.
6. **Comment/doc drift.** Comments describing code that changed, cites naming
   a file:line that has moved, docstrings for removed parameters. (Again --
   read the exception below. Long comments are NOT automatically drift.)

## What is NOT cruft here -- do not flag these, you will be wrong

These are deliberate and each one has cost someone real time to re-learn:

* **Long explanatory comments are the house style and are load-bearing.** This
  codebase deliberately records WHY a thing is the way it is, including
  operator rulings, past production failures, and "do not re-open" notes. A
  comment that explains a non-obvious decision is doing its job. Flag a comment
  only when it is factually WRONG about the code beside it -- never merely for
  being long.
* **Tombstones are intentional.** Names ending `_RETIRED`, blocks headed
  "tombstone", and short "X was removed on DATE because Y" notes exist so the
  next reader does not re-add the thing. Do not propose deleting a tombstone
  unless it is older than a year AND names nothing that still exists.
* **Video engine lane duplication is a RULING, not an accident** (operator,
  2026-08-23): each video lane owns its own prompt composer and helpers ON
  PURPOSE, so lanes can diverge without touching each other. Do not propose
  consolidating `nodes/_otr_video_engines/eng_*.py` helpers into a shared
  module. Duplication *within* one lane is still fair game.
* **`nodes/_otr_video_engines/acceptance.py` must import nothing but
  `__future__`.** That is a ratified structural rule with its own test -- it
  exists so the grader CANNOT consult live routing state. Do not propose
  "cleaning up" its literal string comparisons into shared-constant imports.
* **Per-node try/except in `__init__.py` is partial-install resilience.** A
  missing dependency must skip ONE node, not zero out the pack.
* **`docs/2026-*/` and `kibitz-runs/` are gitignored** working notes. They are
  not part of the shipped pack; do not audit them for cruft.
* **`.comfyignore` deliberately excludes `scripts/` from the published
  package.** A script not shipping is not the same as a script being dead.

## Rules of evidence

* **Every finding cites `file:line`** and quotes the smallest relevant snippet.
* **Say how you verified it.** "grepped `<symbol>` across nodes/ scripts/
  tests/ workflows/, 0 hits outside its definition" is evidence. "Appears
  unused" is not.
* **Mark each finding CONFIRMED / LIKELY / UNVERIFIED.** An honest UNVERIFIED
  is more useful than a confident guess -- the driver checks every claim
  against the real files before acting on it, and a wrong claim costs that
  check for nothing.
* **Dynamic reachability is the trap in this repo.** Things are resolved by
  STRING at runtime: engine ids in `workflows/*.json` `widgets_values`, node
  names in `NODE_CLASS_MAPPINGS`, registry lookups, `getattr` dispatch, and
  per-class `compose_prompt` resolution. **Before calling anything dead, grep
  the workflow JSONs and the registries for its string name**, not just for
  Python references. This is the single most likely way to be wrong.
* **Estimate the payoff.** Lines removed, and whether deletion is mechanical or
  needs a behaviour decision.

## Output format

A ranked list. For each finding:

    ### <short title>
    CONFIDENCE: CONFIRMED | LIKELY | UNVERIFIED
    WHERE: path/to/file.py:123-145
    WHAT: one sentence -- what it is and why it is dead or redundant
    EVIDENCE: the greps/parses you ran and what they returned
    RISK: what would break if you are wrong, and how a reviewer checks fast
    PAYOFF: ~N lines, mechanical | needs-a-decision

End with **"WHAT I COULD NOT CHECK"** -- anything you ran out of time on, or
that needs runtime behaviour to settle. That section is genuinely valuable;
do not pad it and do not omit it.

Prefer twelve findings you can defend over forty you cannot. A single confirmed
dead module beats a page of "consider refactoring".
