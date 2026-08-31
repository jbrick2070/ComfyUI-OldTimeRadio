# Dead code, stale claims and slop — final sweep (v3)

**Hand to each reviewer INDEPENDENTLY. Do not show them each other's answers.
Two blind samples have already beaten one confirmed sample twice on this repo:
the second run found a live cross-machine bug the first missed entirely.**

You are auditing a real, shipping ComfyUI custom-node pack on Windows. Find
code that can be DELETED or CORRECTED without changing behaviour — and be
honest about what you could not verify.

    REPO:   C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
    BRANCH: v2.0-alpha
    PYTHON: C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe  (set PYTHONUTF8=1)
    TESTS:  python -m pytest -q -p no:cacheprovider tests   (~12,400 tests, ~7 min)
    CORPUS: C:\Users\jeffr\Documents\ComfyUI\output\otr\episodes\*\audio\*_ledger.json
            (~2000 frozen production ledgers — REAL EVIDENCE, use them)

**READ-ONLY. Do not edit files, do not run renders, do not start a server** — a
GPU render queue is often active. CPU greps, AST parses, corpus scans, git
archaeology and running tests are all fine and encouraged. PowerShell chains
with `;` not `&&`. Work from the CURRENT working tree, not git history.

---

## THE REACHABILITY RULE — three layers, then one more question

This is where audits of this repo go wrong. Everything else is detail.

**A blocking dependency is a CLAIM, not a verdict.** Walk it:

    A looks alive because B references it.
    B looks alive because C references it.
    If C is dead, the whole chain is dead — and at ONE hop it looks
    perfectly referenced from the inside.

Walk until you reach something genuinely live: a registered node in
`NODE_CLASS_MAPPINGS`, a string named in `workflows/*.json`, a CLI entry point,
or a test asserting PRODUCTION behaviour rather than exercising the orphan.
Report the chain as
`symbol -> caller -> its caller -> reached: <what makes this live>`.
If you cannot reach something live within three hops, say so plainly rather
than assuming a fourth would rescue it.

**Then the question that catches what reachability cannot: DOES IT ACTUALLY DO
ANYTHING?** A function that is called, computes a value, and whose return
nobody reads is dead in the way that matters — and it survives every
reachability check ever written, because it genuinely has a caller. Same for a
parameter that is accepted and never read, and a widget that is displayed and
never consumed. **This is the highest-value shape in this codebase.** A real
example found this way: a node's `normalize_dbfs` widget drove a `_normalize()`
function that was never called, while the actual delivery level was set
downstream in LUFS — the dial, the function, and the docstring step were all
theatre.

**Counting traps, both of which produced wrong first answers tonight:**
* **Recursion inflates grep counts.** A helper that walks dicts/lists calls
  itself three times; five hits can mean one real caller.
* **A mention is not a use.** An import never called, a name in a comment or
  docstring, a string literal, a `getattr` no live path reaches. Prefer AST
  over grep for the final call at every layer.

---

## The six things to hunt, in value order

**1. STALE CLAIMS — comments and docstrings that LIE about wiring.** Highest
value here and easiest to miss, because a confident comment reads as
documentation rather than as a claim. A banner saying "used by the writer's
cast-lock path" when nothing calls it does active harm: it sends the next
auditor down a false trail. **Hunt directly:** grep `used by`, `called from`,
`consumed by`, `routes through`, `feeds`, `wired into`, `is dormant`, then
verify each against a real call graph. A wrong comment is a defect even when
the code beside it is alive. Four were found and fixed this way already.

**2. UNWIRED FIXES — code that fixes a real bug and was never reconnected.**
The most valuable thing you can find, in either direction. Someone wrote a fix,
wired it, and a later refactor deleted the call site — silently taking the fix
out of service. For each, answer three questions SEPARATELY:
(a) what defect did it fix, (b) **is that defect still reachable today, or did
something else supersede it**, and (c) does it appear in the ~2000 ledgers?
A fix for a defect that can no longer occur is deletable. **A fix for one that
still occurs is a BUG REPORT, not a cleanup item** — say which you found, and
do not delete the second kind.

**3. Unreachable / unreferenced code.** Apply the three-layer rule.

**4. Retired mechanisms that left debris.** Env flags nobody sets, config keys
nothing reads, engines registered but not routable, receipt fields no consumer
reads, tests asserting behaviour that no longer exists.

**5. Duplicate implementations of one idea.** Two helpers computing the same
thing; a local reimplementation of something already in a shared module. Read
the deliberate-duplication exception below first.

**6. Inert controls — propose REMOVING them, WITH the migration.** A widget,
env var or parameter accepted, displayed, and never used to control the thing
it names. These are dishonest controls: they promise a knob that does nothing.
**Do not settle for "document it as inert"** (operator ruling 2026-08-28:
*"that's being lazy not to remove an inert widget"*). Work out the migration
and put it in the finding — see the `widgets_values` note below. If the honest
answer is that the control should be MADE TO WORK, say that instead and say
what it should control.

---

## What is NOT cruft here — you will be wrong

* **Long explanatory comments are the house style and load-bearing.** This
  codebase deliberately records WHY, including operator rulings and past
  production failures. Flag a comment only when it is factually WRONG about the
  code beside it (category 1) — never for being long.
* **Tombstones are intentional.** "X was removed on DATE because Y" exists so
  nobody re-adds X.
* **Video engine lane duplication is a RULING** (2026-08-23): each video lane
  owns its own prompt composer and helpers ON PURPOSE. Do not propose
  consolidating `nodes/_otr_video_engines/eng_*.py` helpers. **Note this ruling
  is scoped to VIDEO lanes — `LEAN_MEAN_CLEANUP.md` separately and still
  authorizes consolidating the TTS sidecar seam.** Check the scope before
  invoking it either way.
* **`nodes/_otr_video_engines/acceptance.py` must import nothing but
  `__future__`** — a ratified structural rule with its own test, so the grader
  CANNOT consult live routing state. Its literal string comparisons are
  deliberate.
* **Per-node try/except in `__init__.py` is partial-install resilience.**
* **`docs/2026-*/` and `kibitz-runs/` are gitignored** working notes.
  `.comfyignore` excludes `scripts/` from the published package — **not
  shipping is not the same as being dead.**

---

## `widgets_values` is POSITIONAL — and that is the migration, not a veto

Removing a widget shifts every later value in every saved graph
(BUG-LOCAL-097). That is WORK, not an impossibility. Any inert-control finding
must name:

* the widget's INDEX in its node's `INPUT_TYPES` order — **and whether it is
  LAST**, because a trailing widget costs no re-index at all and a mid-list one
  costs it everywhere;
* every workflow JSON carrying that node — `workflows/otr_canonical.json`,
  `workflows/otr_story_only.json`, and everything under `workflows/variants/`
  (~62 files);
* the re-index each `widgets_values` array needs;
* that it all lands in ONE change, verified by
  `python scripts/build_variants.py --check` plus
  `tests/test_widget_value_alignment.py` (which pins that one node type
  declares one widget ORDER everywhere — it is what catches a migration that
  updated the canonical and missed a variant).

The only unrecoverable case is a workflow saved OUTSIDE this repo. Note it;
do not let it veto a fix.

---

## Before you propose a deletion

1. **Check `docs/LEAN_MEAN_CLEANUP.md`** — the live cleanup-plan authority.
   A `RE-GROUND` row is a gate, not a green light. Quote the row.
2. **Check `docs/OTR_STANDING_RULINGS.md` and `docs/PROD_BUG_LOG.md`.** If a
   standing ruling protects it, that outranks your finding — say so loudly.
3. **Name every file that must change in the SAME commit.** Deleting a module
   breaks its importers, and **if the importer is a SHARED test covering
   several unrelated things, its whole collection fails and silently takes that
   unrelated coverage with it.** This nearly happened once tonight.
4. **Watch for tests that are the deleted thing's own scaffolding.** A parity
   test comparing a live implementation against a duplicate goes WITH the
   duplicate — the check existed only because there were two copies.
5. **Deleted symbols named in comments need a forensic marker.** This repo has
   a guard (`tests/test_legacy_audit_clean.py`) that fails on unclassified
   mentions of retired symbols; a tombstone must carry `legacy` / `deleted` /
   `removed in` on the same line, or it trips the very guard it was written to
   satisfy.

---

## Already removed — do not re-report these

`_load_canon_for_writer`, `_otr_voice_resolver`, `compute_cache_key`, the
fuzzy cast-consolidation cluster, `nsfw_frame_qc`, `refine_target_grade`,
`optimization_profile` (both sites), `_OPTIMIZATION_PROFILE_CHOICES`,
`normalize_dbfs` + `_normalize`, `_generate_bark_for_line` and its private
helper chain, `GemmaHeartbeatStreamer`, `_normalize_dialogue_names`,
`_pick_accent` / `_ACCENTS`.

Known-open and deliberately NOT dead code: the Bark preset-health cluster in
`story_orchestrator.py` (`_bark_test_presets` / `_bark_health_check` /
`_bark_health_check_for_cast`) — an unwired safety fix under operator review.
Leave it alone.

---

## Output format

Ranked, most confident first:

    ### <short title>
    CATEGORY: stale-claim | unwired-fix | unreachable | debris | duplicate | inert-control
    CONFIDENCE: CONFIRMED | LIKELY | UNVERIFIED
    WHERE: path/to/file.py:123-145
    WHAT: one sentence
    CHAIN: symbol -> caller -> its caller -> reached: <what makes it live, or "nothing">
    CONSUMED: does anything read what it produces? (the "does it do anything" test)
    EVIDENCE: the greps/AST scans/corpus queries you ran and what they returned
    SUPERSEDED-BY: (unwired-fix only) what covers this now, or "nothing"
    CORPUS: (unwired-fix only) occurrences in ~2000 ledgers, with the query
    ATOMIC-WITH: every file that must change in the same commit
    RISK: what breaks if you are wrong, and the fast way to check
    PAYOFF: ~N lines, mechanical | needs-a-decision

End with **"WHAT I COULD NOT CHECK"** — genuinely valuable; do not pad it and
do not omit it.

**Twelve findings you can defend beat forty you cannot.** One confirmed stale
claim that would have misled the next reader is worth more than three tiny
deletions — and one unwired fix that is still needed is worth more than all of
them, because that one is a bug, not debt.
