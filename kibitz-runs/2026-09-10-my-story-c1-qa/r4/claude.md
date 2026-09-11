VERDICT: yes-with-fixes. The span-scoped conservation machinery in
nodes/_otr_ledger_clean.py matches the r3 final.md Section C design and closes
the live PBUG-20260829-14 / BUG_BIBLE 11.64 defect ("Until next time" ->
"That's a wrap" is now blocked by `test_whole_row_false_accusation_preserves_
valid_coda_and_outer_whitespace` and `test_collective_whole_row_complaints_
cannot_bypass_authorization`), but one genuine, untested safety gap survived
all three prior rounds inside the candidate reread path.

MUST-FIX BEFORE BUILD:
1. [nodes/_otr_ledger_clean.py:2146, function `_repair_row`] A malformed/
   exhausted REREAD is silently treated as "the judge confirmed it clean."
   `still_judged, _ok = _judge_row(...)` discards the second return value.
   `_judge_row` returns `([], False)` (never raises) exactly when the
   structured-call ladder is exhausted on a malformed reply
   (`_is_exhausted_clean_response`, lines 185-193, 1365-1373). When that
   happens on THIS reread call, `still_judged` is `[]`, and
   `remaining = still_judged or _as_findings(still_patterns)` (line 2155)
   falls through to `still_patterns` alone -- the free pattern list, which by
   this module's own stated purpose cannot see judge-only defects ("The door
   closes" catches nothing per the module docstring, lines 11-30). If
   `still_patterns` is also empty (the common case for a judge-only finding),
   the outcome is recorded `"clean"` (line 2159) and the candidate is
   committed via `set_line_text_metrics` + `receipt["repaired"] += 1` (lines
   2166-2173) -- never actually reread and confirmed. This directly
   contradicts the comment immediately above it ("READ IT BACK THE SAME WAY
   IT WAS READ THE FIRST TIME. A repair graded by a weaker check than the one
   that condemned it is a repair that can pass by moving the defect somewhere
   the check cannot see," lines 2135-2137) -- an unreachable reread IS a
   strictly weaker check than the one that condemned the original, and the
   code does not distinguish "reread said clean" from "reread never ran."
   No test in tests/test_ledger_clean_stage.py exercises this: every
   provider-failure test (`test_cleaner_jobs_do_not_swallow_real_runtime_
   failures`, parametrized `job="judge"`) raises on the FIRST per-row judge
   call in `run_ledger_clean`, before `_repair_row` is ever entered, and
   never on the reread call inside the repair loop; the malformed/"exhausted"
   branch (as opposed to a real runtime error) is untested here entirely.
   Concrete fix: capture the flag (`still_judged, reread_ok = _judge_row(...)`)
   and gate the "clean" outcome on it, e.g. change line 2166's condition to
   `if not still_judged and not still_patterns and (not judge_reachable or
   reread_ok):` -- this preserves every existing green test (all of them run
   with a `_Slot` that never raises `_is_exhausted_clean_response`, so
   `reread_ok` is always `True` in the current suite) while closing the gap.
   Add one new test: a scripted `_Slot` subclass whose reread reply is
   schema-malformed (exhausts `structured_call`) for a judge-only-found
   defect (e.g. reuse `INVISIBLE`/"The door closes behind him."), asserting
   the row does NOT ship `"repaired"` and instead ships `"unclean"`/
   `"improved"` with the flag set -- mirroring the existing
   `test_the_judge_reads_the_repair_back_before_it_is_accepted` shape.

SHOULD-FIX:
1. [tests/test_ledger_clean_stage.py, `test_cleaner_jobs_do_not_swallow_
   real_runtime_failures`] The parametrization covers `job="judge"` for the
   FIRST per-row judge call only (the `_FailingJob` raises on any prompt
   containing "DO THIS, IN ORDER:", but `run_ledger_clean` never reaches
   `_repair_row` in that fixture because the exception propagates out of the
   very first `_judge_votes` call). Add a fourth job value, e.g.
   `job="reread"`, that lets the first judge call and the first repair call
   succeed and only fails the SECOND "DO THIS, IN ORDER:" prompt, to prove
   real runtime failures on the reread specifically still propagate rather
   than being swallowed by the same `except Exception` in `_judge_row`
   (verify: they do, by code inspection -- `_judge_row` only swallows
   `_is_exhausted_clean_response`, everything else re-raises -- but this path
   is currently proven by reading the code, not by a red-then-green test).

OPTIONAL / NICE-TO-HAVE:
1. `_authorize_repair_scope` (nodes/_otr_ledger_clean.py:1408-1527) folds the
   judge's own free-text complaint into `intervals`/`_covers_spoken_row`
   unfiltered by `admissible` -- only PATTERN-sourced findings are gated to
   `MARKUP_KINDS` on `shakespeare`/`public_domain` (via `repairable_kinds`,
   `_otr_spoken_text_policy.py:232-246). A judge that flags the author's own
   third-person prose as "not speech" on a fidelity lane can still reach a
   real span-scoped repair through `judge_complaint`, which sits in tension
   with `_otr_spoken_text_policy.py`'s module docstring ("the fidelity lanes
   repair markup and merely REPORT language," lines 47-55). This is almost
   certainly the explicitly-deferred "F1 source semantics" item named in the
   task brief, not a C1 regression -- `test_the_author_s_own_language_
   cannot_trigger_a_pattern_repair` only proves the default `_Slot()` (which
   never flags anything) stays untouched, it does not prove a judge that DOES
   flag author language is blocked. Leaving this as a named, tracked
   follow-up rather than opening it now matches the task brief's own scope
   fence; flagging here only so it is not lost.

CUT THESE:
None -- the span/scope machinery (`_RepairSpan`, `_exact_interval`,
`_merge_repair_spans`, `_covers_spoken_row`, `_splice_replacements`,
`_ScopeAuthorization`) is exactly load-bearing for the four-verdict contract
in r3 final.md Section C and BUG_BIBLE 11.64's fix description; nothing here
is unused or speculative. The `JUDGE_PER_SENTENCE`/`JUDGE_VOTES`/
`REPAIR_READS_BRIEF_ONLY`/`JUDGE_ATTRIBUTION` knobs are pre-existing (module-
level, documented as measurement levers for `scripts/otr_clean_stage_lab.py`)
and untouched by this diff's scope-conservation logic -- not new surface
introduced by C1.

VERIFY-AT-BUILD checklist:
1. The 177-focused / 14,197-passed / 51-existing-failure / 183-skipped /
   1-xfailed suite numbers in the task brief do not match r4/judgment.md's
   convergence-time numbers (14,135 passed / 53 pre-existing failures) from
   kibitz-runs/2026-09-10-my-story-cross-machine/r4/judgment.md:90-91. That
   is expected drift (different checkpoints, days apart) rather than a
   contradiction, but this review could not run pytest (out of scope per the
   task brief) -- confirm the stated full-suite numbers with a real run
   before treating C1 as closed, and confirm the 51-failure baseline is
   identical in membership to A2's baseline, not just count.
2. `docs/PROD_BUG_LOG.md` PBUG-20260829-14 addendum (line 14144) and
   BUG_BIBLE.yaml 11.64 (line 11133, confirmed present, `legacy_id:
   PBUG-20260829-14-my-story-coda`) both read as intended; the portable Bible
   regression `tests/bug_bible_regression.py:1673-1706` (in the separate
   comfyui-custom-node-survival-guide repo) AST-extracts and exec()s
   `_RepairSpan`/`_exact_interval`/`_merge_repair_spans`/`_covers_spoken_row`/
   `_splice_replacements` verbatim from the real `nodes/_otr_ledger_clean.py`
   and exercises the same conservation contract without a model. Confirmed
   present and internally consistent with the production code at review
   time; still worth a real `pytest` run in that repo (out of scope here) to
   confirm it collects and passes against the CURRENT file, not a stale copy.
   README.md's "341 bible entries" claim was independently verified by
   grepping `^- id:` in BUG_BIBLE.yaml -- 341, matches exactly.
3. `test_the_clean_stage_is_wired_into_the_one_shared_producer_boundary`
   (tests/test_ledger_clean_stage.py:1155-1183) proves `run_ledger_clean` is
   called before `run_ledger_cleanup` inside `_run_writer_tail` by source-
   text search on `tests/fixtures/writer_family.family_source()`. This review
   did not independently open `nodes/_otr_writer_tail.py` to confirm the call
   site still exists verbatim outside the fixture's own claim -- confirm at
   build time that the fixture is not itself stale (it is pinned "byte-
   identically" per the docstring after a 2026-08+ file move, per the
   comment at line 1163, which is a plausible but unverified-by-me claim).
4. Mac memory-kill root cause and the C2 "five model-dirty rows versus six
   repairs" count remain UNVERIFIABLE per r1/driver_anchor.md:14,34 from the
   broader cross-machine campaign. Neither is part of this C1 diff and
   neither should be treated as resolved by this review -- both need their
   own live-artifact evidence, unrelated to nodes/_otr_ledger_clean.py.

Mark [ASSUMPTION]: I could not run `git diff` (no shell/git tool available in
this review session and the task brief forbids running anything besides
reading); `diff.txt`/`diff_utf8.txt` in the repo root are stale/unrelated
(video_engine.py font-loading content from a different session) and were
NOT used as the basis of this review. Instead I read the full current state
of nodes/_otr_ledger_clean.py (2381 lines), nodes/_otr_spoken_text_policy.py,
tests/test_ledger_clean_stage.py, and tests/test_d1_protected_fact_component.py
directly, and cross-checked every design claim in the task brief's "Contract"
paragraph against that code and against kibitz-runs/2026-09-10-my-story-
cross-machine/r3/final.md Section C and r4/judgment.md. [ASSUMPTION] that the
git-status-reported "modified" state of these four files at the time of
review is the same content I read (I have no prior-commit baseline to diff
against, so I cannot independently confirm the SIZE or SHAPE of the change,
only that the current file content is internally consistent and matches the
stated design).
