VERDICT: yes-with-fixes. The correction genuinely closes the prior MUST-FIX.
`nodes/_otr_ledger_clean.py:2146` now reads `still_judged, reread_ok =
_judge_row(...)` and gates on it (`if not reread_ok:` at line 2155, appending
outcome `"reread_unresolved"` and `continue`-ing without touching
`best_text`/`best_count`/`current`) before the old "clean" condition at line
2175 can ever fire. This is verified against the real file, not the task
brief's description of it, and is backed by three real, non-vacuous tests
(`test_malformed_reread_cannot_be_clean_or_best_progress`,
`test_pattern_repair_gets_a_real_reread_after_initial_judge_exhaustion`, and
the widened `test_cleaner_jobs_do_not_swallow_real_runtime_failures` with a
new `job="reread"` parametrization) at
tests/test_ledger_clean_stage.py:329-380. One coverage gap remains (below),
smaller than the one just closed, plus two claims in the brief that cannot be
confirmed from static reading alone.

MUST-FIX BEFORE BUILD:
None -- the specific defect named in the prior review's MUST-FIX #1 is fixed
and tested. No new must-fix defect found in the corrected code.

SHOULD-FIX:
1. [nodes/_otr_ledger_clean.py:2155-2202, `_repair_row`] The mixed-history
   case is untested: an EARLIER attempt gets a valid reread showing measured
   progress (`best_text`/`best_count` updated at line 2199-2200), a LATER
   attempt's reread comes back malformed (`reread_unresolved`), and the
   budget exhausts. By code inspection this correctly ships the earlier
   `best_text` via the "improved" fallback (lines 2209-2230) because
   `best_text` is untouched by the `continue` at line 2163 -- consistent
   with the task brief's own stated intent ("retain original/flag unless an
   EARLIER actually judged candidate had shown progress"). But no test
   exercises this interleaving: `test_malformed_reread_cannot_be_clean_or_
   best_progress` (tests/test_ledger_clean_stage.py:349-364) has every
   attempt malformed (best_text never set, ships "unclean"), and the
   pre-existing `test_the_best_rewrite_ships_when_it_cannot_be_made_
   spotless` (tests/test_ledger_clean_stage.py:600-624) has every attempt
   validly judged (reread_ok always True). Neither proves the boundary where
   a validly-judged improvement survives a later malformed reread and still
   ships as "improved" rather than being discarded back to "unclean". Add
   one test: attempt 1 reduces the finding count with a real reread, attempt
   2's reread is schema-malformed (reuse the `_MalformedReread` shape from
   line 350), assert `receipt["improved"] == 1` and
   `ledger["lines"][1]["text"]` equals attempt 1's candidate, not the
   original. Low risk (the code path is a simple untouched local variable),
   but it is exactly the kind of interaction a future edit to this loop
   could silently break without a red test to catch it.
2. [tests/test_d1_protected_fact_component.py] This file is one of the four
   files the task brief and c1_snapshot.json (below) both name as part of
   this change, but the brief's narrative only describes edits to
   `nodes/_otr_ledger_clean.py` and its reread path. The file imports
   `_Slot`/`_dirty_judgement` from tests/test_ledger_clean_stage.py (line
   134-136), so a change to those shared fixtures plausibly required a
   matching touch here -- but the brief does not say what changed in this
   file or why, and its own text ("A new missing slot-label comment in one
   failure was corrected") does not map to anything grep-able as "slot
   label" in this file or in nodes/_otr_ledger_clean.py /
   nodes/_otr_spoken_text_policy.py (checked: the only 3 files in the repo
   matching slot.label/slot_label are nodes/_otr_writer_inputs.py,
   tests/test_writer_input_resolve.py, nodes/_otr_source_payload.py -- none
   of which are in this diff's four-file scope). Confirm at build time
   (or ask the author directly) what specifically changed in
   test_d1_protected_fact_component.py and nodes/_otr_spoken_text_policy.py
   for this correction, since the brief's own "no new work" claim cannot be
   checked against a description this vague.

OPTIONAL / NICE-TO-HAVE:
1. The brief's characterization of the fidelity-lane pattern/model
   distinction as pre-existing is confirmed accurate by direct read: the
   module docstring at nodes/_otr_spoken_text_policy.py:47-55 ("the fidelity
   lanes repair markup and merely REPORT language") and `repairable_kinds`
   at nodes/_otr_spoken_text_policy.py:232-246 (gates only PATTERN findings
   to `MARKUP_KINDS` on `FIDELITY_BANKS`; the judge/`_judge_row` path is not
   filtered by this function at all) are both unchanged in substance from
   what the prior review cited, and the OPTIONAL item in
   kibitz-runs/2026-09-10-my-story-c1-qa/r4/claude.md was explicitly framed
   as a deferred F1-source-semantics item, not a bug. Correctly left
   untouched here.

CUT THESE:
None -- the correction is a minimal, surgical fix (capture a discarded
return value, gate on it, add an outcome branch) with proportionate new
test coverage. Nothing speculative was added.

VERIFY-AT-BUILD checklist:
1. Full-suite numbers ("183 focused tests pass"; "14,241 passes/51 existing
   failed IDs", said to still be rerunning at brief-writing time) are
   asserted, not shown, and this review has no shell/pytest access (task
   brief scope: "No tests, models, edits or hardware coordination"). Run the
   suite and confirm: (a) the three new/widened tests at
   tests/test_ledger_clean_stage.py:329-380 pass, (b) the full-suite pass/
   fail counts match what the brief claims once the rerun finishes, and (c)
   the 51-failure set is the same IDs as the pre-existing baseline (not just
   the same count) -- this is a repeat of VERIFY-AT-BUILD #1 from
   kibitz-runs/2026-09-10-my-story-c1-qa/r4/claude.md, still open, now with
   a third, different checkpoint number (183/14,241/51) layered on top of
   the earlier two (177/14,197/51 and 14,135/53) from that same file's
   VERIFY-AT-BUILD #1 -- confirm which is the true current baseline.
2. docs/2026-09-11-my-story-cross-machine/c1_snapshot.json exists and its
   `"files"` key lists exactly the same four paths as this diff
   (nodes/_otr_ledger_clean.py, nodes/_otr_spoken_text_policy.py,
   tests/test_ledger_clean_stage.py, tests/test_d1_protected_fact_component.py)
   with 64-hex-char SHA-256 values -- confirmed present and well-formed by
   direct read and a `[0-9a-f]{64}` regex match on every entry. Its `"base"`
   field (440573962d53521e...) matches the current git HEAD short hash
   `44057396` shown in this session's git status. Still worth confirming at
   build time that these hashes are the hashes of the files AFTER this
   correction landed (i.e. a fresh sha256sum on disk), not a stale snapshot
   from an earlier pass in the same campaign -- this review has no hash
   tool and took the file's own content as given.
3. SHOULD-FIX #2 above: get a concrete description of what changed in
   nodes/_otr_spoken_text_policy.py and tests/test_d1_protected_fact_component.py
   for this correction and confirm it is genuinely collateral to the
   reread-tuple fix (e.g. a shared-fixture touch) and not unrelated scope
   creep, before treating "no new work... is inferred" as settled.
4. Bible/PROD_BUG_LOG claims (legacy ID PBUG-20260829-14, 341 entries, 146
   unchanged metadata issues) live in a separate repo
   (comfyui-custom-node-survival-guide) this review did not open; the
   PBUG-20260829-14 addendum at docs/PROD_BUG_LOG.md:14144 was confirmed
   present by direct read and is unrelated to this specific reread
   correction (it documents the original My Story coda defect, already
   verified by the prior round). The Bible-repo claims are unverified here
   and were already flagged UNVERIFIABLE by the prior round's
   VERIFY-AT-BUILD #2 -- still open.
5. nodes/_otr_writer_tail.py:1383-1404 was independently re-read this round
   (not just cited): confirmed `run_ledger_clean` (line 1389) still runs
   before `run_ledger_cleanup` (line 1397) inside the same transaction
   window opened at line 1384 and reconciled at line 1404. This closes
   kibitz-runs/2026-09-10-my-story-c1-qa/r4/claude.md's VERIFY-AT-BUILD #3
   for real -- no longer open.

Mark [ASSUMPTION]: no shell/git tool was available in this session either
(same constraint as the prior round), so this review also could not diff
against a pre-correction baseline; it read the current file content of
nodes/_otr_ledger_clean.py, nodes/_otr_spoken_text_policy.py,
tests/test_ledger_clean_stage.py, and nodes/_otr_writer_tail.py directly and
cross-checked every claim in the task brief's paragraph against that code.
[ASSUMPTION] that the current on-disk content is the corrected state the
brief describes (matches: the discarded-flag pattern from the prior review
is gone, `reread_ok` is threaded through exactly as described, and the three
named tests exist with the described assertions) -- but the diff's exact
boundary (what changed, versus what was already this shape before the
correction) could not be independently confirmed without `git diff` or the
prior commit's content, so "the correction is minimal" is inferred from the
absence of any inconsistency in the current file, not from seeing a diff.
