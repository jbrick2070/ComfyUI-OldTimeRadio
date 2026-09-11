# Bible 11.63: grounded finished-diff judgment

Single CLI reader: Cursor / Grok 4.6 High. Actual four-file separate-repo diff
against 77bd6012 was inspected by root: BUG_BIBLE.yaml, otr_coverage_index.yaml,
README.md and tests/bug_bible_regression.py. Reviewer is read-only.

Accepted: stale index count/head and README range. Index now states 340 and
77bd6012 plus appended 11.63, with the entry range through 11.63. Removed the
binding-error sentence from the portable fix because the selected regression
anchors verify retry exhaustion, not a binder exception. Runtime still lets a
real binding exception propagate. Keep pushes together and update the OTR
production log: implemented wiring proof, repeated real-model test pending.

Rejected: missing snapshot / copied delta assertions. These were reviewer path
misreads. The real receipt is in the OTR repository at
docs/2026-09-11-my-story-cross-machine/grammar_snapshot.json. The two actual
pack directories are C:/Users/jeffr/Documents/ComfyUI/_worktrees/
otr-my-story-grammar-baseline (clean e07476eb) and otr-my-story-grammar-candidate
(e07476eb plus the 10 copied and hash-verified OTR Python files). No OTR source
was copied into the Bible repo. Both Bible runs used the same current coverage
guard and explicit --pack-dir. Exact XML comparison: candidate 30 pass/10 fail,
baseline 29 pass/11 fail, both 11 skip/3 xfail; no new or changed failures.
The sole flip is TestPhase07To12ProductionRegressionCatalog::
test_otr_my_story_current_repair_and_parent_identity_coverage. Its extended
11.63 guard fails on the clean baseline's missing P1 binding test and passes
on the candidate. Matching totals from earlier work are not copied evidence.

New rule admission is live PBUG-20260910-05, 4060 Steps 117--121, not static
M1 history or a Mac-memory cause. Actual legacy_id PBUG-20260910-05 matches
the validator format. Root's metadata comparison found 146 existing issues
in both baseline and candidate, zero new issues. Prose cross-references remain;
do not add unsupported xref tags or claim globally clean legacy metadata.

CONVERGED. See grammar_bible_comparison.json, grammar_snapshot.json and
grammar_bible_metadata_comparison.json in the OTR receipt directory. Preserve
the original input as the immutable dispatch record; this judgment corrects
its count claim and expands its compressed paths/word spacing.
