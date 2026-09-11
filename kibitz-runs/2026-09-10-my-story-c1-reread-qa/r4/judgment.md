# Root judgment: reread correction converged

Sonnet confirms the first must-fix is closed and finds no new must-fix. Root
re-read the actual branch: reread_ok=False records unresolved and continues before
remaining counts, clean acceptance or best_text/best_count. It cannot win either
path. The four runtime-failure kinds now have candidate-reread coverage; malformed
and recovered-read cases pass in the 183-test focused set.

The suggested mixed-history test is optional coverage, not an open production
bug: the unresolved branch never mutates best_text/best_count, and the existing
verified-progress and new unresolved-only tests exercise the two dispositions.
No additional retry/review loop or behavior change is justified by that comment.

The collateral-file question is a scope misunderstanding. The first review's
complete four-file C1 snapshot remains the comparison surface. The reread
correction changes only _otr_ledger_clean.py and test_ledger_clean_stage.py.
_otr_spoken_text_policy.py and test_d1_protected_fact_component.py have identical
hashes in c1_initial_snapshot.json and c1_snapshot.json. Their original C1 changes
are the exact raw-pattern helper and the protected-fact negative control's scoped
replacement payload/unchanged prefix. No protected-row exemption was weakened.

The call audit's literal label is '# LLM slot:', not the review's slot.label
or slot_label search. Root's actual pytest failure and targeted recheck verify
the new scope authorization call is tagged and only the two baseline request_slot
sites remain. This was a comment correction, not another editing mechanism.

Validation is the driver's responsibility, not a reason to multiply reviewer
calls. Baseline is A2 44057396:14197passed/51fail/183skip/1xfail. Intermediate
and final focused/full runs are distinct, recorded in c1_receipt.md and JUnit
comparisons; never substitute one count for another. Root verifies source hashes,
candidate copy, final Bible results, 341 parsed entries and 146 unchanged metadata
issues before push. Sonnet independently read the actual writer-tail calls in
this follow-up, closing its earlier fixture uncertainty.

Two actual Claude Code Sonnet/high CLI calls were made: initial QA and this
blocker-specific follow-up. No exact resolved model version or dollar receipt is
available. No further review is required; complete final regression verification
and continue GO_FORWARD's remaining coding. Live machines remain on hold.
