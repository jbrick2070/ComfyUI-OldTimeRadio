VERDICT: yes-with-fixes. The F1 source-repair chunk is correct, root-caused,
and matches the operator's checker+rewriter/no-infinite-loop amendment
exactly as specified; nothing found blocks the commit. One orphaned helper
and one confusing-but-deliberate receipt field should be cleaned up or
explicitly justified, and the "233 tests" claim needs a real pytest run
before anyone repeats it as fact.

Grounding performed: direct reads of nodes/_otr_story_source.py (new, full
file), nodes/_otr_my_story.py (_call at L369-406, P0-P3 at L437-636,
episode driver L960-1090), nodes/_otr_ledger_clean.py (L117-131 _exact_interval,
L247-268 protected/attempt-budget constants, L500-508 narrative source
injection, L1990-2054 final-seam call site), nodes/_otr_writer_tail.py
(L1360-1432 transaction/rollback seam, L1539-1542 final_spoken_sha256), and
both test files (tests/test_story_source_review.py full, relevant ranges of
tests/test_my_story_runner.py). Also grep-verified CREATIVE_FIELDS,
build_source_document callers, and build_raw_documents callers across the
tree, and read docs/GO_FORWARD_PLAN.md L89-120 for the operator amendment
text quoted in input.md.

MUST-FIX BEFORE BUILD:
None -- no correctness, budget-multiplication, or wiring defect found.

SHOULD-FIX:
1. [input.md para 3, "_otr_story_source.rewrite_story_source"] Dead helper
   shipped with no production caller. `build_raw_documents`
   (nodes/_otr_story_source.py L48-53) wraps `build_source_document` per
   CREATIVE_FIELDS field and is exercised only by
   tests/test_story_source_review.py L48-58
   (`test_raw_documents_keep_exact_unicode_whitespace_and_exclude_author`).
   Grep across the tree (`build_raw_documents`) shows zero callers in
   nodes/_otr_my_story.py, nodes/_otr_ledger_clean.py, or
   nodes/_otr_writer_tail.py -- the actual production path builds its own
   hashes inline (rewrite_story_source L98-102) and never calls this
   function. It reads as pre-staged infrastructure for the "Adaptive source
   organization" chunk input.md itself flags as not-yet-done. Per this
   repo's own standing rule (CLAUDE.md OTR file, "Orphans: rip fully or wire
   back," 2026-09-04) an unreferenced symbol either gets deleted with a grep
   receipt or is wired in deliberately with a comment saying why it exists
   unused -- right now it is neither. Fix: delete it (and the now-unused
   `SourceDocument`-shaped return type, since RAW_COORDINATE_VERSION itself
   stays live via the receipt field at L97) or add a one-line comment noting
   it is staged for the adaptive-organization chunk and not yet called.
2. [input.md para 5, "writer tail ... retained ordered/speaker text ...
   receive hashes"] The `source_rewrites[].status` field can read
   "rewritten"/"applied": true for a correction the transaction then rolled
   back. Confirmed by
   tests/test_story_source_review.py L295-315
   (`test_tail_rollback_keeps_attempt_history_and_actual_retained_hash_without_rechecking`):
   after a forced reseal failure, `story["source_rewrites"][0]["status"] ==
   "rewritten"` (comment in the test itself: "# attempted correction") while
   the actual line text is restored to its pre-correction wording and
   `story["retained_spoken"]["clean_outcome"] == "restored_pre_clean"` is the
   only field that tells the truth about what was actually persisted. This is
   consistent with how _otr_writer_tail.py L1412-1421 documents the design
   ("Reattach attempted corrections ... without another model call") and is
   not a correctness bug -- the saved ledger text is right, only the audit
   trail is ambiguous in isolation. Fix: rename the field (e.g.
   `status: "attempted_rewritten"` or add a sibling `persisted: bool` set
   from the same clean_outcome check) so a reader of `source_rewrites` alone,
   without cross-referencing `retained_spoken.clean_outcome`, cannot conclude
   a correction survived when it did not.

OPTIONAL / NICE-TO-HAVE:
1. [input.md para 3] `receipt["qualified"]` (nodes/_otr_story_source.py L107)
   is initialized False and never set True anywhere in the file -- confirmed
   by grep, single hit. This is evidently deliberate (the whole point is
   "never PASS," matching driver_anchor.md's "no semantic qualification
   claim"), but a one-line comment at the declaration saying so would save
   the next reader a grep to confirm it isn't a forgotten TODO.
2. [input.md para 2/4] Every one of the four author prompts (interpret,
   treatment, each act, frame) now carries the full raw_source_block
   (nodes/_otr_my_story.py L375), not just the source-correction call. This
   is a deliberate root-cause fix (authors work from verbatim text instead of
   a normalized/summarized view) and not scope creep, but it does mean the
   raw fields are duplicated once per pass in the transcript. Harmless at My
   Story's field sizes and not a defect; flagging only because it is the kind
   of fixed per-call overhead that would be worth remembering if raw fields
   ever grow large.

CUT THESE:
1. `build_raw_documents` (nodes/_otr_story_source.py L48-53) -- see
   SHOULD-FIX 1. Safe to cut: it has no production caller, its own test does
   not exercise any code path that would break without it (the test targets
   the function directly), and RAW_COORDINATE_VERSION -- its only external
   dependency -- is independently used and tested via the receipt field at
   L97, so removing this wrapper does not touch the live path at all.

VERIFY-AT-BUILD checklist:
1. Run the actual test suite and confirm "233 focused tests passed" from
   input.md para 6. I read and spot-verified the individual test bodies in
   tests/test_story_source_review.py (18 test functions, several
   parametrized) and the ~20 newly-relevant functions in
   tests/test_my_story_runner.py (L889-1056), and every assertion I traced
   matches the production code it exercises -- but I have no shell/pytest
   access in this review pass, so the aggregate count and green status are
   unverified by me. Run:
   `$env:PYTHONUTF8=1; C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_story_source_review.py tests/test_my_story_runner.py tests/test_ledger_clean*.py`
   and diff the printed count against 233.
2. [driver_anchor.md] "Independent Einstein review found the failure-history
   guard originally stopped before the second cleanup helper... A rollback
   test exposed a projection mismatch." I confirmed the CURRENT code already
   has the fix (spoken_projection filters to character/announcer roles only,
   nodes/_otr_story_source.py L220-221; the failure-history guard covers both
   ledger_clean and ledger_cleanup via the single try/except spanning
   nodes/_otr_writer_tail.py L1391-1432) and that the cited regression test
   passes its own assertions on paper. I did not execute it; confirm at
   build time that `pytest tests/test_story_source_review.py -k rollback` is
   green on this exact checkout, not a prior one.
3. [input.md para 6] "No node interface/widget/wiring change; canonical
   workflow is unchanged." Consistent with the session's git status, which
   does not list workflows/otr_canonical.json among modified/untracked
   files. No further action needed unless a later commit in this same chunk
   touches that file -- if so, re-run the canonical-workflow verification
   chain from CLAUDE.md section 0 (OTR_WorkflowValidator + link/widget
   audit) before push.
4. [prior UNVERIFIABLE, driver_anchor.md] "semantic accuracy of a real model
   and large-story capacity" remains explicitly out of scope for this
   read-only pass and for the code itself (the whole design deliberately
   never asserts semantic correctness -- see OPTIONAL 1). No code-level fix
   applies; this is a live-hardware verification item for whenever this
   component gets its own soak/obs-publish leg, not a build gate on the
   commit itself.
