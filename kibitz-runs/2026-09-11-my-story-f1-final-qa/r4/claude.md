VERDICT: yes-with-fixes. Both SHOULD-FIX items from the prior round
(kibitz-runs/2026-09-10-my-story-f1-rewrite-qa/r4/claude.md) are correctly and
completely resolved in the current snapshot, and the rollback-path test now
proves the fix rather than just asserting the old ambiguous field. Nothing
found blocks the commit. One new, small documentation inconsistency was
introduced by this round's own fix and should be cleaned up before it is
copied elsewhere.

Grounding performed (direct reads against the current checkout, not the prior
round's quoted line numbers, which have since shifted): full read of
nodes/_otr_story_source.py (199 lines covering rewrite_story_source,
build_raw_documents, spoken_projection, _apply_spoken_edits,
rewrite_spoken_from_source); nodes/_otr_writer_tail.py L1330-1442 (the
transaction open/try/except block, the retained/clean_outcome stamping, the
exception-path re-save); nodes/_otr_clean_transaction.py full file (restore(),
reconcile(), _reseal(), _degrade(), open_transaction()); nodes/_otr_my_story.py
L360-407 (_call, the existing "per-sub-pass" LLM-slot tag precedent) and
L369-406 for the non-ledger pass_ids (interpret/treatment/act/frame), which
never enter a transaction and so have no analogous retained/clean_outcome
need; tests/test_story_source_review.py L1-70 (build_raw_documents /
source_intervals assertions), L230-322 (both tail tests, including the
rollback test body in full); tests/test_my_story_runner.py L1000-1043
(source_rewrites assertions for the P0-P3 pass_ids). Grepped every
"# LLM slot:" comment across nodes/ (62 hits) to check the new tag against
the project's actual established vocabulary, and grepped
"creative_writing_model" usage in _otr_writer_tail.py to confirm the
configured_model_id passed into the spoken-correction call is the right slot.

MUST-FIX BEFORE BUILD:
None -- no correctness, budget-multiplication, transaction, or wiring defect
found in this follow-up diff.

SHOULD-FIX:
1. [nodes/_otr_story_source.py L175] The new comment reads "# LLM slot:
   creative/technical -- the caller supplies the artifact's author owner."
   input.md's mechanical-fixes paragraph describes this as using "the
   project's recognized creative/technical tag," but grepping every other
   "# LLM slot:" comment in nodes/ (62 occurrences) shows the tag vocabulary
   is strictly single-valued: "creative" or "technical" alone, never a
   slash-joined compound, anywhere in the tree. The one place that already
   has this EXACT same situation -- a function called from both creative and
   technical contexts, with the caller deciding which -- is
   nodes/_otr_my_story.py L388: "# LLM slot: per-sub-pass -- caller supplies
   the creative or technical slot." That is the actual established
   convention for this case, one file away in the same lane, and the new
   comment coined a different, novel spelling instead of reusing it. Not a
   behavior bug (the comment is inert), but it is the kind of small
   inconsistency CLAUDE.md's "clean names, the reader matters" line is about,
   and it is easy to propagate to the next caller that copies the nearest
   example. Fix: change L175 to "# LLM slot: per-sub-pass -- the caller
   supplies the artifact's author owner" (or otherwise match the
   _otr_my_story.py L388 wording) so there is one tag vocabulary, not two.

OPTIONAL / NICE-TO-HAVE:
1. [nodes/_otr_story_source.py L111-114 + nodes/_otr_writer_tail.py
   L1419-1426] The budget-dedup receipt ("budget_already_spent", set when a
   second rewrite_story_source call reuses a pass_id already present in
   `receipts`) leaves `output_sha256` at its pre-update default --
   `candidate_sha256(candidate)`, i.e. a hash of the INPUT to that second,
   refused call, not of anything ever actually attempted or applied. The
   writer-tail loop at L1419-1426 computes `retained` for every receipt whose
   `pass_id == "ledger_clean_spoken"` unconditionally, so if this pass_id
   were ever invoked a second time against the same `receipts` list (e.g. a
   retried writer-tail run reusing an already-populated
   meta.my_story.source_rewrites), the second receipt's `retained` flag would
   compare the current ledger against a hash that was never a real
   correction attempt, reading as a false "not retained." I could not find a
   path in the current code that calls `run_ledger_clean` (hence
   `rewrite_spoken_from_source`) more than once within a single writer-tail
   execution -- the call site is a single, unconditional invocation at
   nodes/_otr_writer_tail.py L1396 -- so this looks unreachable today and I
   am not raising it as a SHOULD-FIX. Flagging only because the dedup
   guard's own design intent ("re-entry cannot reset the budget") implies
   repeat pass_id entries are an anticipated shape, and the reconciliation
   code added this round does not yet account for more than one. A one-line
   guard (only stamp retained/clean_outcome on the LAST matching receipt, or
   skip any receipt whose status is "budget_already_spent") would close the
   gap cheaply if this path is ever exercised. [ASSUMPTION: I did not find a
   caller that re-invokes the writer tail against an already-populated
   source_rewrites list; verify this is genuinely unreachable before treating
   it as settled.]
2. [nodes/_otr_story_source.py L108] `"qualified": False,  # application is
   not semantic proof` -- confirms the prior round's OPTIONAL 1 is done;
   no further action.

CUT THESE:
None. The prior round's CUT item (`build_raw_documents`) is no longer
cuttable -- it is now a real production caller at
nodes/_otr_story_source.py L93 (`documents = build_raw_documents(raw)`), whose
`.char_count` values populate the receipt's `source_intervals` field at
L102-103, and this is directly asserted against production code (not just the
helper in isolation) by tests/test_story_source_review.py L70
(`receipt["source_intervals"] == [...]`). Both prior-round findings are
resolved, not merely silenced.

VERIFY-AT-BUILD checklist:
1. [input.md para "Current focused execution"] "236 passed in 8.67 seconds"
   against tmp/my_story_f1_final_focus.xml, and the separate in-progress full
   suite repeat ("one new failure and its expanded pre-existing slot-tag
   failure," baseline 51 failures). I read and traced every assertion in
   tests/test_story_source_review.py and the relevant ~20 functions in
   tests/test_my_story_runner.py (L1000-1043) against the current production
   code and found them internally consistent and grounded -- but I have no
   shell/pytest access in this review pass, so the aggregate counts and
   green/red status are unverified by me. Run:
   `$env:PYTHONUTF8=1; C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_story_source_review.py tests/test_my_story_runner.py tests/test_ledger_clean*.py tests/test_writer_tail*.py tests/test_clean_transaction.py`
   and diff the printed total against both 236 (focused) and the full-suite
   number once the driver's repeat finishes; confirm the "one new failure"
   and "expanded pre-existing slot-tag failure" mentioned in input.md are
   gone, not just smaller.
2. [carried from 2026-09-10 round, now closed by test] "status can read
   rewritten/applied for a rolled-back correction." This is now directly
   proven, not just argued from reading: run
   `pytest -q tests/test_story_source_review.py -k rollback` on this exact
   checkout and confirm
   test_tail_rollback_keeps_attempt_history_and_actual_retained_hash_without_rechecking
   (L296-322) is green, which exercises `retained is False`,
   `clean_outcome == "restored_pre_clean"`, the `retained_spoken` summary
   fields, and a post-rollback `validate_receipt(saved)` all together.
3. [input.md para "Bible and canonical checks are also being reverified by
   the driver"] Out of scope for this read-only pass (explicitly named as
   separate, driver-owned work in input.md) -- confirm at build time that
   the Bug Bible delta-scrape and `workflows/otr_canonical.json` validation
   chain (CLAUDE.md section 0) both come back clean, since this review did
   not re-run either.
4. [carried, still open by design] Semantic accuracy of a real model on
   large-source capacity, live media delivery, and hardware memory behavior
   remain explicitly out of scope for this component and for this review, as
   input.md itself states. No code-level fix applies here; this is a
   live-hardware verification item for whenever My Story gets its own
   soak/obs-publish leg.
