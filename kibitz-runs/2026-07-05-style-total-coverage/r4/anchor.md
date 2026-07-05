# ANCHOR REVIEW (Cowork Claude) -- style total-coverage r4 (convergence)

Doc: STAGE3_TOTAL_COVERAGE_SUBPLAN.md v4.

VERDICT: SHIP. v4 is build-ready pending the listed verify-at-build items. Fresh-pass
residuals (non-blocking):

1. Convergence check: r1-r3 accepted items are all reflected (template placeholders,
   two-axis look fields, 15-field final schema from A1, all-packs syntactic upgrade,
   dispatch-arg rename, substitution order vs talking swap, static key set, provenance
   additive keys + trace allowlist, threading list, still_word concrete fields). No
   cross-round conflicts found.
2. The A1 "sci-fi defaults in non-default packs" temporary state is HONEST only if the
   delta tests in B assert the fields CHANGED from the A1 defaults -- pin that in B's
   test list so a lazy authoring pass can't ship sci-fi text in an anime pack.
3. The 3B 45-test forced-meta suite must keep passing UNCHANGED through A1 (tails
   untouched); any 3B pin that enumerates pack keys byte-for-byte re-points in the SAME
   A1 commit (r2 AG OPT2 already covers the schema_version pin).
4. Build risk register (for the builder): the writer INPUT_TYPES load path, the
   trace-copy allowlist, and the char-scene builder split are the three highest-churn
   edits; each has a named test.
