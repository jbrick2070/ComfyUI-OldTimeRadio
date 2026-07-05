# ANCHOR REVIEW (Cowork Claude) -- style total-coverage r3 (wiring / sequencing)

Doc: STAGE3_TOTAL_COVERAGE_SUBPLAN.md v3.

VERDICT: SHIP WITH FIXES (wiring-level).

MUST-FIX:
1. **A1 loads all 11 fields but consumes only the image lane -- the A2 fields are
   load-validated DORMANT content for one chunk.** State explicitly (the 3B "addressable
   but dormant" precedent) so the A1 review doesn't flag dead fields, and pin an A1 test
   that the A2 fields load + lint (budget/placeholder) even though unconsumed.
2. **Workflow JSON: NO change in any chunk** (the visual_style widget shipped in 3C; no
   new widgets/inputs). Run OTR_WorkflowValidator + widget audit per chunk with EXPECTED
   NO DIFF -- assert, don't assume.
3. **Sequencing risk: the extraction step (literals -> *_GEOMETRY + fixture constants)
   touches the SAME lines as the re-route.** Do extraction+re-route per surface in one
   edit (never a half state where a composer reads a deleted constant); the AST
   fixture-guard lands with the LAST surface of each chunk, not before (or it flags the
   yet-unrouted surfaces).

SHOULD-FIX:
4. The 3A de-swallowed composer seams (ImportError-only shims) must be re-audited after
   the re-routes -- new style reads inside those composers must sit inside the SAME shim
   posture (no new bare except; the swallow-hunt lens from the 3A fan-out).
5. Chunk B authoring: still_word lettering consistency (operator 2026-07-04) -- add a
   per-pack test that the typography selection for a fixed episode meta is deterministic
   across calls (no per-card randomness through the new fields).

CONFIRMED-OK: chunk split A1/A2/B/C; provenance additive-keys approach; the retired :1656
fallback with exact-key indexing (the "" no-op arm preserved).
