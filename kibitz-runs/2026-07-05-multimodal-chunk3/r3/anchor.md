# ANCHOR REVIEW (Cowork Claude) -- chunk 3 r3 (wiring / integration / sequencing)

Doc: CHUNK3_SOURCE_PAYLOAD_SUBPLAN.md v3. Extra grounding:
tests/test_writer_input_resolve.py read IN FULL (:1-100).

VERDICT: SHIP WITH FIXES (wiring-level).

MUST-FIX:
1. **The S31 B6 invariant must SURVIVE the re-route, not just have its test deleted.**
   test_writer_input_resolve.py pins that the RSS path routes the TECHNICAL model
   (slot-label/id agreement in differing-slots mode). The replacement test must pin the
   invariant across the NEW topology: (a) `_resolve_inputs` passes
   `technical_model=technical_model` into `entry.fetch(...)` (AST or mock pin), AND
   (b) the science_rss wrapper forwards it as `_fetch_rss_seed_or_die`'s 2nd positional.
   Deleting the old test without both pins loses a load-bearing invariant. (CONFIRMED
   :36-72.)
2. **Workflow JSON: NO change required -- must be ASSERTED, not assumed.** Chunk 3 adds
   zero widgets/inputs (module + pipelines.json field + writer internals). Add to
   acceptance: OTR_WorkflowValidator + widget audit run anyway post-change (hard rule 0
   discipline), expected diff = none.
3. **Sequencing within the single commit:** (1) `_otr_source_payload.py` module + its
   unit tests green in isolation; (2) routing field + sweep + fixture updates; (3) writer
   re-routes + Sprint-2.2 test re-points; (4) full suite + Bug Bible + B7. The
   pipelines.json field and `_pipe_row` fixture MUST land in the same edit step as the
   parser change or the suite is red mid-build (fine locally, but do not commit between).

SHOULD-FIX:
4. `_resolve_inputs` currently maps style->slug BEFORE the fetch (:1374, rss_style_slug
   incl. the _LLM_STYLE_FALLBACK). That mapping is science-lane semantics living in the
   writer; fine for v1 (the contract passes style_slug through), but NOTE it as a known
   science-ism the media/PD lanes will simply ignore (their fetchers ignore style_slug).
   No code change.
5. The sweep's fetcher/interpreter registration check calls
   `_otr_source_payload.registered_*_ids()` -- ensure the check does NOT import/execute
   wrapper bodies (ids only; lazy law preserved). Trivial but state it.

CONFIRMED-OK: bank binding at :2605 is a one-line change; top-level import posture safe
both edges; degrade-branch variables (casting_brief=""/script_brief=""/key_terms_tuple=())
untouched by the re-route.
