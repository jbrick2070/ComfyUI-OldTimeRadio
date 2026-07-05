# ANCHOR REVIEW (Cowork Claude, code-grounded) -- chunk 3 r1 (arc/coherence)

Doc: docs/multimodal-story-schema/CHUNK3_SOURCE_PAYLOAD_SUBPLAN.md (DRAFT v1)
Grounding: OTR_LedgerScriptWriter.py (:1131-1215 fetch, :1354-1392 branch, :2605 gate,
:2991-3102 interpret), nodes/_otr_story_routing.py (full), nodes/story_packs/banks.json.

VERDICT: SHIP WITH FIXES. The arc is right: registry-owned contract ids (JSON) + Python
registries + fail-loud resolution + sweep guarantee on runnable flips is exactly the
chunk-1/2 pattern extended to execution callables. Science byte-identity strategy
(verbatim wrappers + registry-owned seed_source label) is sound. Scope discipline (no
fake non-science curation) matches the Stage-4 precedent.

MUST-FIX (all CONFIRMED against code):
1. **Halt-reason stamp fidelity.** run() :3083-3085 stamps
   `meta["news_briefs_halt_reason"] = f"{type(exc).__name__}: {exc}"`. Catching the
   translated `SourceInterpretError` changes that stamp from "NewsInterpreterError: ..."
   to "SourceInterpretError: ...". Fix: stamp from `exc.__cause__` when present
   (`type(exc.__cause__).__name__`), or format both. Pin with a test. (CONFIRMED :3083)
2. **Translation scope.** The news_interpreter wrapper must translate ONLY
   `NewsInterpreterError`; any other exception from build_news_briefs propagates
   untouched today (:3039 catches NIE only) and must continue to. State it as a hard
   contract line + test. (CONFIRMED :3039)
3. **Import-cycle posture must be explicit both ways.** The science_rss wrapper lazily
   imports the writer at CALL time (fine), but the writer must also import
   `_otr_source_payload` in a way that cannot recurse at module-import time.
   `_otr_story_routing` importing `_otr_source_payload` top-level is safe ONLY if
   `_otr_source_payload` imports neither the writer nor news_interpreter at top level.
   Add the sys.modules lazy-guard test for all three edges. (Plan has the guard for one
   edge; extend.)
4. **AST guard (A) wording.** `_fetch_rss_seed_or_die` is DEFINED in the writer; the
   guard must ban production CALL SITES outside `_otr_source_payload.py` (definition +
   test files exempt), same for `build_news_briefs` (defined in news_interpreter.py,
   called today at writer :3016 only -- CONFIRMED via repo grep).

SHOULD-FIX:
5. `_resolve_inputs` default `source_bank or "science_news"` (:1444) keeps direct/test
   callers on science; the fetch re-route must preserve that default so the 9-part
   self-test (:5996+) and refine paths stay green without threading changes.
   (CONFIRMED :1444, :2605 -- run() gates runnable FIRST, so production non-science
   selection dies before _resolve_inputs; MissingContract raise is defense-in-depth.)
6. Q1: keep the OWN hierarchy (routing errors = registry-load problems; payload errors =
   execution problems). Q2: registry metadata is right (payload shape stays frozen).
   Q4: EXACT key set, per registry precedent.

UNVERIFIABLE (verify-at-build): none material; run() except-clause exact indentation and
the news degrade branch variable names to be re-read at edit time.
