# ANCHOR REVIEW (Cowork Claude) -- chunk 3 r2 (coding plan / implementability)

Doc: CHUNK3_SOURCE_PAYLOAD_SUBPLAN.md v2. Extra grounding this round:
tests/test_story_routing_stage2.py (:36-90 fixtures), writer :131 (routing import style).

VERDICT: SHIP WITH FIXES (small, all mechanical).

MUST-FIX:
1. **`requires_source_contract` breaks every synthetic `_pipe_row()` fixture.**
   test_story_routing_stage2.py builds pipeline rows WITHOUT the new key (:48-57), and
   `_check_unknown_keys`/required-bool parsing is exact. Decide explicitly: the field is
   REQUIRED (fail-loud posture, matches `executable`) and `_pipe_row()` gains
   `"requires_source_contract": False` SAME COMMIT (one line, all ~30 call sites inherit),
   + shipped pipelines.json rows updated. State this in the plan so the builder doesn't
   discover it mid-suite. (CONFIRMED :48-57.)
2. **Writer import style is TOP-LEVEL package-relative** (`from . import
   _otr_story_routing ...` :131) -- the plan says "match file style"; pin it: add
   `from . import _otr_source_payload as _otr_source_payload` at :131-adjacent, NOT a
   local import (the module is import-light by contract, so top-level is safe and matches
   the routing import precedent). (CONFIRMED :131.)
3. **The interpreter wrapper's payload->kwarg mapping must be pinned against the payload
   VALIDATOR, not hand-typed twice.** build kwargs from the validated dict keys
   (outlet=payload["source"], pub_date=payload["date"]) -- the two RENAMED keys are the
   foot-gun; the mock-pin test (2.4) must assert the RENAME mapping explicitly.

SHOULD-FIX:
4. Test 2.3's "runnable:true + requires_source_contract=false pipeline + empty ids LOADS"
   needs the synthetic bank to also satisfy the EXISTING cross-refs (default pack exists,
   required_seams present) -- reuse `_mk_registry` wholesale; note it.
5. AST guard (B) scope: the writer calls resolve_fetcher inside `_resolve_inputs` (which
   IS called inside run()'s body; run has broad try/except? verify at build -- if run()
   wraps _resolve_inputs in a try/except Exception for ledger stamping, guard (B) must
   target the IMMEDIATE lexical context, same as the chunk-2 pin).

CONFIRMED-OK (no change): three-edge lazy guard is implementable (routing is already
lazy; source_payload has no top-level heavy imports by construction); halt-reason
__cause__ fix is a 3-line change at :3083; seed_source registry metadata threads through
`_resolve_inputs` cleanly (entry.seed_source replaces the literal).
