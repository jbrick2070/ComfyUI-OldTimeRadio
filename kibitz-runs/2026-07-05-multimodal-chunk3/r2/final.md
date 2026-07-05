# r2 JUDGMENT (Cowork Claude, anchor + judge) -- chunk 3, coding plan round

## Accepted
- CODEX M1 (CONFIRMED :2605/:3005): run() discards require_runnable_bank's return; v3
  binds `bank = require_runnable_bank(source_bank)` at the gate and D.2.5 uses it (or
  re-fetches via get_bank(resolved["source_bank"]) -- builder's choice, one object).
- CODEX M2 (CONFIRMED :3090 -- the halt path re-raises the CAUGHT exception): wrapping
  would change the exception TYPE that surfaces to the graph on halt. v3: halt path
  stamps from `__cause__` AND re-raises `exc.__cause__` when present (science surfaces
  NewsInterpreterError byte-identically); raises the SourceInterpretError itself only
  when no cause.
- CODEX M3: new `validate_interpreter_result(result, origin)` enforcement API (direct
  attrs + model_dump required keys); a violation raises SourcePayloadContractError which
  is NOT caught by the SourceInterpretError except-clause -- contract bugs propagate hard,
  never degrade. Test 2.8 asserts the error type.
- CODEX M4 (verify-at-build CONFIRMED by codex citation): tests/test_writer_input_resolve.py
  :34-66 pins the direct `_fetch_rss_seed_or_die` call in `_resolve_inputs`; replaced SAME
  COMMIT with the wrapper-forwarding pin + a no-direct-call assertion.
- CODEX S1 = AG M1 = ANCHOR M1: `_pipe_row()` gains `"requires_source_contract": False`;
  AG M2/M3: _PIPELINE_KEYS + StoryPipeline dataclass + _parse_pipeline threading -- all
  SAME COMMIT, field REQUIRED bool (matches `executable` posture).
- CODEX S2: validate `notes` as list-of-str while touching _parse_pipeline.
- CODEX OPT: `__all__` on the new module.
- CODEX CUT-1: AST guard (B) REPLACED by behavior tests (SourceContractMissingError
  propagates un-degraded through _resolve_inputs; SourcePayloadContractError propagates
  through D.2.5). Guard (A) (no direct calls) stays.
- AG S1: explicit single-base inheritance stated. AG S2: routing imports
  `from . import _otr_source_payload` top-level.
- ANCHOR M2: writer imports the new module TOP-LEVEL beside :131 (routing precedent).
- ANCHOR M3: the source->outlet / date->pub_date RENAME mapping is asserted explicitly in
  the mock-pin test.

## Rejected
- AG OPT (local import inside _resolve_inputs/run): conflicts with the :131 top-level
  routing-import precedent and the module is import-light by contract. Top-level.

## Verify-at-build
- run()'s broader try/except topology around D.2.5 (behavior tests cover the semantics).
- _pipe_row call-site count (~30) all inherit the fixture default.
