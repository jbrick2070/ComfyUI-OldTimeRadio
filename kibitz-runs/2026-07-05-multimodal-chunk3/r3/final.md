# r3 JUDGMENT (Cowork Claude, anchor + judge) -- chunk 3, wiring round

## Accepted
- CODEX M1 + AG M2 (dump content types): `validate_interpreter_result` validates the
  DUMP VALUES -- casting_brief/script_brief/news_close_brief are str, key_terms is
  list[str] (ledger-freeze expects a list, _otr_ledger_freeze :232-250 per codex; spot-
  check at build) -- and RETURNS the validated dump; the writer assigns that exact object
  to meta["news"] (single validation point, no double dump).
- CODEX M2: direct `.key_terms` contract = NON-STRING iterable of non-empty strings
  (str/bytes rejected -- tuple(str) would char-split at :3033).
- CODEX M3 (reshaped): the sweep rule for runnable banks becomes EITHER
  (pipeline.requires_source_contract=true AND both ids non-empty+registered) OR
  (pipeline.requires_source_contract=false AND pipeline.executable=true -- the runner
  must exist before a non-contract bank can flip runnable; pipelines.json notes already
  bind executable's flip to the runner shipping). This is a VALIDATION-time read of
  `executable` (same class as the precedence equality check), NOT a runtime gate -- the
  metadata-only runtime law stands. Fixture matrix updated accordingly.
- CODEX S1 = AG OPT: delete the stale `from . import news_interpreter as _OTRNI` local
  import at :2773 (verify no remaining _OTRNI use in run() at build).
- CODEX S2: implement the sweep addition AFTER the precedence-equality check (:361-367).
- CODEX OPTs: validate_source_payload returns a shallow copy; add the tuple-dump
  negative test.
- CODEX CUT (partial): guard (A) SPLIT -- keep the simple Name-call AST ban for
  `_fetch_rss_seed_or_die` (writer-internal name, alias-free); replace the
  `build_news_briefs` half with the focused writer test: run() contains no
  `_OTRNI.build_news_briefs` attribute call and D.2.5 calls resolve_interpreter.
- AG M1 (CONFIRMED :3015): the interp() call STAYS inside
  `with slot_scheduler.helper_context("build_news_briefs"):` -- helper-label telemetry
  byte-identical (label string kept for stamp compatibility).
- AG S1: notes validation stays optional-key-tolerant (validate only when present,
  default ()).
- ANCHOR M1: S31 B6 invariant survives via BOTH pins (writer kwarg pin + wrapper 2nd-
  positional pin). ANCHOR M2: post-change OTR_WorkflowValidator + widget audit with
  expected NO diff. ANCHOR M3: build sequencing (module green in isolation -> routing ->
  writer -> full suite; single commit). ANCHOR S5: sweep reads ids only, never executes
  wrapper bodies.

## Rejected
- None material this round (all four reviews converged on overlapping fixes).

## Verify-at-build
- _otr_ledger_freeze.py :232-250 key_terms list expectation (codex citation).
- _otr_story_spine.py :212 news_close_brief read (AG citation).
- No remaining _OTRNI references in run() after the reroute.
