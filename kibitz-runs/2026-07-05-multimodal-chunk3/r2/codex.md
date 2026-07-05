VERDICT: yes-with-fixes. Core shape is implementable, but the plan has two undefined enforcement points and at least one existing test will fail unchanged.

MUST-FIX BEFORE BUILD:
1. [1c] `bank` is not defined at the D.2.5 interpreter reroute point. Current `run()` discards `_otr_story_routing.require_runnable_bank(source_bank)` and later only has `resolved["source_bank"]` before the news block. See `nodes/OTR_LedgerScriptWriter.py:2605` and `nodes/OTR_LedgerScriptWriter.py:3005`. Concrete fix: bind `bank = _otr_story_routing.require_runnable_bank(source_bank)` at the gate, pass that object into `_resolve_inputs` or re-fetch once with `get_bank(resolved["source_bank"])` before `resolve_interpreter(bank)`.

2. [1a/1c/3] Hard-halt exception identity will drift. Today the writer catches `_OTRNI.NewsInterpreterError` and re-raises that same exception on required halt: `nodes/OTR_LedgerScriptWriter.py:3039`, `nodes/OTR_LedgerScriptWriter.py:3090`. The plan wraps it as `SourceInterpretError` and catches/re-raises that, while acceptance says science halt/degrade semantics stay byte-identical. Concrete fix: on required halt, if `SourceInterpretError.__cause__` is a `NewsInterpreterError`, stamp from the cause and re-raise the cause; only raise `SourceInterpretError` directly for non-science/no-cause interpreters.

3. [1a/2.8] Interpreter output contract has no enforcement API. The plan lists required attributes and `model_dump()` keys, but D.2.5 currently consumes outputs directly (`model_dump`, `.casting_brief`, `.script_brief`, `.key_terms`, `.attempts`) at `nodes/OTR_LedgerScriptWriter.py:3030-3037`. A missing field would become `AttributeError`/bad downstream data, not `SourcePayloadContractError`. Concrete fix: add `validate_interpreter_result(result, origin)` in `_otr_source_payload.py`, verify direct attrs plus `model_dump()` key/types, call it before returning/using briefs, and make test 2.8 assert that error type.

4. [2.4/2.6] Existing AST test will fail after removing the direct writer call. `tests/test_writer_input_resolve.py:34-66` currently requires `_resolve_inputs` to contain a direct `_fetch_rss_seed_or_die(...)` call with `technical_model` as the second positional arg. The new design intentionally moves that call into `_otr_source_payload.py`. Concrete fix: replace that test in the same commit with a wrapper-forwarding test and/or an AST guard that `_resolve_inputs` no longer calls `_fetch_rss_seed_or_die` directly.

SHOULD-FIX:
1. [1b/2.3] Name the required synthetic-registry updates explicitly. `tests/test_story_routing_stage2.py:48-55` builds pipeline rows without `requires_source_contract`; once `_parse_pipeline` requires the bool, this helper must add it or most routing tests will fail before reaching the new assertions.

2. [1b] Tighten `notes` while touching `_parse_pipeline`. Current parser only does `notes = tuple(obj.get("notes", []))` with no list/string validation in `nodes/_otr_story_routing.py:267`; adding a new strict bool beside an untyped sequence leaves an avoidable schema hole.

OPTIONAL / NICE-TO-HAVE:
- Add `__all__` exports for the new source-payload public API so tests and future callers do not reach into private registry names.

CUT THESE (over-engineering):
1. [2.6] The AST guard for resolver calls being outside any `try/except` that catches `SourcePayloadError` is brittle. A direct unit test that `SourceContractMissingError` propagates through `_resolve_inputs`/D.2.5 without degrade is smaller and checks behavior, not formatting.