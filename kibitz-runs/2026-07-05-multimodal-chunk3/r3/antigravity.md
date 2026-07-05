VERDICT: build-ready as-is? yes-with-fixes. The plan is sound but requires two critical fixes to prevent telemetry recording failures and downstream typing crashes.

MUST-FIX BEFORE BUILD:
1. [Section 1c] Missing slot-scheduler helper context: The plan replaces the direct news_interpreter call in D.2.5 of nodes/OTR_LedgerScriptWriter.py with the new interp() contract call. However, it does not specify wrapping interp() inside `with slot_scheduler.helper_context("build_news_briefs"):`. Failing to do so will bypass slot tracking, breaking `meta["slot_calls_by_helper"]` and slot transition telemetry invariants.
   Fix: Wrap `briefs = interp(...)` inside the `with slot_scheduler.helper_context("build_news_briefs"):` context block in `run()` D.2.5.
2. [Section 1a] Lack of type validation for dictionary values in validate_interpreter_result: The duck-typed validation contract in nodes/_otr_source_payload.py checks only the presence of keys in the dictionary returned by `model_dump()`. It does not check if their values are of the correct types. Since downstream consumers (e.g., nodes/OTR_LedgerScriptWriter.py:4330 and nodes/_otr_story_spine.py:212) read `news_close_brief` directly from the news dictionary and expect a `str`, returning non-string types will lead to unhandled late runtime crashes.
   Fix: In `validate_interpreter_result`, assert that `model_dump()["casting_brief"]`, `model_dump()["script_brief"]`, and `model_dump()["news_close_brief"]` are instances of `str`, and `model_dump()["key_terms"]` is an iterable of `str`.

SHOULD-FIX:
1. [Section 1b] Optional notes field list-of-str validation in _parse_pipeline: The plan adds list-of-str validation for notes in nodes/_otr_story_routing.py:_parse_pipeline (line 272), but notes is a defaulted field (defaulting to `()` in `StoryPipeline`). Using a naive `_require_str_list` on a missing key will fail.
   Fix: Validate notes using `_require_str_list(obj, "notes", origin)` only if `"notes"` in `obj`, otherwise default to `()`.

OPTIONAL / NICE-TO-HAVE:
1. [Section 1c] Unused import cleanup in writer: Since `except _OTRNI.NewsInterpreterError` in nodes/OTR_LedgerScriptWriter.py is replaced by `except SourceInterpretError`, the local import `from . import news_interpreter as _OTRNI` at line 2773 is no longer required and should be removed.

CUT THESE (over-engineering):
None.

[ASSUMPTION] We assume that `slot_scheduler` is always initialized and present during the `run()` call, which is true for the current production ledger writer implementation (nodes/OTR_LedgerScriptWriter.py:2826).
