VERDICT: yes-with-fixes. Two contract holes can let registry/runtime wiring pass while downstream breaks.

MUST-FIX BEFORE BUILD:
1. [1a, 1c] `validate_interpreter_result` must validate the dumped `meta["news"]` payload, not only direct attrs/key presence. Writer stamps `meta["news"] = briefs.model_dump()` at `nodes/OTR_LedgerScriptWriter.py:3030`, while freeze validation requires `meta.news.key_terms` to be a list at `nodes/_otr_ledger_freeze.py:232-250`. Current plan only requires dump keys, so an interpreter can pass direct attrs but dump `key_terms` as tuple/string/null and break later. Fix: require dump fields `casting_brief`, `script_brief`, `news_close_brief` are `str` and `key_terms` is `list[str]`; have `validate_interpreter_result` return that validated dump and make writer assign that exact object.

2. [1a] Direct `.key_terms` contract says `iterable[str]`, which accidentally admits a bare `str`; writer immediately does `tuple(briefs.key_terms)` at `nodes/OTR_LedgerScriptWriter.py:3033`, which would split a string into characters. Existing science contract is `key_terms: list[str]` at `nodes/news_interpreter.py:175`. Fix: reject `str`/`bytes`; require a non-string iterable of non-empty strings, materialized once.

3. [1b, 2.3] The planned sweep test allowing `runnable:true + requires_source_contract:false + empty ids` creates a runtime-broken state unless a simple_4 runner dispatch is added in the same change. Today `require_runnable_bank` checks only `bank.runnable` at `nodes/_otr_story_routing.py:443-453`; `pipeline.executable` is explicitly metadata-only at `nodes/_otr_story_routing.py:14-17`; the production writer still enters `_resolve_inputs` and, on empty custom premise, fetches via the source path at `nodes/OTR_LedgerScriptWriter.py:1369-1392`. [ASSUMPTION] No hidden simple_4 runtime dispatcher exists; searched production references and found only registry routing. Fix: either drop that runnable-true fixture until the simple_4 runner ships, or add a validation rule that `requires_source_contract:false` runnable banks are allowed only when their pipeline has an actual executable runner wired.

SHOULD-FIX:
1. [1c] Remove the stale late `from . import news_interpreter as _OTRNI` import when rerouting D.2.5. It currently lives at `nodes/OTR_LedgerScriptWriter.py:2773`; leaving it defeats the new module’s lazy ownership of `news_interpreter` and is easy to miss because the call moves out of the writer.

2. [1b] Implement the `requires_source_contract` sweep after the existing default pack/default pipeline equality check at `nodes/_otr_story_routing.py:361-367`, or explicitly document why `bank.default_story_pipeline` is trusted before that check. This prevents validating against one pipeline while the default pack declares another.

OPTIONAL / NICE-TO-HAVE:
- Make `validate_source_payload` return a shallow copy so fetcher-owned dicts cannot be mutated after validation.
- Add one test where `model_dump()["key_terms"]` is a tuple while direct `.key_terms` is valid; it should fail before ledger freeze.

CUT THESE (over-engineering):
1. [2.6] The broad AST guard banning every production call to `build_news_briefs` outside `_otr_source_payload.py` is brittle if it scans helper aliases or future wrappers. Keep a focused writer test proving D.2.5 calls `resolve_interpreter` and no longer calls `_OTRNI.build_news_briefs`; that closes the real wiring risk.