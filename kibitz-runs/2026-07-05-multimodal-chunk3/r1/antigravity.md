VERDICT: yes-with-fixes. The core routing and wrapper structure is sound, but critical omissions in the interpreter contract (news_close_brief) and logical conflicts between the registry sweep and runtime resolution will cause test failures and runtime breakages.

MUST-FIX BEFORE BUILD:
1. [1a + 2.8] Missing news_close_brief in the InterpretedSource protocol and test.
   - Defect: The duck-typed InterpretedSource protocol contract and its corresponding test pin (2.8) omit news_close_brief. However, news_close_brief is a critical LLM-authored brief that is stored in meta["news"] and consumed downstream in announcer outro generation (OTR_LedgerScriptWriter.py:4330), video rendering (video_engine.py:1787), and fallback/coda composition (_otr_line_composer.py:3111, 3431). Without including it, custom interpreters could drop it, passing the contract-surface test but breaking the run downstream.
   - Fix: Add news_close_brief: str to the required attributes of the InterpretedSource protocol in 1a and check it in the test 2.8.

2. [1a + 1b + 2.2] Contradiction between registry load sweep and Unknown*Error test.
   - Defect: The plan defines UnknownFetcherError and UnknownInterpreterError for unregistered fetcher/interpreter IDs resolved at runtime (1a). However, the sweep additions in 1b raise RegistryValidationError if a bank carries an unregistered non-empty ID at registry load time. This makes the runtime Unknown*Error classes dead code in production. The test in 2.2 (verifying that synthetic bank rows with unregistered IDs raise Unknown*Error) will fail during registry load with RegistryValidationError instead.
   - Fix: Permitting unregistered non-empty IDs in the sweep for non-runnable banks (so they only raise Unknown*Error during runtime resolution), or cut the redundant runtime Unknown*Error classes entirely and have the test assert RegistryValidationError.

3. [1a + 1d] Over-constrained exact key-set prevents custom metadata on non-science lanes.
   - Defect: validate_source_payload enforces an exact key set (SOURCE_PAYLOAD_KEYS), raising a hard error on any unknown key. This prevents future non-science fetchers/interpreters [ASSUMPTION: We infer that future non-science fetchers/interpreters (like media_archive or public_domain_story) will require custom/dynamic metadata (e.g. creator, catalog IDs) or dynamic seed sourcing based on their definitions in banks.json and STAGE2_SUBPLAN.md] from carrying source-specific metadata (e.g. archive catalog IDs, creator names, publisher, volume/chapter info) in the payload, defeating the purpose of custom lanes.
   - Fix: Relax validation in validate_source_payload to allow supersets (additional keys) while strictly checking the presence and type of the required 7 base keys.

4. [1a + 4 Q2] Statically bound seed_source prevents dynamic fetcher attribution.
   - Defect: Placing seed_source in the FetcherEntry dataclass as static registry metadata (e.g. seed_source="rss_fetch") prevents fetchers from dynamically reporting their source at runtime (e.g. Gutenberg vs Internet Archive). This rigid design was selected to work around the exact-key-set limitation.
   - Fix: Allow the fetcher to dynamically return seed_source inside the payload dictionary (e.g., payload["seed_source"]), falling back to the registry entry only if absent.

SHOULD-FIX:
1. [1c + 1a] Hardcoded science news nomenclature in general contract.
   - Defect: The plan uses news-centric naming (e.g. meta["news"]) at the general contract wrapper and re-route boundary. For non-science lanes (like public domain stories or media archives), this will store non-news payload metadata under a misleading key.
   - Fix: Rename meta["news"] to a lane-agnostic name (e.g. meta["source_payload"] or meta["interpreted_source"]), or document it clearly as a legacy back-compat key.

2. [1a] Hardcoded fetcher keyword arguments block generic fetchers.
   - Defect: The contract signature fetch(*, bank, style_slug: str, technical_model: str) forces all future fetchers to accept style_slug and technical_model, even if they are local text loaders or random pickers that have no use for them.
   - Fix: Accept **kwargs in the fetch signature, or make style_slug and technical_model optional.

OPTIONAL / NICE-TO-HAVE:
1. Consider defining a formal typing.Protocol for InterpretedSource in nodes/_otr_source_payload.py to make the duck-typing contract explicit and self-documenting for new lane developers.

CUT THESE (scope / over-engineering):
1. [1a + 1c] SourceContractMissingError runtime check.
   - Why: Since the registry sweep ensures that all banks with runnable: true must have both fetcher and interpreter IDs registered (and non-empty), and the writer executes require_runnable_bank first thing in run(), a runnable bank can never have a missing contract at runtime. The check inside resolve_fetcher / resolve_interpreter is redundant and can be replaced with a simple assert or cut entirely.
