# Lane-Enablement CHUNK 3 -- Source-Payload Fetcher/Interpreter Contracts (DRAFT v1, pre-kibitz)

Date: 2026-07-05. Branch: `v2.0-alpha`. Parent: `docs/multimodal-story-schema/STAGE2_SUBPLAN.md`
section 4b item 3. Status: DRAFT -- kibitz r1..r4 pending (codex + antigravity panel,
Cowork Claude anchor + judge).

## 0. Problem (grounded)

The source-payload stage is the LAST science-hardwired stage before a non-science bank can
flip `runnable:true`. Today:

- **Fetch** is hardwired in `OTR_LedgerScriptWriter._resolve_inputs` (:1388):
  `_fetch_rss_seed_or_die(rss_style_slug, technical_model)` -> the canonical article dict
  `{headline, summary, full_text, source, date, link, seed_text}` (all str; seed_text
  non-empty). The `custom_premise` widget path (:1354-1368) synthesizes the SAME dict shape
  (bank-agnostic bypass) and stamps `seed_source="custom_premise"`; the RSS path stamps
  `seed_source="rss_fetch"`.
- **Interpret** is hardwired in `run()` D.2.5 (:3016):
  `news_interpreter.build_news_briefs(technical_fn=..., full_text, headline, summary,
  outlet, pub_date, style, seed=0, model_id)` -> a briefs object consumed via
  `.model_dump()` (-> `meta["news"]`), `.casting_brief`, `.script_brief`, `.key_terms`,
  `.attempts`. Failure type `NewsInterpreterError` drives the Sprint-2.2 halt/degrade
  switch (`news_briefs_required` widget + `OTR_NEWS_BRIEFS_REQUIRED=0` escape hatch).
- `banks.json` rows already carry `fetcher` / `interpreter` string ids
  (science: `"science_rss"` / `"news_interpreter"`; the 3 non-science banks: `""`) --
  pure metadata today, consulted by nothing.

Chunk 3 = make those two fields LIVE routing coordinates behind a typed, fail-loud
contract, with the science lane byte-identical. It does NOT build any non-science
fetcher/interpreter (no fake curation -- each lane's concrete implementation rides its own
lane-enablement, same law as the Stage-4 rules packs).

## 1. Design (one new module + two writer re-routes + sweep rules)

### 1a. New module `nodes/_otr_source_payload.py` (import posture: stdlib-only at import
time; every heavy dependency imported LAZILY inside the wrapper bodies -- same posture as
`_otr_story_routing.py`; zero I/O at import)

- **Typed errors:** `SourcePayloadError(Exception)` base;
  `UnknownFetcherError` / `UnknownInterpreterError` (non-empty id not registered);
  `SourceContractMissingError` (bank has an EMPTY id at run-intent -- names the bank and
  the lane-enablement checklist item); `SourcePayloadContractError` (payload shape
  violation); `SourceInterpretError` (interpreter execution failure -- the contract's
  replacement surface for `NewsInterpreterError` at the writer boundary).
- **Payload contract:** `SOURCE_PAYLOAD_KEYS = frozenset({"headline", "summary",
  "full_text", "source", "date", "link", "seed_text"})`.
  `validate_source_payload(payload, origin) -> dict`: exact key set (unknown key = hard
  error), every value `str`, `seed_text` non-empty after strip. Fail-loud, no coercion.
- **Fetcher registry:** `_FETCHERS: dict[str, FetcherEntry]` where `FetcherEntry` =
  frozen dataclass `(fetch: callable, seed_source: str)`.
  Contract signature (keyword-only): `fetch(*, bank, style_slug: str,
  technical_model: str) -> dict` -- returns a payload that the RESOLVER (not the
  wrapper) passes through `validate_source_payload`.
  v1 registry content: `"science_rss"` -> lazy wrapper around the writer's
  `_fetch_rss_seed_or_die(style_slug, technical_model)` verbatim, `seed_source="rss_fetch"`
  (byte-identical stamps).
- **Interpreter registry:** `_INTERPRETERS: dict[str, callable]`.
  Contract signature (keyword-only): `interpret(*, bank, payload: dict, technical_fn,
  style: str, model_id: str) -> InterpretedSource` where the returned object MUST expose
  `.model_dump() -> dict`, `.casting_brief: str`, `.script_brief: str`,
  `.key_terms` (iterable[str]), `.attempts: int` (duck-typed protocol; pinned by a
  contract test, not an ABC).
  v1 registry content: `"news_interpreter"` -> lazy wrapper calling
  `build_news_briefs(technical_fn=..., full_text=payload["full_text"],
  headline=payload["headline"], summary=payload["summary"], outlet=payload["source"],
  pub_date=payload["date"], style=style, seed=0, model_id=model_id)` -- kwargs verbatim
  -- and translating `NewsInterpreterError` -> `SourceInterpretError` (chained, message
  preserved) so the writer's halt/degrade boundary keys on the CONTRACT type.
- **Resolution API:** `resolve_fetcher(bank) -> FetcherEntry` /
  `resolve_interpreter(bank) -> callable`:
  empty id -> `SourceContractMissingError` naming the bank (this is how a selected bank
  without a built lane fails the episode LOUD -- never a silent slide into the science
  path); unknown non-empty id -> `UnknownFetcherError`/`UnknownInterpreterError`.
  Plus `registered_fetcher_ids()` / `registered_interpreter_ids()` for the sweep.

### 1b. Routing sweep additions (`_otr_story_routing._sweep_and_crossref`)

- Any bank with a NON-EMPTY `fetcher`/`interpreter` id must be registered in
  `_otr_source_payload` (dangling id = `RegistryValidationError`).
- Any bank with `runnable: true` must have BOTH ids non-empty (and therefore registered).
  Science satisfies; this is the registry-level guarantee that keeps every future
  `runnable:true` flip honest about item 3 of the 4b checklist.
- Import note: `_otr_story_routing` imports `_otr_source_payload` top-level (both are
  stdlib-only at import; lazy posture preserved).

### 1c. Writer re-routes (both science byte-identical)

- `_resolve_inputs` (RSS branch only): replace the hardcoded call with
  `entry = resolve_fetcher(get_bank(source_bank))`;
  `news_article = validate_source_payload(entry.fetch(bank=..., style_slug=rss_style_slug,
  technical_model=technical_model), origin=...)`; `seed_source = entry.seed_source`.
  The `custom_premise` branch is UNCHANGED except its synthesized dict also passes
  `validate_source_payload` (uniform surface; the dict already conforms).
  NOTE `_resolve_inputs` already receives `source_bank` (2C threading) and run() already
  gated `require_runnable_bank` FIRST -- so resolve_fetcher on the production path can
  only see a runnable bank; the MissingContract raise is the belt-and-suspenders for
  direct/test callers and future flips.
- `run()` D.2.5: `interp = resolve_interpreter(bank)`; call with the payload dict;
  `except SourceInterpretError` replaces `except _OTRNI.NewsInterpreterError` with the
  Sprint-2.2 halt/degrade logic VERBATIM (news_briefs_required + env escape hatch +
  meta["news"]=None stamp + halt-reason stamp). `meta["news"]` key unchanged.

### 1d. Explicitly OUT of chunk 3

- No non-science fetchers/interpreters (media RSS, public-domain text loader, custom
  packet) -- each rides its lane's enablement with real curation.
- No banks.json content change (ids already present; science already correct).
- No change to the custom_premise semantics (it stays a bank-agnostic source override).
- No widget/JSON change (`workflows/otr_scifi_16gb_full.json` untouched -- verified no
  new node inputs; the source_bank widget shipped in 2C).

## 2. Tests (new file `tests/test_source_payload_chunk3.py` + same-commit updates)

1. Payload validator matrix: conforming dict passes; missing key / extra key / non-str
   value / empty seed_text each raise `SourcePayloadContractError`.
2. Registry resolution: science bank resolves both entries; empty-id bank (media_archive)
   raises `SourceContractMissingError` naming the bank; unknown non-empty id raises
   Unknown*Error (via a synthetic SourceBank row -- no registry mutation).
3. Sweep rules: a temp registry with a dangling fetcher id fails load; a temp registry
   with runnable:true + empty interpreter fails load (monkeypatched _STORY_PACKS_ROOT,
   same fixture pattern as the 2A fail-loud matrix).
4. Science byte-identity: the science_rss wrapper forwards EXACTLY
   `(style_slug, technical_model)` to `_fetch_rss_seed_or_die` (mock pin);
   the news_interpreter wrapper forwards the EXACT kwarg set to `build_news_briefs`
   (mock pin, incl. seed=0); `seed_source` stamps "rss_fetch"/"custom_premise" unchanged.
5. Error translation: NewsInterpreterError inside the wrapper surfaces as
   SourceInterpretError with `__cause__` preserved; the writer halt path (required=True)
   and degrade path (required=False / env hatch) behave identically (existing Sprint-2.2
   tests re-pointed at the contract type SAME COMMIT).
6. AST guards: (A) no production call site outside `_otr_source_payload.py` references
   `_fetch_rss_seed_or_die` or `build_news_briefs` (writer + orchestrator scan; test
   files exempt); (B) `resolve_fetcher`/`resolve_interpreter` calls in the writer are
   NOT wrapped in any try/except that swallows SourcePayloadError (the
   resolve-outside-try pattern from chunk 2).
7. Lazy-import guard: importing `_otr_source_payload` performs no file I/O and does not
   import the writer / news_interpreter / story_orchestrator (sys.modules assertion).
8. Contract-surface pin: the InterpretedSource protocol attributes consumed by run()
   (`model_dump/casting_brief/script_brief/key_terms/attempts`) are pinned so a wrapper
   swap cannot silently drop one.

## 3. Acceptance

- Science lane byte-identical (stamps, meta["news"], halt/degrade semantics, seed_source).
- A non-science bank selected at run-intent still fails at `require_runnable_bank`
  (unchanged); a hypothetical runnable flip without contracts is now IMPOSSIBLE at
  registry load (sweep rule) -- item 3 of the 4b checklist is DONE, leaving only item 4
  (remaining seam audit) + per-lane curation.
- Full suite + Bug Bible + B7 green; UTF-8 no BOM; commit + push to v2.0-alpha.

## 4. Open questions for the panel

- Q1: Should `SourceInterpretError` subclass `StoryRoutingError` for a single catchable
  family, or stay its own hierarchy (draft: own hierarchy; routing errors are registry
  problems, payload errors are execution problems)?
- Q2: Is the FetcherEntry.seed_source label the right home for the "rss_fetch" stamp, or
  should the fetcher return it in-band (draft: registry metadata -- keeps the payload
  shape frozen)?
- Q3: Does `validate_source_payload` on the custom_premise branch risk any behavior
  change (draft: no -- the synthesized dict conforms by construction; test pins it)?
- Q4: Exact-key-set vs superset-allowed on the payload (draft: EXACT, per the registry
  row precedent -- unknown key = hard error).
