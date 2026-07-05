# Lane-Enablement CHUNK 3 -- Source-Payload Fetcher/Interpreter Contracts (v3, post-r2)

Date: 2026-07-05. Branch: `v2.0-alpha`. Parent: `docs/multimodal-story-schema/STAGE2_SUBPLAN.md`
section 4b item 3. Status: v3 after kibitz r1+r2 (codex + antigravity, Claude anchor+judge --
judgments `kibitz-runs/2026-07-05-multimodal-chunk3/{r1,r2}/final.md`). r3+r4 pending.

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
  `.attempts`. Downstream consumers ALSO read `news_close_brief` out of the
  `meta["news"]` dump (writer :4330 outro, video_engine :1787 HUD, line_composer coda) --
  the dump contents are part of the contract, not just the direct attributes.
  Failure type `NewsInterpreterError` drives the Sprint-2.2 halt/degrade switch
  (`news_briefs_required` widget + `OTR_NEWS_BRIEFS_REQUIRED=0` escape hatch); the halt
  path stamps `meta["news_briefs_halt_reason"] = f"{type(exc).__name__}: {exc}"` (:3083).
- `banks.json` rows already carry `fetcher` / `interpreter` string ids
  (science: `"science_rss"` / `"news_interpreter"`; the 3 non-science banks: `""`) --
  pure metadata today, consulted by nothing.

**SCOPE DECLARATION (r1, codex M3):** the payload contract below is the
`legacy_many_pass` ARTICLE ADAPTER -- the uniform surface that pipeline's
`source_interpret` pass consumes. It is NOT a universal source packet. Lane-specific
provenance (public-domain author/rights, media-archive URL/sha) rides each lane's
enablement WITH its consumer, per the Stage-2 "no fields without consumers" law. The
simple_4 pipeline never touches this contract (see 1b).

Chunk 3 = make the two bank fields LIVE routing coordinates behind a typed, fail-loud
contract, with the science lane byte-identical. It does NOT build any non-science
fetcher/interpreter (no fake curation).

## 1. Design (one new module + one pipeline field + two writer re-routes + sweep rules)

### 1a. New module `nodes/_otr_source_payload.py`

Import posture: stdlib-only at import time; imports NEITHER the writer NOR
news_interpreter NOR `_otr_story_routing` at module level (three-edge cycle guard,
test-pinned); heavy imports happen LAZILY inside wrapper bodies; zero I/O at import.
The bank parameter is DUCK-TYPED (`.fetcher`/`.interpreter`/`.source_bank_id`
attributes); no runtime routing import (r1 codex S1).

- **Typed errors (own hierarchy -- NOT StoryRoutingError subclasses; r1 codex CUT-1):**
  `SourcePayloadError(Exception)` base; ALL others EXPLICITLY subclass it (r2 AG S1):
  `UnknownFetcherError` / `UnknownInterpreterError` (non-empty id not registered);
  `SourceContractMissingError` (empty id at resolution -- names the bank + the 4b
  checklist item); `SourcePayloadContractError` (payload/interpreter-result shape
  violation); `SourceInterpretError` (interpreter execution failure). Module exports
  `__all__` (r2 codex OPT).
  Layering (r1 AG M2 clarification): the SWEEP raises `RegistryValidationError` at
  registry load for production registries; the resolver's Unknown*/Missing errors are
  DEFENSE-IN-DEPTH for direct/synthetic callers -- both exist, deliberately.
- **Payload contract:** `SOURCE_PAYLOAD_KEYS = frozenset({"headline", "summary",
  "full_text", "source", "date", "link", "seed_text"})`.
  `validate_source_payload(payload, origin) -> dict`: EXACT key set (unknown key = hard
  error; r1 rejected superset -- no consumer, typo guard, registry precedent), every
  value `str`, `seed_text` non-empty after strip. No coercion.
- **Fetcher registry:** `_FETCHERS: dict[str, FetcherEntry]`, `FetcherEntry` = frozen
  dataclass `(fetch: callable, seed_source: str)`. `seed_source` is REGISTRY metadata
  (r1 rejected in-band dynamic seed_source -- payload shape stays frozen).
  Contract signature (keyword-only, fixed): `fetch(*, bank, style_slug: str,
  technical_model: str) -> dict`; fetchers may IGNORE inputs irrelevant to them (a local
  text loader ignores style_slug). The RESOLVER's caller passes the result through
  `validate_source_payload`.
  v1 content: `"science_rss"` -> lazy wrapper around the writer's
  `_fetch_rss_seed_or_die(style_slug, technical_model)` verbatim,
  `seed_source="rss_fetch"` (byte-identical stamps).
- **Interpreter registry:** `_INTERPRETERS: dict[str, callable]`.
  Contract signature (keyword-only): `interpret(*, bank, payload: dict, technical_fn,
  style: str, model_id: str) -> briefs-like object`.
  **Contract surface (minimal duck pin, r1 codex CUT-2 + AG M1):** direct attributes
  `.model_dump() -> dict`, `.casting_brief: str`, `.script_brief: str`,
  `.key_terms` (iterable[str]), `.attempts: int`; AND `model_dump()` MUST contain keys
  `{casting_brief, script_brief, news_close_brief, key_terms}` (downstream reads
  news_close_brief out of the dump). Pinned by a contract test, no Protocol class.
  **Enforcement API (r2 codex M3):** `validate_interpreter_result(result, origin)` in
  this module checks the direct attributes + the model_dump required keys and raises
  `SourcePayloadContractError` on violation; the writer calls it on the interp() return.
  Contract violations are NOT caught by the SourceInterpretError except-clause -- they
  propagate as hard bugs, never degrade.
  v1 content: `"news_interpreter"` -> lazy wrapper calling `build_news_briefs(
  technical_fn=..., full_text=payload["full_text"], headline=payload["headline"],
  summary=payload["summary"], outlet=payload["source"], pub_date=payload["date"],
  style=style, seed=0, model_id=model_id)` -- kwargs verbatim -- translating
  `NewsInterpreterError` -> `SourceInterpretError` (chained via `raise ... from exc`,
  message preserved). The wrapper translates ONLY `NewsInterpreterError`; ANY other
  exception propagates untouched (today's behavior; anchor M2, test-pinned).
- **Resolution API:** `resolve_fetcher(bank) -> FetcherEntry` /
  `resolve_interpreter(bank) -> callable`; empty id -> `SourceContractMissingError`
  naming the bank (a selected bank without a built lane fails LOUD -- never a silent
  slide into science); unknown non-empty id -> Unknown*Error. Plus
  `registered_fetcher_ids()` / `registered_interpreter_ids()` for the sweep.

### 1b. Pipeline capability flag + routing sweep additions

- **NEW pipelines.json field `requires_source_contract: bool`** (r1 codex M2) on every
  pipeline row: `legacy_many_pass` = true (its source_interpret pass consumes the
  contract); `simple_4_prompt_experimental` = false (its 4-pass runner defines its own
  source mechanism when it ships). REQUIRED bool (matches `executable` posture). Threads
  SAME COMMIT through (r2 AG M1-M3 + codex S1): `_PIPELINE_KEYS`, `_parse_pipeline`
  (required-bool parse + pass into the dataclass), `StoryPipeline` dataclass (field
  before the defaulted `notes`), the shipped pipelines.json rows, AND the synthetic
  fixture `tests/test_story_routing_stage2.py::_pipe_row` (gains
  `"requires_source_contract": False` -- all call sites inherit). While touching
  `_parse_pipeline`, `notes` gains list-of-str validation (r2 codex S2). Metadata for
  VALIDATION-time rules only -- never consulted at run time (same law as `executable`).
- Sweep additions (`_otr_story_routing._sweep_and_crossref`):
  (a) ANY bank with a NON-EMPTY `fetcher`/`interpreter` id must be registered in
  `_otr_source_payload` (dangling id = `RegistryValidationError` -- typo guard, all
  banks);
  (b) any bank with `runnable: true` WHOSE default pipeline has
  `requires_source_contract: true` must have BOTH ids non-empty (and per (a),
  registered). Science satisfies; a future simple_4 runnable flip is NOT forced to carry
  legacy fetcher ids.
- Import note: `_otr_story_routing` gains `from . import _otr_source_payload` top-level
  (r2 AG S2; safe: the latter is stdlib-only at import; lazy posture preserved,
  test-pinned). The WRITER also imports it TOP-LEVEL beside the :131 routing import
  (r2 anchor M2; the module is import-light by contract -- AG's local-import suggestion
  rejected).

### 1c. Writer re-routes (both science byte-identical)

- `_resolve_inputs` (RSS branch only): replace the hardcoded call with
  `entry = _otr_source_payload.resolve_fetcher(_otr_story_routing.get_bank(source_bank))`;
  `news_article = validate_source_payload(entry.fetch(bank=..., style_slug=rss_style_slug,
  technical_model=technical_model), origin=...)`; `seed_source = entry.seed_source`.
  The `custom_premise` branch is UNCHANGED except its synthesized dict also passes
  `validate_source_payload` (uniform surface; conforms by construction, test-pinned).
  `source_bank` default `"science_news"` (:1444) is PRESERVED so direct/test/refine
  callers stay green. run() gates `require_runnable_bank` FIRST (:2605), so the
  production path only sees runnable banks; MissingContract is defense-in-depth.
- **Bank binding (r2 codex M1):** run() currently DISCARDS require_runnable_bank's
  return (:2605); v3 binds `bank = _otr_story_routing.require_runnable_bank(source_bank)`
  at the gate (or re-fetches once via `get_bank(resolved["source_bank"])` before D.2.5 --
  one object, builder's choice) so `resolve_interpreter(bank)` has a real bank row.
- `run()` D.2.5: `interp = resolve_interpreter(bank)` (resolved OUTSIDE the try);
  inside the try: call interp with the payload dict, then
  `validate_interpreter_result(briefs, origin)`. `except SourceInterpretError` replaces
  `except _OTRNI.NewsInterpreterError` with the Sprint-2.2 halt/degrade logic VERBATIM
  (news_briefs_required + env escape hatch + meta["news"]=None + halt-reason stamp),
  with TWO identity fixes (r1 M1 + r2 codex M2):
  (a) the halt-reason stamp derives from `exc.__cause__` when present (science stamp
  stays "NewsInterpreterError: ...", byte-identical);
  (b) the required-halt RE-RAISES `exc.__cause__` when present (science surfaces
  NewsInterpreterError to the graph exactly as today, :3090), raising the
  SourceInterpretError itself only when no cause exists.
  `meta["news"]` key name KEPT (legacy back-compat key -- documented, not renamed).

### 1d. Explicitly OUT of chunk 3

- No non-science fetchers/interpreters; no banks.json content change; no change to
  custom_premise semantics; no workflow-JSON change (verified at build: no new node
  inputs/widgets); no meta key renames.

## 2. Tests (new `tests/test_source_payload_chunk3.py` + same-commit updates)

1. Payload validator matrix: conforming passes; missing key / EXTRA key / non-str value /
   empty seed_text raise `SourcePayloadContractError`.
2. Registry resolution: science resolves both; empty-id bank (media_archive row) raises
   `SourceContractMissingError` naming the bank; synthetic bank rows with unknown
   non-empty ids raise Unknown*Error (direct resolver calls -- no registry load involved).
3. Sweep rules (monkeypatched _STORY_PACKS_ROOT fixture, 2A pattern): dangling fetcher id
   fails load; runnable:true + requires_source_contract pipeline + empty interpreter
   fails load; runnable:true bank on a requires_source_contract=false pipeline with empty
   ids LOADS (simple_4 future-proof pin).
4. Science byte-identity: science_rss wrapper forwards EXACTLY (style_slug,
   technical_model) to `_fetch_rss_seed_or_die` (mock pin); news_interpreter wrapper
   forwards the EXACT kwarg set incl. seed=0 with the RENAME mapping asserted explicitly
   (payload["source"]->outlet, payload["date"]->pub_date; r2 anchor M3); seed_source
   stamps "rss_fetch"/"custom_premise" unchanged; custom_premise dict passes validation.
   SAME COMMIT: `tests/test_writer_input_resolve.py:34-66` (pins the direct
   `_fetch_rss_seed_or_die` call in `_resolve_inputs`) replaced with the
   wrapper-forwarding pin + a no-direct-call assertion (r2 codex M4).
5. Error translation + halt fidelity: NewsInterpreterError -> SourceInterpretError with
   `__cause__`; non-NIE exceptions propagate UNtranslated; halt path stamps
   "NewsInterpreterError: ..." AND re-raises the cause (byte-identical surface); degrade
   path (required=False / env hatch) behavior unchanged. Existing Sprint-2.2 tests
   re-pointed SAME COMMIT.
6. Guards: (A) AST -- no production CALL node outside `_otr_source_payload.py` calls
   `_fetch_rss_seed_or_die` or `build_news_briefs` (definitions + tests exempt);
   (B) BEHAVIOR (r2 codex CUT-1 replaced the AST variant): SourceContractMissingError
   propagates un-degraded out of `_resolve_inputs`; SourcePayloadContractError from
   validate_interpreter_result propagates out of run()'s D.2.5 (never caught by the
   degrade branch).
7. Lazy/cycle guard: importing `_otr_source_payload` does no file I/O and imports none of
   {writer, news_interpreter, story_orchestrator, _otr_story_routing} (sys.modules pin).
8. Contract-surface pin: direct attributes AND required model_dump() keys incl.
   news_close_brief (a briefs-like stub missing one fails).

## 3. Acceptance

- Science lane byte-identical: stamps (seed_source, meta["news"], halt reason string),
  halt/degrade semantics, kwargs into build_news_briefs.
- A non-science bank at run-intent still dies at `require_runnable_bank` (unchanged); a
  runnable flip without contracts on a source-contract pipeline is IMPOSSIBLE at registry
  load. 4b item 3 = DONE; remaining = item 4 (seam audit) + per-lane curation.
- Full suite + Bug Bible + B7 green; UTF-8 no BOM; commit + push to v2.0-alpha.

## 4. Resolved questions (r1)

Q1 own hierarchy (codex CUT-1). Q2 registry metadata (codex + judge). Q3 no behavior
change, test-pinned. Q4 EXACT key set (codex + judge; AG superset rejected).

## 5. Verify-at-build

- Exact except-clause shape + degrade-branch variable names at writer :3039-3102.
- pipelines.json field addition vs any test that pins pipeline row keys byte-for-byte.
- The writer's local import convention for the two new modules (match file style).
