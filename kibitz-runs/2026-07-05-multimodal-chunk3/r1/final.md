# r1 JUDGMENT (Cowork Claude, anchor + judge) -- chunk 3 source-payload contracts

Panel: codex + antigravity (claude CLI dropped per operator 2026-07-02). Anchor: anchor.md.

## Accepted (folded into v2 of the subplan)
- ANCHOR M1 = CODEX M1 (CONFIRMED :3083): halt-reason stamp must preserve the underlying
  exception identity -- stamp from `exc.__cause__` when present; test-pinned.
- CODEX M2 (CONFIRMED vs pipelines.json): runnable=>both-ids rule over-reaches -- simple_4
  is a 4-pass runner lane that never uses the legacy fetcher/interpreter. Reshaped: NEW
  pipeline field `requires_source_contract: bool` (legacy_many_pass=true, simple_4=false);
  sweep rule keys on the bank's default pipeline's flag. Dangling NON-EMPTY ids stay a
  load error on every bank (typo guard).
- CODEX M3 (CONFIRMED vs lane packs): payload contract v1 is explicitly SCOPED as the
  legacy_many_pass article adapter, not a universal source packet; provenance fields ride
  each lane's curation WITH their consumer (Stage-2 visual_style-cut precedent).
- CODEX S1: `_otr_source_payload` duck-types the bank param; NO runtime import from
  `_otr_story_routing` (cycle guard); TYPE_CHECKING only.
- CODEX S2 = ANCHOR M4: AST guard bans CALL nodes only; definitions + tests exempt.
- CODEX CUT-1 (=Q1): own error hierarchy, no StoryRoutingError subclassing.
- CODEX CUT-2: no formal Protocol class; minimal duck-typed attribute pin (also rejects
  AG's typing.Protocol optional).
- AG M1 (CONFIRMED, reshaped): downstream consumes `news_close_brief` via
  meta["news"]=model_dump() (writer :4330, video_engine :1787, line_composer coda). The
  contract pin covers BOTH the direct attributes (casting_brief/script_brief/key_terms/
  attempts/model_dump) AND required model_dump() keys {casting_brief, script_brief,
  news_close_brief, key_terms}.
- AG S1 (half): meta["news"] key name documented as legacy back-compat (NOT renamed --
  byte-identity law).
- AG S2 (as doc line): fixed keyword-only fetch signature stays; fetchers may ignore
  irrelevant inputs.
- ANCHOR M2: wrapper translates ONLY NewsInterpreterError; everything else propagates.
- ANCHOR M3: lazy-import guard extended to all three edges (source_payload imports neither
  writer nor news_interpreter nor routing at import time).

## Rejected (with reason)
- AG M2 as stated: test 2.2 uses SYNTHETIC bank rows (no registry load), so no
  contradiction/test failure. The layering is clarified in v2 (sweep=load-time
  RegistryValidationError; resolver Unknown*/Missing = defense-in-depth for direct
  callers), Unknown*Error NOT cut.
- AG M3 (superset payload keys): no consumer exists for extra keys today; EXACT key set is
  the registry-row precedent and the typo guard; the scoped-contract declaration covers
  future growth. (Codex Q4 answer agrees: EXACT.)
- AG M4 (dynamic seed_source in payload): would unfreeze the payload shape for a consumer
  that does not exist; registry metadata suffices (codex concurs); revisit at lane
  curation.
- AG CUT-1 (drop SourceContractMissingError): rejected -- defense-in-depth for direct/test
  callers and future refactors; the no-fallback law prefers a typed loud error over an
  assert.

## Verify-at-build
- Exact except-clause shape + degrade-branch variable names at writer :3039-3102.
- pipelines.json schema addition threads through _PIPELINE_KEYS + _parse_pipeline + all
  existing pipeline tests (row addition is positional-safe -- JSON objects, not widgets).
- The 2C guardrail tests do not pin pipelines.json byte-for-byte (check before adding the
  field).
