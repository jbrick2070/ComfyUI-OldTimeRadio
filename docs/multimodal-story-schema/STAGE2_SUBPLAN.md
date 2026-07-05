# Multi-Modal Story Schema -- STAGE 2 HARDENED SUB-PLAN (v3 FINAL, post-kibitz r1+r2)

Date: 2026-07-05. Branch: `v2.0-alpha`. Parent: `docs/multimodal-story-schema/BUILD_PLAN.md`.
Status: CONVERGED (kibitz r1+r2, codex + antigravity, Claude anchor+judge;
artifacts `kibitz-runs/2026-07-05-multimodal-stage2/{r1,r2}/`). BUILD-READY.
Precondition SHIPPED @843ced43 (outline-resolver swallow removed + AST pin).

## 0. Scope

Stage 2 = story-path ROUTING + lane packs:
1. Registries: `nodes/story_packs/banks.json` + `nodes/story_packs/pipelines.json`.
2. Fail-loud `source_bank`/`story_model`/`story_pipeline` resolution (unknown id =
   hard error, no fallback); router drops `_SCIENCE_PACK_PATH` for bank routing.
3. Lane packs authored (public_domain_story / media_archive / custom_source_bank
   simple_4): addressable + validated, NOT executable yet.

API: `resolve_story_pack(source_bank_id, story_model_id=None)` (model defaults
from the bank; pipeline always from pack/bank in Stage 2 -- no override surface).
NOT in Stage 2: visual_style entirely CUT from the registries (codex r2 cut --
no `default_visual_style` field at all; Stage 3 adds it with its consumer); lane
execution; the widget until 2C.

## 1. Identifier model (converged, canonical)

- **Path IS the coordinate:** `nodes/story_packs/<source_bank_id>/<story_model_id>.json`;
  header triple MUST match path (hard error). simple_4 lives at
  `custom_source_bank/simple_4_prompt_experimental.json`.
- **Run gating (codex r2 M2): `bank.runnable` is the ONLY runtime gate.**
  `pipeline.executable` is METADATA-ONLY (documents whether a JSON-directed pass
  runner exists; legacy_many_pass=false forever by design, simple_4=false until
  its runner ships) and is NEVER consulted at run time. Only `science_news`
  ships `runnable: true` (its execution path is the hardcoded production
  writer). Tests prove science_news+legacy_many_pass runs and that run-intent on
  a runnable:false bank raises `StoryBankNotRunnableError` naming the bank.
- **Pipeline precedence (codex r2 M4):** registry validation REQUIRES
  `default_pack.story_pipeline_id == bank.default_story_pipeline`; mismatch =
  hard error (no "which wins" ambiguity).
- **Seam vocabulary is PRODUCTION-ONLY in all repo JSON.** Lab names never land.
- **Two seam namespaces, no global growth:** `PRODUCTION_SEAM_ALLOWLIST` stays
  EXACTLY the Stage-1 set (exact-equality test unchanged). A pipeline row may
  carry `declared_seams` (list[str], loaded as frozenset). A pack's
  `prompt_stages` keys must be within `PRODUCTION_SEAM_ALLOWLIST UNION
  declared_seams(pack.story_pipeline_id)`.
- **Cache-safe loader split (anchor r2 M1 = codex r2 M1):** `load_pack(path)`
  is UNCHANGED (strict production-only validation, `_PACK_CACHE` by path).
  NEW `load_pack_with_seams(path, extra_seams: frozenset)` validates against
  the union and caches in its OWN cache keyed `(resolved_path, extra_seams)`.
  Its ONLY sanctioned production caller is `_otr_story_routing.py`
  (consumer-guard test pins this). Standalone `load_pack` of the simple_4 pack
  raises `UnknownSeamError` -- correct and intended (antigravity assumption
  confirmed).
- **`bank.required_seams` (antigravity r2 M2):** must be production-allowlisted
  seams ACTUALLY present in the default pack. For the non-runnable lanes that is
  `["line_composer_system", "coda_system"]` only. Seams with no production
  consumer are not required, not authored, not invented.

## 2. Chunk 2A -- registries + resolver (science byte-identical)

- `banks.json` (LIST of bank objects; explicit uniqueness check on
  source_bank_id -- `_reject_dup_keys` cannot see list-element dups, antigravity
  r2 M5). Row schema (exact, unknown key = hard error): source_bank_id (id str),
  label (str), source_kind (str), interpreter (str, may be ""), fetcher (str,
  may be ""), default_story_model (id str), default_story_pipeline (id str),
  defaults (dict; values must be scalars, contents otherwise opaque),
  required_seams (list[str], production names), runnable (bool), guide_ref (str,
  opaque, type-check only).
- `pipelines.json` (LIST; uniqueness on story_pipeline_id). Row schema:
  story_pipeline_id, label, executable (bool, metadata-only), declared_seams
  (list[str], default []), passes (list of {pass_id, slot in
  {creative,technical}, seam_refs list[str], description}), notes (list[str]).
  seam_refs cross-checked against PRODUCTION UNION declared_seams for ALL
  pipelines (cheap + keeps the descriptive row honest; the check is
  load-time-only, never a runtime gate).
- `nodes/_otr_story_routing.py` (stdlib-only): **LAZY** load-once registries --
  ZERO I/O at module import (antigravity r2 M6 + anchor r2 S1); first
  `get_bank`/`get_pipeline`/`resolve_story_pack` call triggers load+sweep,
  cached. Typed errors: `StoryRoutingError` base + `UnknownBankError`,
  `UnknownPipelineError`, `RegistryValidationError`, `StoryBankNotRunnableError`
  (tests assert types, not message text). `require_runnable_bank(bank_id)`
  helper for run-intent gating (consumed in 2C). `_clear_caches()` test hook
  clears the routing registries AND both pack caches (codex r2 S2).
- **Sweep rule (precise; antigravity r2 M3 + codex r2 S3):** top-level files
  `banks.json`/`pipelines.json` are the registries, not packs. Every immediate
  SUBDIRECTORY of `nodes/story_packs/` must be a registered source_bank_id
  (unknown dir = hard error); every `*.json` inside a bank dir must validate
  (with that pack's pipeline's declared_seams) and match path coordinates.
  Registry error messages include registry path + offending bank/model/pipeline
  ids (codex r2 optional folded -- these surface at node-registration time).
- Cross-refs: bank.default_story_model -> on-disk pack; pipeline ids exist;
  precedence equality (section 1); required_seams present in the default pack.
- Router: drop `_SCIENCE_PACK_PATH`; `resolve_story_pack("science_news")`
  (still transitional -- 2C threads the widget selection). Byte-identity pinned
  by the existing equivalence tests.
- Same-commit test updates: `test_stage1b_router_fail_loud_on_missing_pack`
  re-pointed from monkeypatching `_SCIENCE_PACK_PATH` to the routing layer
  (antigravity r2 M4 -- CONFIRMED, the test patches that attr today); consumer
  guard allows `_otr_story_routing.py` + pins load_pack_with_seams callers;
  caller-count pin untouched (routing never calls the resolver).
- New tests: fail-loud matrix (unknown bank/model/pipeline, missing pack,
  header/path mismatch, unknown top-level dir, orphan pack, list-dup ids,
  malformed JSON, precedence mismatch, required_seams absent), lazy-import
  guard (importing _otr_story_routing performs no file I/O -- assert via a
  monkeypatched Path.read_text sentinel or equivalent), non-runnable run-intent
  raise, cache-split guard (strict load_pack of the simple_4 pack STILL raises
  UnknownSeamError even after a routed load cached it).

## 3. Chunk 2B -- lane packs (exact key contract, codex r2 M5)

Exact `prompt_stages` key sets (pinned by per-pack exact-key-set tests):
- `public_domain_story/faithful_radio_adaptation.json`:
  {line_composer_system, coda_system}.
- `media_archive/media_restoration_adventure.json`:
  {line_composer_system, coda_system}.
- `custom_source_bank/simple_4_prompt_experimental.json`:
  {pass_1_creative_story, pass_2_creative_ledger_fill,
  pass_3_technical_schema_cleanup, pass_4_technical_ledger_audit} (declared
  pipeline seams ONLY -- no production seams; the simple_4 lane never uses the
  legacy writer's seams).
Content adapted from the lab prose to the production seam meanings; the lab's
`outline_rules_extra` key is STRIPPED (not in _KNOWN_FIELDS -- it would fail
validation; antigravity r2 M1); status kept as inert metadata
("ready_fixture"/"experimental").
Tests: registry sweep covers each pack; per-pack exact key set;
resolve_story_pack reaches each lane; science byte-identity + audio unchanged.

## 4. Chunk 2C (GATED, last) -- selector surface

- `source_bank` widget appended at END of the writer's optional inputs +
  `widgets_values` (BUG-LOCAL-097): workflow JSON gains slot 25 = "science_news"
  (plain widget value -- no `inputs[]` entry unless converted to input; codex r2
  S4 verify against the real JSON at build).
- `tests/test_workflow_json_guardrails.py:673-733` updated SAME COMMIT: length
  25 -> 26, slot 24 stays "auto" (story_scaffold), slot 25 == "science_news"
  (codex r2 M6 -- CONFIRMED pins). Plus an INPUT_TYPES positional test that
  source_bank is the LAST optional entry (antigravity r2 S1).
- `OTR_LedgerScriptWriter.run` gains `source_bank="science_news"` before the
  keyword-only refine args (codex r2 M3; exact signature verify-at-build);
  selection threads EXPLICITLY through to the resolver (request field /
  argument -- resolve_creative_system_prompt gains a source_bank_id parameter,
  default "science_news" keeping all 4 existing callers byte-identical);
  `require_runnable_bank(source_bank)` called before story execution.
- Boot posture: dropdown choices from the lazy registry inside INPUT_TYPES; a
  broken registry fails node registration LOUD (no baked-in fallback list --
  rejected r1). Non-runnable banks ARE listed (honest error on use).
- Gate: kibitz on the wiring + OTR_WorkflowValidator + JSON round-trip +
  link/widget audit before commit.

## 4b. Lane-enablement checklist (2C kibitz r3/r4 -- GATES any future
## `runnable:true` flip of a non-science bank)

The 2C widget threads the selection to the ONLY pack-routed seam today
(`line_composer_system`). Before ANY non-science bank flips `runnable:true`,
each of these seams must be made bank-aware (or explicitly bypassed for the
lane) -- the run-intent gate is what keeps the 2C contract honest until then:
1. Outline seams -- DONE (lane chunk 1 @69afbd83, 2026-07-06): the three
   outline stage prompts are pack-routed via the router repo=None lane;
   science byte-identical (constants = extraction fixture, pinned); a bank
   without the seams fails LOUD (media_archive pinned).
2. Exchange seam -- DONE (lane chunk 2 @9809e36f, 2026-07-05): the STATIC
   system prompt is pack-routed via the new `exchange_system` seam (router
   repo=None lane, resolved OUTSIDE the prepass PD1 swallow -- a bank without
   the seam fails the episode LOUD); dynamic craft bullets stay Python-owned.
   Science byte-identical (EXCHANGE_SYSTEM_PROMPT extraction fixture, pinned).
3. Source payload: RSS fetch (`_fetch_science_news`) + `news_interpreter.
   build_news_briefs` are science-hardwired; banks' fetcher/interpreter
   fields are metadata until their contract is built.
4. Remaining seams: audit `_PHASE_TO_PACK_SEAM` coverage (announcer intro/
   outro, coda, style-pick seams ship in packs but are not yet routed).

2C judgment + arc record: `kibitz-runs/2026-07-05-multimodal-2c/` (r1-r4;
antigravity credit-bugged and dropped -- panel = codex, Claude anchor+judge).
Bug found by the gate: BUG-LOCAL-416 (refine locals() capture TypeError).

## 5. Invariants

JSON owns content; Python owns validation/routing/execution; NO fallbacks;
unknown id = hard error; audio spine FROZEN; science lane byte-identical through
2A/2B; suite + Bug Bible + B7 green per chunk; UTF-8 no BOM; commit per green
chunk (push only on operator instruction this session); prod/main gated.

## 6. Acceptance

- 2A: lazy registries load fail-loud w/ sweep + cross-refs + precedence; router
  routes science via the bank; zero episode change; all test pins updated.
- 2B: 3 lane packs at canonical coordinates with the EXACT key sets above;
  non-runnable run-intent raises; science unchanged.
- 2C: widget in the real JSON (slot 25, default science_news), guardrail test
  updated same commit, selection threaded explicitly, validator green.

## 7. Judgment log (r1+r2)

Accepted: codex r1 M1-M4/S1/cuts; codex r2 M1-M6, S1-S4, visual_style cut,
guide_ref/defaults opacity; antigravity r1 M3/M4 observations + legacy-pipeline
insight (reshaped: seam_refs validated against union, never a runtime gate);
antigravity r2 M1-M6 + S1; anchor r1 M1/M2/S1-S3, anchor r2 M1/S1/S2.
Rejected: antigravity r1 INPUT_TYPES silent fallback list, placeholder allowlist
entries, subset-relaxed allowlist test (no-fallback/allowlist-law violations).
Verify-at-build: run() signature insert point; widget serialization shape in the
real JSON; story_scaffold append point.
