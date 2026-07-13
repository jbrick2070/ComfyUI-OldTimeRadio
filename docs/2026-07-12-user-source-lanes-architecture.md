# User Source Lanes -- Replacement Architecture & Coding Plan (v1 DRAFT)

- **Date:** 2026-07-12 (late). **Status:** DRAFT FOR ARCHITECTURE APPROVAL -- no code,
  no GO_FORWARD_PLAN change until this converges (operator directive).
- **Kibitz arc COMPLETE (r1-r4), but NOT CONVERGED.** r1 @ `d724e08a`, r2 @ `f7c6902c`,
  r3 (wiring) @ `1af3d2bc`, r4 (convergence) folded in this change. Panel = codex
  (`gpt-5.6-sol`, pinned) + antigravity; Claude anchor + judge. Artifacts + per-round
  judgment logs: `kibitz-runs/2026-07-12-user-source-lanes/{r1..r4}/`.
  **r4 surfaced TWELVE grounded must-fixes rather than confirming convergence** -- three of
  them (PackRef owner loss, the staging boundary, the SDK import contradiction) were defects
  introduced by the CONFIDENT r3 fold itself. All twelve are folded here. The architecture
  SHAPE was never challenged: no round overturned an earlier decision, and no panelist
  proposed a different design.
  **Therefore: ONE r5 confirmation pass is required before the coder slot.** Releasing now
  would trust a fold that has not been adversarially read even once -- which is exactly the
  failure r4 just caught twice.
- Re-grounded against live HEAD this session; the fast-moving-base precondition in §14
  governs every line pin below (r3 re-pinned the §3 runner-map block, which had drifted
  ~+37 lines).
- **SUPERSEDES FOR SCOPE:** `docs/2026-07-12-vibe-coder-extensibility-r2-coding-plan.md`
  (@ 97d4f9eb). That plan's "content packs only, NO new lanes" ruling is RETIRED by
  operator correction. Its useful work is carried forward explicitly in §13 -- nothing
  else from it is binding.
- **Sibling plan affected:** `docs/2026-07-12-randomizer-rolls-r2-coding-plan.md` --
  its converged `_otr_lane_specs` design is ABSORBED here (this build lands first and
  CREATES the authority; the randomizer build shrinks to `_otr_bank_roll` + eligibility
  on top). §9 states the contract; that doc gets its delta note only after this plan
  converges.
- **Binding:** `AGENTS.md`, `CLAUDE.md`, `docs/PRODUCTION_SPRINT_LESSONS.md`,
  `docs/SOURCE_BANK_GUIDE.md`, `docs/SOURCE_BANK_PREFLIGHT.md`.

---

## 1. Product behavior (plain language)

A user drops a folder under `user_packs/source_lanes/<lane_id>/`, runs ONE command to
validate and activate it, restarts ComfyUI, and their lane appears in the existing
**Source Bank** dropdown next to the shipped ones. No shipped registry file, no Python
module, no canonical workflow is hand-edited. Two authoring paths share that flow:

- **Path A -- feed variant (simple, low-risk):** "science_news, but MY feed." A
  manifest-only bundle that reuses a shipped, approved lane's entire fetch/interpret/
  write/safety architecture and changes ONLY the feed URLs, the source identity shown
  in credits/HUD, and (optionally) which compatible story pack it defaults to. No
  Python in the bundle.
- **Path B -- original lane (advanced):** a genuinely new source strategy. The bundle
  ships its own `lane.py` (fetcher / interpreter / compatibility hook / optionally a
  full runner), its own story rules and packs -- all behind OTR's stable outer
  contracts (SourcePayload, two LLM slots, production Ledger, writer tail, asset
  paths, fail-loud). Full qualification ladder.

Bundles with ACTIVATION-DETECTABLE defects (schema, contract, collision, staleness,
path escape, import/signature failure) NEVER appear in any dropdown and NEVER break
ComfyUI boot: they are quarantined with the same structured reason in the console and
in `otr_check`. RUNTIME failures (a dead feed, a lane bug mid-render) abort THAT run
loudly with no fallback; they do not retroactively quarantine the lane -- the
activation state machine only transitions on check/byte evidence (r1: the promise is
scoped to what the state machine can actually guarantee). **Promise boundary (r3):** a
user lane's own `fetch_source` runs IN-PROCESS and user-trusted (§5.5) -- OTR does not
wall it off, so a lane that hangs its own network call hangs that render. That is a lane
defect, aborted loud when it trips a bound, not something OTR guarantees against. The
bounded-fetch seam (§4) is PROVIDED to lanes; using it is an SDK requirement, not an
enforced wall. (codex r3 proposed a killable child for runtime fetches; rejected -- it
contradicts the no-sandbox non-goal for the same in-process trust posture every custom
node already has.)

### Exact end-user steps

Path A (feed variant):
1. Copy the template: `user_packs/source_lanes/my_space_news/lane.json` (from
   `docs/templates/`).
2. Fill in `base_lane_id: science_news`, your `feed_urls`, your `source_identity`.
3. Run `scripts\otr_check.bat lane user_packs/source_lanes/my_space_news --activate`.
   Fix anything it names (file + field + fix) and re-run until it writes the receipt.
4. Restart ComfyUI. Pick `my_space_news` in the Source Bank dropdown. Render.

Path B (original lane):
1. Copy the example bundle (`docs/templates/example_lane/`) to
   `user_packs/source_lanes/<lane_id>/`; it contains `lane.json`, `lane.py`,
   `story_rules.json`, `story_packs/default.json`, `fixtures/`.
2. Implement the typed entry points in `lane.py` (§5.3). Point your LLM assistant at
   `docs/EXTENDING_OTR.md` -- the generated contract tables are written for that.
3. Run `scripts\otr_check.bat lane <path> --activate`. The checker validates every JSON,
   then imports and contract-tests your Python in a bounded child process against your
   fixtures. Fix and re-run until the receipt lands.
4. Restart. Select the lane. Render.
   **Restart semantics, stated honestly (r4, codex -- the old parenthetical overclaimed):**
   editing a bundle file does NOT de-activate a lane in an ALREADY-RUNNING ComfyUI -- the
   registries do not live-rescan, and the running process keeps executing from the immutable
   snapshot it admitted at boot. What is true: `otr_check lane <id> --status` sees your edit
   immediately and reports STALE, and the NEXT restart drops the stale lane from every
   dropdown until you `--activate` again. Byte hashes are the truth; the restart is when the
   truth is re-read.

## 2. Widget / default / restart behavior (exact)

- **Source Bank dropdown (exists):** default stays `science_news`; shipped banks
  unchanged; activated user lanes (both kinds) appear by `lane_id`. Values are DATA --
  a changed choice list produces ZERO canonical-JSON diff (proven: the guardrail pins
  `wv[23] == "science_news"` and membership via `list_bank_ids()`,
  `tests/test_workflow_json_guardrails.py:702-714`; the headless path validates combo
  membership against the LIVE `/object_info` schema, `scripts/otr_api.py:198-216`, so
  activated lanes are automatically accepted there too).
- **Story Pack dropdown (operator-required; carried design):** appended at the live
  END of the writer widget vector; default `"(bank default)"`; activated lane packs
  appear as `<lane_id>/<story_model_id>`; a VARIANT with no bundled packs shows its
  base's packs under the BASE's coordinates (honest -- the pack IS the base's; §4
  aliasing law). An explicit pack must belong to the selected lane OR its declared
  base -- anything else fails loud. This is the ONLY canonical-workflow delta in the
  whole feature (§7). (r1 panel note: both panelists proposed cutting/deferring this
  widget; REJECTED -- the operator's requirement specifies it. It stays its own late
  wave and every lane runs via manifest defaults without it, so it never blocks the
  lanes themselves.)
- **Story Pack THREADING (r3 -- as drafted the widget would have been SILENTLY
  IGNORED; this is the wiring, not a nicety).** Grounded chain today:
  `compose_line(...)` / `generate_outline(source_bank_id=...)` never resolve a pack --
  they pass a BANK ID to `resolve_creative_system_prompt`
  (`_otr_creative_prompt_router.py:157-206`), which calls
  `resolve_story_pack(source_bank_id)` and lands on `bank.default_story_model`. So a
  selected non-default pack would be dropped on EVERY router-sourced seam
  (line-composer, outline macro/phase/beat, exchange, coda, the three announcer seams).
  (antigravity r3 flagged the symptom but named `_otr_line_composer.py` /
  `_otr_outline.py` as the offenders -- both have ZERO `resolve_story_pack` calls; the
  choke point is one hop deeper.) The fix threads a **full `PackRef`** through FOUR
  surfaces and nothing else:
  **(r4 CORRECTION -- r3 threaded only `story_model_id`, which is a build-breaker.)**
  `resolve_story_pack(bank, model)` resolves via `_pack_path(bank.source_bank_id,
  model_id)` = `story_packs/<bank>/<model>.json`. So a VARIANT selecting a BASE-owned pack
  would resolve to `story_packs/<variant_id>/<model>.json` -- **a path that cannot exist**
  (§4's own aliasing law keeps base packs under the BASE's coordinates) -> hard
  `UnknownStoryModelError` on the first variant that uses its base's pack, which is the
  DEFAULT case. The threaded unit is therefore
  `PackRef{owner_lane_id, story_model_id, pipeline_id, sha256}`, resolved by
  **`owner_lane_id`, NEVER the selected bank**:
  1. `resolve_creative_system_prompt` gains an optional `PackRef` -- the SINGLE
     choke point for every creative seam (composer + outline + announcer need no new
     `resolve_story_pack` call, only signature pass-through);
  2. the three writer pack-resolution sites (`OTR_LedgerScriptWriter.py:1836`, `:1874`,
     `:3778`);
  3. the replay path `_otr_freeze_cascade.py:316`, which re-resolves from
     `meta["source_bank"]` -- it must read the STAMPED PackRef (owner + id + sha) back out
     of meta, or a replay silently reverts to the default pack, or fails outright for a
     base-owned pack (§10; grounding confirms this is load-bearing, not belt-and-braces);
  4. `run()` / `_resolve_inputs` gain the `story_pack` parameter (refine re-entry then
     carries it for free -- it rebuilds args from
     `inspect.signature(type(self).run).parameters`, `:3446-3457`, so a new kwarg rides
     through automatically as long as it avoids the exclusion tuple).
  Every episode -- shipped lanes INCLUDED -- stamps all four PackRef fields, and replay
  resolves by stamped OWNER + digest, never by selected `source_bank`.
  REJECTED (antigravity): a thread-local `set_active_story_model_override`. Hidden
  ambient state is invisible to the freeze cascade and would replay wrong. Thread the
  PackRef; never hide it.
- **Story-pack vs MODEL PROMPT-PROFILE precedence (r4, codex -- undefined, and the live
  router silently wins).** Grounded `resolve_creative_system_prompt` (`:190-207`): if the
  selected model's curated row has `prompt_profile == "otr_1940s_v1"`, the router returns
  `OTR_PERIOD_SYSTEM_PROMPT` at `:197-198` and **never consults the pack seam or
  `source_bank_id` at all**. So on a period model, an explicit story_pack selection would
  be silently discarded. LAW (smallest compatibility-preserving rule): the
  `"(bank default)"` sentinel PRESERVES today's model-profile precedence exactly; an
  EXPLICIT story_pack selection OVERRIDES the model profile for pack-routed seams. Both
  directions pinned by tests.
- **Falsy-coercion law (r3, executable -- the trap nobody on the panel caught):**
  `resolve_story_pack` (`_otr_story_routing.py:522`) tests `story_model_id is not None`.
  A blank `""` is therefore NOT a default -- it is a lookup for a pack literally named
  `""`, and it raises `UnknownStoryModelError`. The `"(bank default)"` widget value MUST
  coerce to `None`, never `""`. A test pins both directions. (This is exactly the
  "falsy coercion" the §13 carry names; it is now an executable law with a line pin.)
- **Rules axis (r3, decided):** `resolve_story_rules(source_bank_id)`
  (`_otr_story_rules.py:296`) has NO model parameter at all. v1 keeps story rules
  BANK-scoped -- packs vary the prompts, not the rules. Recorded so it is a decision,
  not an oversight.
- **Early request-resolution gate (r3, codex -- it has an exact seat).** Grounded order
  in `run()`: `require_runnable_bank` at `:3380`, `_resolve_inputs` (which performs the
  RSS fetch, `:1427-1435`) at `:3516`, and the FIRST pack resolution not until `:3778`
  -- 262 lines downstream of the network call. So today nothing validates pack ownership
  or compatibility before OTR goes to the network and the models. ONE gate lands BETWEEN
  `:3380` and `:3516`: parse `PackRef` -> validate lane/base pack ownership -> reject
  roll + explicit pack (§9) -> enforce `source_ref_mode` + `supports_custom_premise` ->
  call `check_compatibility`. Only validated objects travel past it; the fetch happens
  after.
- **Source Reference (exists, run param `:1236`, threaded `:1401/:1473`, stamped
  `:3645`):** a feed variant with blank `source_ref` uses its manifest's feeds; a lane
  accepts URL/file/ID overrides only when its declared `source_ref_mode` permits.
- **Custom Premise (exists):** available when the selected lane's manifest declares
  `supports_custom_premise: true` (shipped rows keep today's behavior).
- **No new widgets beyond Story Pack.** No "lane kind" widget (the bundle declares its
  kind); no `lane_options_json` in v1 (manifest defaults / named presets must carry
  it; if a grounded case ever proves otherwise, that is a v2 proposal).
- **Restart contract:** activation changes are picked up at ComfyUI restart (lazy
  registry singletons). The model dropdowns keep their live-rescan behavior.

## 3. One lane authority -- `_otr_lane_specs` (the map)

**Decision (re-grounded): YES -- the randomizer plan's converged `_otr_lane_specs`
becomes THE authority, and THIS build creates it** (that plan already specifies:
"the ONE lane authority ... It REPLACES `_RUNNER_BY_PIPELINE` -- there is no second
table and no view").

**Seed state RE-PINNED at live HEAD (r3 -- every pin in the r2 draft had drifted ~+37
lines; §14's re-read law caught it, and it MUST be re-derived again at the coder slot):**
`_RUNNER_BY_PIPELINE` (`:1693-1699`), lazy `_run_*_lane` wrappers (`:1637-1684` -- there
are FIVE: fable2, scifi_codex, scifi_gemini, scifi_sonnet, original_codex56sol),
`_LEGACY_INLINE_PIPELINES` (`:1702-1704`), `_resolve_lane_runner` (`:1755-1773`) -- plus
`_FETCHERS`/`_INTERPRETERS` (`_otr_source_payload.py:512-532`) and the banks/pipelines
registry (`_otr_story_routing.py:465-482`). Consumer count is re-counted at build:
`_RUNNER_BY_PIPELINE` is read TWICE inside `run()` alone -- the early refine rejection
(`:3404`) and dispatch via `_resolve_lane_runner` (`:3768`) -- so the r1 "exactly four
consumers" figure is treated as a floor, not a fact.

**The effective row IS a `SourceBank` (r3, codex -- LaneSpec as drafted did not expose
the interface consumers actually read).** The live record is
`_otr_story_routing.SourceBank` (`:104-116`), ELEVEN fields: `source_bank_id`, `label`,
`source_kind`, `interpreter`, `fetcher`, `default_story_model`, `default_story_pipeline`,
`defaults`, `required_seams`, `runnable`, `guide_ref` -- mirrored by the `_BANK_KEYS`
allowlist (`:193-197`). Consumers read THOSE names. So a LaneSpec's effective row is a
real `SourceBank` -- same field names, NO renames, `_BANK_KEYS` untouched -- and the lane
metadata below rides ALONGSIDE it, never instead of it. (`require_runnable_bank` `:533`
returns that row; post-L1 it returns the EFFECTIVE row for user lanes, so credits/HUD
stamps, fetch dispatch, and runner dispatch inherit correctness with no per-site edit.)
Note `list_story_pack_choices` does NOT exist today (it is not in routing's `__all__`);
L4 CREATES it.

`LaneSpec` (one record per lane, shipped AND user):

| field | meaning |
|---|---|
| `lane_id`, `label`, `kind` | identity; kind = `shipped` / `rss_variant` / `original` |
| `runnable` | the ONLY run gate (registry law preserved) |
| `execution_kind` | `common_writer` \| `own_runner` -- the DISPATCH authority (§5.6). **(r4: was described in prose but missing from this table.)** With `kind`, it also gates the L5b staging path: staging requires `kind == original` AND `execution_kind == own_runner`, never `execution_kind` alone |
| `pipeline_id` | **(r4 -- this row contradicted §5.6 and would have broken every user pack.)** `legacy_many_pass` (the REAL registered row) for ALL common-writer lanes, variants included -- their packs declare it and the parity law holds natively. A synthetic `user:<lane_id>` row is composed ONLY for `own_runner` lanes. It is NOT "synthetic for every user lane" |
| `feed_urls` | variants only: the per-feed provenance RECORDS (§4), threaded explicitly to the fetch chain. **(r4: missing from this table.)** |
| `pack_owner` / default `PackRef` | the lane's default pack as a full `PackRef` (owner + id + pipeline + sha) -- resolution NEVER derives the path from the selected bank (§2, r4) |
| `runner` | lazy ref: shipped wrapper, common-writer marker, or `lane.py:run_lane` |
| `fetcher` / `interpreter` | lazy refs: shipped ids, base-lane inherit, or `lane.py` entry points |
| `source_ref_mode` | `none` / `url` / `file` / `id` (declared, enforced at fetch) |
| `supports_custom_premise` | widget gating |
| `default_story_model`, `story_rules_ref` | pack routing |
| `word_range` + `check_compatibility` ref | request compatibility (randomizer + manual) |
| `random_eligible` | manifest opt-in AND runnable AND compat -- randomizer pool |
| `base_lane_id` + `base_hash` | variants only |
| `activation` | receipt id + per-file sha256 map + checker/schema versions (user lanes) |

Laws:
- ONE registry composed at lazy load: shipped seed FIRST (any error = hard fail, a
  broken shipped lane is a build error), then activated user bundles (any error =
  quarantine, §6). `resolve_fetcher` / `resolve_interpreter` / lane-runner resolution /
  `list_bank_ids` / `list_story_pack_choices` / randomizer filters ALL consult this
  one authority. No user-only shadow registry, no second runner/compat table.
- **Atomic registration:** a user lane's bank row equivalent, pipeline identity,
  rules, packs, handlers, runner, and compatibility contract activate ALL-or-NONE. A
  selectable lane with an incomplete implementation is a contract violation the
  authority makes unrepresentable (the LaneSpec cannot be constructed partially).
- The four writer consumer sites move onto the authority in the same change that
  creates it (the "lane-spec rip" the randomizer plan already scoped -- ownership
  transfers to this build; §9).
- **Every id-membership surface consults the authority (r1, agy's three boot-crash
  classes):** `resolve_story_pack`/`_pack_path` resolve user-lane packs via the
  LaneSpec's bundle paths (never the shipped root alone); `_otr_story_rules._load_all`
  validates runnable-lane rules coverage THROUGH the authority (shipped runnable
  banks -> shipped rules dir, user lanes -> bundle `story_rules.json` or base-inherit
  ref) instead of asserting every runnable id has a file under `nodes/story_rules/`;
  `registered_fetcher_ids()`/`registered_interpreter_ids()` cross-refs admit
  user-lane entry-point refs. Without these three, the first activated lane crashes
  boot.
- **Reset seam (r1, codex; r3-corrected -- the existing hooks are ASYMMETRIC):**
  grounded, `_otr_story_routing._clear_caches` (`:547-552`) clears `_REGISTRY` AND both
  `_otr_story_pack` caches but NOT `_RULES`; `_otr_story_rules._clear_caches`
  (`:317-320`) clears ONLY `_RULES`. Neither knows about the other, so a test that
  mutates runnable flags today can validate rules against a stale routing registry.
  `_otr_lane_specs._clear_caches()` therefore FANS OUT to all three (authority, routing,
  rules) -- one internal reset for tests and restart admission; no live rescanning.
- **Import DAG + load order (r2, 3-way convergent):** a new DEPENDENCY-FREE leaf
  module `nodes/_otr_lane_contracts.py` owns the shared records/enums (LaneSpec,
  PackRef, CompatRequest/CompatDecision, InterpreterResult, LaneTailParts, the
  HOST_CONTRACT surface) -- stdlib only, imports nothing from OTR. `_otr_lane_specs`
  is import-leaf-safe (every cross-module reach is a LAZY inner import, the writer's
  own :1632-1647 pattern) and NEVER calls routing APIs during authority
  CONSTRUCTION; strict load-order invariant: routing loads its raw
  banks/pipelines `_REGISTRY` FIRST, then the authority composes, then consuming
  modules' validation paths call authority FUNCTIONS (function-local deferred
  imports). This kills the circular-init loop all three reviews flagged.
- **Raw seed accessor -- a REQUIRED NEW L1 surface (r3).** Grounding correction: there is
  NO circular-init loop in the code TODAY. `_ensure_loaded` (`:465-482`) runs
  `_sweep_and_crossref(banks, pipelines)` on raw dicts PASSED AS ARGUMENTS and only then
  assigns `_REGISTRY`; the sweep never calls `get_bank`/`resolve_story_pack`, and
  `_load_routed_pack` (`:336-347`) calls `_read_pack_data` + `load_pack_with_seams`, not
  `resolve_story_pack`. (antigravity r3 claimed the reverse edge; MISREAD -- the real
  dependency runs `resolve_story_pack -> _ensure_loaded -> get_bank`.) The loop only
  becomes possible in the FUTURE state, once `get_bank`/`list_bank_ids` delegate to the
  authority -- and the load-order invariant above is what prevents it. What grounding DOES
  prove is that the mechanism the invariant needs is **missing**: no `_load_shipped_raw`
  or equivalent exists -- `_ensure_loaded` is the ONLY path from banks.json/pipelines.json
  to rows, and its sweep (`:480`) is unconditional and inseparable. L1 therefore CREATES a
  private raw seed accessor that parses banks + pipelines WITHOUT the cross-ref sweep; the
  authority composes from it, and the sweep then validates the completed map. This is a
  build deliverable, not an assumption.
- **Construction must be PURE and the publish ATOMIC -- two callbacks still re-enter (r4,
  codex + agy convergent; the r3 raw-seed accessor closed the banks/pipelines edge but not
  these).** Grounded, two paths still call back into routing during load: the routing sweep
  calls `_osp.registered_fetcher_ids()` / `registered_interpreter_ids()`, and
  `_otr_story_rules._load_all` calls `_runnable_bank_ids()` (`:233-236`), which
  deferred-imports routing's `list_bank_ids` / `get_bank`. Once those APIs delegate to the
  authority, a load re-enters itself. LAW: parsing and validation are PURE FUNCTIONS taking
  banks, pipelines, PackRecords, handler ids and rules rows as EXPLICIT PARAMETERS; all
  locals are validated; the singleton is then published ATOMICALLY; consumer APIs may consult
  the authority only AFTER publication. `_otr_story_rules._load_all` / `_validate_row` accept
  the runnable-id set as a parameter instead of importing routing.
- **HOST_CONTRACT hash (r2, hybrid judged):** derived PROGRAMMATICALLY from the live
  surface objects -- `repr(sorted(SOURCE_PAYLOAD_KEYS))` + the contracts-module
  dataclass field signatures + the checker policy version -- not hand-bumped strings
  (forgettable) and not raw file bytes (comment edits would stale every receipt). A
  guardrail test pins the derivation so a silent surface change fails the suite.

## 4. Path A -- safe feed variant (manifest-only)

`user_packs/source_lanes/<lane_id>/lane.json`:

```json
{
  "schema_version": "lane-v1",
  "lane_id": "my_space_news",
  "kind": "rss_variant",
  "label": "My Space News",
  "base_lane_id": "science_news",
  "feed_urls": ["https://example.com/space.rss"],
  "source_identity": {
    "source_material_label": "Space story",
    "credits_source_line": "Source: My Space Feed"
  },
  "default_story_model": "",
  "story_rules": "inherit",
  "random_eligible": true
}
```

(Field-name note: the pack field is `default_story_model` -- the canonical pack id
IS the `story_model_id` filename stem everywhere: manifest, LaneSpec, dropdown,
receipts, replay stamps (r1: one identity, no `default_story_pack` synonym). The
WHITELIST is the contract: feed URLs, source-identity strings, an optional compatible
`default_story_model`, `random_eligible`, an optional NARROWED `word_range`. Nothing
else.)

- **Compatibility inheritance (r1):** a variant inherits the base lane's word_range
  and `check_compatibility` surface verbatim; the manifest may NARROW word_range,
  never widen it. Anything beyond that is original-lane territory.
- **Pack namespace aliasing (r1, anchor):** pack validation enforces
  `pack.source_bank_id == bank_id` by path coordinates (`_otr_story_routing.py:
  373-378`), so base packs can never be re-stamped under a variant id. A variant's
  effective pack namespace = its bundled packs (validated under `<lane_id>`) UNION
  the base lane's packs (under the base's own coordinates). Blank
  `default_story_model` = the base's default pack.
- **Source-identity projection (r2 enumeration; r4-CORRECTED -- one key was on the wrong
  side of the line):** overridable IDENTITY keys = `source_material_label`,
  `credits_source_line` (stamp path `OTR_LedgerScriptWriter.py:3648-3659`),
  `key_terms_label`, `close_brief_label`. FORBIDDEN (behavior-bearing, inherit-only):
  `coda_mode`, `source_develop_verb`, `source_grounding_label`, `story_form_label`, **and
  `title_form_label` (r4, codex -- it was classed as "identity", but it STEERS TITLE
  GENERATION rather than describing the source; identity overrides are attribution/display
  ONLY)**. No generic defaults overlay exists -- the whitelist IS the projection.
  [RATIFY the split] Precedence: variant manifest > base defaults.
- **Per-feed provenance + rights (r4, codex -- required by BINDING project docs, and a
  single lane-level string cannot satisfy them).** `docs/SOURCE_BANK_GUIDE.md` §5 and
  `docs/SOURCE_BANK_PREFLIGHT.md` Gate 2 are binding on this plan and demand canonical URL,
  digest, outlet/author, rights status, license URL and attribution, failing CLOSED on
  unknown or incompatible adaptation rights. One `credits_source_line` cannot truthfully
  describe a multi-publisher feed set. So `feed_urls` becomes a list of per-feed RECORDS
  (url + outlet + rights status + license URL + attribution), and the SELECTED feed's and
  article's provenance is stamped into `meta.source_meta` / `meta.source_rights`. The
  plumbing already exists: `SourceFetchResult` carries `source_meta` and `source_rights`
  sidecars precisely for this (`_otr_source_payload.py:85-95`). [RATIFY -- this expands the
  manifest beyond a bare URL list.]
- **PackRef data model (r2, codex):** the widget string parses ONCE into
  `PackRef{owner_lane_id, story_model_id}`; resolution threads the PackRef, never
  re-parses. A variant bundle pack whose filename stem collides with a base-lane stem
  is an ACTIVATION ERROR (no precedence guessing between local and base namespaces).

- `base_lane_id` must be in the APPROVED feed-capable base set -- v1:
  `science_news` + `media_archive`, BOTH now grounded (normative default, ratify at
  approval; agy's science-only minority recorded). The base must be shipped AND
  runnable.
- **Enabling change (one parameterization, TWO enumerated files -- r1; the science chain
  RE-GROUNDED to THREE hops in r3):**
  - science (**THREE hops, not two -- the middle one lives in the WRITER, and the r2 draft
    omitted it, so `feed_urls` could not physically have reached the fetcher**):
    `_fetch_science_rss` (`_otr_source_payload.py:265-289`)
    -> `_fetch_rss_seed_or_die` (**`OTR_LedgerScriptWriter.py:1144-1146`**, calling out at
    `:1170-1174`)
    -> `_fetch_science_news` (`story_orchestrator.py:1677-1679`), which iterates the
    `SCIENCE_NEWS_FEEDS` constant (`:1228-1264`);
  - media_archive: `DEFAULT_MEDIA_ARCHIVE_FEEDS` + the EXISTING `OTR_MEDIA_ARCHIVE_FEEDS`
    env resolver (`_otr_media_archive_sources.py:16-19, 156-164`), fetcher
    `fetch_media_archive_rss` (`:175-197`), reached via `_otr_source_payload.py:398-406`.
  ALL THREE science signatures + the media fetcher gain
  `feed_urls: tuple[str, ...] | None = None`, threaded EXPLICITLY from the LaneSpec.
  Precedence: LaneSpec feeds (variants) > `OTR_MEDIA_ARCHIVE_FEEDS` env (legacy operator
  override, media only, shipped lane unchanged) > shipped constants. None = byte-identical
  shipped behavior.
  (r3 REJECTED antigravity's `getattr(bank, "feed_urls", None)` shortcut: grounded,
  `fetch_media_archive_rss` does `del bank, technical_model, source_ref` at `:184` -- it
  deliberately discards the row. Smuggling feeds back through an object the fetcher
  explicitly drops re-couples what the code decoupled. Explicit parameter; codex
  independently concurs, and NO temporary env mutation.)

- **Network hardening is NOT inherited -- it does not exist yet, and this build is what
  makes that dangerous (r3, codex; the single most important correction of the arc).**
  The r2 draft claimed "All Gate-2 network laws (timeouts, bounded retries,
  size/status/content-type checks) are INHERITED CODE -- the variant cannot alter them."
  That was FALSE. Grounded:
  - science (`story_orchestrator.py:1714-1724`): `feedparser.parse(feed_url)` guarded only
    by `socket.setdefaulttimeout(7)` -- which is **process-global, not thread-local**, and
    is set/restored concurrently by every pool thread (the code comment claiming "locally
    for this thread" is simply wrong; one thread's `finally` can restore `None` = infinite
    while another is mid-parse). No retry budget, no HTTP status check, no content-type
    check, no body-size cap.
  - media (`_otr_media_archive_sources.py:129-137`): `feedparser.parse(raw_or_url)` with
    **no timeout of any kind**, no retry, no status/content-type/size bound. The
    `max_chars=6000` truncation at `:29` happens AFTER the full body is downloaded and
    parsed -- it is not a network boundary.
  - there is NOTHING to route through: no shared bounded-HTTP helper exists.
    `_otr_google_api/client.py` and `_otr_comfy_backend.py` have timeouts + retry sets but
    are service-keyed (inject `x-goog-api-key`, resolve against a fixed base) and cannot
    fetch an arbitrary URL; `story_orchestrator._fetch_full_article` (`:1267`) is the
    closest existing pattern (timeout + `raise_for_status`) and still has no retry budget,
    no content-type check, and no streamed byte cap.
  Feeding shipped constants to that code is one thing; feeding USER-SUPPLIED URLs to it is
  another, and that is exactly what Path A does. So the bounded-fetch seam is a REQUIRED
  L3 DELIVERABLE, not an inheritance:
  - **new `nodes/_otr_feed_fetch.py`** -- one bounded fetch used by BOTH lanes: explicit
    connect + read timeouts, capped redirects, HTTP status check, content-type allowlist,
    STREAMED body-size cap, a small bounded retry budget. Both call sites then hand
    `feedparser.parse()` pre-fetched, size-capped BYTES instead of a URL -- which also
    retires the racy global-socket-timeout hack.
  - **IT MUST COVER THE SECOND HOP -- and that hop is the actual hole (r4, codex + agy
    convergent; the r3 seam hardened only the feed DOCUMENT and would have left the
    dangerous request wide open).** Grounded: after parsing the feed, the science lane
    makes a SECOND, entirely unhardened request to a URL taken from INSIDE the feed entry
    -- `_fetch_full_article` (`nodes/story_orchestrator.py:1267`,
    `requests.get(url, timeout=timeout, headers=...)` + `raise_for_status`), called from
    `_resolve_body` (`:1904`) on `entry.get("link", "")` (`:1768`), executed in PARALLEL
    across up to 10 candidates. It has NO host allowlist, NO redirect restriction
    (`requests` follows redirects by default), NO content-type check, and NO streamed size
    cap (the 12000-char truncation runs AFTER `resp.text` has already materialized the
    whole body).
    Today this is incidentally safe: `SCIENCE_NEWS_FEEDS` is a hardcoded allowlist of
    institutional domains, so the links inside are trustworthy. **A user-supplied feed
    removes that guarantee entirely** -- a Path A variant can point `link` at ANY host, and
    OTR will fetch it, follow its redirects, and read an unbounded body. So `_fetch_full_article`
    routes through `_otr_feed_fetch` too, with a purpose-specific content-type policy
    (article = `text/html`; feed = the feed types), and the same redirect/size/retry bounds.
  - **Numeric bounds (r4 -- both panels converged on these independently; RATIFY, do not
    assume):** connect 5s, read 10s, max 3 redirects, 2 MiB decoded body cap, 2 retries on
    the 408/429/5xx set, max 32 feed_urls, max 2048-char URLs, max 8 concurrent workers.
  - **concurrency cap:** `story_orchestrator.py:1783` builds
    `ThreadPoolExecutor(max_workers=len(shuffled_feeds))` -- one worker per feed. Harmless
    today (the list is a hardcoded 29-entry constant with no user path), a real
    unbounded-fan-out bug the moment `feed_urls` is user-supplied. Clamp it in the SAME
    change (the sibling pool at `:1933` already does: `max_workers=max(1, len(attempts))`).
  - **manifest limits:** maximum `feed_urls` count and maximum URL length, enforced at
    activation (§6.4), alongside the existing scheme / no-embedded-credentials checks.
  The variant still cannot ALTER any of this -- the bounds are inherited code once they
  exist. The honesty repair is that they must be BUILT, and the cost is in §15.
- The variant's effective lane = base LaneSpec + whitelisted overrides. It can NEVER
  override runner, pipeline, interpreter, provider, ledger behavior, or safety
  checks -- the whitelist makes that unrepresentable, not merely forbidden.
- `story_rules: "inherit"` (default) uses the base's rules pack; a bundle MAY ship
  `story_rules.json` instead (validated with the same code as shipped rules). **`story_rules`
  IS part of the whitelist (r4 -- the manifest example used it while the whitelist prose
  omitted it).** Packs: optional `story_packs/` (validated per the carried content-pack
  laws); blank `default_story_model` = base's default pack. **(r4: this line previously said
  `default_story_pack` -- the exact synonym the doc itself outlaws twelve lines earlier.
  There is ONE identity: `default_story_model`.)**
- **Feed failover -- the honest law (r4, codex; the absolute "never a fallback to another
  feed" was contradicted by BOTH shipped base lanes).** Grounded: science swallows a failed
  feed and continues (`except Exception: return []` per feed); media_archive accumulates
  entries ACROSS its configured feeds and raises only when NONE yield payloads (`:187-196`).
  So bounded failover is inherited behavior, not a violation. LAW: bounded selection and
  failover are permitted ONLY WITHIN the receipt-bound declared feed set; the SELECTED feed
  and the SELECTED article are stamped into the ledger; exhaustion of the declared set fails
  the run LOUD. Cross-lane, cross-bank, undeclared-feed, and different-model fallback remain
  absolutely forbidden -- that is what the no-fallback law actually protects.
- **Escalation rule:** anything needing Python, a new payload shape, or different
  interpretation is NOT a variant -- the checker rejects manifest fields outside the
  whitelist with "use an original lane" in the fix text.
- **Receipt binds the base:** the activation receipt records the base lane's version
  hash (base row + its fetcher/interpreter code digests). A material base change
  invalidates the variant's activation (re-run `--activate`; §11).

## 5. Path B -- original source lane (plug-in bundle)

### 5.1 Bundle shape

```
user_packs/source_lanes/<lane_id>/
  lane.json          # manifest: kind: original
  lane.py            # entry points (lazy; never imported at boot)
  story_rules.json   # REQUIRED (runnable-lane law, _otr_story_rules.py:274-280)
  story_packs/
    default.json     # >=1 pack; content-pack laws apply
  fixtures/          # deterministic activation/preflight samples
```

### 5.2 Manifest (`kind: "original"`)

Adds to the common fields: `source_kind`; `entry_points`; `reuse_common_writer: true` XOR
`run_lane` declared; `required_seams` (when reusing the common writer); `source_ref_mode`;
`supports_custom_premise`; `word_range: {min, max}`; `random_eligible`.

**Entry-point matrix -- ENUMERATED and enforced (r3, codex; "only `check_compatibility` is
required" was underspecified and would have let a lane activate with no way to produce a
payload):** the legal combinations are exactly two, keyed on `execution_kind` (§5.6):

| `execution_kind` | REQUIRED | FORBIDDEN |
|---|---|---|
| `common_writer` | `fetch_source` + `interpret_source` + `check_compatibility` | `run_lane` |
| `own_runner` | `fetch_source` + `run_lane` + `check_compatibility` | `interpret_source` |

Anything else = activation error with the missing/extra name in the fix text. Rationale:
`run_lane` ALWAYS consumes `payload`, so a fetcher is never optional; and a lane with no
fetcher/interpreter would fall into the source-contract-free dispatch path built for
`original_radio`, not a generic user lane. Activation ships ONE fixture per permitted
combination.

### 5.3 Typed runtime interfaces (grounded against live contracts; exact signatures
re-derived at build)

**Typed contracts law (r2, codex):** every surface below is a repository-owned
dataclass/TypedDict in `_otr_lane_contracts.py` -- `CompatRequest`, `CompatDecision`,
`InterpreterResult`, `LaneTailParts` -- each with a constructor, a validator, exact
null rules, ONE known-valid fixture, and specified rejection behavior. The live
dispatch dereferences `outline_view`/`canon`/`run_story_spine`/`final_title_override`
/`tail_finalizer` immediately (`OTR_LedgerScriptWriter.py:3739-3779`) and interpreter
validation requires attribute/`model_dump()` agreement
(`_otr_source_payload.py:204-257`) -- the types are pinned to those consumers, not
invented.

**SDK import contradiction -- RESOLVED (r4, codex + agy convergent; the r3 fold created it:
§1 promised lanes the bounded-fetch seam while §6.3 forbade importing anything but stdlib +
the contracts leaf, and `SourceFetchResult` lives in neither).** Grounding overturned BOTH
panelists' premise and made the fix cheaper than either proposed: `_otr_source_payload.py` is
**already stdlib-only at import time** -- its entire top-level import block is
`from __future__ import annotations` + `from dataclasses import dataclass` (`:33-35`); every
heavy import is lazy, inside function bodies, by documented design ("Zero file I/O at
import"), and `nodes/__init__.py` is a single comment line. Importing it from a bare child
process drags in NOTHING. It is a de-facto leaf that simply is not named like one. So:
- `SourceFetchResult` (frozen dataclass: `payload: dict`, `source_meta: dict | None`,
  `source_rights: dict | None`, `:85-95`) and `SOURCE_PAYLOAD_KEYS` (`:80-82`) are DEFINED in
  `_otr_lane_contracts.py` and **RE-EXPORTED** from `_otr_source_payload` -- one definition,
  zero duplication, no import-graph change, no consumer edits.
- the bounded fetch (§4) is handed to `fetch_source` as an **INJECTED TYPED CAPABILITY** on
  the run context, NOT as an import. The SDK's stdlib+leaf rule stays intact instead of being
  punctured, and the lane physically cannot reach around it to a raw socket without leaving
  the contract.
Same principle for `run_lane`: it receives a typed `LaneRunContext` + `SlotPort` capabilities,
never the writer's raw internal `resolved` dict or `slot_scheduler` (codex r4 -- exposing
those would freeze writer internals as public SDK surface). All injected capabilities are
part of HOST_CONTRACT and therefore stale a receipt when they change.

- `fetch_source(*, lane, technical_model, source_ref="", fetch) -> SourceFetchResult`
  -- MUST return the writer's exact seven-key `SOURCE_PAYLOAD_KEYS` envelope
  (`_otr_source_payload.py:80-82`; unknown key = hard error), mirroring the
  registered-fetcher contract (`SOURCE_BANK_GUIDE.md` §5). `fetch` is the injected bounded
  fetcher; network access outside it is an SDK violation.
- `interpret_source(*, lane, payload, technical_fn, model_id)` -- returns the
  interpreter surfaces (`casting_brief`, `script_brief`, `news_close_brief`,
  `key_terms`, `attempts`, `model_dump()`), mirroring the `legacy_many_pass`
  interpreter contract.
- `check_compatibility(*, lane, request) -> CompatDecision` -- request carries
  words/refine/source_ref/custom_premise; returns accept or a structured refusal
  (used by manual selection AND the randomizer filter; today's word gates live INSIDE
  runners at differing depths -- fable2 at entry :3351-3356, scifi_* inside runners --
  this hook standardizes the surface WITHOUT moving shipped gates in v1).
- `run_lane(*, payload, pack, resolved, led, meta, creative_fn, technical_fn,
  slot_scheduler, source_bank_row, story_rules, episode_root, episode_id) ->
  LaneTailParts` -- ONLY when the common writer cannot be reused; the signature is the
  LIVE dispatched-runner interface (`SOURCE_BANK_GUIDE.md` §4: outline_view/title/
  premise, EpisodeCanon-compatible canon, final_title_override, run_story_spine,
  optional before_save/after_save finalizer). Behind it the lane may implement any
  internal pass graph.

**Boundary honesty (r1, codex):** `lane.py` runs IN-PROCESS with the user's
permissions and receives mutable `led`/`meta`/callables -- exactly like shipped
dispatched runners (`OTR_LedgerScriptWriter.py:3739-3779`). The §5.4 restrictions are
therefore COOPERATIVE SDK RULES, not enforced walls (consistent with the no-sandbox
non-goal). What OTR actually ENFORCES: post-runner artifact validation -- the returned
tail parts and ledger state are validated against the production contracts BEFORE
shared-tail entry (schema, required fields, authoritative-producer receipts), and any
violation aborts loud. The lane gets for free: the two slot callables + runtime
policy, the production Ledger contract, the canonical writer tail, asset paths
(`otr\episodes\<ep>\`, `otr\obs\`), fail-loud everywhere.

**LaneTailParts contract law (r1, codex -- production history makes this
non-optional):** content-owned runners have already failed at provenance boundaries
(`docs/PROD_BUG_LOG.md:427-455`; producer-receipt rules in the Bug Bible). Before
`run_lane` is exposed, the exact `LaneTailParts` schema, the required ledger state at
handoff, the authoritative PRODUCER for every provenance field (delivery text,
episode seed, cast provenance, title, source identity, freeze state), forbidden
replay claims, and the before_save/after_save lifecycle are specified and validated
-- r2 carries the field-level detail; the LAW lands now.

**run_lane staging (r1, judged):** the operator requires original lanes to optionally
ship their own runner, so run_lane STAYS in v1 scope -- but it is the LAST wave
(L5b), an EXPERIMENTAL tier shipping only after the fetch/interpret+common-writer
path (L5a) is proven by the reference lane, and only with the LaneTailParts law
satisfied. (codex proposed deferring run_lane entirely; partially rejected --
operator requirement wins, staging adopted.)

**run_lane durable-state isolation (r2, codex -- validation alone cannot protect
disk; r3 RE-SEATED, because the staging facade could not exist where L5b put it):**
shipped runners save incrementally and `Ledger.save()` merges-with-disk and REPLACES both
the canonical file and `led.data` (`production_ledger.py:1287-1346`; writer warning at
`:3793-3797`) -- a failing user runner would otherwise leave invalid durable state behind
before post-run validation.

**The r2 design was unbuildable as sequenced.** Grounded order inside `run()`:
`new_ledger()` at `:3672` -- which BOTH creates the canonical episode directory
(`Ledger.__init__` makedirs, `production_ledger.py:551-557`) AND installs the global
`_CURRENT` (`production_ledger.py:365-368`; constructing the ledger IS publishing it) --
then meta stamps (`:3677-3709`), then the skeleton `led.save()` (`:3721`), and only THEN
the dispatched-runner call (`:3766-3790`). By the time a runner is reached, canonical
state is already on disk and globally published. A staging facade bolted on at the
dispatch point protects nothing.

**And the r3 fix was STILL not far enough (r4, codex -- grounded, and worse than the panel
claimed).** r3 promoted the staging dir after validating the runner result but BEFORE the
shared tail. That still leaks: the tail's `tail_finalizer` runs USER code on both sides of
persistence -- `before_save(ctx)` (`:6933`) -> `led.save()` (`:6963`) -> `after_save(...)`
(`:6969`) -- and shipped finalizers RAISE on both sides in production today
(`_CodexTailFinalizer.before_save` -> `CodexPreTailAuditError`, `_otr_scifi_codex.py:2679`;
`after_save` -> `CodexSavedLedgerAuditError`, `:2714`). Worse still, `led.save()` is called
~10 times BEFORE the tail is even entered (skeleton `:3721`, then `:4662, :4797, :5920,
:6003, :6133, :6237, :6277, :6348, :6374`), and the tail writes the CANON file before the
finalizer. So a failure anywhere in the tail would expose canonical partial state.

L5b therefore does this -- **promotion is the LAST act, after the tail, not before it**:
1. `execution_kind` + `kind` are known from the LaneSpec BEFORE any ledger exists, so
   **user-own-runner selection moves ahead of `new_ledger()`**;
2. the lane gets a **STAGING episode directory + a NON-GLOBAL staging ledger** (never
   installed as `_CURRENT`, so no downstream reader can see it);
3. the ENTIRE chain runs INSIDE staging: runner -> shared tail -> `before_save` -> final
   contract validation (LaneTailParts + provenance receipts) -> `save()` -> `after_save`;
4. only THEN: **atomic rename** into the canonical destination, rebase `Ledger.out_dir` /
   `meta.paths` / every returned path, and bind `_CURRENT` through a NEW `production_ledger`
   API. Assert no staging path survives in memory, in the ledger JSON, or in returned
   outputs;
5. on any failure: abort loud, staging dir removed, **NO canonical episode ever existed**.
User code never gets `led.save()` against the canonical episode path. L5b consequently
OWNS `nodes/OTR_LedgerScriptWriter.py` + `nodes/production_ledger.py` (§12).

**The staging gate is `kind == "original"` AND `execution_kind == "own_runner"` -- BOTH,
never `execution_kind` alone (r4, agy + anchor, convergent).** FIVE SHIPPED lanes are
dispatched own-runners today (`_RUNNER_BY_PIPELINE:1693-1699`: fable2, scifi_codex,
scifi_gemini, scifi_sonnet, original_codex56sol) and they save incrementally against the
canonical ledger. Gating on `execution_kind` alone would drag all five onto the staging path
and re-baseline proven production. Shipped dispatched lanes keep today's exact sequence,
byte-for-byte. This is stated as LAW so that a plausible-sounding "unify the two paths"
refactor cannot quietly land it.
(Common-writer lanes are unaffected -- fetch/interpret never touch the ledger.)

**Validate BEFORE the dereference (r3):** the dispatch site reads `_parts.outline_view`,
`_parts.canon`, `_parts.run_story_spine`, `_parts.final_title_override` as HARD attribute
access (`:3799-3813`) -- only `tail_finalizer` is a soft `getattr`. A `LaneTailParts`
missing any of the four raises a bare `AttributeError`, not a structured failure. The L5b
validator therefore runs BEFORE that dereference and emits the structured contract error.

**Provenance ownership, executable (r2):** the contract enumerates, per
execution_kind, the pre-tail ledger fields user code MUST produce, MAY produce, and
MUST NEVER claim -- the forbidden list starts from production history (delivery +
generic `episode_seed` receipts required before finalization; fabricated
`cast_contract.cast_seed` replay claims forbidden -- `docs/PROD_BUG_LOG.md:427-455`
+ the Bible's producer-receipt rules). The fable2 row-level merge-ownership model is
the implementation precedent; field-level lists are pinned in
`_otr_lane_contracts.py` and enforced by the L5b validator.

### 5.4 v1 prohibitions (separate core/expert campaigns, enforced by the checker)
No ComfyUI nodes/links from bundles; no third LLM slot; no new model provider or
credential system; no production-ledger schema changes; no bypass of freeze,
validation, receipts, or publishing.

### 5.5 Trust boundary (documented in EXTENDING_OTR + the activation banner)
`lane.py` is LOCAL, USER-TRUSTED code running with the user's own permissions --
exactly like any custom node they install. OTR does not sandbox it; OTR controls WHEN
it runs (never at boot; at activation in a bounded child process, and lazily at
render when the lane is selected). Running `--activate` is the consent act.

### 5.6 Pipeline + execution identity (r2-corrected -- the naive synthetic id was a
build-breaker)
Every pack's `story_pipeline_id` must resolve in the loaded pipeline registry and
match its lane's default pipeline (`_otr_story_routing.py:336-347, 382-425` -- a bare
synthetic id would raise `UnknownPipelineError` at the first user pack). Fix, two
parts (codex r2, both halves adopted):
1. **`execution_kind` on LaneSpec** (`common_writer` | `own_runner`) is the DISPATCH
   authority -- runtime never dispatches on pipeline metadata flags (registry law
   preserved).
2. **Pipeline identity:** common-writer lanes (all variants + reuse_common_writer
   originals) reference the REAL registered `legacy_many_pass` row -- their packs
   declare it and the parity law holds natively. Own-runner lanes get an
   authority-COMPOSED complete synthetic `StoryPipeline` row (`user:<lane_id>`,
   executable=true, requires_source_contract=false, declared_seams + passes from the
   manifest) published in the authority's pipeline VIEW, which is what pack
   validation consults post-L1 -- so their packs validate against a real, complete
   row. `pipelines.json` and `banks.json` are NEVER edited by activation.

## 6. Activation + quarantine state machine

States: `UNCHECKED -> ACTIVATED -> (bytes changed) STALE -> re-activate` and
`* -> QUARANTINED(issue)` from any validation failure.

**`otr_check lane <path-or-id> [--activate]`** (extends the carried checker):
1. Manifest + every JSON contract (schema, whitelist for variants, packs, rules,
   duplicate keys) -- shared production validator code, `ValidationIssue` diagnostics.
2. Path containment: per-entry resolution against the resolved
   `user_packs/source_lanes/` root (root itself MAY be an external junction -- the
   sanctioned Manager-update survival mechanism; entries may not escape it).
3. Original lanes: bounded CHILD PROCESS imports `lane.py`, verifies declared entry
   points exist with the exact keyword-only signatures (`inspect.signature`), runs
   deterministic fixture preflights. **Child environment spec (r2):**
   `sys.executable` (portable-safe), `PYTHONPATH` = repo root + `OTR_TEST_MODE=1`
   (the test suite's import discipline -- `nodes._otr_lane_contracts` resolves,
   heavy `nodes/__init__` never runs), `CUDA_VISIBLE_DEVICES=""`, fixed timeout with
   process-TREE termination (**Windows mechanism named, r3: `taskkill /F /T /PID <pid>`
   -- a bare `Popen.terminate()`/`kill()` orphans grandchildren**), capped stdout/stderr,
   exit-code protocol (0 pass / 1 contract issues / 2 harness error). SDK law: `lane.py`
   is a SINGLE FILE in v1 (no package, no relative imports) and may import stdlib + the
   contracts leaf; importing `comfy.*` or ComfyUI runtime modules fails activation
   BY DESIGN.
   **Verdict channel (r3, codex -- NOT bare stdout):** imported user code prints during
   import and fixture execution, so a JSON-verdict-on-stdout protocol is corruptible by
   any `print()` in a user lane. The verdict goes to a DEDICATED RESULT FILE; the child's
   stdout/stderr are captured separately, capped, and surfaced in the `ValidationIssue`
   on failure. **Result-file path (r4):** the PARENT mints a unique path via `tempfile` in
   the system temp dir, truncates it, passes it to the child by env var, and reads it after
   the child exits -- so parallel activations and aborted runs cannot collide or read a
   stale verdict. (agy proposed hardcoding the operator's repo `tmp\` dir; rejected -- an
   SDK contract must not bake in one machine's layout.)
   **"Network-free" is a HARNESS promise, not an enforced wall (r4, codex -- wording fix):**
   the child runs user-trusted Python, which can open a socket if it wants. What OTR
   guarantees is that the HARNESS initiates no network call and that activation's own
   verdict never depends on one. Enforcing a socket ban is not attempted (it would
   contradict §5.5) -- so the honest claim is the one stated here.
   **`technical_fn` for the fixture preflight (r3, antigravity -- a real hole: the r2
   draft said `interpret_source` EXECUTES but never said what it calls):**
   `interpret_source` needs the technical LLM slot, and activation is network-free and
   deterministic. The harness injects a RECORDED-RESPONSE mock -- `fixtures/llm_mocks.json`
   in the bundle maps a hash of the prompt/messages to a recorded response string. An
   unmocked prompt is a CLEAR activation error ("your fixture calls the technical slot with
   a prompt that has no recorded response"), never a generic or empty stub answer, which
   would merely crash the lane's own parser and report a confusing failure.
   **Execution scope (r1):** `fetch_source` is SIGNATURE-INSPECTED ONLY (it is a
   network function; activation stays network-free); `interpret_source` and
   `check_compatibility` EXECUTE against `fixtures/` samples (pre-recorded payloads
   -> surfaces shape check). Import-time discipline is enforced as: the import must
   complete inside the timeout and the fixture preflights must pass deterministically
   (best-effort I/O instrumentation is optional hardening, not the contract -- r1
   codex wording fix).
4. Variants: base-lane approval + whitelist + feed URL shape checks (scheme, no
   embedded credentials); the real-feed parse test is a QUALIFICATION smoke, not an
   activation step (activation stays deterministic).
**PUBLICATION ORDER (r3, codex -- the r2 numbering had a crash window: it published the
receipt at step 5 and only built the snapshot it points at in step 7; a crash between them
leaves a VALID-LOOKING receipt aimed at a missing or half-copied bundle, and boot would
admit it).** `--activate` is therefore strictly: validate (steps 1-4) -> build the snapshot
into a TEMP SIBLING dir -> verify the snapshot's own complete hash -> ATOMIC RENAME it into
place -> **publish the receipt LAST**. Boot admits a lane IFF **both** the receipt AND its
snapshot exist and agree. Steps 5 and 7 below describe the two objects; the order of
publication is snapshot-then-receipt, always.

**Activation is a COMPLETE crash-safe transaction (r4, codex -- "publish the receipt last"
was necessary but not sufficient):**
- `activation_id` is derived from CONTENT, independent of timestamp (two activations of
  identical bytes are the same activation).
- fsync + atomic rename the snapshot; fsync + `os.replace` a TEMP receipt file. Every
  crash point resolves to either the OLD valid activation or the NEW one -- never a
  valid-looking receipt with missing content. Crash-injection tests sit at all four points
  (after temp-snapshot write, after snapshot rename, during receipt write, after receipt
  replace).
- **NEVER delete an older content-addressed snapshot during activation.** A resident ComfyUI
  process is still importing from it (registries do not live-rescan). Re-activating while the
  server is up must leave the running process able to finish on its admitted snapshot; GC is
  DEFERRED until no process can reference it. ("Exactly one live snapshot per lane" from r2 is
  hereby corrected: exactly one CURRENT snapshot; older ones are retained, not unlinked.)
- **`__pycache__` must not mutate the snapshot.** Importing `lane.py` FROM the snapshot would
  otherwise write bytecode INTO the hashed tree and break its own byte-identity on the next
  admit. Bytecode writing is suppressed for snapshot imports; a test asserts the snapshot tree
  is byte-identical after a full render.
- **`lane_id` grammar (r4, codex -- unstated, and Windows makes it bite):** `lane_id` MUST equal
  the directory basename and match `[a-z][a-z0-9_]{0,63}`; Windows reserved names rejected;
  duplicate and protected-id checks performed with casefold/normcase. Without this, `Foo` and
  `foo` are one directory but two receipt/snapshot coordinates.

5. `--activate` writes `user_packs/receipts/lanes/<lane_id>.json`: the COMPLETE
   per-file SHA-256 map INCLUDING file count + relative paths (an added stray file
   flips the bundle STALE rather than riding along -- r1 anchor), manifest sha,
   schema+checker versions, base-lane version hash (variants), entry-point
   fingerprint (originals), **host-contract hash (r1, codex): a digest over the
   surfaces the bundle executes against -- SOURCE_PAYLOAD_KEYS schema version, the
   runner-adapter/LaneTailParts schema version, ledger/tail validator version,
   checker policy version. A core update that changes any of these flips every
   original-lane receipt STALE instead of silently reinterpreting old activations**,
   timestamp.
6. **Boot admit (restart):** the authority sweeps bundles WITHOUT importing any user
   Python. A bundle enters the registry IFF its current bytes match a successful receipt
   exactly (including the host-contract hash) AND that receipt's snapshot exists.
   Mismatch = STALE quarantine ("re-run otr_check lane --activate").
   **The boot guard is USER-SIDE ONLY -- and that split is load-bearing (r3).** The user
   sweep guards EVERY failure mode, not just malformed JSON: OS/IO errors, permission
   failures, unreadable dirs, corrupt receipts, malformed `lane.json` -- each logs and
   QUARANTINES that lane; ComfyUI always boots (r2 agy, extended in r3).
   But this guard must NOT be generalized to the shipped registry, which antigravity's r3
   fix proposed. Grounded: `OTR_LedgerScriptWriter.INPUT_TYPES` (`:2398`) calls
   `list_bank_ids()` (`:2894`) and `list_style_ids()` (`:2919`) UNGUARDED **on purpose** --
   the code comment at `:2886-2893` states that a broken `banks.json` MUST fail node
   registration LOUD, a deliberate exception to the "INPUT_TYPES must never raise"
   convention (no-fallback law). Blanket try/except there would silently hide shipped-
   registry corruption. So the §3 law holds exactly as written and is now the boot
   contract: **SHIPPED seed error = hard fail, loud, node registration dies. USER bundle
   error = quarantine, boot survives.** `list_story_pack_choices()` (NEW in L4) becomes the
   third INPUT_TYPES registry call and obeys the same split. Hashing discipline (r2, codex): an allowlist of extensions/paths,
   maximum file count and aggregate size, streamed hashing, reparse-point/device
   entries rejected -- `fixtures/` cannot smuggle unbounded assets into the hash set.
7. **Runtime (TOCTOU actually closed -- r2 supersedes r1's re-hash-then-import,
   which left a race and fought Python's module cache):** `--activate` copies the
   verified bundle into a CONTENT-ADDRESSED IMMUTABLE SNAPSHOT
   (`user_packs/.activated/<lane_id>/<receipt_hash>/`); boot admits against it;
   runtime imports `lane.py` FROM THE SNAPSHOT under a receipt-specific module name
   (importlib, unique per receipt -- stale sys.modules entries can never shadow a
   re-activation). The mutable bundle dir is authoring surface only. Cache
   invalidation: exactly one live snapshot per lane; a successful re-activate
   replaces it. Any import/contract/fetch/runner failure aborts the run loud. NEVER
   a fallback to a shipped lane, another user lane, another model, or another feed.
   Runtime failures do NOT change activation state (§1).
8. `scripts\otr_check.bat lane <id> --status` reports **UNCHECKED / ACTIVATED / STALE /
   QUARANTINED** (r4: UNCHECKED is a declared state of the machine and had been dropped from
   this list) with the receipt + host-contract hashes, WHY-stale (the first mismatching
   relative path or the moved host-contract surface), the admitted SNAPSHOT hash vs the
   mutable authoring bytes (so an operator can tell what the resident server is actually
   running), and receipt/preflight timestamps. Receipts store RELATIVE paths only
   (portable; r2). `--explain-hash` (r4, both panels) itemizes each component of
   `activation_id`, `base_hash` and HOST_CONTRACT so a stale receipt is self-diagnosing.

Quarantine laws (carried + extended -- console, resolve, and `otr_check` all emit the
IDENTICAL stored `ValidationIssue`):
- user bundle claiming a shipped lane id -> quarantined (protected-id), shipped lane
  stays selectable;
- duplicate user lane ids -> ALL claimants quarantined + coordinate tombstoned (never
  filesystem-order winner);
- partial bundles, stale receipts, schema errors, path escapes, import failures,
  contract failures, missing entry points -> quarantined;
- quarantined lanes absent from every dropdown; ComfyUI always boots.

## 7. Canonical-workflow delta

Exactly ONE: the `story_pack` optional COMBO appended at the live end of the writer
widget vector (input descriptor + widgets_values entry per the node's serialization;
positions RE-DERIVED at build from live JSON -- the lean-mean-rip interplay law
carries). Source Bank choice-list changes are data (zero diff, §2). Every widget
change: append-only; canonical JSON updated in the SAME commit; widget-count, live
INPUT_TYPES, link, JSON-round-trip, OTR_WorkflowValidator, generated-variant, and
headless name-based patching audits all green (the harness's loud length guards,
`otr_api.py:524/:613`, machine-enforce the same-change law).

## 8. dynamic_story + Engine Matrix

- `dynamic_story` stays LANE-AGNOSTIC: it consumes the valid common frozen ledger;
  any activated lane that produces one works automatically. No lane-conditional code
  in dynamic_story; no coupling in either direction.
- The Engine Matrix remains audio/image/video capabilities ONLY. `emit-lane-table`
  is CUT from v1 (r1, codex) -- `otr_check lane --status` covers readiness; a
  generated Source Lane table can follow once the authority lifecycle stabilizes,
  and it would still be a SEPARATE table, never mixed into ENGINE_MATRIX.md.

## 9. Randomizer interaction (resolved NOW, not at implementation time)

- Ownership transfer: THIS build creates `_otr_lane_specs` + performs the writer
  lane-spec rip (the randomizer plan's §1 design, absorbed verbatim as the seed);
  the randomizer build then adds `_otr_bank_roll` + the roll widget value on top and
  re-grounds its writer pins (its own PRECONDITION already mandates that re-ground).
- Roll pool = `runnable` AND manifest `random_eligible: true` AND
  `check_compatibility(request)` accepts. Rights do NOT filter (D-1 operator ruling
  stands). No silent redraw: the pool is filtered BEFORE the draw; a post-draw
  failure aborts loud like a manual pick. Manual picks preserve native errors.
- One submission = one roll; refine re-entry reuses the carried `bank_roll` receipt
  (randomizer law, unchanged).
- **`source_bank = roll` x `story_pack` (v1 rule, adopted):** rolling REQUIRES
  `story_pack == "(bank default)"`; an explicit pack combined with roll fails early
  and clearly (a pack pins a lane, a roll denies one -- contradiction by
  construction).
- **Roll RUNTIME tests belong to the randomizer build, not this one (r4, codex).** This
  build lands FIRST and `_otr_bank_roll` does not exist yet, so scheduling pool-draw,
  roll x pack and refine-roll BEHAVIOR tests here schedules tests against absent code. This
  build exposes and tests only the DETERMINISTIC contract the randomizer will consume:
  LaneSpec `random_eligible`, `runnable`, and `check_compatibility`. The roll x pack RULE
  stays stated above (it is a contract the randomizer must honor and this doc is where it is
  decided); its runtime tests move to that build.

## 10. Replay / receipt contract

Every user-lane episode stamps (extending the carried `story_model_id` +
`story_pack_sha256` stamps, Stage-2C/3C block `:3644-3665`): `lane_id`, `lane_kind`,
`lane_manifest_sha256`, `activation_receipt_id` + checker version; variants:
`base_lane_id` + base version hash; originals: fetcher/interpreter/runner code
hashes (from the receipt's file map); `story_rules_sha256`; selected pack id + sha;
source payload sha; the `CompatDecision`; selected models/providers (existing
stamps); the ledger + publishing receipts (existing). Re-entry law (r2-corrected mechanics): an explicit ACTIVATION-RECEIPT VALIDATOR runs
at every stamped-ledger re-entry gate -- it resolves the stamped lane, compares the
stamped receipt id + bundle hashes against the CURRENT activation receipt, and on
stale/missing/mismatch routes to `resolve_freeze_policy`'s existing non-raising
`terminal_error` branch (the real path is `_otr_freeze_cascade.py:302-341`; today it
only catches bank/pack resolution failure -- the receipt comparison is NEW code with
stale/missing/mismatched receipts each pinned by a test). Pack-seam re-resolution at
the cascade uses the stamped `story_model_id` (`resolve_story_pack(bank,
stamped_id)`), never the bare bank default. A changed bundle can never replay
silently under an old receipt.

## 11. Qualification ladders (two, distinct)

**Feed variant (light -- the execution path is inherited):**
manifest/schema validation; protected-id + duplicate collision tests; inheritance
whitelist tests (rejected override attempts); real feed parse -> seven-key payload
test (first live step, qualification not activation); canonical 30-word and 120-word
runs on the variant lane -- named set (r2): the shipped `DEFAULT_LLM`
(mistral-nemo) local family + ONE operator-configured cloud/frontier creative lane,
technical slot exercised as in every existing qualification (the architecture under
test is the FEED, not the writer); ledger + episode asset + `obs_publish OK` + final
OBS proof per PRODUCTION_SPRINT_LESSONS §§6-8. **720 receipts REUSE the base lane's** while (a)
the variant's activation receipt's base hash still matches the live base lane and
(b) no structured prompt/schema/validator change post-dates those receipts (the
GO_FORWARD_PLAN receipt-reuse law, applied verbatim). A grounded contract change in
the base = re-qualify.

**Original lane (full):**
activation child-process tests; lazy-import + no-import-time-I/O tests;
fetcher/interpreter/runner contract tests against fixtures; source-payload +
story-rules validation; compatibility + failure-identity tests (console == resolve ==
otr_check); randomizer-interaction tests (pool membership, roll x pack rule);
replay/hash invalidation tests; the FULL model-family ladder -- 30 words on two local
families + one configured cloud/frontier lane, same pairings at 120, then one frozen
720 leg -- with ledger, episode, OBS, full suite, and Bug Bible gates. (Ladder law per
GO_FORWARD_PLAN §4 / PRODUCTION_SPRINT_LESSONS.)

**Ladder prerequisites named (r1, codex):** the model-family set is the standing
qualification law's -- two local families (the shipped `DEFAULT_LLM`
mistral-nemo + the compact technical family already used by routing tests) plus ONE
operator-configured cloud/frontier lane (operator supplies credentials, as for every
existing qualification); qualification is GRANTED by the artifact set (ledger +
episode asset + `obs_publish OK` + final OBS file + suite/Bible receipts), nothing
softer. Missing prerequisites = the ladder does not start; there is no skip policy.

Static findings never create PBUG/Bible entries; only live production evidence does.

## 12. File ownership / change table + waves

| Wave | Surfaces (owned changes) | Content |
|---|---|---|
| L0a | `.gitignore`, new `nodes/_otr_user_packs.py`, new `nodes/_otr_validation_issue.py`, new `nodes/_otr_lane_contracts.py` | user_packs core: junction stance, containment, quarantine store, ValidationIssue, typed lane contracts leaf |
| L0b | `nodes/_otr_visual_styles.py` | styles overlay (operator-preserved; OFF the lane critical path -- parallel any time after L0a) |
| L1 | new `nodes/_otr_lane_specs.py`, `nodes/OTR_LedgerScriptWriter.py` (consumer sites), `nodes/_otr_source_payload.py`, `nodes/_otr_story_routing.py` (+ **NEW private raw seed accessor, r3**), `nodes/_otr_story_rules.py` | THE AUTHORITY: LaneSpec + execution_kind, effective row IS a `SourceBank` (r3), shipped seed absorption (lane-spec rip per randomizer design), authority pipeline view, PackRecord map (carried), resolution APIs consult authority, load-order invariant, 3-way `_clear_caches` fan-out (r3) |
| L2 | new `scripts/otr_check.py` + `otr_check.bat`, **`nodes/_otr_lane_specs.py` (r3 -- boot admit lives in the AUTHORITY; it cannot be wired from the checker files alone)**, `user_packs/receipts/` contract | checker + `lane --activate` + child-process harness (result-file verdict, recorded-response `technical_fn` mock, `taskkill /F /T`) + receipts + snapshot-then-receipt publication + boot admit + quarantine wiring |
| L3 | **new `nodes/_otr_feed_fetch.py` (r3, REQUIRED)** + `nodes/story_orchestrator.py` (r4: path corrected -- there is no root-level copy) + **`nodes/OTR_LedgerScriptWriter.py` (r3 -- `_fetch_rss_seed_or_die` is the science chain's MIDDLE HOP; without this file `feed_urls` cannot reach the fetcher)** + `nodes/_otr_media_archive_sources.py` + `nodes/_otr_source_payload.py` threading | Path A: bounded-fetch seam covering **BOTH network hops -- feed documents AND `_fetch_full_article` article bodies (r4)** -- timeouts / redirect cap / status / content-type / streamed byte cap / retry budget / concurrency clamp; THEN rss_variant end-to-end (BOTH approved bases) + per-feed provenance records. **Implementation + focused tests only (r3): the canonical 30/120 variant smokes move AFTER L4.** **Includes the shipped-lane RE-BASELINE (r4): routing science + media through the seam is not byte-identical, so their 30/120 receipts refresh here** |
| L4 | `nodes/OTR_LedgerScriptWriter.py` + `workflows/otr_canonical.json` + **`nodes/_otr_creative_prompt_router.py` (the single creative-seam choke point)** + **`nodes/_otr_line_composer.py` / `nodes/_otr_outline.py` (signature pass-through only -- neither calls `resolve_story_pack`)** + **`nodes/_otr_freeze_cascade.py` (stamped-pack replay)** + `nodes/_otr_story_routing.py` (`list_story_pack_choices`, NEW) + contract audit | story_pack widget + the FOUR-surface threading (§2, r3) + falsy-coercion law + stamps (carried, operator-required). **Then the canonical 30/120 variant smokes from L3 run here.** |
| L5a | lane.py adapter seams in `_otr_lane_specs`, example bundle in `docs/templates/example_lane/` | Path B SDK core: fetch/interpret/compat entry-point adapters (enumerated matrix, §5.2) + common-writer reuse, trust doc, reference lane |
| L5b | run_lane dispatch + LaneTailParts validation + **`nodes/OTR_LedgerScriptWriter.py` + `nodes/production_ledger.py` (r3 -- own-runner selection moves BEFORE `new_ledger()`; the staging ledger cannot be bolted on at the dispatch point)** | Path B EXPERIMENTAL tier: own-runner lanes with staging dir + non-global staging ledger + validate-before-dereference + atomic promotion; only after L5a's reference lane is proven (r1 staging) |
| L6 | `docs/templates/`, `docs/EXTENDING_OTR.md` generator, `README.md` | templates + generated tables + recipes (incl. carried local-LLM discovery recipe, docs-only) |
| L7 | tests + live proofs | full suite + Bible + BOTH ladders' live legs (§11). **(r4) Adds an INERT own-runner example + failure-injection + canonical 30/120 legs THROUGH staging -> promotion -> freeze -> episode -> OBS before `execution_kind=own_runner` is exposed: L5a's reference lane is common-writer, so it proves nothing about L5b's materially different ledger/promotion path** |

**Moved OUT of this campaign (r1, codex):** the carried VALIDATE_INPUTS suffix fix +
shipped-model-ID manifest (old W3) -- unrelated to source lanes; it becomes its own
micro-plan executing against the superseded doc's converged design, scheduled
independently in GO_FORWARD_PLAN. The user visual-styles overlay STAYS in L0
(partially rejected cut: the user_packs foundation + quarantine store are genuinely
shared, the marginal cost is near zero, and styles are part of the operator's
original product ask).

Dependency graph (r3-corrected): ACTIVATION-GATE(coder slot + ownership receipts) ->
**L0a** -> L1 -> L2 -> **L3 -> L4** -> L5a -> L5b -> L6 -> L7, with **L0b parallel to
everything after L0a** (it was never a blocker; the r2 `L0 -> L1` edge wrongly made it
one). L3 and L4 are now SEQUENTIAL, not parallel: L3's canonical smokes need L4's stamps
to prove anything (r3).

**Tests are per-wave, not deferred to L7 (r3, codex -- the green-commit contract makes
this non-negotiable).** EVERY wave L0-L6 runs its focused tests + the full Windows suite +
the Bug Bible before it is pushed. L7 owns ONLY cross-wave qualification and the live
artifacts (§11) -- it is not where testing starts.

Each wave = one green pushed commit. Collision note for the queue: L1 touches the SAME
writer surfaces the randomizer plan reserved (the rip moves here); L4 must land outside the
bakeoff code-freeze; sequencing stays "this feature before Randomizer, final qualification,
and the one bakeoff" (operator).

## 13. Carried forward from the superseded plan (verbatim design, re-homed)

Local causal-LLM discovery + recipes; user visual styles overlay; content packs
inside lanes + per-pack parity laws + PackRecord map + TOCTOU sha; structured
`ValidationIssue`; templates + generated `EXTENDING_OTR.md` (tables-only, byte-pinned);
quarantine/collision taxonomy incl. tombstones + protected ids; external junction
support; replay hashes; append-only widget discipline + story_pack widget design
(serialization form, two-channel threading, run() signature, falsy coercion);
`.bat` 4-probe resolution; receipts location/containment; CP-gate receipt style
(re-numbered per §11's two ladders). VALIDATE_INPUTS + shipped-ID baseline: moved to
its own micro-plan (r1; design stands in the superseded doc, execution decoupled).
CUTS carried: models.d, scaffold CLI, hand-written schema docs, live rescan, generic
GGUF walker, model probe, within-file all-errors, otr_check.sh, LOC-table estimates;
NEW r1 cut: emit-lane-table.

## 14. Precondition -- fast-moving base

All line pins were re-read this session but the base moves daily (the randomizer
plan burned this lesson: three HEAD moves in one planning window). Standing rule
carried: claim the coder slot in GO_FORWARD_PLAN, record actual HEAD, re-read every
pin at that HEAD, THEN edit.

## 15. Estimates (replace the retired 4-7 day figure)

| Track | Estimate |
|---|---|
| Shared registry/checker/UI (L0-L2) | 5-6 coder-days |
| Story-pack widget + FOUR-surface threading + replay stamps (L4) | 2-3 coder-days (r3: was folded into the row above at ~1; the router/composer/outline/cascade threading is real work) |
| Bounded-fetch seam `_otr_feed_fetch.py` (L3, r3 -- NEW, was wrongly assumed inherited) | 1.5-2.5 coder-days |
| Safe feed-variant path (L3, both bases, 3-hop science threading) | 2-3 coder-days |
| Original-lane SDK (L5a) + own-runner staging/promotion (L5b, r3-reseated) + templates/docs (L6) | 7-10 coder-days (r3: +1 for moving own-runner selection ahead of `new_ledger()` and the atomic promotion) |
| Verification wave (L7) coding share | 1-2 coder-days |
| Article-body hop + per-feed provenance/rights records (L3, r4) | 1.5-2.5 coder-days |
| Own-runner staging: full tail-inside-staging + atomic promotion + path rebase (L5b, r4 -- promotion moved after the tail) | +1-2 coder-days |
| **Total coding** | **~21-31 coder-days** (r3: +3-4 for the bounded-fetch seam and pack threading; r4: +3-5 for the second network hop, per-feed rights, and the corrected staging boundary -- all latent work the earlier drafts had priced at zero) |
| GPU qualification: variant smokes 30/120 | 0.5-1 elapsed GPU day |
| GPU qualification: **shipped-lane RE-BASELINE** (science + media 30/120 refresh after the fetch seam, r4) | 0.5-1 elapsed GPU day |
| GPU qualification: reference original lane full ladder + 720 | 2-4 elapsed GPU days |
| GPU qualification: own-runner tier 30/120 through staging + promotion (r4) | 0.5-1 elapsed GPU day |

**Honest note on the trend:** the estimate has grown from ~15-22 to ~21-31 coder-days across
the arc. None of that growth is scope creep -- every increment is work the earlier drafts had
priced at ZERO because they asserted an inheritance that did not exist (network bounds), or
threaded an identity that could not resolve (PackRef), or promoted state at a boundary that
could not hold it (staging). A plan that got cheaper each round would have been the warning
sign.

## 16. Normative defaults (folded r1; operator ratifies at approval, may override)

The r1 panel required these to be normative rather than open (§16 items alter
security/scope/ownership). The recommendations are now the PLAN; each carries a
ratification flag:

1. **Approved feed-capable base set (v1) = `science_news` + `media_archive`** --
   both parameterization seams grounded (§4). [RATIFY]
2. **Running `--activate` IS the consent act** for executing bundle Python in the
   bounded child; the command prints the §5.5 trust-boundary banner first. [RATIFY]
3. **Variant story_rules default = inherit base**; bundles may override with their
   own `story_rules.json`. [RATIFY]
4. **Reference example lane ships INERT** under `docs/templates/example_lane/`
   (a test pins that nothing under docs/ can enter the authority). [RATIFY]
5. **`_otr_lane_specs` ownership transfers to this build** (randomizer re-grounds on
   top; its plan gets a delta note after this doc converges). [RATIFY]
6. **The bounded-fetch seam (`nodes/_otr_feed_fetch.py`) is IN SCOPE and blocking for
   Path A (r3).** This is the one r3 item that ADDS cost (+1.5-2.5 coder-days, §15) rather
   than re-sequencing existing cost, so it gets its own flag. The alternative -- shipping
   user-supplied feed URLs into `feedparser.parse()` with no timeout, no status check, no
   content-type check and no byte cap (the grounded state today, §4) -- is not a variant
   OTR can honestly offer. Ratifying this ratifies the estimate change. [RATIFY]

**Added by r4 (these three change SCOPE or COST -- they are operator calls, not panel
calls):**

7. **The seam ALSO covers `_fetch_full_article`, and the SHIPPED lanes route through it
   -- accepting a re-baseline.** The article-body hop (§4) is the real hole: with a
   user-supplied feed, `link` can point anywhere, and OTR currently follows it with no host
   allowlist, no redirect cap, and no size cap. Hardening it means the shipped science +
   media lanes change behavior (a content-type/size policy can drop an entry
   `feedparser.parse()` accepts today), so their 30/120 receipts must refresh in L3. The
   alternative -- a user-only fetch path -- leaves the PRODUCTION lane on the racy
   global-socket-timeout path we just exposed, i.e. exactly the "dead lever" the operator
   ruled against. [RATIFY the re-baseline]
8. **Numeric network bounds:** connect 5s / read 10s / 3 redirects / 2 MiB decoded body /
   2 retries (408, 429, 5xx) / max 32 feeds / max 2048-char URLs / max 8 workers. Both
   panels proposed these independently; they are DEFAULTS, not measurements. [RATIFY]
9. **Per-feed provenance + rights records** (§4) rather than a single lane-level credits
   string. Required by the binding `SOURCE_BANK_GUIDE.md` §5 + `SOURCE_BANK_PREFLIGHT.md`
   Gate 2 (fail CLOSED on unknown/incompatible adaptation rights), and it expands the
   manifest beyond a bare URL list. [RATIFY]

## 17. Non-goals (v1)

Marketplace/sharing; sandboxing user Python; new providers/credential systems; third
LLM slot; **top-level / versioned ledger-SCHEMA changes (r4 narrowing: namespaced `meta`
receipts ARE allowed -- §10 deliberately adds metadata fields, so a blanket "no ledger
schema changes" contradicted the plan)**; ComfyUI nodes/links from bundles; per-lane UI
panels; `lane_options_json`; Engine Matrix mixing; Design B style roll; relaxing fail-loud,
freeze, SFW, or publishing gates.

## 18. Normative contracts appendix (r4 -- BINDING; the coder builds from THIS, not from
## a promise that "r2 carries the field-level detail")

These are repository-owned types in `nodes/_otr_lane_contracts.py` (dependency-free leaf,
stdlib only). Each ships a constructor, a validator, exact null rules, ONE known-valid
fixture, and specified rejection behavior. Field-level provenance ownership per
`execution_kind` (§5.3) is enforced by the L5b validator against these types.

```python
SOURCE_PAYLOAD_KEYS = frozenset({          # defined HERE, re-exported by _otr_source_payload
    "headline", "summary", "full_text", "source", "date", "link", "seed_text"})

@dataclass(frozen=True)
class SourceFetchResult:                   # moved from _otr_source_payload.py:85-95
    payload: dict                          # EXACTLY the seven keys above; unknown key = hard error
    source_meta: dict | None = None        # provenance sidecar (selected feed + article)
    source_rights: dict | None = None      # rights sidecar; fail CLOSED on unknown/incompatible

@dataclass(frozen=True)
class PackRef:                             # the threaded unit -- parsed ONCE, never re-parsed
    owner_lane_id: str                     # resolves the PATH. NEVER the selected bank (r4).
    story_model_id: str                    # pack filename stem == the one canonical identity
    pipeline_id: str
    sha256: str

@dataclass(frozen=True)
class CompatRequest:
    target_words: int
    refine_active: bool
    source_ref: str = ""
    custom_premise: str | None = None

@dataclass(frozen=True)
class CompatDecision:
    accepted: bool
    reason: str = ""                       # REQUIRED and non-empty when accepted is False

@dataclass(frozen=True)
class InterpreterResult:
    casting_brief: str
    script_brief: str
    news_close_brief: str
    key_terms: tuple[str, ...]
    attempts: int
    def model_dump(self) -> dict: ...      # attribute/model_dump agreement is VALIDATED
                                           # (_otr_source_payload.py:204-257)

@dataclass(frozen=True)
class LaneTailParts:
    outline_view: dict                     # the four below are HARD dereferences at
    canon: object                          # OTR_LedgerScriptWriter.py:3799-3813 --
    run_story_spine: object                # a missing one raises bare AttributeError, so
    final_title_override: str | None       # the validator MUST reject before that point
    tail_finalizer: object | None = None   # the ONLY soft (getattr) field
```

`LaneRunContext` / `SlotPort` (the injected capabilities handed to `run_lane` and
`fetch_source`) are typed here too -- user code never receives the writer's raw internal
`resolved` dict or `slot_scheduler`, so writer internals never become public SDK surface.
Every type in this appendix is part of the HOST_CONTRACT hash: changing any field stales
every original-lane receipt (§6.5), and a mutation test proves it.

---

**Next step (r4-revised):**
1. **Operator ratifies §16 -- now NINE flags.** Flags 7-9 (article-body hop + shipped-lane
   re-baseline, numeric network bounds, per-feed rights records) change SCOPE and COST, so
   they are operator calls, not panel calls. Everything else in the r4 fold is mechanical and
   already decided.
2. **One r5 confirmation pass** on this folded doc. NOT a fifth opinion for its own sake:
   r4 proved that a confident fold introduces defects (it caught three of mine), and the r4
   fold has not yet been read adversarially by anyone.
3. If r5 comes back clean -> update GO_FORWARD_PLAN (replacing the superseded queue entry)
   -> claim the coder slot -> re-derive EVERY line pin at the recorded HEAD (§14) -> code.

**Do not skip step 2.** The doc is now internally consistent and grounded, which is exactly
what it looked like at the end of r3 -- one round before r4 found twelve things wrong with it.

Kibitz artifacts + the per-round judgment logs: `kibitz-runs/2026-07-12-user-source-lanes/{r1..r4}/`.

**Standing panel note (so r5 does not spend a round re-litigating):** the user visual-styles
overlay (L0b) has now been proposed for CUT by a panelist in r1, r3, and r4 -- and rejected
all three times for the same grounded reasons (operator product ask; shared `user_packs` +
quarantine foundation; already off the critical path at `L0a -> L1`, blocking nothing). It is
not scope creep. Likewise, stripping the `(rN ... REJECTED because ...)` annotations from this
doc was proposed and rejected: those annotations are the only thing preventing a future
reviewer from re-introducing a fix that was already grounded and killed -- the dead-lever
failure mode. They stay.
