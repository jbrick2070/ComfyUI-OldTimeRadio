# User Source Lanes -- Replacement Architecture & Coding Plan (v1 DRAFT)

- **Date:** 2026-07-12 (late). **Status:** DRAFT FOR ARCHITECTURE APPROVAL -- no code,
  no kibitz arc yet, no GO_FORWARD_PLAN change until this converges (operator
  directive). Re-grounded against live HEAD this session; the fast-moving-base
  precondition in §14 governs every line pin below.
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
scoped to what the state machine can actually guarantee).

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
3. Run `otr_check lane <path> --activate`. The checker validates every JSON, then
   imports and contract-tests your Python in a bounded child process against your
   fixtures. Fix and re-run until the receipt lands.
4. Restart. Select the lane. Render. (Editing ANY bundle file de-activates the lane
   until you `--activate` again -- byte hashes are the truth.)

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
table and no view"). Grounded seed state it replaces/wraps (all in the writer today):
`_RUNNER_BY_PIPELINE` (:1656), lazy `_run_*_lane` wrappers (:1632-1647),
`_LEGACY_INLINE_PIPELINES` (:1665-1667), `_resolve_lane_runner` (:1718-1735), with
exactly four consumers (telemetry :1684, resolution :1724, refine rejection :3357,
dispatch :3721-3743) -- plus `_FETCHERS`/`_INTERPRETERS`
(`_otr_source_payload.py:512-532`) and the banks/pipelines registry
(`_otr_story_routing.py:465-482`).

`LaneSpec` (one record per lane, shipped AND user):

| field | meaning |
|---|---|
| `lane_id`, `label`, `kind` | identity; kind = `shipped` / `rss_variant` / `original` |
| `runnable` | the ONLY run gate (registry law preserved) |
| `pipeline_id` | shipped row id, or synthetic `user:<lane_id>` (§5) |
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
- **Reset seam (r1, codex):** `_otr_lane_specs._clear_caches()` mirrors the existing
  hooks (styles :393, routing :547, rules :317) -- one internal reset for tests and
  restart admission; no live rescanning.

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
  "default_story_pack": "",
  "story_rules": "inherit",
  "random_eligible": true,
  "notes": ""
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
- **Source-identity projection (r1, exact keys):** the manifest's `source_identity`
  entries overlay the base row's `defaults` dict keys consumed downstream --
  `source_material_label`, `credits_source_line` (stamped at
  `OTR_LedgerScriptWriter.py:3648-3659` and rendered by credits/HUD), plus the other
  `defaults` label keys enumerated in the whitelist at build. Precedence: variant
  manifest > base defaults. Nothing outside the enumerated keys projects.

- `base_lane_id` must be in the APPROVED feed-capable base set -- v1:
  `science_news` + `media_archive`, BOTH now grounded (normative default, ratify at
  approval; agy's science-only minority recorded). The base must be shipped AND
  runnable.
- **Enabling change (one parameterization, TWO enumerated files -- r1):**
  - science: `SCIENCE_NEWS_FEEDS` constant (`story_orchestrator.py:1228-1263`)
    iterated by `_fetch_science_news(...)` (:1677+), reached via the wrapper
    `_fetch_science_rss` (`_otr_source_payload.py:265`);
  - media_archive: `DEFAULT_MEDIA_ARCHIVE_FEEDS` + the EXISTING
    `OTR_MEDIA_ARCHIVE_FEEDS` env resolver (`_otr_media_archive_sources.py:16-19,
    156-164`), fetcher `fetch_media_archive_rss` (:175+), reached via
    `_otr_source_payload.py:398-406`.
  Both fetch chains gain `feed_urls: tuple[str, ...] | None = None` threaded from the
  LaneSpec. Precedence: LaneSpec feeds (variants) > `OTR_MEDIA_ARCHIVE_FEEDS` env
  (legacy operator override, media only, shipped lane unchanged) > shipped constants.
  None = byte-identical shipped behavior. All Gate-2 network laws (timeouts, bounded
  retries, size/status/content-type checks, untrusted-data delimiting) are INHERITED
  CODE -- the variant cannot alter them.
- The variant's effective lane = base LaneSpec + whitelisted overrides. It can NEVER
  override runner, pipeline, interpreter, provider, ledger behavior, or safety
  checks -- the whitelist makes that unrepresentable, not merely forbidden.
- `story_rules: "inherit"` (default) uses the base's rules pack; a bundle MAY ship
  `story_rules.json` instead (validated with the same code as shipped rules). Packs:
  optional `story_packs/` (validated per the carried content-pack laws); blank
  `default_story_pack` = base's default pack.
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

Adds to the common fields: `source_kind`; `entry_points` (which of
`fetch_source` / `interpret_source` / `check_compatibility` / `run_lane` exist --
`check_compatibility` REQUIRED); `reuse_common_writer: true` XOR `run_lane` declared;
`required_seams` (when reusing the common writer); `source_ref_mode`;
`supports_custom_premise`; `word_range: {min, max}`; `random_eligible`.

### 5.3 Typed runtime interfaces (grounded against live contracts; exact signatures
re-derived at build)

- `fetch_source(*, lane, technical_model, source_ref="") -> SourceFetchResult | dict`
  -- MUST return the writer's exact seven-key `SOURCE_PAYLOAD_KEYS` envelope
  (`_otr_source_payload.py:80-82`; unknown key = hard error), mirroring the
  registered-fetcher contract (`SOURCE_BANK_GUIDE.md` §5).
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

### 5.4 v1 prohibitions (separate core/expert campaigns, enforced by the checker)
No ComfyUI nodes/links from bundles; no third LLM slot; no new model provider or
credential system; no production-ledger schema changes; no bypass of freeze,
validation, receipts, or publishing.

### 5.5 Trust boundary (documented in EXTENDING_OTR + the activation banner)
`lane.py` is LOCAL, USER-TRUSTED code running with the user's own permissions --
exactly like any custom node they install. OTR does not sandbox it; OTR controls WHEN
it runs (never at boot; at activation in a bounded child process, and lazily at
render when the lane is selected). Running `--activate` is the consent act.

### 5.6 Pipeline identity
Activation synthesizes `pipeline_id = "user:<lane_id>"` mapped onto the two EXISTING
pipeline classes (grounded `pipelines.json`: `requires_source_contract` metadata for
common-writer lanes; `executable: true` for run_lane lanes -- runtime dispatch stays
on lane shape, never on those flags, preserving the registry law). `pipelines.json`
and `banks.json` are NEVER edited by activation.

## 6. Activation + quarantine state machine

States: `UNCHECKED -> ACTIVATED -> (bytes changed) STALE -> re-activate` and
`* -> QUARANTINED(issue)` from any validation failure.

**`otr_check lane <path-or-id> [--activate]`** (extends the carried checker):
1. Manifest + every JSON contract (schema, whitelist for variants, packs, rules,
   duplicate keys) -- shared production validator code, `ValidationIssue` diagnostics.
2. Path containment: per-entry resolution against the resolved
   `user_packs/source_lanes/` root (root itself MAY be an external junction -- the
   sanctioned Manager-update survival mechanism; entries may not escape it).
3. Original lanes: bounded CHILD PROCESS (`sys.executable`, timeout, no GPU --
   `sys.executable` guarantees portable-install compatibility, r1 agy) imports
   `lane.py`, verifies declared entry points exist with the exact keyword-only
   signatures (`inspect.signature`), runs deterministic fixture preflights.
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
   Python; a bundle enters the registry IFF its current bytes match a successful
   receipt exactly (including the host-contract hash). Mismatch = STALE quarantine
   ("re-run otr_check lane --activate").
7. **Runtime (TOCTOU-closed, r1 codex):** selecting an activated original lane
   RE-HASHES the bundle files against the admitted receipt IMMEDIATELY BEFORE the
   lazy import -- a file changed after boot fails STALE before any user code runs.
   Then `lane.py` imports in-process; any import/contract/fetch/runner failure aborts
   the run loud. NEVER a fallback to a shipped lane, another user lane, another
   model, or another feed. Runtime failures do NOT change activation state (§1).
8. `otr_check lane <id> --status` reports ACTIVATED / STALE / QUARANTINED with the
   receipt + host-contract hashes (r1, codex optional -- adopted).

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

## 10. Replay / receipt contract

Every user-lane episode stamps (extending the carried `story_model_id` +
`story_pack_sha256` stamps, Stage-2C/3C block `:3644-3665`): `lane_id`, `lane_kind`,
`lane_manifest_sha256`, `activation_receipt_id` + checker version; variants:
`base_lane_id` + base version hash; originals: fetcher/interpreter/runner code
hashes (from the receipt's file map); `story_rules_sha256`; selected pack id + sha;
source payload sha; the `CompatDecision`; selected models/providers (existing
stamps); the ledger + publishing receipts (existing). Re-entry law (carried, r4):
stamped hashes MUST match the current activation receipt or the re-entry fails loud
(freeze cascade via its non-raising `terminal_error` path, `:302-305`). A changed
bundle can never replay silently under an old receipt.

## 11. Qualification ladders (two, distinct)

**Feed variant (light -- the execution path is inherited):**
manifest/schema validation; protected-id + duplicate collision tests; inheritance
whitelist tests (rejected override attempts); real feed parse -> seven-key payload
test (first live step, qualification not activation); canonical 30-word and 120-word
runs on the variant lane (one local family + the configured lane, since the
architecture under test is the FEED, not the writer); ledger + episode asset +
`obs_publish OK` + final OBS proof. **720 receipts REUSE the base lane's** while (a)
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
| L0 | `.gitignore`, new `nodes/_otr_user_packs.py`, `nodes/_otr_validation_issue.py`, `nodes/_otr_visual_styles.py` | user_packs foundation: junction stance, containment, quarantine store, ValidationIssue, styles overlay (carried) |
| L1 | new `nodes/_otr_lane_specs.py`, `nodes/OTR_LedgerScriptWriter.py` (4 consumer sites), `nodes/_otr_source_payload.py`, `nodes/_otr_story_routing.py`, `nodes/_otr_story_rules.py` | THE AUTHORITY: LaneSpec, shipped seed absorption (lane-spec rip per randomizer design), PackRecord map (carried), resolution APIs consult authority |
| L2 | new `scripts/otr_check.py` + `otr_check.bat`, `user_packs/receipts/` contract | checker + `lane --activate` + child-process harness + receipts + boot admit + quarantine wiring |
| L3 | `story_orchestrator.py` + `_otr_media_archive_sources.py` + `_otr_source_payload.py` threading | Path A: rss_variant end-to-end (BOTH approved bases, r1) + 30/120 variant smokes |
| L4 | `nodes/OTR_LedgerScriptWriter.py` + `workflows/otr_canonical.json` + contract audit | story_pack widget + two-channel consumer threading + stamps (carried, operator-required) |
| L5a | lane.py adapter seams in `_otr_lane_specs`, example bundle in `docs/templates/example_lane/` | Path B SDK core: fetch/interpret/compat entry-point adapters + common-writer reuse, trust doc, reference lane |
| L5b | run_lane dispatch + LaneTailParts validation | Path B EXPERIMENTAL tier: own-runner lanes, only after L5a's reference lane is proven (r1 staging) |
| L6 | `docs/templates/`, `docs/EXTENDING_OTR.md` generator, `README.md` | templates + generated tables + recipes (incl. carried local-LLM discovery recipe, docs-only) |
| L7 | tests + live proofs | full suite + Bible + BOTH ladders' live legs (§11) |

**Moved OUT of this campaign (r1, codex):** the carried VALIDATE_INPUTS suffix fix +
shipped-model-ID manifest (old W3) -- unrelated to source lanes; it becomes its own
micro-plan executing against the superseded doc's converged design, scheduled
independently in GO_FORWARD_PLAN. The user visual-styles overlay STAYS in L0
(partially rejected cut: the user_packs foundation + quarantine store are genuinely
shared, the marginal cost is near zero, and styles are part of the operator's
original product ask).

Dependency graph: ACTIVATION-GATE(coder slot + ownership receipts) -> L0 -> L1 ->
L2 -> {L3, L4} -> L5a -> L5b -> L6 -> L7. Each wave = one green pushed commit.
Collision note for the queue: L1 touches the SAME writer surfaces the randomizer
plan reserved (the rip moves here); L4 must land outside the bakeoff code-freeze;
sequencing stays "this feature before Randomizer, final qualification, and the one
bakeoff" (operator).

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
| Shared registry/checker/UI (L0-L2 + L4) | 6-8 coder-days |
| Safe feed-variant path (L3, both bases) | 2-3 coder-days |
| Original-lane SDK (L5a + L5b) + templates/docs (L6) | 6-9 coder-days |
| Verification wave (L7) coding share | 1-2 coder-days |
| **Total coding** | **~15-22 coder-days** (old W3 moved out, LaneTailParts law added -- net wash) |
| GPU qualification: variant smokes 30/120 | 0.5-1 elapsed GPU day |
| GPU qualification: reference original lane full ladder + 720 | 2-4 elapsed GPU days |

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

## 17. Non-goals (v1)

Marketplace/sharing; sandboxing user Python; new providers/credential systems; third
LLM slot; ledger schema changes; ComfyUI nodes/links from bundles; per-lane UI panels;
`lane_options_json`; Engine Matrix mixing; Design B style roll; relaxing fail-loud,
freeze, SFW, or publishing gates.

---

**Next step:** operator architecture approval -> kibitz r1-r4 arc on THIS doc ->
fold -> only then update GO_FORWARD_PLAN (replacing the superseded queue entry) and
release to the coder.
