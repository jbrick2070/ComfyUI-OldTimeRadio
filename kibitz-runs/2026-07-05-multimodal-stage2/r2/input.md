# Multi-Modal Story Schema -- STAGE 2 HARDENED SUB-PLAN (v2, post-kibitz r1)

Date: 2026-07-05. Branch: `v2.0-alpha`. Parent: `docs/multimodal-story-schema/BUILD_PLAN.md`.
Status: r1 CONVERGED (codex + antigravity, Claude anchor+judge; artifacts in
`kibitz-runs/2026-07-05-multimodal-stage2/r1/`). Precondition SHIPPED @843ced43
(outline-resolver swallow removed + AST pin test).

## 0. Scope (what Stage 2 is -- and is not)

Stage 2 = **story-path ROUTING + the new lane packs**:

1. **Registries:** `nodes/story_packs/banks.json` + `nodes/story_packs/pipelines.json`.
2. **Resolution (Python):** fail-loud `source_bank` / `story_model` / `story_pipeline`
   resolution -- unknown id = hard error, no fallback. Replaces the router's
   transitional `_SCIENCE_PACK_PATH` with bank -> pack-coordinate routing.
3. **Lane packs authored:** public_domain_story, media_archive, custom_source_bank
   (simple_4), addressable + validated; NOT executable yet.

Stage 2 exposure model (r1 codex S1): the API takes explicit
`resolve_story_pack(source_bank_id, story_model_id=None)` -- model defaults from
the bank; `story_pipeline` always resolves from the bank/pack default in Stage 2
(no per-episode pipeline override surface). The only user surface is the 2C
`source_bank` widget.

NOT in Stage 2:
- No visual_style anything (Stage 3); `default_visual_style` in banks.json is
  TOLERATED INERT (type-checked string, never resolved) -- same posture as
  Stage 1's inert pack fields (r1 codex cut #1 folded).
- New lanes do NOT execute end-to-end (interpreters/fetchers + the simple_4 pass
  runner are later work). Selecting a `runnable:false` bank at run time FAILS
  LOUD naming the bank -- never a silent fall-through to the science path.
- Workflow-JSON widget = chunk 2C, last, own gate.

## 1. Identifier model (r1 MUST-FIX resolutions -- canonical, no exceptions)

- **Path IS the coordinate:** every pack lives at
  `nodes/story_packs/<source_bank_id>/<story_model_id>.json`, and the pack's
  header triple MUST match its path (validator hard error on mismatch). The
  simple_4 pack therefore lives at
  `nodes/story_packs/custom_source_bank/simple_4_prompt_experimental.json`
  (NOT `experimental/` -- the lab's directory name dies here; codex r1 M1).
- **Runnable truth (codex r1 M2):** in the SHIPPED banks.json, ONLY
  `science_news` is `runnable: true`. media_archive / public_domain_story /
  custom_source_bank ship `runnable: false` (their interpreters/fetchers/pass
  runner do not exist). Flipping a bank to true requires its execution lane to
  land first. Resolver: resolving a pack from a non-runnable bank is ALLOWED
  (validation/addressability); RUNNING an episode from one raises
  `StoryBankNotRunnableError` naming the bank + what is missing.
- **Seam vocabulary is PRODUCTION-ONLY everywhere (codex r1 M3):** banks'
  `required_seams`, pipelines' `seam_refs`, and pack `prompt_stages` all use the
  production seam names. The lab names (`outline_system`, `pitch_room_system`,
  `line_grounding`, `style_pick_inventor`, ...) are normalized during adaptation
  and never enter the repo's JSON.
- **Two seam namespaces, no global growth (codex r1 M4 + anchor M1; antigravity's
  "placeholder allowlist entries" REJECTED as an allowlist-law violation):**
  - `PRODUCTION_SEAM_ALLOWLIST` stays EXACTLY the Stage-1 set; the exact-equality
    test (`test_allowlist_equals_authored_set`) is NOT relaxed.
  - A pipeline may declare its OWN seams (`declared_seams` on the pipeline row,
    e.g. simple_4's `pass_1_creative_story..pass_4_technical_ledger_audit`). A
    pack's `prompt_stages` keys must each be in
    `PRODUCTION_SEAM_ALLOWLIST UNION pipeline.declared_seams` for the pack's own
    `story_pipeline_id`. Loader change: `_validate`'s seam check gains an
    optional `extra_seams: frozenset` parameter supplied by the ROUTING layer
    (pack loaded standalone without routing context keeps the strict
    production-only check -- i.e. Stage-1 behavior is unchanged for the science
    pack and for direct `load_pack` calls).
  - `seam_refs` are cross-checked ONLY for `executable: true` pipelines
    (antigravity cut folded); `legacy_many_pass` is DESCRIPTIVE metadata
    (`executable: false` -- rename of the lab's `executable_in_lab`), its
    seam_refs normalized for hygiene but not load-bearing.

## 2. Chunking

**Chunk 2A -- registries + resolver (science lane byte-identical).**
- `nodes/story_packs/banks.json` (4 banks per section 1; fields: source_bank_id,
  label, source_kind, interpreter, fetcher, default_story_model,
  default_story_pipeline, defaults{}, required_seams[], runnable, guide_ref,
  default_visual_style [inert]). `defaults{}` contents opaque (structural
  type-check only; consumers come with lane execution).
- `nodes/story_packs/pipelines.json` (legacy_many_pass executable:false
  descriptive; simple_4_prompt_experimental executable:false until its runner
  exists, with declared_seams = the 4 pass seams).
- `nodes/_otr_story_routing.py` (stdlib-only): load-once cached, strict
  (dup-key/unknown-key/non-empty ids), typed `StoryRoutingError` hierarchy
  (+ `StoryBankNotRunnableError`), `get_bank`, `get_pipeline`,
  `resolve_story_pack(source_bank_id, story_model_id=None)`, and an explicit
  `_clear_caches()` test hook (anchor S3).
- **Registry sweep validation (anchor M2):** every `*.json` under
  `nodes/story_packs/<bank>/` must validate and its header triple must match its
  path coordinates; an orphan/misfiled pack = hard error at registry load.
  Cross-refs: bank.default_story_model resolves to an on-disk pack;
  bank.default_story_pipeline + pack.story_pipeline_id exist in pipelines.json;
  bank.required_seams (production names) present in the default pack;
  executable pipelines' seam_refs resolve.
- Router: `_otr_creative_prompt_router.py` drops `_SCIENCE_PACK_PATH`, calls
  `resolve_story_pack("science_news")`. The fixed "science_news" binding is
  STILL transitional -- threading the 2C widget selection into the resolver is
  2C's job (antigravity r1 M1, deferred by design; verify-at-build there).
  Byte-identical output pinned by existing equivalence tests.
- Same-commit test updates (anchor S1): the router caller-count pin
  (test_creative_prompt_router.py:173-188) if the call count changes, and the
  sanctioned-consumer guard extended to allow `_otr_story_routing.py`.
- Tests: routing fail-loud matrix (unknown bank/model/pipeline, missing pack,
  header/path mismatch, orphan pack, dup key, malformed JSON, non-runnable-run
  raise), science byte-identity + audio suite unchanged.

**Chunk 2B -- lane packs authored + validated.**
- `nodes/story_packs/public_domain_story/faithful_radio_adaptation.json`,
  `nodes/story_packs/media_archive/media_restoration_adventure.json`,
  `nodes/story_packs/custom_source_bank/simple_4_prompt_experimental.json`.
- Content adapted from the lab prose to PRODUCTION seams: at minimum
  `line_composer_system` (the live consumed seam) per lane + `coda_system`;
  lane seams with no production consumer are NOT invented. simple_4's pack
  carries its 4 declared pipeline seams. `status: "ready_fixture"` /
  `"experimental"` kept as inert metadata.
- Tests: each lane pack loads via the registry sweep; exact-key-set per pack;
  resolve_story_pack reaches each lane; StoryBankNotRunnableError on run-intent
  for each non-runnable bank; science-lane byte-identity unchanged.

**Chunk 2C (GATED, last) -- the selector surface.**
- `source_bank` widget on the writer node, wired IN
  `workflows/otr_scifi_16gb_full.json` same commit; saved default =
  `science_news`. APPEND at the END of widgets_values (BUG-LOCAL-097); expected
  append point after the current optional tail (`story_scaffold`,
  OTR_LedgerScriptWriter.py ~:2191 -- VERIFY-AT-BUILD, codex r1 S2).
- Selection threads from `run()` into the resolver EXPLICITLY (argument /
  request field) -- no thread-local unless the call graph forces it
  (antigravity r1 M1 verify-at-build).
- Boot posture: the dropdown choices come from banks.json at INPUT_TYPES time;
  a broken registry fails node registration LOUD. NO try/except with a baked-in
  choice list -- a silent `["science_news"]` fallback is a no-fallback-law
  violation (antigravity r1 M2 fix REJECTED; risk noted, answered fail-loud).
- Non-runnable banks ARE listed (visible lanes, honest error on use) --
  simplest consistent no-hidden-state surface; revisit at operator eyeball.
- Gate: kibitz on the wiring + validator + widget audit before commit.

## 3. Invariants (unchanged)

JSON owns content; Python owns validation/routing/execution; NO fallbacks;
unknown id = hard error; audio spine FROZEN (`test_audio_byte_identical` green);
science lane byte-identical through 2A/2B; suite + Bug Bible + B7 green per
chunk; UTF-8 no BOM; commit per green chunk (push per operator instruction this
session); prod/main gated.

## 4. Acceptance

- 2A: registries load fail-loud w/ sweep + cross-refs; router routes science via
  the bank; zero episode change; caller-count + consumer-guard pins updated.
- 2B: 3 lane packs on disk at canonical coordinates, validated, addressable;
  run-intent on a non-runnable bank raises loud; science unchanged.
- 2C: dropdown in the real JSON, default science_news, validator green, selection
  threaded explicitly.

## 5. r1 judgment note

Accepted: codex M1-M4 + S1 + both cuts; antigravity M3/M4 observations, S1, its
legacy-pipeline cut. Rejected (with reason): antigravity's INPUT_TYPES silent
fallback list + "placeholder allowlist entries" + subset-relaxed allowlist test
(all no-fallback/allowlist-law violations). Verify-at-build: story_scaffold
append point; 2C selection threading.
