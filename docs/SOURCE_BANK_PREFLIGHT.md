# Source Bank Preflight

Run this checklist after the bank's design is locked, its code exists, and its
canonical integration is complete. Use `SOURCE_BANK_GUIDE.md` as the
normative brief.

## Acceptance protocol

Every hard item receives `PASS`, `FAIL`, or an explicitly allowed `N/A`,
plus evidence: a file and line, test name, validator output, ledger path, or
published asset path.

When executing the checklist, number items top-to-bottom within each gate
(`G1.1`, `G1.2`, and so on) and save an `ID | status | evidence` matrix.
The final receipt must name that matrix and its SHA-256.

- Any hard `FAIL` stops the release.
- `N/A` is legal only where this checklist says so.
- Warnings are recorded but do not change the verdict.
- "Probably", "looks right", and an unverified reviewer claim are not evidence.
- Finish with the receipt template at the end of this file.

## Gate 1 -- Independent design

For this gate, a material choice changes one fingerprint dimension;
normalization ignores only cosmetic names and IDs; and a near-copy preserves
the same non-contract instructions or logic under surface rewording. Save the
pre-comparison fingerprint and a six-dimension comparison matrix with paths
and hashes.

- [ ] **Hard:** The architecture was written down before existing bank
  implementations were studied. Its fingerprint records the source strategy,
  pass DAG and slot assignments, role/authority graph, artifact handoffs,
  retry/audit topology, and ledger-write strategy.
- [ ] **Hard:** At least three material design choices are traced directly to
  this bank's source constraints or intended listener experience.
- [ ] **Hard:** After design lock, discretionary structure was compared with
  registered banks after cosmetic names were normalized. Fail if the pass DAG
  matches an existing lane, four or more of the six fingerprint dimensions
  match one lane, or any non-contract prompt or implementation block was
  copied or near-copied.
- [ ] **Hard:** Similarity is limited to mandatory shared interfaces. There is
  no existing-bank runner import, bank-to-bank fallback, renamed role system,
  or "existing lane plus different prompts" architecture.
- [ ] **Hard:** The design names its orientation and closure mechanism, and a
  test or receipt identifies the resulting artifact or line IDs. Whether the
  result works artistically remains taste, not a fatal gate.
- [ ] **Hard:** The creative contract forbids guns, blood, violence, and
  swearing without using deterministic prose censorship as a substitute for a
  model repair.

Creative quality remains a taste decision, not a runtime validator.

## Gate 2 -- Source, access, evidence, and rights

- [ ] **Hard:** The source mode is explicit: no-source, local/package,
  operator-pinned `source_ref`, or automatic public/keyless selection.
- [ ] **Hard:** A no-source bank has an explicit bank-specific initialization
  path. It does not use empty `fetcher` plus empty `interpreter` and
  accidentally enter the reserved `original` architecture. **N/A**
  only for a source-backed bank.
- [ ] **Hard:** The bank introduces no API key, login, paid service, protected
  browser session, or endpoint that normally returns a paywall, CAPTCHA, or
  anti-bot challenge. Existing model credentials remain behind the supplied
  LLM slots.
- [ ] **Hard:** Runtime source I/O occurs only inside the declared fetch path,
  never at module import, pack discovery, or node `INPUT_TYPES` evaluation.
- [ ] **Hard:** Network code enforces timeouts, bounded retries, status and
  content-type checks, and response-size limits. Tests use deterministic
  fixtures; the live run records the real selected source. **N/A** only when
  the bank performs no network retrieval.
- [ ] **Hard:** An explicit source failure stops loudly. Any automatic
  next-candidate behavior is declared, bounded, and records the selected
  candidate. No unrelated source, synthetic text, or other bank is silently
  substituted.
- [ ] **Hard:** Fetched text is delimited as untrusted data, embedded
  instructions are ignored, and source claims are verified against data
  rather than model confidence.
- [ ] **Hard:** Every source-backed runner receives the writer's exact seven
  live `SOURCE_PAYLOAD_KEYS`, all strings, with non-empty `seed_text`;
  provenance and rights remain in sidecars. A lane-specific typed artifact is
  derived only after ingress. **N/A** only for a no-source bank or an
  intentional shared-ingress change with its own tests.
- [ ] **Hard:** The registered fetcher matches
  `fetch(*, bank, technical_model, source_ref="")` and returns `dict |
  SourceFetchResult`. A `legacy_many_pass` interpreter matches its live call
  signature and returns coherent brief attributes plus `model_dump()`.
  **N/A** only where the selected architecture does not use that component.
- [ ] **Hard:** The ledger preserves `meta.source_bank`,
  `meta.source_ref`, `meta.source_meta`, and `meta.source_rights` as
  applicable. Source identity, canonical URL, retrieval date, digest,
  author/outlet, license status/URL, and attribution are present when the
  source supplies them.
- [ ] **Hard:** Public accessibility is not treated as public domain or
  commercial clearance. Unknown or incompatible rights stop an adaptation.
- [ ] **Hard:** Every on-air quotation, claim, number, and proper noun
  presented as real or source-derived fact resolves to validated evidence.
  Fictional invention is distinguishable from source fact.
- [ ] **Hard:** Every evidence span satisfies the literal-identity contract:
  each quoted span equals `payload[field][start:end]` exactly, offsets are computed
  against the actual slice (never a normalized copy), span and evidence IDs are
  zero-padded to a fixed width (`F01`, not `F0`/`F00`), and no quote exceeds its
  payload field. Span-bearing source fields are whitespace-normalized AT ADMISSION,
  upstream of any offset computation, so a dirty-source (leading whitespace/tabs)
  cannot shift a literal span. **N/A** only for a no-source bank. This is the
  most-hit source-backed P0 (literal-span / offset-integrity).
- [ ] **Hard:** The exact selected body is preserved without silent post-fetch
  clipping and receipted with its route, raw RSS content index/count when
  applicable, character count, UTF-8 byte count, and SHA-256. If it exceeds one
  model context, overlapping windows cover the complete normalized body with no
  gaps; every local span is validated before exact rebasing, and the merged
  artifact validates against complete A0. **N/A** only for a no-source bank.
- [ ] **Hard:** A no-source bank stamps honest original/synthetic provenance
  and contains no fabricated citation, URL, author, or license. **N/A** for a
  source-backed bank.

## Gate 3 -- LLM slots, prompts, and authorship

- [ ] **Hard:** Every generation call uses only the supplied
  `creative_fn` or `technical_fn`. There is no model loader, inference
  backend import, direct model API, third slot, or new credential path.
- [ ] **Hard:** Each pass has a justified slot assignment. Creative writing
  and creative revision use the creative slot; extraction and structured
  verification use the technical slot.
- [ ] **Hard:** All model instructions live under the pack's
  `prompt_stages`. Every referenced seam exists.
- [ ] **Hard:** Each structured seam, worked example, typed schema, parser,
  and repair prompt agree exactly. Worked examples validate in a test.
- [ ] **Hard:** Every schema BOUND is model-visible and never tighter than its
  validator. Each seam restates the field's max length, exact count, and pattern:
  the automatic shape instruction emits required paths but NOT min/max/pattern, so
  a seam and schema can "agree exactly" on shape while the bound stays invisible and
  the model overruns it. No seam or projection caps a field below what its
  post-validator demands. Each model-authored value is requested AS a typed field,
  never as prose describing the value -- a value written in prose makes the model
  emit prose and drop the field. This is the `string_too_long` / wrong-count /
  `Field required` family; a bank must not pass preflight with the bound unstated.
- [ ] **Hard:** Every model-authored collection defines a concrete item model.
  No collection of things is typed as `list[dict[...]]`,
  `dict[str, Any]`, or `Any`. Identifier-keyed mappings are allowed.
  Item structure is pinned; descriptive category vocabulary remains open
  unless a closed enum is a real downstream contract.
- [ ] **Hard:** Every input-list-to-output-row transformation declares and
  tests exact ownership: one row per input item, one singular owned value per
  row, and complete downstream reference coverage. Base and repair prompts
  forbid numbered, secondary/tertiary, and suffixed pseudo-fields; Python
  validates the cross-artifact multiset and closure.
- [ ] **Hard:** Every canonical spoken line is traceable to an accepted model
  artifact. Parser extraction may remove declared serialization delimiters
  but does not rewrite the content field.
- [ ] **Hard:** Python creates only mechanical data such as IDs, order,
  references, enums, counts, hashes, and validated routing metadata. It does
  not create or alter spoken prose; mechanical serialization of already
  accepted verbatim rows is allowed.
- [ ] **Hard:** Invalid creative content returns to a model through a finite
  per-candidate repair ladder. A designated liveness route declares whether
  exhaustion retires that candidate and requests a fresh complete model-authored
  candidate until acceptance or operator cancellation. Such a route has no
  fixed outer model-output ceiling and no canned story fallback. Deterministic
  configuration, source/security, provider, I/O, compiler, ownership, graph,
  freeze, and proof failures remain loud.
- [ ] **Hard:** Every rejection names the item, evidence, defect, and allowed
  correction. A model reviewer cannot create a fatal finding that
  deterministic code cannot corroborate.
- [ ] **Hard:** Base, structural-retry, and typed-repair prompts fit the
  resolved context cap of their actual slot/model. Provenance-sensitive calls
  fail loudly rather than truncate.
- [ ] **Hard:** Output reservations scale from the artifact's real size driver
  such as line count or evidence count, not only `target_words`.
- [ ] **Hard:** Models are never asked to calculate, report, or enforce exact
  word, line, item, or coverage counts. Python measures them
  deterministically and supplies measured defects to any creative repair.
  No model-produced or unused count field can gate production.
- [ ] **Hard:** Creative randomness uses OS entropy. Reproducibility comes
  only from the existing explicit seed overrides; the bank does not plant a
  fixed seed.
- [ ] **Hard:** `target_words` is advisory and recorded. It does not trigger
  deterministic trim, padding, culling, rewriting, or a fatal quota gate.

## Gate 4 -- Ledger closure and delivery

- [ ] **Hard:** The provided `Ledger` is the only ledger. The root
  `cast`, `lines`, `beats`, `scenes`, `shots`, `music`, and
  `clips` values are present lists, never null.
- [ ] **Hard:** The bank populates non-empty cast, scenes, shots, beats, and
  lines. An empty music list is schema-legal but is not treated as a request
  for silence by the current canonical theme node. A music-free design has
  explicit supported behavior and tests. Announcer-only behavior uses only
  the live explicit contract.
- [ ] **Hard:** IDs are non-empty and unique within their tables.
- [ ] **Hard:** A bank-owned graph test proves scene-owned shot -> scene,
  scene-owned beat -> shot and matching scene, beat.line_ids -> lines, voiced
  scene line -> exactly one beat and shot, and character line/beat -> cast.
  Declared bookend, frame, or music sentinels outside that graph have focused
  tests.
- [ ] **Hard:** Every optional `music.anchor_line_id` resolves to a real
  line when used.
- [ ] **Hard:** Line and beat speaker identities agree. Every character
  `char_id` resolves to cast; announcer identity follows the declared live
  convention.
- [ ] **Hard:** Every `speaker_role` is one of `character`,
  `announcer`, `music_open`, `music_close`, or `music_inter`.
- [ ] **Hard:** Every non-skipped voiced row has non-empty canonical text.
  Music sentinels may have empty text without being skipped. Every row marked
  `skip=True` has empty text and a non-empty `tts_skip_reason`. Spoken
  text has no speaker label, stage direction, or whole-line quotation wrapper.
- [ ] **Hard:** Every voiced line `boundary` is `shot_start`,
  `beat_start`, or `continue`, and agrees with the actual shot/beat
  transition.
- [ ] **Hard:** Counts and hashes are stamped from final canonical content,
  after all accepted model repairs.
- [ ] **Hard:** The ledger is assembled once from accepted artifacts. Rejected
  candidate prose, hashes, seals, and readiness state never enter ledger
  metadata. Final line/hash proofs establish the accepted ledger's identity and
  downstream integrity, not fidelity to an abandoned fictional draft.
- [ ] **Hard:** Evidence maps and authorship receipts live in typed artifacts
  or namespaced `meta`; the fixed line schema contains no ad hoc provenance
  fields.
- [ ] **Hard:** The pack deliberately selects its live freeze policy: non-empty
  `line_composer_system` means `legacy_full`; absence means
  `content_owned_readonly`. A test proves the expected policy.
- [ ] **Hard:** A content-owned runner assigns valid character `tts_model`
  and `voice_preset` values that satisfy the declared reuse policy
  (unique when reuse is disabled). Its proof survives while the shared writer
  tail stamps fresh `text_for_tts` and canonical-text source hashes. A
  legacy lane proves the shared CastLock/readiness path owns delivery.
- [ ] **Hard:** The lane-defined return object exposes `outline_view.title`
  and `outline_view.premise`; an EpisodeCanon-compatible `canon` with
  title, premise, setting, time of day, and sound palette;
  `final_title_override`; and `run_story_spine`. An optional
  `tail_finalizer` implements `before_save` and `after_save`.
- [ ] **Hard:** The bank-owned closure test passes independently of the freeze
  audit. Freeze warnings alone are not proof of full
  scene/shot/beat/line/cast closure.

## Gate 5 -- Registry, runner, and canonical wiring

- [ ] **Hard:** The pack is duplicate-key-safe JSON at
  `nodes/story_packs/<source_bank_id>/<story_model_id>.json`, uses the live
  schema version, and its header coordinates match its path.
- [ ] **Hard:** Every new or edited JSON -- the bank and pipeline rows, the pack
  `<story_model_id>.json`, and `story_rules/<source_bank_id>.json` -- is UTF-8 with
  NO byte-order mark. A leading BOM survives the duplicate-key, round-trip, and
  validator checks but breaks downstream string matching, so a `head -c3` no-BOM
  check is explicit (matches the Teardown gate).
- [ ] **Hard:** The exact bank row and pipeline row schemas validate. Defaults,
  declared seams, required seams, pass slots, and cross-references resolve.
- [ ] **Hard:** Custom seams live in pipeline `declared_seams` and pass rows
  and are supplied by the pack. Bank `required_seams` contains only live
  shared production seams.
- [ ] **Hard:** Every required fetcher and interpreter ID is registered.
  **N/A** only for a valid no-source or independent-runner contract that
  deliberately declares neither.
- [ ] **Hard:** The execution runner exists and is registered explicitly in
  `_otr_lane_specs.LANE_SPECS` (by MODULE + ATTRIBUTE NAME, resolved lazily),
  with that lane's request-compatibility policy DECLARED -- `compat_attr=""`
  means "accepts any request", stated on purpose, never left blank by
  omission. No plugin-style discovery or fallback is assumed.
- [ ] **Hard:** `runnable=true` lands only with the runnable lane. A custom
  non-source-contract pipeline has `executable=true` in the same change.
- [ ] **Hard:** `resolve_story_pack` and `require_runnable_bank` succeed
  for the new coordinates, and an unknown or disabled coordinate fails loud.
- [ ] **Hard:** The existing canonical `source_bank` selector reaches the
  new bank. There is no copied, generated, or parallel workflow.
- [ ] **Hard:** If no node, widget, input, link, or default changed, tests
  prove registry-driven selection and the canonical workflow remains
  unchanged, including the shipped `science_news` default. If any did
  change, `workflows/otr_canonical.json` changed in the same commit.
- [ ] **Hard:** Any canonical JSON change passes
  `OTR_WorkflowValidator`, JSON round-trip, link referential integrity,
  wired-input-name, and live `INPUT_TYPES`/widget-count audits. Optional
  widgets were appended, not inserted.
- [ ] **Hard:** Importing routing, pack, source, and runner modules performs no
  network request, model load, GPU allocation, or unrelated file mutation.

## Gate 6 -- Gates, tests, and live proof

- [ ] **Hard:** Every fatal gate is objectively checkable, repairable by the
  responsible component, and a real contract defect. Taste, pacing, register,
  and warnings remain notes.
- [ ] **Hard:** Validators are role-aware. They do not reject a role for
  following its declared authority.
- [ ] **Hard:** Cast-identity guards distinguish generic role labels from
  personal names through the shared bounded classifier. A newly introduced
  generic cast label either fits the declared role grammar or extends that
  grammar and its all-bank regression matrix in the same change.
- [ ] **Hard:** Story-brief input anonymization and output name validation use
  complementary projections of the same cast classification: articles and
  honorifics never become identities, generic roles remain legal visual nouns,
  and meaningful personal-name components remain protected.
- [ ] **Hard:** Tests cover registry loading, pack/seam schema parity, source
  success and failure, rights/provenance, prompt fit, retry exhaustion,
  authorship, ledger graph closure, tail handoff, import safety, and
  no-fallback behavior.
- [ ] **Hard:** After every code change, the full Windows regression suite and
  Bug Bible regression pass as required by `AGENTS.md` and `CLAUDE.md`.
- [ ] **Hard:** The final canonical workflow validation and link/widget audit
  pass against the live node definitions.
- [ ] **Hard:** Before the live run, the machine is reset using the selective
  process and port procedure in `AGENTS.md`; no blanket Python kill is used.
- [ ] **Hard:** A live 30-word run loads
  `workflows/otr_canonical.json`, selects the new bank, exercises its real
  source policy and real two-slot path, and reaches the shared writer tail.
- [ ] **Hard:** Before long-form qualification, 30-word canonical smokes pass
  with at least two materially different local LLM families and one configured
  frontier/cloud creative lane, while the technical slot remains independently
  exercised. Record each concrete model label and prompt ID.
- [ ] **Hard:** After those 30-word smokes pass, repeat the same model pairings
  at 120 words and save the ledger and published-asset receipt for each. Do not
  begin a 720-word qualification or bakeoff until every 120-word leg is green.
- [ ] **Hard:** The saved ledger passes the lane-owned closure proof and shared
  freeze path with no hard errors. Its source, rights, slot-call, authorship,
  and word-count receipts are present.
- [ ] **Hard:** The final episode is published to `otr/obs`, and the exact
  asset path exists on disk. A resident server or VRAM allocation is not proof
  of completion.
- [ ] **Hard:** No deliverable remains in a temporary directory, and no
  temporary probe or generated workflow is included in the change.

If an automated review panel is used, follow the current `AGENTS.md` and
`CLAUDE.md` rules. Ground every finding in live code. Review may find
implementation defects after the design lock; it may not redesign the bank's
creative premise by consensus.

## Final receipt

```text
SOURCE BANK PREFLIGHT: PASS | FAIL

bank:
story_model:
story_pipeline:
design_fingerprint:
design_fingerprint_path_sha256:
comparison_matrix_path_sha256:
completed_check_matrix_path_sha256:
target_derived_choices:
source_mode:
source_and_rights_evidence:
llm_slot_evidence:
authorship_evidence:
ledger_graph_test:
tail_handoff_test:
registry_and_canonical_evidence:
full_suite:
bug_bible:
workflow_validator:
live_ledger:
published_asset:
hard_failures:
warnings:
```

`PASS` means `hard_failures: 0` and every hard item above has concrete
evidence.

## Teardown protocol -- removing a bank (the inverse of Gate 5)

Removing a bank is a coder-window change with the same rigor as adding one. A bank lives in
~10 wired surfaces, not just `banks.json` -- rediscovering them by hand each time is the failure
this section prevents. Playbook proven by the `499386aa` roster trim and the 2026-07-18 4-bank rip
(`docs/2026-07-18-rip-4-banks-plan.md`).

**CLEAN RIP (operator law).** The removed bank leaves ZERO footprint: no half-rip items, NO
negative/absence tests (nothing asserts a ripped id is "gone"/"unknown"/"not runnable"), and NO
"retired-variant coverage" migrated to a survivor to keep the ripped thing alive. If a test's SUBJECT
is the ripped bank, DELETE the case; tests reference ONLY surviving banks; roster/bijection tests assert
the survivors POSITIVELY (what exists), never the absence of what was removed. The final grep for the
ripped id across `nodes`/`tests`/`workflows` returns nothing -- test bodies included.

**Step 0 -- decide the removal DEPTH (this drives everything below):**
- **Variant removal** (a base or sibling version of the same lane SURVIVES, e.g. the 2026-07-19 rip
  of base `scifi_codex` while its shared runner + the renamed `scifi_news` sibling stayed): remove only the bank's OWN row/pack/rules + its
  DEDICATED pipeline. KEEP the shared lane runner module and any shared pipeline.
- **Full-family removal** (NO surviving sibling of that lane, e.g. `scifi_sonnet_v3` is the only sonnet
  bank): ALSO delete the lane runner module + its interpreter/source-kind registration + the dedicated
  lane test. "Only version of its family" is the tell that a rip goes deep.

**Surfaces to clean (each: PASS + a file:line / test / grep evidence):**
- [ ] **Hard:** Bank row deleted from `nodes/story_packs/banks.json`.
- [ ] **Hard:** Pack dir `nodes/story_packs/<id>/` and `nodes/story_rules/<id>.json` deleted.
- [ ] **Hard:** Pipeline removed from BOTH registries when dedicated -- `LANE_SPECS` in
  `nodes/_otr_lane_specs.py` AND the JSON catalog `nodes/story_packs/pipelines.json`
  (**the easy-to-miss one** -- a retired pipeline left in the JSON is a semantic registry failure even
  with no bank pointing at it). KEEP any pipeline a surviving bank still uses (e.g. `legacy_many_pass_adapt`).
- [ ] **Hard:** Runner + routes -- delete the bank's `LANE_SPECS` entry and any
  `if base == "<family>":` route. **Full-family only:** delete the lane module `nodes/_otr_<family>.py`
  and its `validate_source_payload("<family>")` / interpreter registration; grep the family id across
  `nodes/` and clean every orphaned import.
- [ ] **Hard:** Registry consistency -- no retired `story_pipeline_id` remains in
  `_otr_story_routing._ensure_loaded().pipelines`; `runnable`(bank) and `executable`(pipeline) stay in
  sync or `_otr_story_routing` raises `RegistryValidationError`.
- [ ] **Hard:** Tests UPDATED, not just deleted -- the roster/bijection test (`tests/test_bank_variants.py`
  counts + id lists) reflects the new runnable roster; guard tests that enumerate banks via `_CURRENT_BANKS`/
  inline lists (`test_placeholder_guard_v4`, `test_scene_guard_v4`, `test_provenance_v4`,
  `test_genre_guard_spoken_v4`, `test_outro_guard_v4`, `test_source_snapshot`, ...) regenerate their lists
  from the surviving roster or pin the exact ids. Do NOT trust a hand list -- `grep _CURRENT_BANKS` (+ the
  ripped ids) across `tests/` to find EVERY such list (2026-07-18: two guard tests were missed on the first
  hand-enumeration). Advisory/positive tests that DRIVE a ripped lane string: DELETE if the subject is the
  ripped bank, MIGRATE to a surviving lane if they test surviving machinery. A dedicated lane test is
  deleted on full-family removal. **Imported AND used:** on full-family removal a surviving test may import
  a symbol FROM the deleted module AND use it downstream (an assertion/fixture, not just the `import` line);
  deleting only the import then NameErrors at COLLECTION (whole file fails, not one case). Grep the deleted
  module name across `tests/` and, per hit, delete the USAGE too -- or re-source the symbol from a surviving
  module if the surviving coverage genuinely needs it -- never just the import line (2026-07-18:
  `test_scifi_source_repair.py` imported a sonnet schema symbol at :5 and used it at :117).
- [ ] **Hard:** Ledger discipline -- if the removed bank carried a repeatable LIVE production failure,
  RECORD a PBUG in `docs/PROD_BUG_LOG.md` with fix = "retired the runnable bank + its pipeline/route" and
  mark any open NEWBUG doc CLOSED-BY-RIP. Never delete the only causal record; a rip is a legitimate fix,
  a hole in the ledger is not.
- [ ] **Hard:** Canonical -- `workflows/otr_canonical.json` stays byte-unchanged (bank removal is
  registry-driven; it must not touch the graph). If it did change, that is a red flag to investigate.

**Gate (identical to adding + the 2026-07-18 QA hardening):**
- [ ] **Import-smoke (Bible 03.01/03.02):** on a full-family removal, after deleting the lane module,
  LOAD the node registry clean ("All N nodes loaded, 0 skips") and grep the REPO-ROOT `__init__.py` for a
  leftover key -- that is the real loader surface (`NODE_CLASS_MAPPINGS` lives at `__init__.py:116` +
  `:351-363`), NOT `nodes/__init__.py` (corrected 2026-07-18). A string grep proves the ids are gone, NOT
  that the pack still imports.
- [ ] **Ledger-ownership (CLAUDE.md "no hole in the ledger"; PBUG-20260712-05):** enumerate every ledger
  field each removed bank stamped -- including COMPUTED keys (`f"{source_bank_id}_..."`) a literal grep
  misses -- and confirm zero surviving readers in the shared writer tail. A green suite does not prove
  ledger completeness.
- [ ] **No dead levers (GO_FORWARD item 5):** every KEPT runner/helper still has a live caller post-rip;
  excise any that went dead (a shared factory used only by the removed pipelines dies with them).
- [ ] Full Windows suite + Bug Bible GREEN -- gate on GREEN **plus retired-id absence**. Scan SOURCE-ONLY
  (exclude `__pycache__` -- stale `.pyc` false-fail) with `Select-String` (the builder runs PowerShell; no
  `grep`). Grep the **BARE** token, NOT the quoted `"<id>"` -- a quoted grep MISSES prose/comment tokens
  (2026-07-18: a bare `scifi_sonnet` in a surviving `_otr_scifi_p0_contract.py` comment slipped a
  quoted-only scan and falsified an "only one ref" invariant). For a FULL-FAMILY removal the family token
  can legitimately SURVIVE in kept code (advisory branches, forensic/PBUG comments) -- ENUMERATE those
  carve-outs and gate on EXACTLY-N surviving hits, never a blind zero. **Record suite/Bible counts as
  evidence, never pin them** (a hardcoded count false-fails or masks a real drop).
- [ ] `OTR_WorkflowValidator` + JSON round-trip **+ a no-BOM/UTF-8 check on EVERY touched text file** (not
  just the JSONs -- edited `.md`/`.py` too; PowerShell `[System.IO.File]::ReadAllBytes($p)[0..2]` must not
  equal `239 187 191`, since the builder has no `head`) (Bible 02.11/12/13); `workflows/otr_canonical.json` byte-unchanged (registry-driven --
  a change there is a red flag; verify it strands no COMBO id, BUG-08.06/12.23); commit + push;
  `HEAD == origin`; AST-parse touched `.py`.
