# Vibe-Coder Extensibility -- Coding Plan (r4-CONVERGED)

> **STATUS 2026-07-15 (baseline): SUPERSEDED -- do not execute (banner below still governs).**
> Replacement update: `docs/2026-07-12-user-source-lanes-architecture.md` has since completed
> its kibitz arc r1-r4 (r3 folded @ 1af3d2bc, r4 @ 34fd5c18) but is NOT converged -- one r5
> confirmation pass + operator ratification of its section 16 (nine flags) are required before
> any coder slot. Its estimate is ~21-31 coder-days, superseding the "4-7 days" that
> GO_FORWARD_PLAN formerly attached to this retired plan.

> **SUPERSEDED FOR SCOPE (operator correction, 2026-07-12 late).** This plan's
> "content packs only -- NO user-created lanes" ruling is RETIRED: the operator
> requires real user source lanes (safe feed variants + original lane plug-ins).
> The replacement architecture is `docs/2026-07-12-user-source-lanes-architecture.md`
> (DRAFT, awaiting architecture approval + its own kibitz arc). Do NOT execute this
> plan's waves as scoped; its carried-forward pieces are enumerated in the
> replacement's §13. The GO_FORWARD_PLAN queue entry pointing here is stale until the
> replacement converges.

- **Date:** 2026-07-12. **Parent:** `docs/2026-07-12-vibe-coder-extensibility-r1.md`
  (@ 852209bf). **Revision:** full kibitz arc r2+r3+r4 folded (codex @ gpt-5.6-sol,
  antigravity @ gemini-3.5-pro, Claude anchor+judge; judgment logs under
  `kibitz-runs/2026-07-12-vibe-coder-extensibility/{r2,r3,r4}/final.md`). r4 verdicts:
  both panelists yes-with-fixes, all fixes folded below -- ARC CONVERGED, plan is
  code-ready. NO code exists yet.

## ACTIVATION GATE (pre-W0 -- r4)
Before ANY W0 file is touched: (1) this build is scheduled in
`docs/GO_FORWARD_PLAN.md` (appended section exists; coder re-prioritizes it into the
queue); (2) the sole coder slot is claimed per the one-window law; (3) clean-or-released
ownership receipts exist for EVERY touched surface -- `nodes/OTR_LedgerScriptWriter.py`,
`nodes/_otr_story_routing.py`, `nodes/_otr_story_pack.py`, `nodes/_otr_visual_styles.py`,
`nodes/_otr_model_catalog.py` (W3), `README.md`, `workflows/otr_canonical.json`. The
bakeoff code-freeze and the dynamic_story/randomizer builds share these surfaces --
sequencing is the coder's call in GO_FORWARD_PLAN, not this doc's.
- **Operator rulings (2026-07-12):** (1) NO TIERS -- two KINDS of addition: a **content
  pack** (story pack inside ANY registered RUNNABLE lane; the vibe path, equal across
  lanes) vs a **new lane** (fetcher/interpreter/runner + full independent-design
  preflight; expert path, out of scope). Gate 1 governs new lanes; content packs are not
  new banks. (2) `user_packs/` overlay ratified. (3) No blessed lane -- everything works
  uniformly across all runnable lanes.

## W0 -- user_packs overlay foundation

- New `nodes/_otr_user_packs.py`: single source for overlay roots. Root =
  `Path(__file__).resolve().parents[1] / "user_packs"` (repo root); surface roots
  `user_packs/visual_styles/`, `user_packs/story_packs/<bank_id>/`. Stdlib only; ZERO
  I/O at import. `.exists()`/`.is_dir()` guards -- an absent overlay tree is a normal
  no-op, never a FileNotFoundError.
- **Junction stance (r4, deliberate):** the `user_packs/` ROOT may be a junction/symlink
  resolving OUTSIDE the repo -- that is the operator's sanctioned survival mechanism
  against Manager reinstalls (better than a backup warning). Per-ENTRY containment is
  enforced against the RESOLVED root: any entry whose resolved path escapes the resolved
  root is quarantined loud. (r3's root-must-be-in-repo rule is superseded.)
- `.gitignore` += `/user_packs/` (anchored -- not any-depth). Docs: recommend the
  external-junction setup for update survival; W5 includes an in-place-update
  preservation test, and destructive-reinstall survival via junction is recorded as the
  operator-approved answer to acceptance criterion 6.
- `_otr_visual_styles`: lazy sweep merges BOTH roots. Identical schema/lint for overlay
  packs; collisions follow the quarantine taxonomy below. `_clear_caches()` (exists,
  :393) clears any new state.
- `_otr_story_routing` -- the heart of W0:
  1. **PackRecord map.** Discovery builds an immutable
     `(bank_id, story_model_id) -> PackRecord(path, pack)` map covering BOTH roots;
     `_Registry` (today only banks+pipelines, :481) gains it; `resolve_story_pack` and
     all listing resolve EXCLUSIVELY through the map (kills the `_pack_path`
     tracked-root reconstruction, :517-530). Deterministic ordering; overlay sweep
     mirrors `_sweep_and_crossref` laws (:350-378): subdir must be a registered bank,
     top-level files forbidden (overlay can NEVER add bank rows / pipelines -- the
     no-tiers ruling made structural), header triple must match path coordinates. On the
     OVERLAY side, layout violations (stray top-level file, unregistered-bank subdir,
     sidecar-named file) QUARANTINE loud instead of raising (r4: a stray file must not
     crash boot); tracked-root layout violations keep today's hard fail.
  2. **Overlay only under RUNNABLE banks.** `custom_source_bank` is registered but
     runnable:false -- overlay dirs under non-runnable banks are quarantined loud
     (unreachable packs must not exist, let alone be advertised).
  3. **Per-pack parity laws (today default-only, :387-425) extend to EVERY pack in the
     map, both roots:** `pack.story_pipeline_id == bank.default_story_pipeline`;
     `bank.required_seams` present in `pack.prompt_stages`; pipeline pass `seam_refs`
     present; three-way custom-seam parity (pack keys / declared_seams / pass_refs).
     A selectable pack is a VALIDATED pack -- selection can never outrun validation.
  4. `_clear_caches()` (exists, :547) additionally resets the PackRecord map.
- `nodes/story_rules/`: NO overlay (lane infrastructure; stem = source_bank_id;
  runnable-bank coverage law `:274-280` unchanged). Content packs never ship rules.
- **Overlay quarantine taxonomy (r3 contract + r4 collision semantics -- the full law):**
  INPUT_TYPES triggers the lazy sweep, so NOTHING overlay-side ever raises at boot.
  Every overlay-side failure (schema, parity, user-content policy, layout, sidecar-name,
  symlink escape, collision) QUARANTINES the offending file: loud console error naming
  file + reason, absent from every choices list, and its `ValidationIssue` is stored in
  a quarantine map SEPARATE from the registry so run-time resolution and `otr_check`
  emit the IDENTICAL structured issue (a bare missing-key error is not acceptable).
  Collision semantics (r4 tombstones, substitution-proof):
  - **Overlay claims a SHIPPED id (protected-id case):** the overlay file is quarantined
    with a protected-id issue ("rename your pack id"); the shipped entry remains
    authoritative and selectable. No substitution occurs -- the shipped id was never
    legitimately claimable, and production lanes can never be disabled by a user drop.
  - **Overlay-vs-overlay duplicate:** BOTH files are quarantined and the coordinate is
    TOMBSTONED -- excluded from choices, and resolution of that coordinate RAISES the
    stored collision issue rather than returning anything (no order-dependent winner,
    no silent pick).
  TRACKED-root failures keep today's hard fail (a broken shipped pack IS a build error).
  Nothing silently succeeds; nothing substitutes.
- **Sidecar law (r3):** `_PACK_SIDECAR_FILENAMES_BY_BANK` exemptions (`:42`, applied
  `:333`) do NOT apply under `user_packs/` -- overlay sidecar-named files are
  quarantined loud (they would otherwise be unreachable, unvalidated content).
- The published PackRecord map is wrapped in `MappingProxyType`, and PackRecord exposes
  DEEP-IMMUTABLE views of the pack payload (r4: `prompt_stages` via mapping proxy, list
  fields as tuples) so executed content cannot diverge from the stamped digest.
- Restart contract unchanged (overlay swept inside the same lazy singletons).
- Tests: overlay style+pack visibility after `_clear_caches()`; protected-id collision
  (shipped survives + overlay quarantined, styles AND packs); overlay-vs-overlay
  tombstone (choices exclude it; resolve raises the stored issue); overlay layout
  violations quarantine (boot survives); non-runnable-bank overlay quarantined;
  symlink-entry escape quarantined; junctioned ROOT accepted with entries contained;
  absent user_packs tree = clean load; per-pack parity failures loud for overlay AND
  tracked secondary packs; quarantine-issue identity across boot log / resolve /
  otr_check; import-isolation.

## W1 -- pack-selection surface (the ONE canonical-JSON change)

Today all three writer call sites resolve the bank DEFAULT (`:1799/:1837/:3741`) and the
lane runner dispatches by `bank.default_story_pipeline` (`:3729-3742`) -- a second pack in
a lane is unreachable. Fix:

- New routing API `list_story_pack_choices()` -> `["(bank default)"] + sorted(
  "<bank_id>/<story_model_id>")` over PackRecords of RUNNABLE banks only. The writer
  formats nothing itself.
- New optional COMBO widget `story_pack`, appended at the very END of the writer's
  optional block (order is load-bearing, `:2362-2366`). Default `"(bank default)"`.
- `run()` gains the parameter `story_pack: str = "(bank default)"` in its SIGNATURE
  (r4: the widget without the parameter is a TypeError at execute) and threads it
  through `_resolve_inputs`; the locked contract audit covers both. Falsy values
  (`None`/`""` from API workflows) coerce to the sentinel -- the writer's own
  `source_bank or "science_news"` precedent (`:1468`).
- `run()` resolves ONCE, immediately after `require_runnable_bank`: sentinel/falsy ->
  `resolve_story_pack(bank_id)` (byte-identical today); else guard `"/" in value` (a
  malformed value fails loud with `story_pack value must be <bank_id>/<story_model_id>`),
  split on the FIRST `/`, hard error if the pair's bank != selected `source_bank` (no
  cross-bank fallback), then resolve via the PackRecord map. Pipeline dispatch stays
  `bank.default_story_pipeline`, which W0's parity law makes provably equal to the
  selected pack's pipeline.
- **CONSUMER INVENTORY -- the r3 headline (both agents independently; code-verified).**
  The r2 "three consumers" list was incomplete: legacy-lane prompts resolve through
  `_otr_creative_prompt_router.resolve_creative_system_prompt`, which re-resolves the
  bank DEFAULT for every pack-routed seam (`:206`), with callers in `_otr_outline`
  (`:1850-1884`) and `_otr_line_composer` (`:2109/:3291/:3602/:3736`); and
  `_otr_freeze_cascade.resolve_freeze_policy` re-resolves the default and branches
  freeze policy on ITS `line_composer_system` seam (`:316-331`). Two-channel threading
  law:
  1. **run()-scope -- ONE contract, object threading (r4 unification):** the selected
     `PackRecord` is passed EXPLICITLY to the original interpreter factory (`:1799`),
     the QA gate (`:1837`), the lane runner (`:3741`), AND down the prompt-router
     chain -- `resolve_creative_system_prompt`, `generate_outline`, `compose_line`, and
     the announcer/coda helpers gain `pack: StoryPack | None = None` (None = resolve
     the bank default = byte-identical for every pre-W1 caller; a passed object is used
     as-is, never re-resolved). The r3 draft's `story_model_id` parameter variant is
     superseded -- ID-based resolution exists ONLY on the ledger channel below. No
     production code re-resolves by bank alone.
  2. **Ledger-scope** (freeze cascade, reroll paths): resolve via the W1 stamps --
     `resolve_story_pack(bank, stamped_story_model_id)` PLUS the r4 replay-integrity
     check: the resolved `record.sha256` MUST equal `meta["story_pack_sha256"]`;
     mismatch fails loud (an overlay edited between runs must never silently execute
     different prompts under an old ledger). Inside `resolve_freeze_policy` the
     mismatch/unresolvable cases take the existing non-raising `terminal_error` receipt
     path (`:302-305` contract). An absent stamp (pre-W1 ledger) = today's behavior.
     Parity test: untagged AND unstamped ledgers behave byte-identically to today;
     stamped-hash-mismatch test pins the loud failure.
  A thread-local "active pack" was considered and REJECTED (hidden global state).
  Enforcement: grep-test pins `resolve_story_pack(` to exactly ONE writer call site;
  integration fixture selects a secondary pack whose `line_composer_system` presence
  differs from the default and proves composer prompts AND freeze policy follow the
  SELECTED pack; refinement re-entry test (`:3232-3400` reconstructs run() args) proves
  `story_pack` survives every re-entry.
- **Stamps (verified absent today, `:3644-3665`):** `meta["story_model_id"]` +
  `meta["story_pack_sha256"]`, stamped with the Stage-2C/3C block before the skeleton
  save. TOCTOU law: pack bytes are read ONCE at discovery -- decode/parse/validate THAT
  payload, hash THOSE bytes, digest immutable in `PackRecord.sha256`; W1 stamps
  `record.sha256` and never re-reads `record.path`.
- **Canonical JSON (exact form, 3-way converged):** `gate_in` is a force-input with NO
  widget slot (`:2996-3008`; node 1 today: 34 widget-inputs + gate_in = 35 inputs, link
  279 -> slot 34, verified by parse). `story_pack` is declared AFTER `gate_in` in
  INPUT_TYPES: it becomes input slot 35 with `{"widget":{"name":"story_pack"}}`,
  `widgets_values[34]` = `"(bank default)"`, link 279 UNTOUCHED. If lean-mean-rip lands
  first (node-1 input 9 removed, link 279 dst 34->33), the SAME append-at-live-END law
  applies -- every count/slot is RE-DERIVED from the live JSON at build, never
  hardcoded. Same-change: OTR_WorkflowValidator + round-trip + link referential
  integrity + widget-count vs live INPUT_TYPES audits, and the locked optional-set
  contract audit (`:6998+`) gains `story_pack`.
- Tests: `test_story_pack_widget.py` (choices shape incl. runnable-only; sentinel
  default; no-slash fail; cross-bank fail; default-path byte-parity; explicit-pack
  resolution), single-resolve grep test, consumer-threading integration fixture,
  re-entry survival, ledger-channel parity, canonical guardrail update, stamp presence
  + hash correctness.

## W2 -- `otr_check.py`: one validator, two entry points

- **Factored pure seams first (the loaders and the CLI call the SAME functions):**
  `validate_style_file(path)` in `_otr_visual_styles`;
  `validate_story_pack_file(path, bank, pipelines)` in `_otr_story_routing`. Runtime
  loading keeps fail-fast semantics by raising the first issue; the CLI runs two-phase:
  per-file validation collecting ONE structured issue per file, then the global phase
  (duplicates, defaults, cross-refs) over accumulated records.
- **Structured diagnostics:** `ValidationIssue(code, path, field, message, fix)`
  attached to the shared validation exceptions (today they are prose-only). Plain-text
  AND `--json` render the same issues; exit codes: 0 clean / 1 validation issues /
  2 internal error. String-parsing exceptions is forbidden.
- **CLI grammar (canonical, r4-complete):** `otr_check.py style <path|id>... |
  pack <path|bank_id/story_model_id>... | model <id> | all [--json] |
  emit-schema-doc [--out docs/EXTENDING_OTR.md]`. Bare story-model ids are REJECTED
  (bank-scoped; no searching, no guessing). `--receipt <out>` is legal ONLY with exactly
  one `pack` target -- rejected for multiple packs, `style`, `model`, `all`, and
  `emit-schema-doc` (a receipt describes one pack + one SHA-256, cardinality is part of
  the contract). `emit-schema-doc` rejects `--json`/`--receipt`; exit codes shared:
  0 clean / 1 validation issues (or doc drift) / 2 internal error. `all` scans both
  roots of both surfaces + the model catalog and REPORTS quarantined files with their
  stored issues. Registry caches are cleared at invocation start. Import discipline =
  the test suite's pattern (OTR_TEST_MODE + direct module import; never through
  ComfyUI-heavy `nodes/__init__`). Model-catalog failures get a ValidationIssue ADAPTER
  at the failure site (`UnknownModelError` is prose-only today, `:1129-1228`) -- never
  string parsing.
- **SFW/user-content policy IN the production validator (parity law):** new
  `nodes/_otr_user_content_policy.py` -- stdlib, whole-word matching. **Scanned field
  paths (r4, exact -- verified against the live `StoryPack` schema `:75-90`):**
  `prompt_stages.*` values, `label`, `status`, `examples[*]`, `tone_guardrails[*]`,
  `source_requirements[*]`, `ledger_validation_notes[*]`. **EXEMPT (self-trigger
  guard):** `forbidden_plot_patterns[*]` and `forbidden_leakage_terms[*]` -- declaration
  fields that legitimately NAME banned content in order to ban it. (The r3 draft's
  "defaults" was not a pack field; bank-row defaults are repo-owned and out of
  user-content scope.) Called by the production pack validator, so load-fail ==
  check-fail with the same exception (overlay-side violations QUARANTINE per W0;
  tracked-side hard-fail); styles keep their own forbidden-terms lint. **Dependency direction (r3):** `_otr_original_radio.py` imports pydantic at
  module load (`:43`), so the policy module OWNS the curated literals and
  original_radio/stage3 consume FROM it (leaf import) -- or pin a copied subset with a
  drift test. Never policy -> original_radio. Seed vocabulary contract:
  `DEFAULT_PROFANITY_TERMS` (`_otr_stage3_validators.py:140`) + a curated
  violence/weapons subset from `FORBIDDEN_TERMS` (`_otr_original_radio.py:88`)
  EXCLUDING lane-specific anachronism/source-framing terms. (Binding to
  `_ALL_FORBIDDEN` rejected -- 1940s anachronism terms would false-fail modern lanes.)
- **Authoring receipt for content packs = its own gate-ID contract** (NOT the bank
  preflight's positional G-numbers). Stable IDs enumerated NOW (r4): CP-1 pack/schema/
  prompt validity; CP-2 selected-lane compatibility (parity laws); CP-3 user-content
  policy; CP-4 full regression suite + Bug Bible; CP-5 canonical 30-word smoke; CP-6
  ledger closure; CP-7 published asset. The CLI asserts and stamps CP-1..CP-3
  (authoring-validated); CP-4..CP-7 are listed UNRESOLVED with their IDs -- the CLI can
  NEVER emit them as passed (two-state law; the UNRESOLVED listing is load-bearing and
  stays -- it is what stops a partial receipt masquerading as qualification). Receipt
  carries SHA-256 of the pack, `banks.json`, `pipelines.json`, the lane's story_rules
  pack, plus schema + checker versions and verified-file/seam counts.
- **Receipt location (r3, convergent):** default and only sanctioned output dir =
  `user_packs/receipts/` (invisible to both surface sweeps by construction). The CLI
  REJECTS any `--receipt` path whose resolved location falls under
  `user_packs/story_packs/` or `user_packs/visual_styles/` (same containment guard as
  the overlay) -- a receipt beside a pack would be swept as a pack on restart. No sweep
  exemptions for receipt-looking names: strays in swept dirs stay loud.
- `scripts/otr_check.bat` resolution order (r4-final): `OTR_PYTHON` env override ->
  `%~dp0..\.venv\Scripts\python.exe` (repo-local venv, if one exists) ->
  `%~dp0..\..\..\.venv\Scripts\python.exe` (the ComfyUI-root venv -- this box's real
  interpreter) -> `%~dp0..\..\..\..\python_embeded\python.exe` (standard portable
  layout: portable root holds `python_embeded` BESIDE `ComfyUI\`, four levels from
  `scripts\`) -> actionable failure. Never an arbitrary system python.
  `otr_check.sh` is CUT from this build (r4: the supported environment is Windows; it
  can follow when a supported non-Windows runtime exists).
- Tests: CLI==loader parity both directions; two-phase ordering; ValidationIssue JSON
  schema; receipt fields + hashes; policy module coverage (word-boundary, field sweep);
  wrapper resolution order (fixture dirs); cache-clear-at-start.

## W3 -- suffix decoupling (Surface A serialization hazard)

- `VALIDATE_INPUTS` on the writer follows the ESTABLISHED REPO PATTERN -- blanket
  `**kwargs -> return True` (precedents: otr_shot_lock:1032, otr_video_director:262,
  otr_master_audio_mux:498, otr_image_director:317): queue-time combo rejection is
  bypassed; EVERY gate already lives in run() (strip + catalog validator for models;
  2C/3C/W1 gates for bank/style/pack). This kills the suffix-flip false-reject AND
  preserves the auto-download admit-path (a strict membership check would reject valid
  org/name ids not yet in choices). Comment cites the precedents + this rationale.
  Verify the installed ComfyUI's VALIDATE_INPUTS contract at build.
- Shipped-ID baseline (r4-pinned): manifest at `tests/fixtures/shipped_model_ids.json`
  listing every curated canonical id; rule: every baseline id must remain present unless
  the change intentionally updates the manifest in the same commit with a review note.
  Renamed/removed detection is a REPO REGRESSION against that manifest (not an end-user
  command). ID-stability test pins selected ids across additions + download-state flips
  (the r1 repro).
- **Sequencing:** `nodes/_otr_model_catalog.py` is CLEAN in git as of this revision
  (verified) -- but W3 gates on the GO_FORWARD_PLAN ownership receipt, not on
  "clean right now."

## W4 -- templates, derived schema doc, recipes

- `docs/templates/visual_style_TEMPLATE.json` + `story_pack_TEMPLATE.json`:
  validator-GREEN fixtures; annotations in `docs/templates/README.md` (JSON has no
  comments). They live OUTSIDE all scanned roots (all three sweeps hard-fail strays).
  Drift-pin tests load both templates through the REAL loaders via a tmp overlay.
- `otr_check.py emit-schema-doc` generates MECHANICALLY DERIVABLE TABLES ONLY (fields,
  types, enums, requiredness, machine-checkable constraints) into
  `docs/EXTENDING_OTR.md`; a generated==committed test pins it (byte compare: UTF-8,
  LF, no BOM). **Documentation ownership boundary (r4):** the root `README.md` is the
  ONLY prose home for the three recipes; `docs/templates/README.md` annotates the two
  templates and nothing else; `docs/EXTENDING_OTR.md` is 100% generated and opens with
  a DO-NOT-EDIT marker naming the generator. The tables are
  the LLM-paste contract; the README section is the human 5-minute path.
- README: three 5-minute recipes (style; content pack -- any runnable lane, EXPLICIT
  restart-to-see-choices note for the `story_pack` dropdown; local HF model --
  causal-LM guard, bare label, UNKNOWN-tier honesty, no restart needed) + one pointer:
  new LANES -> SOURCE_BANK_GUIDE.md + full preflight. Quarantine behavior documented:
  a rejected draft shows a console error + `otr_check` explains the fix.

## W5 -- verification wave

- Full Windows regression suite + Bug Bible after every package (standing law); tracker
  / GO_FORWARD_PLAN updates; no new PBUG/Bible entries without live production proof
  (the suffix-flip repro is a TEST until a live artifact shows it).
- **Live canonical proof (r4 -- unit tests alone do not qualify the selection path):**
  one reset-and-boot (selective reset per AGENTS.md/CLAUDE.md §4) canonical 30-word
  headless smoke that loads `workflows/otr_canonical.json`, selects a SECONDARY
  user-overlay story pack AND an overlay visual style, then verifies: realized inputs
  match the UI mapping, selected-pack prompts actually drove the run (integration
  marker), `meta.story_model_id` + `meta.story_pack_sha256` stamps, ledger closure,
  episode asset at `otr\episodes\<ep>\`, `obs_publish OK`, and the final `otr\obs\`
  asset (Test-Path before declaring success).
- In-place-update preservation test (r4): a simulated repo update leaves `user_packs/`
  content and quarantine behavior intact.

## Laws (unchanged)
No fallbacks; unknown id = hard error. No I/O at import or INPUT_TYPES beyond the
existing catalog live-scan pattern (overlay rides the existing lazy singletons;
`story_pack` choices come from the cached registry). Restart contract for banks/styles;
models live-rescan. Widgets append-at-END only; canonical JSON changes in the same
commit as the code (only W1 touches it). UTF-8 no BOM. SFW. LLM-first.

## Cuts (locked, r1+r2+r4)
models.d; interactive scaffold CLI; hand-written schema docs (derived tables only);
live no-restart rescan; generic GGUF walker; model load/generate probe; within-file
all-errors collection; end-user renamed-ID detection; generated recipe prose;
`otr_check.sh` (Windows is the supported environment); per-wave LOC estimates (behavior
and verification constrain the build, line counts drift).

## Order
Dependency graph (r3-fixed, r4-gated): ACTIVATION GATE -> `W0 -> W1 -> W2 -> W4`, plus
`ownership receipt -> W3`, then `{W3, W4} -> W5`. W5 certifies nothing until BOTH
branches are green. Each W = one green commit+push. Coder-day estimates live in the
GO_FORWARD_PLAN appended section, not here.

## Arc resolutions (r3 wiring + r4 convergence)
1. Serialization: LOCKED (story_pack after gate_in, input 35 / widget slot 34, link 279
   untouched; re-derive everything from live JSON at build).
2. Consumer handoff: run()-scope = `pack: StoryPack | None` object threading;
   ledger-scope = stamped id + REQUIRED sha256 match (r4).
3. `ValidationIssue`: dependency-leaf stdlib module (`nodes/_otr_validation_issue.py`) +
   `.issue` attribute on each module's EXISTING exception families; no shared base.
4. Receipts: `user_packs/receipts/`, one-pack cardinality, CP-1..CP-7 gate IDs (r4).
5. Collisions: protected-id quarantine (shipped survives) + overlay-vs-overlay
   tombstones (r4).
6. Junction root: ALLOWED as the update-survival mechanism; per-entry containment
   against the resolved root (r4 supersedes r3).

## Verify-at-build (carried into W-execution -- r4 checklist)
- [W3] Inspect the installed ComfyUI VALIDATE_INPUTS implementation; prove the blanket
  contract with an API suffix-flip test (no value_not_in_list; run() receives the
  stripped canonical id).
- [W1] Re-derive writer input/widget positions from live INPUT_TYPES +
  `workflows/otr_canonical.json`; verify link 279 still gates `gate_in`, `story_pack`
  owns the appended widget slot, API-realized inputs match the UI.
- [W1] Refinement re-entry retains the selected PackRecord/id/digest across every run()
  reconstruction path (`:3232-3400`).
- [W1] Untagged AND unstamped ledgers byte-identical to today; stamped ledgers select
  the secondary pack; stamped hash mismatch fails loud (freeze cascade via its
  non-raising terminal_error path, `:302-305`).
- [W2] Pin SOURCE_PAYLOAD_KEYS to the exact seven-key set (`_otr_source_payload.py:80-82`);
  templates/receipts use those names.
- [W2] Exercise every story-pack exception family (`_otr_story_pack.py:55-72`); `.issue`
  populated without changing public exception identity; JSON and plain renderings match.
- [W2] Quarantine console output, runtime resolution, and `otr_check --json` emit
  IDENTICAL structured diagnostics for the same overlay defect.
- [W2] Confirm whether a repo-local `.venv` exists for the wrapper's first probe (probe
  is harmless either way).
- [W4] Generated `docs/EXTENDING_OTR.md` byte-compare: UTF-8, LF, no BOM.
- [W4] A discovered causal-LM snapshot appears bare in both writer dropdowns without
  restart and loads through the actual generic loader path.
- [W5] OTR_WorkflowValidator, JSON round-trip, link referential integrity, live
  INPUT_TYPES widget audit, locked optional-set audit, full suite, Bug Bible, and the
  canonical 30-word overlay-selection smoke (W5 bullet 2).
