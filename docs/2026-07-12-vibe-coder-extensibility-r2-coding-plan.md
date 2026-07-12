# Vibe-Coder Extensibility -- R2 Coding Plan (r3-HARDENED, wiring-converged)

- **Date:** 2026-07-12. **Parent:** `docs/2026-07-12-vibe-coder-extensibility-r1.md`
  (@ 852209bf). **Revision:** kibitz r2 AND r3 folded (codex @ gpt-5.6-sol,
  antigravity @ gemini-3.5-pro, Claude anchor+judge; judgment logs:
  `kibitz-runs/2026-07-12-vibe-coder-extensibility/r2/final.md` + `r3/final.md`).
  Operator directed r2-r3 only; r4 convergence pass not yet run. NO code yet.
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
  no-op, never a FileNotFoundError. Entries whose RESOLVED path escapes
  `user_packs_root()` (symlink/junction) are rejected loud.
- `.gitignore` += `/user_packs/` (anchored -- not any-depth). Docs note the caveat: a
  Manager "reinstall" that deletes the node folder removes user_packs too; back it up.
- `_otr_visual_styles`: lazy sweep merges BOTH roots. Duplicate `style_id` anywhere
  (incl. shipped-vs-overlay shadowing) = hard error naming BOTH paths. Identical
  schema/lint for overlay packs. `_clear_caches()` (exists, :393) clears any new state.
- `_otr_story_routing` -- the heart of W0:
  1. **PackRecord map.** Discovery builds an immutable
     `(bank_id, story_model_id) -> PackRecord(path, pack)` map covering BOTH roots;
     `_Registry` (today only banks+pipelines, :481) gains it; `resolve_story_pack` and
     all listing resolve EXCLUSIVELY through the map (kills the `_pack_path`
     tracked-root reconstruction, :517-530). Deterministic ordering; overlay sweep
     mirrors `_sweep_and_crossref` laws (:350-378): subdir must be a registered bank,
     top-level files forbidden (overlay can NEVER add bank rows / pipelines --
     the no-tiers ruling made structural), header triple must match path coordinates.
  2. **Overlay only under RUNNABLE banks.** `custom_source_bank` is registered but
     runnable:false -- overlay dirs under non-runnable banks are rejected loud
     (unreachable packs must not exist, let alone be advertised).
  3. **Per-pack parity laws (today default-only, :387-425) extend to EVERY pack in the
     map, both roots:** `pack.story_pipeline_id == bank.default_story_pipeline`;
     `bank.required_seams` present in `pack.prompt_stages`; pipeline pass `seam_refs`
     present; three-way custom-seam parity (pack keys / declared_seams / pass_refs).
     A selectable pack is a VALIDATED pack -- selection can never outrun validation.
  4. `_clear_caches()` (exists, :547) additionally resets the PackRecord map.
- `nodes/story_rules/`: NO overlay (lane infrastructure; stem = source_bank_id;
  runnable-bank coverage law `:274-280` unchanged). Content packs never ship rules.
- **Overlay quarantine contract (r3, agy's boot-crash find judged in):** INPUT_TYPES
  triggers the lazy sweep, so a malformed OVERLAY draft must not crash the node pack at
  boot. Any overlay-side validation failure (schema, parity, user-content policy,
  duplicate id, sidecar-name, symlink escape) QUARANTINES that overlay file: loud
  console error naming file + reason, absent from the registry and every choices list,
  run-time selection fails loud (not in the map), and `otr_check` reports the identical
  issue. TRACKED-root failures keep today's hard fail (a broken shipped pack IS a build
  error). Nothing silently succeeds; nothing substitutes -- fail-loud preserved, and a
  vibe coder's draft can never brick the extension.
- **Sidecar law (r3):** `_PACK_SIDECAR_FILENAMES_BY_BANK` exemptions (`:42`, applied
  `:333`) do NOT apply under `user_packs/` -- overlay sidecar-named files are
  quarantined loud (they would otherwise be unreachable, unvalidated content).
- **Junction rule tightened (r3):** the RESOLVED `user_packs` root itself must live
  under the resolved repo root (a junction root pointing elsewhere is rejected), and
  every entry's resolved path must stay under the resolved root.
- The published PackRecord map is wrapped in `MappingProxyType` (a frozen dataclass
  holding a mutable dict is not immutable; `_Registry` `:120-127` has the same
  weakness -- fix it for the new map, leave the old fields as-is).
- Restart contract unchanged (overlay swept inside the same lazy singletons).
- Tests: overlay style+pack visibility after `_clear_caches()`; duplicate-id rejection
  both directions; overlay top-level registry file rejected; non-runnable-bank overlay
  rejected; unregistered-bank subdir rejected; symlink escape rejected; absent
  user_packs tree = clean load; per-pack parity failures loud for overlay AND tracked
  secondary packs; import-isolation.

## W1 -- pack-selection surface (the ONE canonical-JSON change)

Today all three writer call sites resolve the bank DEFAULT (`:1799/:1837/:3741`) and the
lane runner dispatches by `bank.default_story_pipeline` (`:3729-3742`) -- a second pack in
a lane is unreachable. Fix:

- New routing API `list_story_pack_choices()` -> `["(bank default)"] + sorted(
  "<bank_id>/<story_model_id>")` over PackRecords of RUNNABLE banks only. The writer
  formats nothing itself.
- New optional COMBO widget `story_pack`, appended at the very END of the writer's
  optional block (order is load-bearing, `:2362-2366`). Default `"(bank default)"`.
- `run()` resolves ONCE, immediately after `require_runnable_bank`: sentinel ->
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
  1. **run()-scope:** the ONE selected `PackRecord` is passed EXPLICITLY to the original
     interpreter factory (`:1799`), the QA gate (`:1837`), the lane runner (`:3741`),
     AND down the prompt-router chain -- `resolve_creative_system_prompt`,
     `generate_outline`, `compose_line`, and the announcer/coda helpers gain
     `story_model_id: str | None = None` (None = bank default = byte-identical for
     every pre-W1 caller). No production code re-resolves by bank alone.
  2. **Ledger-scope** (freeze cascade, reroll paths): resolve via the W1
     `meta.story_model_id` stamp -- `resolve_story_pack(bank, stamped_id)`; an absent
     stamp (pre-W1 ledger) = today's behavior. Backwards-compatible by construction;
     parity test: untagged AND unstamped ledgers behave byte-identically to today.
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
- **CLI grammar (decided now):** `otr_check.py style <path|id>... |
  pack <path|bank_id/story_model_id>... | model <id> | all [--json] [--receipt <out>]`.
  Bare story-model ids are REJECTED (they are bank-scoped; no searching, no guessing).
  `all` scans both roots of both surfaces + the model catalog. Registry caches are
  cleared at invocation start. Import discipline = the test suite's pattern
  (OTR_TEST_MODE + direct module import; never through ComfyUI-heavy `nodes/__init__`).
  Model-catalog failures get a ValidationIssue ADAPTER at the failure site
  (`UnknownModelError` is prose-only today, `:1129-1228`) -- never string parsing.
- **SFW/user-content policy IN the production validator (parity law):** new
  `nodes/_otr_user_content_policy.py` -- stdlib, whole-word matching, authored-field
  coverage (prompt seams + defaults + labels). Called by the production pack validator,
  so load-fail == check-fail with the same exception (overlay-side violations
  QUARANTINE per W0; tracked-side hard-fail); styles keep their own forbidden-terms
  lint. **Dependency direction (r3):** `_otr_original_radio.py` imports pydantic at
  module load (`:43`), so the policy module OWNS the curated literals and
  original_radio/stage3 consume FROM it (leaf import) -- or pin a copied subset with a
  drift test. Never policy -> original_radio. Seed vocabulary contract:
  `DEFAULT_PROFANITY_TERMS` (`_otr_stage3_validators.py:140`) + a curated
  violence/weapons subset from `FORBIDDEN_TERMS` (`_otr_original_radio.py:88`)
  EXCLUDING lane-specific anachronism/source-framing terms. (Binding to
  `_ALL_FORBIDDEN` rejected -- 1940s anachronism terms would false-fail modern lanes.)
- **Authoring receipt for content packs = its own gate-ID contract** (NOT the bank
  preflight's positional G-numbers): stable IDs over {pack/schema/prompt validity,
  selected-lane compatibility (parity laws), user-content policy, regression suite,
  canonical 30w smoke, ledger closure, published asset}. The CLI asserts and stamps the
  first three (authoring-validated); the rest are listed UNRESOLVED with their IDs --
  the CLI can NEVER emit them as passed (two-state law). Receipt carries SHA-256 of the
  pack, `banks.json`, `pipelines.json`, the lane's story_rules pack, plus schema +
  checker versions and verified-file/seam counts.
- **Receipt location (r3, convergent):** default and only sanctioned output dir =
  `user_packs/receipts/` (invisible to both surface sweeps by construction). The CLI
  REJECTS any `--receipt` path whose resolved location falls under
  `user_packs/story_packs/` or `user_packs/visual_styles/` (same containment guard as
  the overlay) -- a receipt beside a pack would be swept as a pack on restart. No sweep
  exemptions for receipt-looking names: strays in swept dirs stay loud.
- `scripts/otr_check.bat` resolution order: `OTR_PYTHON` env override ->
  `%~dp0..\..\..\.venv\Scripts\python.exe` -> `%~dp0..\..\..\python_embeded\python.exe`
  -> actionable failure. (THREE levels from `scripts\`: repo sits at
  `ComfyUI\custom_nodes\<repo>` -- the r2 draft's two-level paths pointed at
  `custom_nodes\` and were wrong.) Never an arbitrary system python. Optional
  `otr_check.sh` mirrors the order for Linux/macOS dev boxes.
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
- Shipped-ID baseline: committed pinned manifest of curated canonical ids;
  renamed/removed detection is a REPO REGRESSION against that baseline (not an end-user
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
  `docs/EXTENDING_OTR.md`; a generated==committed test pins it. Recipe prose and laws
  live in ONE owned README section -- never generated, never duplicated. The tables are
  the LLM-paste contract; the README section is the human 5-minute path.
- README: three 5-minute recipes (style; content pack -- any runnable lane, EXPLICIT
  restart-to-see-choices note for the `story_pack` dropdown; local HF model --
  causal-LM guard, bare label, UNKNOWN-tier honesty, no restart needed) + one pointer:
  new LANES -> SOURCE_BANK_GUIDE.md + full preflight. Quarantine behavior documented:
  a rejected draft shows a console error + `otr_check` explains the fix.

## W5 -- verification wave

Full Windows regression suite + Bug Bible after every package (standing law); tracker /
GO_FORWARD_PLAN updates; no new PBUG/Bible entries without live production proof (the
suffix-flip repro is a TEST until a live artifact shows it).

## Laws (unchanged)
No fallbacks; unknown id = hard error. No I/O at import or INPUT_TYPES beyond the
existing catalog live-scan pattern (overlay rides the existing lazy singletons;
`story_pack` choices come from the cached registry). Restart contract for banks/styles;
models live-rescan. Widgets append-at-END only; canonical JSON changes in the same
commit as the code (only W1 touches it). UTF-8 no BOM. SFW. LLM-first.

## Cuts (locked, r1+r2)
models.d; interactive scaffold CLI; hand-written schema docs (derived tables only);
live no-restart rescan; generic GGUF walker; model load/generate probe; within-file
all-errors collection; end-user renamed-ID detection; generated recipe prose.

## Order + rough size
Dependency graph (r3-fixed): `W0 -> W1 -> W2 -> W4`, plus `ownership receipt -> W3`,
then `{W3, W4} -> W5`. W5 certifies nothing until BOTH branches are green. ~2.5-3k LOC
(W0 ~550+tests incl. quarantine, W1 ~400+tests incl. consumer threading, W2 ~750+tests,
W3 ~150+tests, W4 ~400). Each W = one green commit+push.

## r3 wiring resolutions (was: questions for the r3 panel)
1. Serialization: LOCKED (see W1 -- story_pack after gate_in, input 35 / widget slot 34,
   link 279 untouched; re-derive everything from live JSON at build).
2. Consumer handoff: EXPANDED (see W1 -- prompt-router chain + freeze cascade via the
   two-channel law; the original "three consumers" were not enough).
3. `ValidationIssue`: dependency-leaf stdlib module (`nodes/_otr_validation_issue.py`) +
   `.issue` attribute on each module's EXISTING exception families; no shared base, no
   inheritance-tree coupling.
4. Receipts: `user_packs/receipts/` with CLI containment rejection (see W2).

## Verify-at-build (carried into W-execution)
Installed ComfyUI's VALIDATE_INPUTS contract; SOURCE_PAYLOAD_KEYS exact names;
story-pack loader error classes at ValidationIssue wiring; freeze-cascade meta-channel
parity (untagged + unstamped ledgers byte-identical to today); emit-schema-doc
generated==committed pin compares BYTES (UTF-8, LF, no BOM) to dodge CRLF flake.
