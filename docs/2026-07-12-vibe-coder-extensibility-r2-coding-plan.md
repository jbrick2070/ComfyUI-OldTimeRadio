# Vibe-Coder Extensibility -- R2 Coding Plan (DRAFT for kibitz r2)

- **Date:** 2026-07-12. **Parent:** `docs/2026-07-12-vibe-coder-extensibility-r1.md`
  (r1-hardened scoper @ 852209bf). **Status:** DRAFT -- kibitz r2 (implementability) then
  r3 (wiring) run on this doc. NO code until the arc converges.
- **Operator rulings folded (2026-07-12):**
  1. **No tiers.** The Tier-1/Tier-2 vocabulary is DEAD. There are two KINDS of addition,
     not ranks: a **content pack** (a story pack inside ANY registered runnable lane -- the
     vibe path, equal across all lanes) and a **new lane** (new fetcher/interpreter/runner
     + full independent-design preflight -- expert path, out of scope here). Gate 1
     continues to govern new lanes; content packs are not new banks and do not trigger it.
  2. **`user_packs/` overlay ratified** -- user-authored packs live outside tracked dirs.
  3. **No blessed lane.** The pack template, check command, and recipes must work
     uniformly for EVERY registered runnable lane.

## W0 -- user_packs overlay foundation

- New `nodes/_otr_user_packs.py`: single source of truth for overlay roots.
  `user_packs_root()` = `<repo>/user_packs`; surface roots `user_packs/visual_styles/`,
  `user_packs/story_packs/<bank_id>/`. Stdlib only; ZERO file I/O at import (custom-node
  import isolation).
- `.gitignore` += `user_packs/`.
- `_otr_visual_styles`: the lazy sweep merges BOTH roots. Duplicate `style_id` across any
  two files (including shipped-vs-overlay shadowing) = hard error naming BOTH paths.
  Schema/lint rules identical for overlay packs.
- `_otr_story_routing`: `_sweep_and_crossref` additionally sweeps
  `user_packs/story_packs/`. Same laws as the tracked root: every subdir must be a
  REGISTERED bank id; every pack validates + matches path coordinates; top-level files are
  forbidden (so overlay can NEVER introduce bank rows or pipelines -- content packs only,
  which is the no-tiers ruling made structural). Duplicate (bank, story_model) across
  roots = hard error. `resolve_story_pack` resolves from the merged map.
- `nodes/story_rules/`: NO overlay -- rules packs are lane infrastructure (stem =
  source_bank_id), repo-owned; a content pack never ships one.
- Restart contract unchanged: overlay dirs are swept inside the SAME lazy singletons.
- Tests: overlay style + pack become visible after `_clear_caches()`; duplicate-id
  rejection in both directions; overlay top-level registry file rejected; unregistered
  bank subdir rejected; import-isolation (importing the module does no I/O).

## W1 -- pack-selection surface (the ONE canonical-JSON change)

Today all three writer call sites call `resolve_story_pack(bank_id)` with no model id
(`OTR_LedgerScriptWriter.py:1799/1837/3741`) -- the lane default always wins, so a second
pack in a lane is unreachable. Fix:

- New optional COMBO widget `story_pack`, appended at the very END of the writer's
  optional block (widget order is load-bearing, `OTR_LedgerScriptWriter.py:2362-2366`;
  append-only law). Choices = `["(bank default)"] + sorted("<bank_id>/<story_model_id>")`
  over the merged registry (from the cached singleton -- no new INPUT_TYPES I/O class).
  Default = `"(bank default)"`.
- `run()`: sentinel -> `resolve_story_pack(bank_id)` (byte-identical to today); explicit
  pair -> split on the first `/`, HARD ERROR if the pair's bank != the selected
  `source_bank` (no silent cross-bank fallback), else
  `resolve_story_pack(bank, model_id)`. Ledger meta stamps the resolved
  `story_model_id` (verify existing stamp at build; add if absent).
- `workflows/otr_canonical.json`: append the new widget value at the END of the writer
  node's `widgets_values` in the SAME change; re-validate (OTR_WorkflowValidator,
  JSON round-trip, link referential integrity, widget-count vs live INPUT_TYPES).
- Update the locked INPUT_TYPES/output contract audit (`OTR_LedgerScriptWriter.py:6998+`)
  to include `story_pack` in the expected optional set, same change.
- Tests: new `test_story_pack_widget.py` modeled on `test_source_bank_widget_2c.py`
  (choices shape, sentinel default, cross-bank hard error, resolution parity with default
  path); canonical-JSON guardrail update.
- Precedent: Stage 2C (`source_bank`) and 3C (`visual_style`) widget additions.

## W2 -- `otr_check.py`: one validator, two entry points

- New `scripts/otr_check.py`, subcommands: `style`, `pack`, `model`, `all`,
  `emit-schema-doc`. It CALLS the production validators -- `_otr_visual_styles` load path,
  `_otr_story_routing` sweep, `_otr_story_rules` load, `_otr_model_catalog` validator --
  never a parallel schema implementation.
- Batch granularity v1 (honest trade, panel please pressure): PER-FILE. The CLI walks each
  candidate file, catches that file's first validation error, reports it with
  file+field+fix, and continues to the next file. Within-file all-errors collection would
  require refactoring every loader's fail-fast raise into issue collection -- deferred
  unless the panel shows a cheap seam.
- SFW/forbidden sweep runs IN the CLI for all user content (overlay packs never meet repo
  tests): styles reuse the loader's own forbidden-terms lint; story packs get a
  string-leaf sweep against the shared lexicon. VERIFY-AT-BUILD which lexicon surface is
  canonical (candidates: the news/machine kill lexicon @ 3d32b265, the story_rules
  vocabulary). The sweep existing in the CLI is mandatory; its vocabulary source is the
  build-time decision.
- Pack runs emit an **authoring-validated receipt** (paths + SHA-256 + lane + pack id +
  check version) that LISTS the production-preflight hard-gate IDs as UNRESOLVED. The CLI
  can never emit a production PASS (r1 two-state law).
- `--json` mirrors every diagnostic machine-readably (LLM one-pass repair).
- `scripts/otr_check.bat`: venv-locating wrapper (portable installs have non-standard
  interpreters).
- Tests: CLI==loader parity both directions (loads-green implies check-green; check-fail
  implies load-fail with the same error class); JSON schema of diagnostics; receipt
  fields; wrapper smoke (locates `.venv` python).

## W3 -- suffix decoupling (Surface A serialization hazard)

- Add `VALIDATE_INPUTS` to the writer: accept a combo value that either matches current
  choices OR strips (`_otr_model_catalog._strip_label_suffix`) to a valid canonical id.
  Kills the saved-workflow break when `[NOT DOWNLOADED]` flips to bare after a download.
- ID-stability test: selected canonical ids stable across (a) new entries appearing,
  (b) download-state label flips (the repro from r1); duplicate/renamed/removed shipped
  ids fail the check command.
- **Sequencing:** `nodes/_otr_model_catalog.py` is dirty in the bake-off window RIGHT NOW.
  W3 lands AFTER that window pushes; rebase, do not cross it.

## W4 -- templates, derived schema doc, recipes

- `docs/templates/visual_style_TEMPLATE.json` + `docs/templates/story_pack_TEMPLATE.json`:
  validator-GREEN fixtures (JSON carries no comments; the annotation lives in
  `docs/templates/README.md` beside them). They live OUTSIDE all scanned roots (the
  story_packs/story_rules/visual_styles sweeps hard-fail strays -- templates must never
  sit in those dirs).
- Drift-pin tests: both templates load GREEN through the REAL loaders from docs/templates/
  (copied to a tmp overlay in the test); template drift = red suite.
- `otr_check.py emit-schema-doc` generates `docs/EXTENDING_OTR.md` (field tables, enums,
  laws, per-surface recipes' contract sections) FROM validator constants; a
  generated==committed test pins it. This is the LLM-ready paste-to-assistant doc -- the
  r1 drift law made executable. No hand-maintained schema prose anywhere.
- README: three 5-minute recipes (style; content pack -- any lane, restart note; local HF
  model -- causal-LM guard, bare label, UNKNOWN-tier honesty, no restart needed). Plus a
  one-line pointer: "new LANES are expert territory -> SOURCE_BANK_GUIDE.md + preflight."

## W5 -- verification wave

- Full Windows regression suite + Bug Bible after every package (standing law); the new
  guardrail tests enumerated above ride their packages.
- Tracker/GO_FORWARD_PLAN update; no new PBUG/Bible entries without live production proof
  (admission rule) -- the suffix-flip repro is a TEST, not a PBUG, until a live artifact
  shows it.

## Laws (unchanged, the code must hold)
No fallbacks; unknown id = hard error. No file/network I/O at import or INPUT_TYPES beyond
the existing catalog live-scan pattern (overlay sweeps ride the existing lazy singletons;
`story_pack` choices come from the cached registry). Restart contract for banks/styles;
models live-rescan. Widgets append-at-END only; canonical JSON changes in the same commit
as the code (only W1 touches it). UTF-8 no BOM. SFW. LLM-first (the CLI judges, never
rewrites content).

## Cuts (carried from r1, operator-visible)
models.d overlay; interactive scaffold CLI; hand-written schema docs (derived-only);
live no-restart rescan for banks/styles; generic GGUF walker; model load/generate
compatibility probe (acceptance = discovery; probe can be its own later item).

## Order + rough size
W0 -> W1 -> W2 -> W4 -> W5, with W3 sliding in after the bake-off window's catalog push.
~1.5-2k LOC total (W0 ~350+tests, W1 ~200+tests, W2 ~500+tests, W3 ~150+tests, W4 templates
+ generator ~400, W5 0 new). Each W = one green commit+push.

## Questions for the r2 panel
1. Per-file batch granularity in W2 -- acceptable v1, or is there a cheap all-errors seam?
2. Which lexicon surface should the pack SFW sweep bind to?
3. `story_pack` combo shape: global `<bank>/<model>` pairs vs any better ComfyUI-native
   bank-scoped pattern that does not require frontend JS?
4. Receipt contents: anything missing for the authoring-validated state to be useful?
5. `VALIDATE_INPUTS` return/signature contract vs stock ComfyUI expectations -- any gotcha?
6. Anything in W0-W4 unimplementable as written against the real code?
