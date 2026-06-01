# Session Handoff -- ComfyUI-OldTimeRadio -- 2026-06-01 (OpenRouter dynamic model list: S0 done, S1-S4 next)

## Core goal
Implement `docs/2026-06-01-openrouter-dynamic-model-list__sprint-plan.md`: expand the
writer model router from 2 dropdowns to 4 -- `creative_writing_model` + `technical_model`
(execution-SLOT selectors: local / `openrouter:slot-a` / `openrouter:slot-b`) and two NEW
slug pickers `openrouter_slot_a_model` + `openrouter_slot_b_model` (choose the real
OpenRouter slug from a cached catalog). **S0 (cache) is shipped + committed; S1-S4 remain.**
Loop: REVIEW -> CODE -> WIRE -> REGRESS -> COMMIT, one green commit per sprint.
(NOTE: the earlier story-spine go-forward refactor ALSO shipped this session and is DONE/
unrelated -- do not reopen it.)

## Tech stack & constraints
Windows, RTX 5080. Venv python: `C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe`.
Branch `v2.0-alpha`. CLAUDE.md rules apply (git via Desktop Commander **cmd** + write msg to
`.git\COMMIT_EDITMSG` then `git commit -F`; no "dummy"; regress after every change; PD1 audio
byte-identical -- this feature never touches the audio path).
Plan hard rules: **INPUT_TYPES() NEVER touches the network** (all 4 dropdowns build from the
disk cache only); **APPEND-never-insert** new widgets (saved workflows bind by widget INDEX --
existing indices [0..18] must not shift; this is the BUG-LOCAL-258/253 index-drift trap);
**never silently swap a saved slug** (only empty/unset falls back; an explicit saved slug is
preserved + warned + attempted even if absent from a stale cache; a failed remote call is a
hard error, never a silent remote->remote swap).
Cowork-env gotchas (learned this session): the Bug Bible repo (`comfyui-custom-node-survival-
guide`) is NOT checked out here -- run it on the host; full `tests/` + module self-tests +
forbidden sweep are the headless gate. File tools (Read/Write/Edit) DO reach the real Windows
FS. **Glob has a path quirk (returns nothing) -> use Grep.** Read a redirected pytest file's
tail with the Read tool (cmd has no `tail`; cmd mangles inline `python -c "..."` -- use a
script file or findstr). Subagent READS of the repo work fine; subagent WRITES are untested
(author on the main thread).

## What's done & decided
- **S0 SHIPPED** (commit `8239f3d`, pushed, full suite 3327/0): `nodes/_otr_openrouter_backend.py`
  gained the catalog cache -- `load_catalog_cache()`, `cached_models()`, `catalog_meta()`
  (source/fetched_at/count/staleness=live|cache|stale|empty), `refresh_catalog_cache()`
  (EXPLICIT-only, atomic temp+`os.replace`, a failed/offline fetch keeps the old cache & never
  raises), `_slim_model()` (keeps id/name/provider/created/context_length/pricing/
  supported_parameters + a derived `supports_json`), `_fetch_models_json()` mockable seam,
  `CATALOG_SCHEMA_VERSION=1`. Cache path `<repo>/models/openrouter_models.json` (override dir
  via `OTR_OPENROUTER_CACHE_DIR`). `__main__` self-test 10/10. Reads are network-free
  (INPUT_TYPES-safe).
- **Review resolved every open question (the high-value de-risking):**
  - **Q1 (preservation, the plan's flagged risk) -- the ComfyUI frontend PRESERVES out-of-list
    saved combo values** (litegraph `configure()` assigns `widget.value` verbatim, no clamp/
    reset), so **NO `web/` JS shim is needed.** The real lever the plan MISSED: the OTR backend
    validator **`otr_api._validate_widget_value` REJECTS an out-of-list value** at execution
    (BUG-LOCAL-280). Fix in pure Python -- an admit-path for `openrouter:*` / saved slugs past
    the static choice-list check (mirror the existing `validate_model_id` openrouter admit-path).
    **This is load-bearing -- fold into S1 (permissive slot dropdown) + S3 (validator admit-path).**
    Frontend is the current Vue-nodes line (~`comfyui_frontend_package` 1.45.14).
  - **Q5 -- `supports_json` is reliable**: OpenRouter `supported_parameters` lists
    `structured_outputs` / `response_format` for capable models; S0 already derives + stores
    `supports_json`, so the per-slot REQUIRE_JSON filter is feasible.
  - **Q2 recommended defaults** = `OPENROUTER_RECOMMENDED_CREATIVE_DEFAULT = anthropic/claude-opus-4.8`,
    `OPENROUTER_RECOMMENDED_TECHNICAL_DEFAULT = deepseek/deepseek-v4-pro` (matches the env set
    this session). Q3 favorites = env (`OTR_OPENROUTER_FAVORITES`) + recency (newest by
    `created`). Q4 cache in-repo `models/` (git-ignore it in S4).
- **Scoping insight:** the creative/technical dropdowns ALREADY exclude the full catalog --
  `_otr_model_catalog._active_curated_models()` = local curated models + the two `slot-a/b`
  virtual rows only. So **S1 is purely ADDITIVE** (add the slot-picker builder); nothing to
  remove from creative/technical.
- `_otr_openrouter_backend.resolve_slug(repo_id)` is STILL env-only today (slot-a ->
  `OPENROUTER_MODEL_A`, slot-b -> `OPENROUTER_MODEL_B`, raises if unset). **S3 demotes env to a
  fallback** and resolves from the new widget slug value.
- **Live env on this machine (set this session) -- remote is ENABLED:** `OTR_ENABLE_OPENROUTER=1`,
  `OPENROUTER_MODEL_A=anthropic/claude-opus-4.8`, `OPENROUTER_MODEL_B=deepseek/deepseek-v4-pro`,
  `OPENROUTER_API_KEY` present. So `openrouter_enabled()` is True and `slot-a/b` show in the
  writer dropdowns after a fresh ComfyUI restart.

## State of the art
- HEAD = `8239f3d` on `v2.0-alpha`, local == origin. Full `tests/` baseline 3327 passed / 12
  skipped / 0 failed (~30s).
- Files + current state:
  - `nodes/_otr_openrouter_backend.py` -- S0 cache (above) + the env-only `resolve_slug` (S3
    reworks it), `OpenRouterBackend`, `make_openrouter_generate_fn`, `openrouter_meta_for`,
    `SLOT_A_ID`/`SLOT_B_ID`, `openrouter_enabled()`.
  - `nodes/_otr_model_catalog.py` -- `build_dropdown_choices()` / `dropdown_choices()` (feed
    creative/technical; built from `_active_curated_models()`), `_openrouter_virtual_rows()`
    (the 2 rows when enabled), `_by_repo_id()`, `validate_model_id()`. **S1 adds
    `openrouter_catalog_dropdown_choices()` here.**
  - `nodes/OTR_LedgerScriptWriter.py` (~4200 lines) -- `INPUT_TYPES` (current model widgets +
    optional widgets [0..18]) + `_resolve_inputs`. The spine call + REJECT raise sit ~L3744-3800.
    **S2 appends 2 widgets + updates `_resolve_inputs`; S3 adds slot resolution.**
  - `nodes/otr_api.py` -- `_validate_widget_value` (BUG-280 out-of-list rejection; **S3 adds the
    openrouter admit-path here**).
  - `workflows/otr_scifi_16gb_full.json` -- node-1 `widgets_values`; **S2 appends 2 trailing
    values**, verify [0..18] unchanged.
  - `docs/_s28_forbidden_sweep.py` -- model_id-widget allowlist (currently the 2 writer picks);
    **S4 expands to 4.** `CLAUDE.md` PD6 -- **S4 amends 2->4.**

## Immediate next steps
1. **S1** (`_otr_model_catalog.py`): add `openrouter_catalog_dropdown_choices(slot)` -- remote
   disabled -> `["(enable OpenRouter)"]` sentinel (UI-only, rejected before resolution); enabled
   -> recommended-default + favorites(env) + recent(newest by `created`) + full `cached_models()`
   filtered by `OTR_OPENROUTER_MODEL_ALLOWLIST` / `DENYLIST` / `PROVIDER_FILTER` + per-slot
   `OTR_OPENROUTER_SLOT_x_REQUIRE_JSON` (uses S0's `supports_json`; default off for A, the
   filter is for B/technical; NEVER global). Leave creative/technical builders unchanged. Extend
   `tests/test_openrouter_catalog_rows.py`: catalog absent from creative/technical; per-slot
   filters narrow only their slot; sentinel present when disabled. Regress + commit.
2. **S2** (`OTR_LedgerScriptWriter.py` + `workflows/otr_scifi_16gb_full.json`): APPEND
   `openrouter_slot_a_model` + `openrouter_slot_b_model` at the END of the optional block
   (indices [0..18] UNCHANGED); update `_resolve_inputs`; conditional defaults (creative:
   remote-on -> `openrouter:slot-a`, else `DEFAULT_LLM`; technical: `DEFAULT_LLM` unless
   overridden -- no auto-flip). Append 2 trailing values to node-1 `widgets_values`.
   **Migration test**: old node-1 widgets_values (no slot entries) -> defaults supplied, NO
   existing value shifted, [0..18] intact. Regress + commit. (This is the node-SURFACE change --
   the consequential sprint.)
3. **S3** (`_otr_openrouter_backend.resolve_slug` + writer `_resolve_inputs` +
   `otr_api._validate_widget_value`): §5 3-case resolution on the STORED widget string -- (1)
   empty/unset/placeholder -> fallback chain `OTR_OPENROUTER_SLOT_x_DEFAULT` -> `OPENROUTER_MODEL_x`
   env -> recommended default -> clear config error; (2) explicit saved slug -> use as-is, warn
   if absent from cache, still attempt, NO substitution; (3) selected call fails -> hard error,
   no remote->remote swap. Add the **validator admit-path** (otr_api) so an out-of-list
   openrouter slug is accepted (BUG-280). Stamp `slot_a_resolved_slug` + `slot_b_resolved_slug` +
   cache source/fetched_at/staleness in run meta; demote env to fallback. Self-test per §5.
   Regress + commit.
4. **S4**: `docs/_s28_forbidden_sweep.py` allowlist 2 -> 4 named writer model widgets; `CLAUDE.md`
   PD6 text 2 -> 4 (rationale: writer-only opt-in slug bindings, no non-writer node gains a pick);
   add `models/openrouter_models.json` to `.gitignore`; write `docs/openrouter-setup.md` (router,
   conditional default, preservation rules, cache, filters, refresh script). Forbidden sweep green
   (exactly 4 picks) + Bug Bible/core/dropdown. Regress + commit.
5. **Final verify**: full `tests/` 0-fail; sweep 4 picks; JSON [0..18] intact + 2 trailing slot
   values; INPUT_TYPES zero-network; all changed files no-BOM/AST-clean; HEAD==origin. Operator
   live check: ComfyUI restart -> slot pickers show real slugs; a saved out-of-list slug survives
   load + execution (the Q1 + validator admit-path payoff).

## Open questions
- Pin the exact installed `comfyui_frontend_package` version on the host (`pip show
  comfyui_frontend_package`) for the record -- preserve behavior is confirmed for the current
  line, and the S3 validator admit-path is the load-bearing piece regardless.
- `refresh_catalog_cache()` ownership: no scheduled refresh yet -- operator runs it (or a small
  script). The cache file does not exist until the first refresh; until then the slot dropdowns
  show only the sentinel + env-pinned defaults, which is the intended safe-empty behavior.

---
## Resume instructions
Open a fresh window, attach this file + `docs/2026-06-01-openrouter-dynamic-model-list__sprint-plan.md`,
and say: "Read this handoff + the sprint plan, verify S0 against the live tree, then execute S1
through S4 with REVIEW -> CODE -> WIRE -> REGRESS -> COMMIT, one green commit per sprint.
Acknowledge when ready."
