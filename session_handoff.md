# Session Handoff -- ComfyUI-OldTimeRadio -- 2026-06-01 (OpenRouter 4-dropdown router: S0-S4 SHIPPED)

## Core goal
Implemented `docs/2026-06-01-openrouter-dynamic-model-list__sprint-plan.md`: the writer
model router went from 2 dropdowns to 4 -- `creative_writing_model` + `technical_model`
(execution-slot selectors: local / `openrouter:slot-a` / `openrouter:slot-b`) plus two NEW
slug pickers `openrouter_slot_a_model` + `openrouter_slot_b_model` (pick the real OpenRouter
slug from the S0 disk cache). **S0-S4 are all shipped + pushed on `v2.0-alpha`.** Only
operator live-validation and the separate README newbie rewrite remain.

## Tech stack & constraints
Windows, RTX 5080, venv `C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe`, branch
`v2.0-alpha`. CLAUDE.md rules apply (git via Desktop Commander **cmd** + `.git\COMMIT_EDITMSG`
then `git commit -F`; no "dummy"; regress after every change). Hard rules HELD this session:
**INPUT_TYPES does zero network** (proven with the fetch seam armed to raise -- builds clean,
21 widgets); **append-never-insert** (`[0..18]` byte-identical; slots at `[19]`/`[20]`);
**never silently swap a saved slug**; PD1 audio path untouched. `CLAUDE.md` is git-ignored
(local-only) -- the PD6 amendment lives there but the committed audit pin is the b6 test.

## What's done & decided (this session)
- **S1 `2c96c3b`** -- `nodes/_otr_model_catalog.py` `openrouter_catalog_dropdown_choices(slot)`:
  disabled -> `(enable OpenRouter)` sentinel; enabled -> recommended default + favorites +
  recent(newest by `created`) + full cached catalog, filtered by ALLOWLIST/DENYLIST/
  PROVIDER_FILTER + per-slot REQUIRE_JSON. creative/technical builders UNCHANGED (catalog never
  appears there). Recommended constants added to the backend:
  `OPENROUTER_RECOMMENDED_CREATIVE_DEFAULT=anthropic/claude-opus-4.8`,
  `OPENROUTER_RECOMMENDED_TECHNICAL_DEFAULT=deepseek/deepseek-v4-pro`.
- **S2 `d5fe8c8`** -- `OTR_LedgerScriptWriter.py`: appended the 2 slot widgets at the END of
  optional; `_resolve_inputs` + `run()` thread them; conditional creative default
  (remote-on -> `openrouter:slot-a`, else `DEFAULT_LLM`; technical never auto-flips). Workflow
  `otr_scifi_16gb_full.json` node-1 widgets_values 19 -> 21 (creative/technical stay local at
  wv[3]/wv[4]; wv[19]=opus-4.8, wv[20]=deepseek-v4-pro). Migration test + the two length gates
  (companions:418, guardrails:666) bumped 19 -> 21; `_writer_schemas`/`_writer_node_fixture`
  mirrored to 21.
- **S3 `579571a`** -- preservation core. `resolve_slug` rewritten (§5): bound slug used
  VERBATIM + warned if absent from a stale/cold cache + never swapped; unbound -> chain
  `OTR_OPENROUTER_SLOT_x_DEFAULT` -> `OPENROUTER_MODEL_x` (env DEMOTED to fallback) ->
  recommended -> config error. `set_slot_bindings`/`clear_slot_bindings` (process-global, set
  from raw widget args at run() start, before `_resolve_inputs`). `scripts/otr_api.py`
  `_validate_widget_value` gained the OpenRouter admit-path (out-of-list slug / `openrouter:`
  handle preserved, BUG-LOCAL-280). `openrouter_run_meta()` stamps slot_a/b_resolved_slug +
  catalog source/fetched_at/staleness. **Behavior change:** unbound resolve_slug now returns
  the recommended default instead of raising -- 2 old tests updated
  (test_openrouter_backend, test_openrouter_meta_stamp).
- **S4 `d43c9b0`** -- `tests/test_b6_wiring_guardrails.py` `_MODEL_WIDGET_KEYS` 2 -> 4 active
  writer picks (now also enforces the slug pickers are writer-only); `scripts/otr_openrouter_refresh.py`
  (NEW, the explicit cache refresh CLI); `docs/openrouter-setup.md` rewritten for the
  four-dropdown router (selectors + slug pickers, conditional default, env-is-fallback order,
  refresh/cache, filters, preservation); `.gitignore` explicit cache entry; `CLAUDE.md` PD6
  amended (local, git-ignored).

## State of the art
- HEAD = `d43c9b0` on `v2.0-alpha`, local == origin. Headless gates ALL GREEN:
  full `tests/` **3366 passed / 12 skipped / 0 failed** (~32s); forbidden sweep **exit 0**;
  Bug Bible regression **23 passed**; backend `__main__` self-test **10/10**;
  every changed file no-BOM + AST-clean.
- The catalog cache `models/openrouter_models.json` does NOT exist yet (git-ignored); until the
  operator runs the refresh script the slot pickers show the recommended default + the
  `(no OpenRouter models cached ...)` sentinel -- intended safe-empty behavior.

## Immediate next steps (OPERATOR -- cannot be done headless)
1. Run the refresh script to populate the slug pickers, then restart ComfyUI:
   `C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe scripts\otr_openrouter_refresh.py`
2. In a fresh `OTR_LedgerScriptWriter` node confirm: `creative_writing_model` defaults to
   `openrouter:slot-a` (remote is enabled in your env); `openrouter_slot_a/b_model` list the
   real cached slugs.
3. Save a workflow with an out-of-list slug in a slot picker, reload it, and run -- confirm the
   slug is preserved through load + execution (the Q1 frontend-preserve + S3 validator
   admit-path payoff). A genuinely failed remote call should error loud (no remote->remote swap).
4. (Separate effort, NOT this plan) the broad README newbie rewrite -- tracked as S6 in the
   OpenRouter go-forward plan / the `project_otr_readme_audience` memory.

## Open questions
- Pin the installed `comfyui_frontend_package` version on the host for the record. Preserve
  behavior is confirmed for the current Vue-nodes line; the S3 validator admit-path is the
  load-bearing piece regardless of the frontend.
- `refresh_catalog_cache()` ownership: the operator runs `scripts/otr_openrouter_refresh.py`
  by hand; no scheduled/auto refresh yet (could be a future scheduled task).

---
## Resume instructions
Open a fresh window, attach this file, and say:
"Read this handoff. S0-S4 of the OpenRouter four-dropdown router are shipped + green on
v2.0-alpha (HEAD d43c9b0). The remaining work is operator live-validation (run the refresh
script, restart ComfyUI, confirm the slot pickers + preservation) -- walk me through it, or
start the separate README newbie rewrite (S6)."
