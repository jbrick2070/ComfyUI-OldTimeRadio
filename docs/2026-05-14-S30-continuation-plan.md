# S30 — continuation plan (Cowork loop, pytest-only)

**Branch:** `s30-two-model-selector` — **stay on this branch for every remaining commit.** No sub-branches, no `s30-b1d-hotfix`, no `s30-b2a-wire`. One linear branch from B1d to B8.
**Branch policy:** v2.0-alpha umbrella. Do NOT bump version labels (commits, doc filenames, BUG_LOG entries, internal "v2.1" or "v2.0-beta" mentions). Per `feedback_dont_promote_version_labels`.
**HEAD at hand-off:** `b12b941` (B1c + ROADMAP refresh + hand-off doc).
**Goal:** dual-LLM workflow code + JSON wired. Runtime verification deferred to a later sprint.
**Loop per commit:** review → code → wire → pytest → commit → push. No ComfyUI execution.

---

## Hard rules (apply to every commit B1d → B8)

1. **No deletion of `_load_llm` / `_generate_with_llm` / `_LLM_CACHE` until B4b lands.**
2. **Bug Bible regression** 23 passed / 1 skipped / 2 xfailed at every commit boundary.
3. **No legacy back-compat reintroduced.** No `_RENAME_ALIASES`, no fallback-on-unknown-model, no soft-landing, no "stamp both legacy + new" meta keys. Clean break is done; do not undo it.
4. **No separate change logs.** Updates flow only to BUG_LOG.md and ROADMAP.md. No new `CHANGELOG.md`, no `docs/changes/*.md`, no per-commit summary docs.
5. **No extra branches.** Every commit lands on `s30-two-model-selector` and pushes to `origin/s30-two-model-selector`.
6. **Tests written before fixes** for every P0 defect. Red-on-parent, green-on-fix.
7. **Forbidden-pattern sweep** stays at 0 runtime hits at every commit boundary.
8. **Audio C7 byte-identical proxy (pytest)** must hold at every commit boundary. Real-pipeline audio gate deferred to a separate operator-driven sprint after B8.

---

## Canonical pytest run between commits

```cmd
cd /d C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe -m pytest "C:/Users/jeffr/Documents/ComfyUI/comfyui-custom-node-survival-guide/tests/bug_bible_regression.py" tests\test_workflow_json_guardrails.py tests\test_core.py tests\test_audio_byte_identical.py tests\test_model_catalog_scan.py tests\test_model_catalog_download.py tests\test_loader_slot_primitives.py -q
```

After every commit, regenerate the forbidden-sweep input and run:

```cmd
git diff s29-clean-slate-gate -- "*.py" > docs\s28_diff_tmp.txt
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe docs\_s28_forbidden_sweep.py
```

Commit message goes to `.git\COMMIT_EDITMSG` via the file tool (never inline via cmd `-m`; cmd.exe mangles parens / pipes / `&&`). Then `git commit -F .git\COMMIT_EDITMSG` and `git push origin s30-two-model-selector`.

---

## B1d — pre-B2a hotfix (~1 d, Low/Medium)

### Review
Seven defects in landed B0–B1c code. All P0.

### Code

**`nodes/_otr_model_loader.py`**
- `unload_llm`: replace `LLM_CACHE = {...}` rebind with `LLM_CACHE.clear(); LLM_CACHE.update({...})`. Remove `global LLM_CACHE`. Module-level imports (`from _otr_model_loader import LLM_CACHE`) currently hold a stale dict after the rebind.
- `request_slot`: reorder to `validate → cache hit → context → size estimate → vram fit → (FAIL? raise) → auto_download → unload (if swap) → load → cache`. VRAM fail must fire before any network / disk work.
- `load_llm`: accept optional `context_cap` kwarg from `request_slot`, skip the second `resolve_context_cap` call.

**`nodes/_otr_model_catalog.py`**
- `auto_download_if_missing`: local-cache short-circuit as FIRST check, before gated / auto / disk. Returns local snapshot path immediately if `on_disk=True`.
- `_estimate_resident_gb`: add `SPECIAL_VRAM_ESTIMATES_GB = {TEST_OVERSIZED_LLM: 42.0}`; accept `safetensors_gb_hint: float | None` kwarg. The 70B-on-16GB case currently returns UNKNOWN (uncurated) instead of FAIL.
- `check_vram_fit`: accept `safetensors_gb_hint`, forward to `_estimate_resident_gb`.
- `estimate_model_size_gb`: wrap `_hf_api.model_info(...)` in try/except; re-raise as `UnknownModelError` with the recovery hint. Catches `RepositoryNotFoundError`, `HfHubHTTPError`, network errors.

### Wire
None. Python-only.

### Pytest

| Test | File | Asserts |
|---|---|---|
| `test_unload_llm_preserves_cache_identity` | `test_loader_slot_primitives.py` | `id(LLM_CACHE)` unchanged across `unload_llm()` |
| `test_auto_download_returns_cached_path_without_token` | `test_model_catalog_download.py` | Cached Mistral-Nemo + `HF_TOKEN=None` → returns local path, no `snapshot_download` call, no `GatedModelError` |
| `test_auto_download_skips_when_on_disk` | same | `on_disk=True` in mocked scan → `snapshot_download` not called |
| `test_check_vram_fit_oversized_returns_fail` | `test_loader_slot_primitives.py` | `check_vram_fit(TEST_OVERSIZED_LLM, 8192).tier == "FAIL"` with NO mocks (real catalog state) |
| `test_request_slot_oversized_fails_before_download` | same | `VRAMFitFailedError` raised, `snapshot_download` never called |
| `test_estimate_model_size_gb_404_raises_unknown` | `test_model_catalog_download.py` | Mocked `model_info` raising `RepositoryNotFoundError` → `UnknownModelError` raised |

### Commit gate
All 6 new tests green. Bug Bible 23/1/2xf. Canonical regression holds. Forbidden sweep clean.

### Commit subject
`B1d: pre-B2a hotfix — 7 P0 defects in B0-B1c (unload identity, request_slot ordering, vram fit oversize, auto_download short-circuit, hf api wrapping)`

---

## B2a — writer two-widget surface (~0.5 d, Medium)

### Review
Writer node currently has one `model_id` widget. Needs `creative_writing_model` + `technical_model` widgets and matching output sockets. Strict rule: writer does NOT call `request_slot("technical", ...)` internally in B2a. Internal routing change lands in B2b. B2a is a pure widget-surface change so audio C7 attribution at B2b is clean.

### Code
- `nodes/OTR_LedgerScriptWriter.py` lines 120, 330, 1163: replace single `model_id` widget + `DEFAULT_MODEL_ID` literal + `_MODEL_CHOICES` literal with two widgets:
  ```python
  "creative_writing_model": (_otr_model_catalog.dropdown_choices(), {"default": _otr_model_catalog.DEFAULT_LLM}),
  "technical_model":        (_otr_model_catalog.dropdown_choices(), {"default": _otr_model_catalog.DEFAULT_LLM}),
  ```
- Update `run()` signature: rename `model_id` → `creative_writing_model`, add `technical_model`. Both still feed the **same** legacy generation path for now.
- Add two new output sockets `creative_writing_model` + `technical_model` (STRING) at the end of `OUTPUT_NAMES` / `RETURN_TYPES`.
- Normalize labels at every output / `meta` stamp site (strip `[NOT DOWNLOADED]` via `_otr_model_catalog._strip_label_suffix`). Raw widget values must NEVER reach a downstream consumer.

### Wire
`workflows/otr_scifi_16gb_full.json` — writer node only (only canonical workflow has the writer placed):
- `widgets_values`: insert `DEFAULT_LLM` at index 6 (the new `technical_model` slot). Existing value at index 5 stays — it becomes `creative_writing_model`. Indices 6..17 shift +1 to 7..18.
- `outputs`: append two new entries:
  ```json
  {"name": "creative_writing_model", "type": "STRING", "links": null, "slot_index": 4},
  {"name": "technical_model",        "type": "STRING", "links": null, "slot_index": 5}
  ```
- `links` stays `null` for both new outputs in B2a — downstream consumers wire in B3 (cascade gets `technical_model`).
- `last_link_id` unchanged in B2a since no new links land yet.
- Other workflow JSONs untouched (no writer placed).
- `tools/validate_workflow_links.py` across all 8 JSONs — 0 violations.

### Pytest
- `test_writer_output_slot_indexes_stable` (`test_workflow_json_guardrails.py`): every existing link's source slot in the canonical JSON resolves to its original output name post-edit.
- `test_writer_widget_migration_preserves_values`: load pre-B2a workflow fixture, run migration helper, assert each old widget value lands on the new same-named widget position.
- `test_writer_broadcasts_normalized_model_ids`: AST scan asserts both broadcast outputs route through `_strip_label_suffix` before emission.
- `test_writer_b2a_surface_only`: AST-walk `OTR_LedgerScriptWriter.run` — assert ZERO calls to `request_slot("technical", ...)` (internal routing change is B2b's job; surface change in B2a only).

### Commit gate
4 new tests green. Bug Bible holds. Workflow link validator green across all 8 JSONs. Forbidden sweep clean.

### Commit subject
`B2a: writer two-widget surface + output sockets (single generation path; technical_model output-only)`

---

## B2b — writer internal routing + slot scheduler (~1.25 d, HIGH)

### Review
14 LLM sub-passes inside the writer need slot tagging + routing through `request_slot`. Reference table (per the parent sprint plan §2c):

| # | Pass | Slot | Reason |
|--:|---|---|---|
| 1 | Outline | creative | narrative |
| 2 | Cast | creative | narrative |
| 3 | Dialogue composer | creative | narrative |
| 4 | Polish | creative | narrative |
| 5 | WORD_EXTEND rescue | technical | structured |
| 6 | FORMAT_NORM | technical | structured |
| 7 | Grammarian | technical | structured |
| 8 | LLM_RESCUE | technical | structured |
| 9 | ANNOUNCER bookends | technical | structured |
| 10 | news_interpreter.build_news_briefs | technical | GBNF + pydantic + V0-V3 validators |
| 11 | Style picker pass 1 (inventor) | creative | recombines seed flavors creatively |
| 12 | Style picker pass 2 (chooser) | technical | rule-based + GBNF + regex grammar |
| 13 | Cast contract `_otr_casting` schema validation | technical | locked pydantic schema, JSON validators |
| 14 | Critic (`script_critic.py`) | technical | verdict-style structured output |

### Code
- Tag every LLM call site `# LLM slot: creative` or `# LLM slot: technical` with a one-line reason.
- Replace direct `load_llm(...)` / `make_generate_fn(...)` calls with a new private `_request_slot(slot)` helper that calls `_otr_model_loader.request_slot(slot, resolved_model_id)`.
- Slot scheduler: where the dependency DAG allows, batch consecutive same-slot passes. Document each unavoidable interleave point with `# slot-interleave: <prior> -> <next>` naming the data dependency that forces it.
- New meta keys per pass: `meta["gen_params_by_phase"][<phase>] = {"slot": "...", "model": "<resolved repo id>", ...other existing per-phase fields}`.
- Top-level `meta["creative_writing_model"]` and `meta["technical_model"]` stamps (the resolved repo IDs). **Delete `meta["model_id"]`** outright — no "stamp both" hedge. Every downstream `meta["model_id"]` reader gets updated in this same commit (grep before commit).
- Per-beat-default mode (preserves critic→polish feedback loop): `meta["slot_transitions"] == compute_expected_transitions(fixture)`.
- Opt-in `OTR_BATCH_PER_BEAT=1` mode (loses per-beat feedback): `meta["slot_transitions"] <= 3`.

### Wire
None. JSON unchanged from B2a end state.

### Pytest
- `test_slot1_neq_slot2_split_routing_full_phase_trace`: table-test all 14 phases. For each, assert correct slot + correct model_id received by the mocked loader.
- `test_no_direct_load_llm_calls_inside_writer`: AST sweep on `OTR_LedgerScriptWriter.py` — zero hits for `load_llm(` (other than as import-name reference).
- `test_writer_uses_request_slot_for_every_llm_pass`: AST sweep — every LLM-call site routes through `_request_slot` / `request_slot`.
- `test_slot_transition_count_from_dag`: `compute_expected_transitions(fixture_config)` walks the enabled-phase DAG; assert measured transition count matches the DAG-computed value (per-beat-default) OR <= 3 (opt-in batched).
- `test_polish_on_fixture_routes_to_creative`: separate fixture with `enable_polish_pass=true`; asserts polish invoked once per line, polish routes to `creative_writing_model` slot, `LLM_CACHE` shows exactly one resident model after the run.

### Commit gate
5 new tests green. Bug Bible holds. Audio C7 byte-identical proxy holds (both slots default to Mistral-Nemo → loader caches one model → no swap). Forbidden sweep clean.

### Commit subject
`B2b: writer internal creative/technical routing + slot scheduler + new meta keys`

---

## B2c — delete `cleanup_model_id` legacy strip (~0.25 d, Low)

### Review
`cleanup_model_id` is a legacy normalizer at `OTR_LedgerScriptWriter.py:2470`. `_strip_label_suffix` from the catalog replaces it.

### Code
- Delete the `cleanup_model_id` loop from `OTR_LedgerScriptWriter.py`.
- Delete any test residue referencing it.
- Add `cleanup_model_id` to the forbidden-sweep marker list.

### Wire
None.

### Pytest
- `test_no_cleanup_model_id_callers`: grep + AST scan returns zero hits across `nodes/`, `visual/`, `scripts/`, `tests/`.

### Commit gate
1 new test green. Bug Bible holds. Forbidden sweep catches future reintroduction.

### Commit subject
`B2c: delete cleanup_model_id legacy-strip loop + arm forbidden-pattern marker`

---

## B3 — cascade widget + technical socket (~0.5 d, Medium)

### Review
`OTR_LedgerFreezeCascade` carries its own `model_id` widget + 6 phase-toggle widgets. Replace `model_id` with input socket from writer. Delete the 6 phase toggles (all default OFF in the workflow; their code paths are dead surface).

### Code
- `nodes/OTR_LedgerFreezeCascade.py`:
  - Delete `model_id` widget (lines 148-155) + `DEFAULT_MODEL_ID` literal.
  - Delete `enable_phase_3_polish` / `polish_announcer_beats` / `enable_phase_4_scene_coherence` / `enable_phase_4_5_smart_suggestion` / `enable_phase_5_voice_drift` / `enable_phase_6_episode_arc` widgets (lines 156-220).
  - Add `technical_model` STRING input socket (no widget).
  - Routing: cascade phases 1/2/9 (reviewer verdicts) → `technical_model` resolved via `_otr_model_inputs.require_model(..., slot="technical")`. Tag each call site `# LLM slot: technical`.
- `nodes/_otr_lfc.py` — **untouched in B3.** Phase functions stay alive until B4 deletes them along with the standalone phase nodes.

### Wire
`workflows/otr_scifi_16gb_full.json` — cascade node only:
- `widgets_values`: delete `model_id` value + 6 phase-toggle values (7 entries removed). Post-B3 cascade `widgets_values` = `[enable_phase_7_audio_readiness, enable_phase_8_video_readiness, vram_ceiling_gb]`.
- `inputs`: add `technical_model` input slot. Allocate new link ID = `last_link_id + 1` (read current value, fail if drift from baseline 114). Bump `last_link_id`. New link entry: `[<new_id>, <writer_node_id>, 5, <cascade_node_id>, <cascade_input_slot>, "STRING"]`.
- Update writer's `outputs[5]["links"]` from `null` to `[<new_id>]`.
- Other workflow JSONs untouched.

### Pytest
- `test_cascade_has_no_local_model_widget`: AST scan asserts `OTR_LedgerFreezeCascade` exposes no `model_id` widget in `INPUT_TYPES`.
- `test_cascade_technical_socket_wired_in_canonical_json`: parse canonical JSON; assert one link with source = writer's `technical_model` slot, target = cascade's `technical_model` socket.
- `test_workflow_json_link_integrity`: re-run link validator across all 8 JSONs; zero violations.
- `test_cascade_phase_toggles_extinct`: AST scan asserts none of the 6 phase-toggle widgets exists in `INPUT_TYPES`.

### Commit gate
4 new tests green. Bug Bible holds. Link validator clean. First commit where canonical workflow runs end-to-end on the new contract; audio C7 proxy must hold.

### Commit subject
`B3: cascade — delete model_id + phase-3..6 toggles, wire technical_model socket (canonical JSON re-wired)`

---

## B4 — delete standalone LFC Phase 4/5/6 nodes (~0.25 d, Low)

### Review
`OTR_LFCPhase4Scene`, `OTR_LFCPhase5Voice`, `OTR_LFCPhase6Arc` orphaned from every shipped workflow JSON per parent plan §2a-bis. Combined with deletion-bias policy, the nodes go entirely.

### Code
- Delete the three node files: `nodes/OTR_LFCPhase4Scene.py`, `nodes/OTR_LFCPhase5Voice.py`, `nodes/OTR_LFCPhase6Arc.py`.
- Remove the three entries from `__init__.py` `NODE_CLASS_MAPPINGS` + `NODE_DISPLAY_NAME_MAPPINGS`.
- Delete from `nodes/_otr_lfc.py`: `_phase_3_per_line_polish`, `_phase_4_scene_coherence`, `_phase_4_5_smart_suggestion`, `_phase_5_voice_drift`, `_phase_6_episode_arc`.
- Prune helpers in `nodes/_otr_lfc_llm_helpers.py` used exclusively by the deleted phases.
- Delete any test files exclusively pinning these standalone nodes.

### Wire
None — nodes orphaned from every JSON.

### Pytest
- `test_lfc_phase_nodes_unregistered`: import `__init__`, assert `OTR_LFCPhase4Scene` / `5Voice` / `6Arc` not in mappings.
- `test_no_lfc_phase_callers`: grep returns zero hits in `nodes/`, `visual/`, `scripts/`.
- `test_lfc_phase_functions_extinct`: AST scan of `_otr_lfc.py` asserts the 5 phase functions do not exist.

### Commit gate
3 new tests green. Bug Bible holds. Forbidden sweep clean (B7 adds these as extinction markers).

### Commit subject
`B4: atomically delete standalone LFC Phase 4/5/6 nodes + phase-3..6 functions in _otr_lfc.py`

---

## B4b — rewire RSS news + collapse legacy LLM stack (~1 d, HIGH)

### Review
RSS news path still uses `_so._load_llm` via `_generate_with_llm`. `_run_with_timeout` invalidates `_LLM_CACHE` directly. Both must rewire before deletion. This commit is the original sprint plan's audit-miss fix (BUG-LOCAL-226).

### Code

**Step 1 — refactor `_generate_with_llm` body in `nodes/story_orchestrator.py`** (do NOT delete yet):
- Replace internal `_load_llm(...)` call with `cache_entry = _otr_model_loader.request_slot("technical", model_id)`.
- Replace `_LLM_CACHE`-based generation with `make_generate_fn(cache_entry)`.
- All four call sites (`_llm_rank_news_candidates`, `_llm_rerank_with_bodies`, both `_do_rank_call` invocations at lines 1501 + 1598) continue to work.

**Step 2 — refactor `_run_with_timeout` timeout-recovery** (lines 352-364):
- Delete the manual `_LLM_CACHE` key invalidation block.
- Replace with: `from ._otr_model_loader import unload_llm; unload_llm()`.
- Preserve the `TIMEOUT_RECOVERY` log line.

**Step 3 — grep verification**:
- `git grep "_LLM_CACHE" nodes/` must return zero hits in `story_orchestrator.py` outside the module-level definition line (3090).
- If hits remain, rewire before proceeding to Step 4.

**Step 4 — delete the legacy stack from `story_orchestrator.py`**:
- `_load_llm` (lines 1974-2586).
- `_unload_llm` (line 3093 onward).
- `_LLM_CACHE` module-level dict (line 3090).
- `_generate_with_llm` (delete only if every caller has moved to direct `request_slot` + `make_generate_fn` use; otherwise defer to a follow-up commit).
- Functions still in the file that default `model_id="mistralai/Mistral-Nemo-Instruct-2407"` and have no remaining callers (`_generate_ltx_style_brief` etc.).

**Step 5 — update three importers**:
- `nodes/batch_bark_generator.py`, `nodes/_otr_bark_lib.py`, `nodes/scene_sequencer.py`: change `from .story_orchestrator import _unload_llm` → `from ._otr_model_loader import unload_llm`.

**Step 6 — clear orchestrator-side teardown delegation in `_otr_model_loader.unload_llm`**:
- Once Step 4 deletes `story_orchestrator._LLM_CACHE`, the `try/except` block in `unload_llm` that touches it becomes dead code. Delete that block.

### Wire
None.

### Pytest
- `test_rss_news_path_uses_request_slot`: fixture seeds an RSS payload, runs news fetch, asserts `_otr_model_loader.request_slot("technical", ...)` invoked and `_so._load_llm` NOT invoked.
- `test_run_with_timeout_calls_new_unload`: force timeout via mock; assert `_otr_model_loader.LLM_CACHE` cleared via the new path.
- `test_legacy_load_llm_symbol_removed`: `hasattr(story_orchestrator, "_load_llm")` returns False.
- `test_legacy_llm_cache_symbol_removed`: `hasattr(story_orchestrator, "_LLM_CACHE")` returns False.
- `test_importers_use_new_unload_path`: AST scan of the three importers asserts `from ._otr_model_loader import unload_llm`, not the legacy path.

### Commit gate
5 new tests green. Bug Bible holds. Forbidden sweep clean post-deletion. `BUG-LOCAL-226` marked `[FIXED <hash> 2026-MM-DD]` in BUG_LOG.md in the same commit.

### Commit subject
`B4b: rewire RSS news LLM path through request_slot + delete legacy orchestrator LLM stack (fixes BUG-LOCAL-226)`

---

## B5 — `_POLISH_CACHE` collapse + `OTR_VisualLLMSelector` delete (~1 d, HIGH)

### Review
Three caches existed before B1c collapsed two into `LLM_CACHE`. The third (`_POLISH_CACHE` in `visual/llm_polish.py`) still exists and could double-load Mistral-Nemo on the 16 GB card (guaranteed OOM, Prime Directive 2). Also delete the visual selector node.

### Code
- **Pre-deletion gate**: forbidden sweep must show zero live hits for `_POLISH_CACHE`, `OTR_VisualLLMSelector`. If any remain, fix callers first.
- `visual/llm_polish.py`:
  - Delete `_POLISH_CACHE` module-level dict and `_load_model()` (L67).
  - Replace polish entry point's internal model load with `_otr_model_loader.request_slot("creative", model_id)` followed by `make_polish_generate_fn(cache_entry, ...)`.
  - Tag `# LLM slot: creative — visual prompt cleanup`.
  - Delete the `"none"` short-circuit (L294).
- `visual/visual_prompt_coercion.py`: swap model_id input source from `OTR_VisualLLMSelector.model_id` to writer's `creative_writing_model` socket (link-only, no widget). Tag `# LLM slot: creative — visual prompt prose`.
- Delete `visual/llm_selector.py` entirely.
- Remove `OTR_VisualLLMSelector` from `__init__.py` `NODE_CLASS_MAPPINGS` + `NODE_DISPLAY_NAME_MAPPINGS`.
- `nodes/_otr_model_loader.make_polish_generate_fn`: only B5 signature change is to accept an already-loaded cache entry. **Polish sampling behavior unchanged.** OTR's hardcoded `_POLISH_TOP_P=0.9` / `_POLISH_DO_SAMPLE=True` defaults still apply unconditionally. (Model-author-config respect deferred entirely to a later audio-intentional sprint per parent plan §B5 audio-discipline note.)

### Wire
None — `OTR_VisualLLMSelector` and `OTR_VisualPromptCoercion` aren't placed in any shipped workflow JSON per parent plan §2a-bis. Deleting the selector class is a Python + `__init__.py` change only.

### Pytest
- `test_polish_path_uses_request_slot`: AST scan of `visual/llm_polish.py` asserts every polish path routes through `request_slot("creative", ...)` + `make_polish_generate_fn(...)`.
- `test_no_polish_cache_symbol`: `hasattr(visual.llm_polish, "_POLISH_CACHE")` returns False.
- `test_visual_llm_selector_unregistered`: import `__init__`, assert `OTR_VisualLLMSelector` not in mappings.
- `test_no_visual_llm_selector_callers`: grep returns zero hits anywhere.
- `test_polish_cache_collapse_no_dup_load`: drive polish path via mocked loader; assert `_otr_model_loader.LLM_CACHE` reports exactly one resident model (no duplicate load).

### Commit gate
5 new tests green. Bug Bible holds. Forbidden sweep clean. Audio C7 byte-identical proxy holds (polish sampling unchanged).

### Commit subject
`B5: delete OTR_VisualLLMSelector + collapse _POLISH_CACHE into single LLM cache`

---

## B6 — wiring tests + structure guardrails (~0.75 d, Medium)

### Review
Defense-in-depth regression layer. Catches structural drift.

### Code
None — tests only.

### Wire
None.

### Pytest
- `test_request_slot_uses_check_vram_fit_pre_download`: regression for the B1d hotfix. Mock `snapshot_download`; pass `TEST_OVERSIZED_LLM` to `request_slot`; assert `VRAMFitFailedError` raised before `snapshot_download` invoked.
- `test_no_legacy_meta_id_keys_in_writer`: AST scan asserts writer never emits legacy `model_id` meta key — only `creative_writing_model` + `technical_model` + per-phase `gen_params_by_phase[*]["model"]`. Also asserts `[NOT DOWNLOADED]` never leaks into `meta`.
- `test_14_phase_routing_table`: table-test iterates the 14-pass routing table from B2b; asserts each phase routes through the expected slot AND emits a `meta.gen_params_by_phase[<phase>]` entry.
- `test_no_widget_form_string_outside_writer`: AST scan distinguishes widget-form STRING (rejected outside writer) vs connectable-input-socket STRING (allowed on consumers).
- `test_slot_scheduler_transitions_match_dag`: same as B2b but here as defense-in-depth at the integration layer.
- `TestNoModelWidgetOutsideWriter` (extend `test_workflow_json_guardrails.py`): for every registered node class other than `OTR_LedgerScriptWriter`, assert `INPUT_TYPES()["required"]` and `INPUT_TYPES()["optional"]` contain no widget keyed `model_id`, `model_creative`, `model_technical`, `creative_writing_model`, `technical_model`, or any `model_*` STRING widget.

### Commit gate
6 new tests green. Bug Bible holds.

### Commit subject
`B6: wiring tests + structure-based guardrails (widget-name not widget-value)`

---

## B7 — arm forbidden-pattern sweep with S30 extinction markers (~0.75 d, HIGH)

### Review
Lock in deletion. Future commits cannot reintroduce.

### Code
- `docs/_s28_forbidden_sweep.py`: append S30 extinction markers:
  - `OTR_VisualLLMSelector` (B5)
  - `_LLM_MODEL_CHOICES` (B5)
  - `_MODEL_CHOICES` (B2a)
  - `DEFAULT_MODEL_ID` — path-aware: forbidden in runtime Python under `nodes/` and `visual/`; allowed in tests' own marker-list literals.
  - `_LLM_CACHE` (B4b)
  - `_load_llm(` — path-scoped: forbidden as `nodes/story_orchestrator.py::_load_llm` (B4b deleted it). The new public `_otr_model_loader.load_llm` is the canonical symbol.
  - `_generate_with_llm` (B4b)
  - `_POLISH_CACHE` (B5)
  - `MODEL_CONTEXT_CAPS` (B1b)
  - `DEFAULT_CONTEXT_CAP` (B1b)
  - `cleanup_model_id` (B2c)
  - `enable_phase_3_polish`, `polish_announcer_beats`, `enable_phase_4_scene_coherence`, `enable_phase_4_5_smart_suggestion`, `enable_phase_5_voice_drift`, `enable_phase_6_episode_arc` (B3)
  - `OTR_LFCPhase4Scene`, `OTR_LFCPhase5Voice`, `OTR_LFCPhase6Arc` (B4)
- Structural sweep rule: AST-walk every `INPUT_TYPES` block; distinguish widget-form STRING / COMBO outside writer (reject) from connectable-input-socket STRING (allow). Non-LLM media nodes opt in via class-level `NON_LLM_MODEL_WIDGET_OK = True`.

### Wire
None.

### Pytest
- `test_forbidden_sweep_runs_clean`: invoke sweep against current `nodes/`, `visual/`, `scripts/`; assert zero runtime hits.
- `test_forbidden_sweep_catches_reintroduction`: introduce a deliberate hit in a temp file under `tmp_path`, run sweep against that path, assert it's flagged.
- `test_forbidden_sweep_exemption_works`: verify the `_otr_model_loader.py` exemption for `load_llm(` is honored.
- `test_forbidden_sweep_non_llm_marker_works`: a fixture class with `NON_LLM_MODEL_WIDGET_OK = True` and a `model_id` STRING widget passes the sweep.

### Commit gate
4 new tests green. Bug Bible holds. Sweep itself runs at 0 runtime hits.

### Commit subject
`B7: arm forbidden-pattern sweep with S30 extinction markers + structural widget-vs-socket rule`

---

## B8 — sprint close (~0.5 d, Low)

### Review
Final pytest gates only. No ComfyUI runtime verification (deferred to a separate operator-driven sprint).

### Code
- Fill in `docs/2026-05-14-S30-final-qa-review.md` (template already in repo). Acceptance table: target vs actual for each row.
- Update `ROADMAP.md`: move S30 from "IN FLIGHT" to "COMPLETE 2026-05-14". Demote S29 from "PRIOR CURRENT" to historical (no new section needed; ROADMAP_HISTORY captures it via git log). Update sprint sequencing — Sprint C (`meta.story_brief` v2) now in next-up position.
- Update `BUG_LOG.md`: any new local bugs surfaced during the sprint get logged in real time during each commit, not batched here. This step is verification that the log is current — no batch-update.

### Wire
None.

### Pytest
Final canonical run:
- Bug Bible 23 passed / 1 skipped / 2 xfailed.
- Combined canonical: target ~250+ passed / ~10 skipped / 2 xfailed (growth from B1d through B7 new tests).
- Forbidden sweep 0 runtime hits across all paths.
- Workflow link validator: 0 violations across all 8 JSONs.

### Commit gate
All gates green. ROADMAP refreshed. Final QA doc complete. No `docs/cleanbreak-deferred.md` resurrection (stays deleted from S29).

### Commit subject
`B8: Sprint S30 close — two-model selector shipped, ROADMAP refreshed, final QA review filed`

---

## Commit table

| Commit | Status | Est. | Risk |
|---|---|---|---|
| B0 / B1a / B1a2 / B1b / B1c | done (HEAD `b12b941`) | — | — |
| **B1d** | hotfix pending | 1.0 d | Low/Medium |
| B2a | pending | 0.5 d | Medium |
| B2b | pending | 1.25 d | HIGH |
| B2c | pending | 0.25 d | Low |
| B3 | pending | 0.5 d | Medium |
| B4 | pending | 0.25 d | Low |
| **B4b** | pending (fixes BUG-LOCAL-226) | 1.0 d | HIGH |
| B5 | pending | 1.0 d | HIGH |
| B6 | pending | 0.75 d | Medium |
| B7 | pending | 0.75 d | HIGH |
| B8 | pending | 0.5 d | Low |

**Total remaining: ~7.75 d.** Pure pytest gates; no ComfyUI runs in this sprint.

---

## Fresh-session pickup checklist

When a new conversation opens to continue this sprint:

1. Read `CLAUDE.md` (standing rules, platform pins, test commands).
2. Read `ROADMAP.md` "CURRENT WORK -- S30 Two-Model Selector" section.
3. Read `BUG_LOG.md` head to get last bug number (currently 226).
4. Read this plan (`docs/2026-05-14-S30-continuation-plan.md`) — execution playbook.
5. Read `docs/2026-05-14-S30-final-qa-review.md` — QA template to fill in at B8.
6. Read `docs/2026-05-14-S30-B1c-handoff.md` — what was shipped in B0-B1c and why.
7. `git checkout s30-two-model-selector && git log --oneline -8` — confirm HEAD matches `b12b941`.
8. Confirm canonical pytest run is green at HEAD (commands above under "Canonical pytest run between commits").
9. Begin B1d using the loop: review → code → wire → pytest → commit → push.

**Do NOT cut a sub-branch.** Every commit lands on `s30-two-model-selector` directly.

**Do NOT introduce legacy back-compat** of any kind. If a deletion needs a transition shim, that's a design hole — fix the consumers in the same commit.

**Do NOT bump version labels.** Stay on the `v2.0-alpha` umbrella. No `v2.1`, `v2.0-beta`, `v3.0`, or similar.

**Do NOT create separate change-log files.** Updates go to BUG_LOG.md (live, per-finding) and ROADMAP.md (S30 in-flight section, updated as commits land).

**Do NOT skip the canonical pytest run between commits.** A green commit boundary is the only safe revert window.

If a commit's pytest gate fails: revert the commit, log to BUG_LOG.md, stop and hand back to Jeffrey. Per CLAUDE.md stop conditions.
