# Voice-Path-Cleanbreak — S15.5 + S16 + S17 + S18 + S21 + S22 + S23 + S19 QA (2026-05-13)

**Branch:** `v2.0-alpha`
**Predecessor HEAD:** `53aa2e3`
**Batch HEAD:** `bed3c4a` (pushed; local == origin)
**Commits in batch:** 14
**Regression delta:** `+25` tests (2096 → 2121). Bug Bible regression 23 passed / 1 skipped / 2 xfailed across the batch. KNOWN-FAIL nodeid set steady at 6 (matches `EXPECTED_FAILED_NODEIDS` exactly).

This document covers every commit shipped in the batch. Companion to the S6-S8 QA (`docs/2026-05-12-voice-path-cleanbreak-S6-S8-qa.md`) and the S10-S15 QA (`docs/2026-05-12-voice-path-cleanbreak-S10-S15-qa.md`).

---

## 0. Commit table

| # | Hash      | Subject |
|--:|-----------|---------|
| 1 | `1f654b8` | docs(preamble): update _bark_lib/_sfx_lib rename note for Sprint 7.2 otr_ prefix |
| 2 | `62f5042` | S15.5.1 pre-flight legacy audit (modification) |
| 3 | `a261a7f` | S16.1 scrub Director-era widget names from production workflow JSON (replacement+deletion) |
| 4 | `6c7e784` | S16.2+S16.3+S16.4+S16.5+S16.6 validator hardening batch (modification) |
| 5 | `5c49d20` | S17.1+S17.2+S17.3+S17.4 cache + render integrity batch (replacement+deletion) |
| 6 | `1ddad72` | S18.1+S18.2+S18.3+S18.4 post-freeze writeback audit batch (modification) |
| 7 | `6d08f63` | S21.1+S21.2 VRAM thresholds + context cap (modification) |
| 8 | `b4a3098` | S22.1+S22.2 LLM timeout workflow pause (modification) |
| 9 | `b443f46` | S23.1-S23.9 legacy scrub: Director removal + audit gate (replacement+deletion) |
| 10 | `32f62eb` | S19.1+S19.2 hook hardening + doc-freshness check (modification) |
| 11 | `d698611` | test fixups for S17.1+S17.2+S18.1 contract changes (modification) |
| 12 | `bfbfd0b` | docs: ROADMAP + BUG_LOG live update for the voice-path-cleanbreak batch (modification) |
| 13 | (push) | -- |
| 14 | `bed3c4a` | docs: deferral markers for S14.2 + S19.3 + S19.2 doc clarification (modification) |

Two commits in the batch are auxiliary (preamble docstring fix at `1f654b8` and the final docs at `bfbfd0b` / `bed3c4a`); the remaining 11 are the sprint sub-tasks.

---

## 1. Per-sprint mechanics

### S15.5.1 — Pre-flight legacy audit (`62f5042`)

**What shipped.** `tests/test_legacy_audit_clean.py`. Uses `git grep -nE` over a word-bounded regex (`\bDirector\b|\bdirector_json\b|...`) across `*.py` + `*.json`. Each hit must (a) be on the `EXCLUDED_PATHS` allowlist, or (b) contain a forensic-marker substring (`legacy`, `deleted`, `removed in`, `after the deletion`, `voice-path-cleanbreak`, `voice-path cleanbreak`, `retired`, ...), or (c) be inside a 5-line forensic context window (catches multi-line comment blocks where only the first line carries the marker).

**Deviations from upload-set spec.**

- **Scope reduced to `*.py` + `*.json`.** The plan's grep included `*.md`, but historical migration docs (`docs/*.md`, `ROADMAP.md`, `ROADMAP_HISTORY.md`, `BUG_LOG.md`, `README.md`) are by design forensic narrative. Including them would have surfaced ~250 spurious hits. README + reference-fixture README cleanup is tracked separately as **S23.10 (deferred next batch)**.
- **Word-boundary anchors on every pattern token.** Plan-spec used `Director` unanchored; that matched `Directory`, `TemporaryDirectory()`, "Directories to scan", "help='Directory containing...'", etc. Word-bounded form (`\bDirector\b`) cuts ~25 false positives.
- **`_CONTEXT_WINDOW = 5` lookback.** Plan-spec used line-only substring match. That trips on multi-line forensic comment blocks where the marker word lands on line 1 and lines 2-5 elaborate (e.g. the freeze-cascade history comments in `scene_sequencer.py:572-582`).
- **Forensic markers extended.** Added `"post-cleanbreak"`, `"voice-path-cleanbreak"`, `"voice-path cleanbreak"` (space variant), `"pre-cleanbreak"`, `"retired"` -- so sprint-citation phrasings pass without manual re-grooming.

**Open round-robin question.** Is the context-window lookback the right primitive, or should it be a structural marker like "inside a class whose name ends in `_LEGACY_TOKENS`"? The lookback is heuristic; a future refactor that moves a marker line out of a block silently breaks downstream lines in the same block.

---

### S16.1 — Workflow JSON widget scrub (`a261a7f`)

**Three line edits** to `workflows/otr_scifi_16gb_full.json`:

- L479: `widget.name "production_plan_json"` → `"script_json"`
- L596: `title "Video Plan (wired from Director)"` → `"Video Plan (from FreezeCascade)"`
- L603: `widget.name "director_json"` → `"script_json"`

**Regression.** New `tests/test_workflow_director_freedom.py::test_no_director_surfaces_in_production_workflow` -- regex scan over `node.title`, `widget.name`, `properties["Node name for S&R"]` for every node. Catches the broader substring-in-title case that the validator's exact-match check 5 (S16.2) doesn't.

---

### S16.2 + S16.3 + S16.4 + S16.5 + S16.6 — Validator hardening batch (`6c7e784`)

Five sub-tasks bundled because S16.2/3/5 all touch `nodes/_workflow_validation.py`, S16.4 patches the workflow JSON, and S16.6 is the cumulative gate.

**S16.2 (IMP-18).** Check 5 (forbidden input sockets) extended from scanning only `inp.get("name")` to scanning four surfaces: `inp.name`, `inp.widget.name`, `node.title`, `properties["Node name for S&R"]`. Uses the same exact-match semantics against `FORBIDDEN_INPUT_SOCKETS`. The original narrow check missed three production cases in S16.1.

**S16.3 plan deviation (IMP-14).** Plan-spec said "positional widget walk + raise on slot None OR slot == ''". Empty string was the intended trip for the FluxPortrait.ledger_json case. **The empty-string rule was wrong for ComfyUI semantics.** `INPUT_TYPES.required` routinely declares `("STRING", {"default": ""})` for fields like `OTR_LedgerScriptWriter.episode_title` where blank means "auto-derive at runtime from news headline + style". Failing on bare "" broke 8 tests in `test_workflow_contract_validation.py` and the canonical-workflow assertion. The shipped check **only raises on explicit None** (Comfy's "dropped socket" marker) or `widgets_values` exhaustion (more required-unwired inputs than slots). Pinned the deviation with `test_check3_empty_string_widget_passes` as a regression.

**S16.4 (IMP-21).** `OTR_BatchFluxPortraitRender.ledger_json` wired from `OTR_LedgerFreezeCascade.script_json` slot 1. New link id `114`; `last_link_id` bumped 113 → 114; FreezeCascade's `script_json` output `links` array extends `[2, 12, 16, 19, 24, 21, 113]` → `[..., 114]`. **Side effect:** the writeback used `json.dumps(indent=2)` so the entire workflow JSON file's whitespace reformatted in lockstep. Data structure preserved; diff is mostly whitespace noise.

**S16.5 (IMP-22, IMP-23).** Link-tuple threshold `len(L) < 5` → `len(L) < 6` per actual ComfyUI link shape `[link_id, src_node, src_slot, dst_node, dst_slot, type]`. Duplicate-ID accumulator replaced with `collections.Counter` so an ID appearing 3 times reports as `[42]`, not `[42, 42]`.

**S16.6 deviation.** Plan-spec test called `validate_workflow_contract(..., strict_unknown_types=True)`. The bare test env can't import several optional-dep OTR classes (HuMo, LTX, RTXUpscale, etc.); their `INPUT_TYPES` doesn't register and strict mode fails on the type-existence check. Shipped test uses **default mode** (`strict_unknown_types=False`); strict mode is exercised at the production loader path (S14.2.1, **DEFERRED -- see §9**) where every class loads. Documented in the test's docstring.

**Open round-robin question.** Should S16.4's `json.dumps(indent=2)` writeback be replaced with a surgical link-by-link patch that preserves the file's original whitespace? Trade-off: surgical = small diff, harder to maintain. `json.dumps` = clean canonical format going forward, but every future hand-edit re-formats.

---

### S17.1 + S17.2 + S17.3 + S17.4 — Cache + render integrity (`5c49d20`)

**S17.1 (IMP-11).** MusicGen cache key uplifted to AudioGen's S12.3 standard:
- Signature keyword-only (was positional).
- Hash truncation `[:8]` → `[:12]` for collision parity.
- JSON-canonical payload via `json.dumps(..., sort_keys=True, separators=(",", ":"))` replaces the f-string pipe-delimited form. Float fields canonicalized as `f"{x:.3f}"` (duration) and `f"{x:.2f}"` (guidance_scale).
- `model_id` + `guidance_scale` now hashed.
- Legacy `_cache_key` back-compat wrapper **deleted** per directive 11. Grep confirmed zero external consumers.

**S17.2 (IMP-19).** AudioGen ImportError on `transformers/AudioGen` now raises `RuntimeError` by default (Directive 1: silent silence is a contract breach). New optional widget `allow_silence_fallback: BOOLEAN = False` opts into the old silence-render path; the fallback honors per-cue `render_queue[idx]["duration"]` instead of the bug-source `default_duration` constant. The fallback path no longer stamps `render_results[idx]["sfx_render_status"]` because that variable name was speculative in the plan -- the actual code uses a downstream writeback block. The fallback still emits silence + logs `WARNING: AudioGen import failed; allow_silence_fallback=True -> silence`.

**S17.3 (IMP-10, IMP-12, IMP-12b).** Drift-guard pins. New `tests/test_musicgen_cache_keys.py` (9 tests covering length, dimension changes, float-canonical pinning, positional-call TypeError, format pin, episode_seed coercion). AudioGen's existing `test_audiogen_cache_keys.py` was already pinned in S12.4; regression checked clean.

**S17.4 (IMP-17).** `episode_seed = str(episode_seed) if episode_seed is not None else ""` at the public-node boundary of both AudioGen and MusicGen. Defensive against future callers passing a dict / int. Cache-key path `str()`s the value internally, but coercion at the boundary makes the contract explicit.

**Open round-robin question.** AudioGen still has its `_cache_key` back-compat wrapper (deleted only from MusicGen this batch). The wrapper has tests asserting its presence (`test_audiogen_cache_keys.py::test_audiogen_cache_key_alias_matches_filename_for_write`). Should AudioGen's wrapper also go in a follow-up "legacy alias prune" sprint, or is the consistency cost not worth the small surface?

---

### S18.1 + S18.2 + S18.3 + S18.4 — Post-freeze writeback audit (`1ddad72`)

**S18.1 (IMP-20 part 1).** `nodes/batch_procedural_sfx.py` -- `wav_path` initializer changed from `Optional[str] = None` to `str = ""`. Writeback dict coerces `r.get("wav_path") or ""` so any accidental None upstream still lands `""` on the ledger row. Matches the §6.16 convention enforced by `_otr_ledger_freeze.py` at freeze time but historically not re-validated post-consumer.

**S18.2 (IMP-20 part 2).** New `audit_post_freeze_writeback(ledger, *, strict=False)` in `nodes/_otr_ledger_consumers.py`. Walks `ledger.lines[]` checking 10 known optional-string fields (`sfx_wav_path`, `sfx_engine`, `sfx_type`, `audio_wav_path`, `audio_cache_key`, `music_wav_path`, `music_cache_key`, `video_clip_path`, `tts_skip_reason`, `sfx_render_status`); returns a list of violations or raises `ValueError` if `strict=True`. Soft-rollout API: consumers can log violations to `batch_log` without halting; flip to `strict=True` per consumer once the audit holds clean for two full pipeline runs.

**S18.3 (IMP-20 part 3).** New optional widget `strict_writeback: BOOLEAN = False` on ProcSFX. Default behavior unchanged (log-and-continue); strict mode raises `RuntimeError` on ledger writeback failures. Pattern is the template for extension to AudioGen / MusicGen / SignalLostVideo in a follow-up sprint once the audit walker proves out.

**S18.4.** ProcSFX `render_results` now stamps `sfx_render_status` with `"ok"` (happy path) or `"error"` (disk-write failure). Writeback dict carries it onto the ledger row. Field added to `_OPTIONAL_STRING_FIELDS` so S18.2's walker audits it. AudioGen S17.2 stamps `"fallback_silence"` separately.

**Open round-robin question.** Should the audit walker also check `sfx_render_status` is in a known-good enum (`{"ok", "error", "fallback_silence", "fallback_default_type", "skipped"}`) rather than just "not None"? The current shape allows a typo to land as a "valid" status. Trade-off: enum check = stricter contract, adds maintenance when new states land.

---

### S21.1 + S21.2 — VRAM thresholds + context cap (`6d08f63`)

**S21.1 (IMP-25).** `nodes/story_orchestrator.py` line ~2325: `total_vram >= 15.0` lowered to `>= 14.5`. RTX 5080 Laptop reports ~14.7 GiB after OS/driver reservations on a nominally 16GB card; the prior cutoff missed this class and let bitsandbytes silently fragment. The else branch now logs `device_map=auto path (total_vram=N.NN GiB < 14.5 GiB)` so future tuning has data.

**S21.2 (IMP-27).** `_MODEL_CONTEXT_CAPS` -- Gemma 4 E2B/E4B aligned 16384 → 8192. The asymmetric 16K caps added 2-3 GiB dynamic VRAM during long generation right when audio models want the envelope.

**Drift guards.** `tests/test_story_orchestrator_vram_calibration.py` AST-walks the file. Pinned: literal `14.5` in a comparison with `total_vram`, every entry in `_MODEL_CONTEXT_CAPS` is 8192, fallback default still 8192.

**Deferred sub-tasks.**

- **S21.3** (workflow preset split into `_16gb_aggressive.json` + `_8gb_safe.json`) **conflicts directly with standing memory rule `feedback_minimum_json_files`** ("keep workflow JSONs to minimum, don't create variants"). Skipped pending explicit opt-in from Jeffrey.
- **S21.4** (LTX prompt clamp 300 → 225 chars) -- plan was speculative (`[VERIFY: find the LTX prompt build site]`). The repo's actual LTX prompt flow is `_build_ltx_role_prompt()` in `batch_ltx_render.py:404` which returns the fixed `_PROMPT_BY_ROLE` dict entry verbatim. No 300-char slice exists. N/A.

---

### S22.1 + S22.2 — LLM timeout workflow pause (`b4a3098`)

**S22.1 (IMP-26).** New exception class `_LLMTimeoutWorkflowPause(_LLMTimeout)` in `story_orchestrator.py`. `_run_with_timeout` raises the subclass instead of the base class on timeout. Existing `except _LLMTimeout` handlers still match (subclass); new consumers can branch on the more specific type.

Solves the **LLM → visual case**. The prior cache invalidation (BUG-LOCAL-111) handled LLM → LLM by forcing a fresh load on the next LLM call. But the next stage in the production workflow isn't always LLM -- it can be FLUX / LTX / HuMo, which would race the orphan LLM worker's still-running CUDA kernels and trip `cudaErrorIllegalAddress`. Raising `_LLMTimeoutWorkflowPause` halts the queue at the node boundary so the orphan can finish naturally with nothing else touching the GPU.

**S22.2.** New `docs/manual-smoke-tests.md` with the executable contract for the queue-halt assertion. Steps: lower the writer's timeout to 1s, queue the workflow, confirm stack trace shows the subclass name (NOT the base class), confirm FluxPortrait does NOT execute, confirm no `cudaErrorIllegalAddress`. Failure-mode triage table included for the three observable failure shapes.

`tests/test_llm_timeout_workflow_pause.py` has 3 unit-level pins (subclass-of-base, raises on timeout, message includes "orphan" + "Re-run") plus a `@pytest.mark.skip` placeholder for the integration test.

**Open round-robin question.** The class docstring says: "ComfyUI's node-execution layer surfaces uncaught exceptions as queue halts. Stable since the 2025 unified-execution refactor; if a future ComfyUI version swallows the exception, this assumption needs revisiting." This is a load-bearing assumption with no test. Worth adding a version-pin check or a smoke test that exercises the actual ComfyUI queue path?

---

### S23 — Legacy scrub Director removal (`b443f46`)

Plan-spec was 5 sub-tasks (S23.1-S23.5); audit-expanded to 9. Single bundled commit because the audit walker in S15.5.1 is the cumulative gate.

**S23.1 -- Plan.** Removed `director_raw_dump_dir` repo-wide:
- `nodes/_otr_paths.py:359` function definition + L361 `__all__` entry deleted. Module docstring at L175 scrubbed of "LLMDirector raw-output dump" mention (the helper that wrote those dumps is gone).
- `nodes/story_orchestrator.py:53` import deleted.
- New `tests/test_director_helpers_removed.py` was supplanted by the broader `test_legacy_audit_clean.py` walker.

**S23.2 -- Plan.** Rewrote `story_orchestrator.py` module docstring. "Script Writer + Director" → "Script Writer + Ledger Writer". Director-class-deleted forensic anchor preserved at line 19.

**S23.3 -- Plan.** Scrubbed Director from orchestrator active-code comments at lines 582 (writer never re-assigns broken voice), 690-691 (writer start / writer locks cast), 3706 (next phase inherits model memory). Originally lines 536/644/645/3639 in the plan -- shifted because of earlier S23.1 + S23.2 edits.

**S23.4 -- Plan.** Rewrote `scene_sequencer.py` module docstring. "Gemma Director voice_map dispatch" → "voice_assignments dispatch from the ledger's cast block".

**S23.5 -- Plan.** Rewrote `scene_sequencer.py` L258 active-code comment to refer to the ledger's cast block.

**S23.6 -- AUDIT NEW (HIGH).** Deleted `production_plan_or_empty(plan_json)` helper from `nodes/_otr_ledger_consumers.py:147`. The function parsed an optional Director-shape `production_plan_json` string and returned `{}` for empty/None/invalid input. Zero production callers (verified via grep across `nodes/`, `scripts/`, `visual/`, `__init__.py`). Directive 11 violation. Deleted the function + `__all__` entry + the helper-list mention in the module docstring. `TestProductionPlanOrEmpty` (9 tests) deleted from `tests/test_otr_ledger_consumers.py` in lockstep.

**S23.7 -- AUDIT NEW (HIGH).** Deleted live `production_plan_json` socket from `visual/bridge.py`. The OTR_VisualBridge node declared this as an optional STRING input in INPUT_TYPES, the execute() signature accepted it, and the body wrote the value to `<job_dir>/production_plan.json`. Grep across sidecar / visual worker confirmed NO downstream consumer read the file -- the bridge wrote it for an audience that didn't exist. Removed the INPUT_TYPES entry, the kwarg, and the `atomic_write_text(production_plan.json)` call. Module + class docstrings rewritten.

**S23.8 -- AUDIT NEW (HIGH).** Deleted 7 stale test/script files:
- `tests/test_director_cast_namespace_merge.py` (tests dead Director merge logic for BUG-LOCAL-068)
- `tests/test_p0_features.py` (tests deleted `_validate_director_plan` -- "Exact copy of LLMDirector._validate_director_plan logic")
- `scripts/_visual_full_pipeline_test.py` (instantiates `OTR_Gemma4Director` which was deleted)
- `scripts/phase_b_smoketest.py` (uses retired `production_plan_json` socket)
- `scripts/smoke_visual_e2e.py` + `.bat` (uses retired socket)
- `scripts/lfc_wiring_smoke.py` (references deleted classes)
- `tests/test_lfc_wiring_smoke_script.py` (tests the deleted `lfc_wiring_smoke.py`)

`test_unload_synchronize_guard.py` was NOT deleted -- it tests a live regression guard for `_unload_llm`; the Director mention is historical context ("OpenClose -> Director transition" log line from 2026-04-26).

**S23.9 -- AUDIT NEW (MEDIUM).** Added forensic markers to retained guardrail tests + active-code comments where the legacy reference is genuinely documentation. Touched 18 files in total (see commit body for the full list).

**S23.10 -- DEFERRED.** README.md (11 hits) + `tests/fixtures/reference_episode/README.md` (8 hits) rewrite. Substantial doc effort; audit-clean test scoped to `*.py` + `*.json` so README cleanup tracked independently. README still describes "Story → Director → SceneSequencer" as current architecture at L34, L196-197, L291, L333, L450, L548, L623.

**Open round-robin question.** S23.6 + S23.7 surfaced two orphan surfaces the original cleanbreak plan missed (deletions waves scoped to one subsystem at a time, leaving sidecar-isolated debris). The mitigation -- the S15.5.1 audit walker -- is now in place, but it only catches Director-era surfaces. Are there other legacy generations (LFC, parser-list) that warrant the same kind of repo-wide gate? `parser_list` / `parser-list` are already in the audit regex; LFC isn't.

---

### S19.1 + S19.2 — Hook hardening + doc-freshness (`32f62eb`)

**S19.1 (IMP-24).** `tests/conftest.py::pytest_sessionfinish` extended from reading only `rep_call.failed` to iterating `("setup", "call", "teardown")` via `getattr` loop. Tests that errored in setup or teardown were leaking past the diff as silent-passes pre-S19.1. `EXPECTED_FAILED_NODEIDS` source-of-truth unchanged; the detection logic broadens.

**S19.2 (IMP-15, IMP-16).** `tests/test_naming_conventions.py` extended with `test_conventions_doc_lists_every_lib_module`. Scans `nodes/_otr_*_lib.py` (strict) and asserts every match stem appears somewhere in `docs/conventions.md`. As of 2026-05-13 there are 2 such modules (`_otr_bark_lib.py`, `_otr_sfx_lib.py`); both are listed.

**S19.2 doc clarification (commit `bed3c4a`).** The conventions doc's "Current modules" table listed 5 entries under a section header that documents only `_otr_*_lib.py`. Only 2 of the 5 carry the strict `_lib` suffix; the other 3 (`_otr_casting.py`, `_otr_ledger_consumers.py`, `_otr_ledger_freeze.py`) predate the suffix convention. Added a note explaining the grandfathering -- the test stays narrow; new private library modules going forward should adopt the suffix.

**S19.3 -- DEFERRED.** Survival-guide promotion gated on 2-3 clean sprints of S15.3 use. Only 1 has passed (S15.3 landed `f813b37` on 2026-05-12). New `docs/known-failures-promotion-pending.md` documents the gate state and unblock procedure; the sibling-repo commit is intentionally NOT made from this repo's automation.

---

## 2. Plan deviation summary

| Sub-task | Plan-spec | Actual disposition |
|---|---|---|
| S15.5.1 scope | grep `*.py *.json *.md` | `*.py *.json` only; *.md docs deferred to S23.10 |
| S15.5.1 pattern | unanchored `Director` | word-bounded `\bDirector\b` (kills 25 false positives) |
| S15.5.1 matcher | per-line substring | per-line + 5-line context window |
| S16.3 widget-drift | "" counts as drift | None-only counts as drift (ComfyUI `default: ""` semantics) |
| S16.6 strict mode | `strict_unknown_types=True` | default mode (bare test env can't import all optional-dep classes) |
| S17.2 fallback stamp | `render_results[idx]["sfx_render_status"]` | dropped (variable name was speculative; downstream block handles it) |
| S21.3 preset split | rename + sibling JSON | deferred (conflicts with no-variants memory rule) |
| S21.4 LTX prompt clamp | find `[:300]` slice | N/A (no such slice exists in repo) |
| S23 sub-task count | 5 (S23.1-5) | 9 (audit added 4: S23.6-9) plus S23.10 deferred |
| S14.2 auto-invoke | wire after `json.loads(workflow_str)` | indefinitely deferred (no central loader exists) |
| S19.3 promotion | this batch | deferred (1/3 sprint cycles complete) |
| S20 stretch | optional | skipped (non-blocking per plan) |

---

## 3. S15.5.1 audit findings (8 NEW surfaces beyond the plan)

The plan anticipated 5 REMOVE entries (S23.1-S23.5). The audit found 4 additional NEW work items + 1 follow-on:

| Severity | Item | Description | Resolution |
|---|---|---|---|
| HIGH | S23.6 | `production_plan_or_empty` orphan helper in `_otr_ledger_consumers.py` | Deleted; zero production callers verified |
| HIGH | S23.7 | Live `production_plan_json` socket on `visual/bridge.py` | Deleted; zero sidecar consumers |
| HIGH | S23.8 | 7 stale test/script files testing or using deleted classes | Deleted |
| MEDIUM | S23.9 | 18 files with active-code comments / docstrings needing forensic markers | Reworded |
| LOW | S23.10 | README + reference fixture README describe Director as current | **DEFERRED next batch** |

Two of these (S23.6, S23.7) are logged as **Bible candidates** -- `BUG-LOCAL-207` and `BUG-LOCAL-208`. See §6.

---

## 4. Test inventory (the +25 net new tests)

| File | Tests | Sprint |
|---|--:|---|
| `tests/test_legacy_audit_clean.py` | 1 | S15.5.1 |
| `tests/test_workflow_director_freedom.py` | 1 | S16.1 |
| `tests/test_workflow_validator_extended.py` | 12 | S16.2 / S16.3 / S16.5 |
| `tests/test_workflow_flux_portrait_wiring.py` | 2 | S16.4 |
| `tests/test_workflow_live_passes_validator.py` | 1 | S16.6 |
| `tests/test_musicgen_cache_keys.py` | 9 | S17.1 / S17.3 / S17.4 |
| `tests/test_audiogen_strict_failure.py` | 5 | S17.2 |
| `tests/test_post_freeze_writeback_audit.py` | 8 | S18.2 |
| `tests/test_procsfx_writeback_convention.py` | 6 | S18.1 / S18.3 / S18.4 |
| `tests/test_story_orchestrator_vram_calibration.py` | 3 | S21.1 / S21.2 |
| `tests/test_llm_timeout_workflow_pause.py` | 3 + 1 skip | S22.1 / S22.2 |
| `tests/test_known_failures_hook_phases.py` | 3 | S19.1 |
| `tests/test_naming_conventions.py` (extended) | +1 | S19.2 |
| `tests/test_otr_ledger_consumers.py` (delta) | -9 | S23.6 (TestProductionPlanOrEmpty deleted) |
| `tests/test_audiogen_cache_keys.py` (no change; existing) | 16 | -- |

Net: `+54 new` − `9 deleted` − `~20 from deleted whole-file tests` = +25 (matches the regression delta).

---

## 5. Drift-guard inventory

| Contract | Pinned by |
|---|---|
| Workflow JSON has no Director-era widget names | `test_workflow_director_freedom.py` |
| Validator scans 4 surfaces for FORBIDDEN names | `test_check5_catches_widget_name/_title/_s_and_r` |
| Widget-drift positional + None-only | `test_check3_*` (4 tests) |
| FluxPortrait.ledger_json wired to FreezeCascade | `test_flux_portrait_ledger_wired` |
| Link tuple is 6 elements | `test_link_tuple_requires_six_elements` |
| Duplicate link IDs dedup | `test_duplicate_link_id_reported_once_per_id` |
| MusicGen cache key includes model_id + guidance | `test_cache_prefix_changes_on_*` |
| MusicGen cache prefix is 12 hex chars | `test_cache_prefix_length_is_12` |
| MusicGen cache prefix is keyword-only | `test_cache_prefix_keyword_only_signature` |
| AudioGen strict failure mode default | `test_strict_failure_raises_runtime_error_on_import_error` |
| AudioGen fallback honors per-cue duration | `test_fallback_silence_uses_per_cue_duration_not_default` |
| ProcSFX wav_path is "" not None on failure | `test_wav_path_default_is_empty_string_not_none` |
| ProcSFX stamps `sfx_render_status` | `test_sfx_render_status_in_writeback` |
| Audit walker covers 10 optional string fields | `test_sfx_render_status_in_audited_fields` |
| Flagship VRAM threshold = 14.5 | `test_flagship_vram_threshold_is_14_5` |
| Every `_MODEL_CONTEXT_CAPS` entry = 8192 | `test_all_model_context_caps_at_8192` |
| `_LLMTimeoutWorkflowPause` subclasses `_LLMTimeout` | `test_pause_is_subclass_of_llm_timeout` |
| Known-failures hook tracks all 3 phases | `test_hook_iterates_all_three_phases` |
| Every `_otr_*_lib.py` is in `conventions.md` | `test_conventions_doc_lists_every_lib_module` |
| Repo has no unclassified legacy Director references | `test_no_unclassified_legacy_references` |

---

## 6. Bibliographically promotable bugs (new this batch)

### BUG-LOCAL-207 -- `production_plan_or_empty` orphan Director-derived fallback

**General lesson.** A "graceful fallback" surface introduced for a now-deleted upstream consumer is dead weight that lulls future contributors into thinking the upstream is still alive. Audit fallbacks tied to deleted upstreams in the same commit that deletes the upstream; or run a periodic "no production callers" sweep on helpers whose docstring mentions a known-deleted class.

### BUG-LOCAL-208 -- `visual/bridge.py` carried a live `production_plan_json` socket

**General lesson.** When a deletion wave is scoped to one subsystem, sidecar-isolated subsystems can carry the deletion's debris forward for sprints. A repo-wide audit grep at the END of every cleanbreak (not just inside the affected subsystem) catches this class of survival.

Both `Bible candidate: yes` pending promotion after v2.0 ships, per `feedback_roadmap_buglog_live_docs`.

---

## 7. Sight improvements (IMP-* candidates for the next round-robin)

| # | Severity | Item | Location | Rationale |
|---|---|---|---|---|
| IMP-31 | LOW | AudioGen `_cache_key` back-compat alias still present | `batch_audiogen_generator.py` | MusicGen's was deleted in S17.1 (zero callers). AudioGen's has tests asserting its presence. Symmetry would say delete AudioGen's too + drop the test assertion. Cost: small. Benefit: standing directive 11 consistency. |
| IMP-32 | MEDIUM | `audit_post_freeze_writeback` enum-check vs not-None-only | `_otr_ledger_consumers.py` | The walker accepts any string for `sfx_render_status`. A typo could land as "valid". Enum check (`{"ok", "error", "fallback_silence", "fallback_default_type", "skipped"}`) would tighten the contract. |
| IMP-33 | MEDIUM | S22.1 assumption test: ComfyUI surfaces uncaught exceptions as queue halts | (no current test) | The class docstring acknowledges this is a load-bearing assumption with no automated check. A ComfyUI-version-pin or smoke test would close the gap. |
| IMP-34 | LOW | S15.5.1 audit context window is heuristic | `test_legacy_audit_clean.py` | Refactor that moves a marker line out of a forensic block silently breaks downstream lines in the same block. A structural marker (class / frozenset name suffix) would be more robust but slower to write per call site. |
| IMP-35 | LOW | S16.4 writeback uses `json.dumps(indent=2)`; every future hand-edit reformats | `workflows/otr_scifi_16gb_full.json` | Surgical link-by-link patches would preserve whitespace at the cost of more complex writeback code. |
| IMP-36 | MEDIUM | S23.10 README rewrite still pending | `README.md`, `tests/fixtures/reference_episode/README.md` | New contributors land on the README and see Director described as current. Audit-clean test is scoped to `*.py + *.json` so this doesn't fail CI; needs explicit sprint. |
| IMP-37 | MEDIUM | Other legacy generations (LFC, parser-list partial) lack a repo-wide audit gate | (audit pattern) | `parser_list` / `parser-list` already in the audit regex, LFC isn't. Adding `LFC` would surface ~XX hits; worth a separate scoping pass. |
| IMP-38 | LOW | Audit-test EXCLUDED_PATHS could grow; needs a doc comment per addition | `tests/test_legacy_audit_clean.py` | Currently 4 paths excluded with inline comments. Adding a 5th should be a deliberate decision with a one-line justification next to it. |

---

## 8. Round-robin questions (in priority order)

The questions in §1 collected:

1. **Audit-test context-window heuristic vs structural marker** (S15.5.1) -- is the 5-line lookback right, or should the validator look for structural anchors (class-suffix, frozenset names) instead?
2. **S16.4 surgical patch vs `json.dumps` rewrite** -- trade-off between small workflow JSON diffs and clean canonical format.
3. **AudioGen `_cache_key` alias consistency** (S17.1) -- delete it too, or keep for the legacy test surface?
4. **Audit walker enum check on `sfx_render_status`** (S18.4) -- tighten the contract or accept the looser shape?
5. **S22.1 ComfyUI uncaught-exception assumption** -- needs a version-pin test or a smoke test?
6. **Other legacy generations to audit** (LFC, others) -- worth a separate scoping pass?

---

## 9. Deferred items (status pinned)

| Item | Status | Gate / unblock condition |
|---|---|---|
| S14.2 -- validator auto-invoke | INDEFINITELY DEFERRED | ComfyUI has no central Python-side workflow loader; needs design call between frontend extension OR opt-in `OTR_WorkflowValidator` first-node. Both are their own sprint. See `docs/cleanbreak-deferred.md`. |
| S19.3 -- survival-guide promotion | DEFERRED 1/3 sprints | Needs 2-3 clean sprints of S15.3 use. S15.3 landed `f813b37` 2026-05-12. See `docs/known-failures-promotion-pending.md`. |
| S21.3 -- workflow preset split | DEFERRED (rule conflict) | Conflicts with `feedback_minimum_json_files` standing memory rule. Reopens when Jeffrey explicitly opts in. |
| S23.10 -- README + reference README rewrite | DEFERRED next batch | Audit-clean test scoped to `*.py + *.json`; cleanup tracked independently. README still has 11 active Director references at L34/L196-197/L291/L333/L450/L548/L623. |
| S20 -- stretch tasks | SKIPPED | Marked non-blocking by the plan. |

---

## 10. Acceptance state for batch closure

All gates green:

- [x] `tests/test_legacy_audit_clean.py` -- `1 passed`
- [x] `tests/test_workflow_live_passes_validator.py` -- `1 passed`
- [x] `tests/test_naming_conventions.py` -- `3 passed`
- [x] Bug Bible regression -- `23 passed / 1 skipped / 2 xfailed` (baseline held)
- [x] Full pytest run -- `2108 passed / 7 skipped / 6 known-fail` (exact match to `EXPECTED_FAILED_NODEIDS`)
- [x] Local HEAD == origin HEAD (`bed3c4a4fa58545fc8755b91a1f2fa358a07e83f`)
- [x] No 0-byte tracked Python files
- [x] No BOM-prefixed tracked Python files
- [x] ROADMAP + BUG_LOG live-updated

**Voice-path-cleanbreak is LOCKED at `bed3c4a`.**
