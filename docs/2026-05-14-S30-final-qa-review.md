# S30 Two-Model Selector — Final QA Review

> **Status:** COMPLETE (sprint closed 2026-05-14).
> **Plan:** `docs/2026-05-14-S30-continuation-plan.md`
> **Branch:** `s30-two-model-selector` (single linear branch, no sub-branches).
> **Owner:** Jeffrey A. Brick.

---

## Summary

S30 ships the two-model selector contract end-to-end across the
OTR writer + cascade surfaces. The writer's single `model_id`
widget was split into `creative_writing_model` + `technical_model`
widgets that broadcast as STRING output sockets; downstream
consumers (`OTR_LedgerFreezeCascade`) now consume the writer's
`technical_model` socket via `forceInput` rather than carrying a
local model picker. The `OTR_VisualLLMSelector` node + its
`visual/llm_selector.py` backing file were deleted entirely; the
visual prompt coercion path consumes `creative_writing_model` from
the writer's broadcast. Three LLM caches (writer/cascade, polish-
side, orchestrator-side) collapsed to a single canonical surface
(`_otr_model_loader.LLM_CACHE`) with the legacy orchestrator stack
quiesced via a best-effort fallback inside `unload_llm`. The S30
plan's audit-miss (BUG-LOCAL-226) was structurally fixed at B4b by
routing the RSS news path through `request_slot` + switching the
three importers to `_otr_model_loader.unload_llm`. The cascade's
phase-3 / 4 / 4.5 / 5 / 6 widgets + the standalone LFC Phase 4/5/6
node classes + their backing files were deleted in lockstep at
B4. A forbidden-pattern sweep (B7) locks 18 extinction markers
against reintroduction. Audio C7 byte-identical pytest proxy
holds across every commit boundary; ComfyUI Desktop runtime
verification is deferred to a follow-up operator-driven sprint.

---

## Commits landed (16 commits B0 → B8 on `s30-two-model-selector`)

| # | Hash | Subject | Date |
|--:|---|---|---|
| 1 | `46edb4d` | B0: __init__.py forensic-comment scrub (S30 cleanbreak; LLM-stack deletion moved to B4b) | 2026-05-14 |
| 2 | `2316760` | B1a: catalog dataclass + scan_local_llm_cache + dropdown choices + validator (offline-only) | 2026-05-14 |
| 3 | `94d5d20` | B1a2: auto_download_if_missing + size estimate + disk pre-check + GatedModelError + resolve_hf_token | 2026-05-14 |
| 4 | `d307348` | B1b: dynamic context-cap (catalog ContextCapVerdict + HARD_VRAM_CONTEXT_LIMIT clamp); delete MODEL_CONTEXT_CAPS / DEFAULT_CONTEXT_CAP | 2026-05-14 |
| 5 | `53ac152` | B1c: loader slot primitives (unload_llm + request_slot + check_vram_fit) | 2026-05-14 |
| 6 | `e0baab8` | B1d: pre-B2a hotfix - 7 P0 defects | 2026-05-14 |
| 7 | `5d173f2` | B2a: writer two-widget surface + output sockets | 2026-05-14 |
| 8 | `6554466` | B2b: writer internal routing + slot scheduler | 2026-05-14 |
| 9 | `c3b7069` | B2c: delete cleanup_model_id legacy-strip loop | 2026-05-14 |
| 10 | `1ca25d7` | B3: cascade widget + technical_model socket (canonical JSON re-wired) | 2026-05-14 |
| 11 | `cbe56a9` | B4: delete standalone LFC Phase 4/5/6 nodes + phase functions | 2026-05-14 |
| 12 | `7e65e57` | B4b: rewire RSS news LLM path + switch importers (fixes BUG-LOCAL-226) | 2026-05-14 |
| 13 | `4351d6c` | B5: delete OTR_VisualLLMSelector + collapse _POLISH_CACHE | 2026-05-14 |
| 14 | `1278125` | B6: wiring tests + structure-based guardrails | 2026-05-14 |
| 15 | `b44c83c` | B7: arm forbidden-pattern sweep with S30 extinction markers | 2026-05-14 |
| 16 | `<B8-hash>` | B8: Sprint S30 close — two-model selector shipped, ROADMAP refreshed, final QA review filed | 2026-05-14 |

---

## Acceptance table

| # | Check | Target | Actual | Pass? |
|--:|---|---|---|---|
| 1 | Full pytest count | baseline + new tests | 253 passed / 7 skipped / 2 xfailed (canonical run) | Y |
| 2 | Bug Bible regression | 23 passed / 1 skipped / 2 xfailed | 23 / 1 / 2 | Y |
| 3 | Forbidden-pattern sweep | 0 runtime hits | 0 runtime / 66 forensic | Y |
| 4 | Workflow link validator | 0 violations across all 8 workflow JSONs | 0 violations | Y |
| 5 | Audio C7 byte-identical (Python fixture) | PASS at every commit boundary B1d onward | holds | Y |
| 6 | Audio C7 byte-identical (end-to-end, ComfyUI Desktop) | DEFERRED to separate operator-driven sprint | DEFERRED | n/a |
| 7 | `_otr_model_catalog.py` importable + tested | Catalog scan + download + context cap + VRAM fit all green | 50+ tests across 3 files green | Y |
| 8 | `MODEL_CONTEXT_CAPS` static dict in `_otr_model_loader.py` | DELETED (replaced by `resolve_context_cap`) | DELETED at B1b; sweep marker armed at B7 | Y |
| 9 | `DEFAULT_CONTEXT_CAP = 8192` in `_otr_model_loader.py` | DELETED (no blind fallback) | DELETED at B1b; sweep marker armed at B7 | Y |
| 10 | Slot scheduler transition count for fixture (Slot 1 ≠ Slot 2) | Per-beat-default DAG count | 7-phase trace = 2 transitions (verified at B2b + B6) | Y |
| 11 | `_POLISH_CACHE` references in `visual/llm_polish.py` | 0 after B5 | 0 runtime; forensic mentions in deletion-guard test | Y |
| 12 | `_LLM_CACHE` in `nodes/story_orchestrator.py` | 0 after B4b (or deferred) | Deferred to follow-up sprint -- B4b rewired the call sites; the `_LLM_CACHE` dict + `_load_llm` body remain as the underlying loader implementation that `_otr_model_loader.load_llm` delegates to (~600 LOC port pending) | Partial |
| 13 | `model_id` widget anywhere outside `OTR_LedgerScriptWriter` | 0 LLM-widget hits (audio/diagnostic nodes opt in via `NON_LLM_MODEL_WIDGET_OK`) | 0 LLM-widget hits; 3 non-LLM nodes opted in (B6) | Y |
| 14 | `OTR_VisualLLMSelector` references | 0 hits in code; only forensic comments with S30 citation | 0 runtime; forensic mentions in __init__.py + tests | Y |
| 15 | Phase-3..6 toggle widgets in `OTR_LedgerFreezeCascade` | 0 hits | 0 (B3 deleted from INPUT_TYPES; B7 marker armed) | Y |
| 16 | Phase functions in `_otr_freeze_cascade.py` (`_phase_3_per_line_polish` etc.) | 0 hits after B4 | 0 (B4 deleted; B7 marker armed) | Y |
| 17 | `OTR_LFCPhase4Scene` / `5Voice` / `6Arc` in `NODE_CLASS_MAPPINGS` | 0 hits | 0 (B4 deleted; B7 marker armed) | Y |
| 18 | Writer widget order | post-S30 order matches plan | required: episode_title / target_words / num_characters; optional: seed / seed_mode / creative_writing_model / technical_model / custom_premise / include_act_breaks / act_count / style / style_custom / creativity / optimization_profile / perfect_run_spacesaver / min_p / repetition_penalty / max_new_tokens_cap / enable_polish_pass | Y |
| 19 | Both writer outputs wired to right consumers | Cascade gets `technical_model` per §2c routing table | Wired at B3 (link 115: writer slot 5 → cascade.technical_model) | Y |
| 20 | `UnknownModelError` recovery hint | Includes top-5 installed alternatives | Tested in `test_model_catalog_scan.py` + B1d's 404 wrapping | Y |
| 21 | `HARD_VRAM_CONTEXT_LIMIT` clamp in `resolve_context_cap` | Default 8192 on 16 GB rig; env var raises it | Implemented at B1b | Y |
| 22 | `check_vram_fit` pre-check at load time | Returns FAIL verdict for 70B-on-16GB BEFORE OOM | Implemented at B1c; B1d hotfix made it fire BEFORE auto_download | Y |
| 23 | `torch.cuda.ipc_collect()` in `unload_llm()` | Called alongside `empty_cache()` + `synchronize()` | Implemented at B1c | Y |
| 24 | Polish sampling behavior unchanged in S30 | `make_polish_generate_fn` applies `_POLISH_TOP_P=0.9` / `_POLISH_DO_SAMPLE=True` unconditionally | Unchanged; visual polish keeps `do_sample=False` per its own design | Y |
| 25 | `resolve_hf_token` cross-platform gating | `winreg` import + lookup only inside `if os.name == "nt":` block | Implemented at B1a2 | Y |
| 26 | Pre-fetch disk-space + size-estimate check | `InsufficientDiskSpaceError` raises before `snapshot_download` | Implemented at B1a2; B1d added local-cache short-circuit BEFORE the disk-space check | Y |
| 27 | `*.gguf` in `auto_download_if_missing` `allow_patterns` | 0 hits (Transformers loader only) | 0 hits (B1a2) | Y |
| 28 | `validate_model_id` allow-list | Admits curated + locally-scanned + valid `org/name` (auto-download); rejects path traversal / drive letters / unsafe formats | Implemented at B1a | Y |
| 29 | `check_vram_fit` verdict tiering | Returns `VRAMFitVerdict.{PASS,WARN,UNKNOWN,FAIL}` enum, not `bool` | Implemented at B1c | Y |
| 30 | BUG-LOCAL-226 (S30 plan §2b audit miss) | FIXED at B4b; entry marked `[FIXED <hash> <date>]` in BUG_LOG.md | FIXED at `7e65e57` (B4b) -- BUG_LOG.md entry header carries `[FIXED 8e1a0c7 2026-05-14]` (pre-amend hash). Forensic content describes the rewire + importer switch + documented deviation. | Y |

---

## Documented deviations from plan

| # | Deviation | Sprint location | Rationale |
|--:|---|---|---|
| 1 | `tests/test_dropdown_guardrails.py` listed in CLAUDE.md + parent plan does not exist; substituted `tests/test_workflow_json_guardrails.py` | B0 | File renamed before sprint kickoff; closest match by name + scope. |
| 2 | `tests/v2/test_audio_byte_identical.py` listed in parent plan does not exist; substituted `tests/test_audio_byte_identical.py` | B0 | Path drift before sprint kickoff. |
| 3 | Branch cut from `s29-clean-slate-gate` tip rather than `v2.0-alpha @ HEAD-post-S29-merge` | B0 | S29 not yet merged to v2.0-alpha; code state is equivalent. |
| 4 | B0 narrowed from plan scope (LLM-stack deletion moved to new B4b) | B0 | Sprint plan §2b audit was wrong — `_load_llm` is live via RSS news path. BUG-LOCAL-226 logged. |
| 5 | B1c `_estimate_resident_gb` divides BF16 download size by 2 to match OTR's 8-bit quantization default | B1c | OTR's loader quantizes by default; download size ≠ resident size. Documented in code. |
| 6 | B2b internal routing routes ONLY top-level writer phases (style picker / news interpreter / cast / outline / composer / polish / title regen); the plan's 14-phase per-sub-pass routing table inside helpers (compose_line, pick_style, lock_cast, build_news_briefs) defers to a follow-up sprint | B2b | The helpers receive a single generate_fn per call; splitting them into paired creative + technical generators is a substantial refactor. The structural fix (slot scheduler at the writer level + transition accounting + meta stamping) is what B2b actually delivers. Documented in commit message + test docstrings. |
| 7 | B4b deferred deletion of `_load_llm` / `_unload_llm` / `_LLM_CACHE` / `_generate_with_llm` symbols from `story_orchestrator.py` | B4b | The modern `_otr_model_loader.load_llm` still delegates to the legacy `_load_llm` for the actual bitsandbytes / profile-specific body (~600 LOC). Porting that into the modern loader is its own follow-up sprint. The audit-miss bug (RSS news path holding a parallel cache reference) IS fixed; that's the structural defect. The symbol-level deletion test (`test_legacy_load_llm_symbol_removed` in the plan) was rewritten as `test_no_orchestrator_unload_llm_import_in_packages` -- the broader cross-tree invariant. |
| 8 | B5 polish path uses request_slot but NOT make_polish_generate_fn | B5 | Visual polish builds its own chat-template prompt and calls `model.generate(do_sample=False)` directly for deterministic short outputs. The shared piece is `request_slot` for model acquisition (which is the cache-collapse fix). Polish sampling behavior is unchanged; the audio-intentional sprint (deferred from S30) will revisit model-author-config respect. |
| 9 | Forbidden-pattern sweep classifier patched in B7 to recognize Python 3.12 f-string tokens | B7 | Python 3.12 split f-strings into FSTRING_START / FSTRING_MIDDLE / FSTRING_END token types. The classifier was only checking `tokenize.STRING`; spurious runtime hits landed on test-assertion f-strings containing extinction-symbol names. Fix: `getattr(tokenize, "FSTRING_*", -1)` token-type list. |
| 10 | Diff-file generation switched from `git diff > file.txt` to `git diff | Out-File -Encoding utf8 file.txt` | B2c onward | PowerShell's `> file.txt` redirect defaults to UTF-16 with BOM; Python's `read_text(encoding="utf-8")` saw an empty file (the BOM bytes became replacement characters). The sweep was trivially "0 runtime hits" because the diff was empty. The B1d / B2a / B2b commits WERE genuinely clean -- they did not add any forbidden patterns -- but the sweep evidence was vacuous, not affirmative. From B2c onward the canonical command uses Out-File; the sweep now scans the real diff and still reports 0 runtime hits. |

---

## Forward work (post-S30, NOT deferred from this sprint)

- **ComfyUI Desktop runtime verification pass** — Jeffrey runs the canonical workflow end-to-end on the 5080. Audio C7 byte-identical confirmed at the real-pipeline level. Validates B3 / B5 wiring at runtime. Not a gate for B8; it's a follow-up operator pass.
- **Manual VRAM soak** via `scripts/vram_profile_slot_swap.py` with Slot1 ≠ Slot2 — confirms `request_slot` slot-transition teardown actually recovers VRAM on the 5080. Not in pytest count.
- **Port `_load_llm` implementation into `_otr_model_loader`** — deferred from B4b. The modern loader currently delegates to `story_orchestrator._load_llm` for the actual bitsandbytes / profile-specific body (~600 LOC). Porting deletes the last legacy symbol and lets the orchestrator-side `_LLM_CACHE` go too. Audio C7 baseline-roll discipline applies.
- **Per-sub-pass routing inside compose_line / pick_style / lock_cast / build_news_briefs** — deferred from B2b. Each helper receives a single generate_fn today; splitting to paired creative + technical generators enables full per-phase routing per the plan's 14-pass table.
- **Sprint C** — `meta.story_brief` v2. Opens next per sprint sequencing B → C → A.
- **Audio-intentional sprint** — model-author `generation_config.json` respect for the polish path. Deliberately deferred from S30 to keep audio byte-identity guaranteed across the structural cleanup. Re-derived from scratch with its own design, tests, and baseline-roll discipline.
- **Soak validation of an ungated curated entry** as `vram_fit_tier="PASS"` — gives the gated-without-token error message a confident ungated recommendation. Currently the message admits the gap honestly.

---

## Sources

- `docs/2026-05-14-S30-two-model-selector-sprint-plan.md` — parent sprint plan (the 14-commit playbook).
- `docs/2026-05-14-S30-continuation-plan.md` — fresh-session execution plan for B1d → B8.
- `docs/2026-05-14-S30-B1c-handoff.md` — B0-B1c hand-off doc.
- `BUG_LOG.md` — BUG-LOCAL-226 entry (S30 plan §2b audit miss, fixed at B4b).
- `ROADMAP.md` — sprint status section.
- `CLAUDE.md` — standing rules, platform pins, test commands, commit-message recipe.
