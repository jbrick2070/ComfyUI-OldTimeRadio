# S30 Two-Model Selector — Final QA Review (TEMPLATE)

> **Status:** TEMPLATE — filled in at B8 sprint close.
> **Plan:** `docs/2026-05-14-S30-continuation-plan.md`
> **Branch:** `s30-two-model-selector` (single linear branch, no sub-branches).
> **Owner:** Jeffrey A. Brick.

Replace every `<…>` placeholder when filling in. Do not delete the placeholders before that — the review is verifiable at a glance only when every row has either a real value or an explicit `<not run yet>`.

---

## Summary

`<one paragraph: what shipped, what's the audio-baseline state, what remaining work exists>`

---

## Commits landed (target: 12 commits B1d → B8 on top of the 5 already in place at B1c)

| # | Hash | Subject | Date |
|--:|---|---|---|
| 1 | `46edb4d` | B0: __init__.py forensic-comment scrub (S30 cleanbreak; LLM-stack deletion moved to B4b) | 2026-05-14 |
| 2 | `2316760` | B1a: catalog dataclass + scan_local_llm_cache + dropdown choices + validator (offline-only) | 2026-05-14 |
| 3 | `94d5d20` | B1a2: auto_download_if_missing + size estimate + disk pre-check + GatedModelError + resolve_hf_token | 2026-05-14 |
| 4 | `d307348` | B1b: dynamic context-cap (catalog ContextCapVerdict + HARD_VRAM_CONTEXT_LIMIT clamp); delete MODEL_CONTEXT_CAPS / DEFAULT_CONTEXT_CAP | 2026-05-14 |
| 5 | `53ac152` | B1c: loader slot primitives (unload_llm + request_slot + check_vram_fit) | 2026-05-14 |
| 6 | `<hash>` | B1d: pre-B2a hotfix — 7 P0 defects | `<date>` |
| 7 | `<hash>` | B2a: writer two-widget surface + output sockets | `<date>` |
| 8 | `<hash>` | B2b: writer internal routing + slot scheduler | `<date>` |
| 9 | `<hash>` | B2c: delete cleanup_model_id legacy-strip loop | `<date>` |
| 10 | `<hash>` | B3: cascade widget + technical_model socket (canonical JSON re-wired) | `<date>` |
| 11 | `<hash>` | B4: delete standalone LFC Phase 4/5/6 nodes + phase functions | `<date>` |
| 12 | `<hash>` | B4b: rewire RSS news LLM path + delete legacy orchestrator LLM stack (fixes BUG-LOCAL-226) | `<date>` |
| 13 | `<hash>` | B5: delete OTR_VisualLLMSelector + collapse _POLISH_CACHE | `<date>` |
| 14 | `<hash>` | B6: wiring tests + structure-based guardrails | `<date>` |
| 15 | `<hash>` | B7: arm forbidden-pattern sweep with S30 extinction markers | `<date>` |
| 16 | `<hash>` | B8: Sprint S30 close — two-model selector shipped, ROADMAP refreshed, final QA review filed | `<date>` |

---

## Acceptance table

| # | Check | Target | Actual | Pass? |
|--:|---|---|---|---|
| 1 | Full pytest count | `<baseline + new tests>` passed / `<N>` skipped / 0 failed | `<actual>` | `<Y/N>` |
| 2 | Bug Bible regression | 23 passed / 1 skipped / 2 xfailed | `<actual>` | `<Y/N>` |
| 3 | Forbidden-pattern sweep | 0 runtime hits | `<actual>` | `<Y/N>` |
| 4 | Workflow link validator | 0 violations across all 8 workflow JSONs | `<actual>` | `<Y/N>` |
| 5 | Audio C7 byte-identical (Python fixture) | PASS at every commit boundary B1d onward | `<actual>` | `<Y/N>` |
| 6 | Audio C7 byte-identical (end-to-end, ComfyUI Desktop) | DEFERRED to separate operator-driven sprint | DEFERRED | n/a |
| 7 | `_otr_model_catalog.py` importable + tested | Catalog scan + download + context cap + VRAM fit all green | `<actual>` | `<Y/N>` |
| 8 | `MODEL_CONTEXT_CAPS` static dict in `_otr_model_loader.py` | DELETED (replaced by `resolve_context_cap`) | `<actual>` | `<Y/N>` |
| 9 | `DEFAULT_CONTEXT_CAP = 8192` in `_otr_model_loader.py` | DELETED (no blind fallback) | `<actual>` | `<Y/N>` |
| 10 | Slot scheduler transition count for fixture episode (Slot 1 ≠ Slot 2) | Per-beat-default: equals DAG minimum (`meta["slot_transitions"]`). Opt-in batched: ≤ 3 | `<actual>` | `<Y/N>` |
| 11 | `_POLISH_CACHE` references in `visual/llm_polish.py` | 0 after B5 | `<actual>` | `<Y/N>` |
| 12 | `_LLM_CACHE` in `nodes/story_orchestrator.py` | 0 after B4b | `<actual>` | `<Y/N>` |
| 13 | `model_id` widget anywhere outside `OTR_LedgerScriptWriter` | 0 hits | `<actual>` | `<Y/N>` |
| 14 | `OTR_VisualLLMSelector` references | 0 hits in code; only forensic comments with S30 citation | `<actual>` | `<Y/N>` |
| 15 | Phase-3..6 toggle widgets in `OTR_LedgerFreezeCascade` | 0 hits | `<actual>` | `<Y/N>` |
| 16 | Phase functions in `nodes/_otr_lfc.py` (`_phase_3_per_line_polish` etc.) | 0 hits after B4 | `<actual>` | `<Y/N>` |
| 17 | `OTR_LFCPhase4Scene` / `5Voice` / `6Arc` in `NODE_CLASS_MAPPINGS` | 0 hits | `<actual>` | `<Y/N>` |
| 18 | Writer widget order | `[episode_title, target_words, num_characters, seed, seed_mode, creative_writing_model, technical_model, custom_premise, include_act_breaks, act_count, style, style_custom, creativity, optimization_profile, perfect_run_spacesaver, min_p, repetition_penalty, max_new_tokens_cap, enable_polish_pass]` | `<actual>` | `<Y/N>` |
| 19 | Both writer outputs (`creative_writing_model`, `technical_model`) wired to right consumers | Cascade gets `technical_model` per §2c routing table | `<actual>` | `<Y/N>` |
| 20 | `UnknownModelError` recovery hint | Includes top-5 installed alternatives | `<actual>` | `<Y/N>` |
| 21 | `HARD_VRAM_CONTEXT_LIMIT` clamp in `resolve_context_cap` | Default 8192 on 16 GB rig; `OTR_HARD_VRAM_CONTEXT_LIMIT` env var raises it | `<actual>` | `<Y/N>` |
| 22 | `check_vram_fit` pre-check at load time | Returns FAIL verdict for 70B-on-16GB BEFORE OOM | `<actual>` | `<Y/N>` |
| 23 | `torch.cuda.ipc_collect()` in `unload_llm()` | Called alongside `empty_cache()` + `synchronize()` | `<actual>` | `<Y/N>` |
| 24 | Polish sampling behavior unchanged in S30 | `make_polish_generate_fn` applies `_POLISH_TOP_P=0.9` / `_POLISH_DO_SAMPLE=True` unconditionally | `<actual>` | `<Y/N>` |
| 25 | `resolve_hf_token` cross-platform gating | `winreg` import + lookup only inside `if os.name == "nt":` block | `<actual>` | `<Y/N>` |
| 26 | Pre-fetch disk-space + size-estimate check | `InsufficientDiskSpaceError` raises before `snapshot_download` | `<actual>` | `<Y/N>` |
| 27 | `*.gguf` in `auto_download_if_missing` `allow_patterns` | 0 hits (Transformers loader only) | `<actual>` | `<Y/N>` |
| 28 | `validate_model_id` allow-list | Admits curated + locally-scanned + valid `org/name` (auto-download); rejects path traversal / drive letters / unsafe formats | `<actual>` | `<Y/N>` |
| 29 | `check_vram_fit` verdict tiering | Returns `VRAMFitVerdict.{PASS,WARN,UNKNOWN,FAIL}` enum, not `bool` | `<actual>` | `<Y/N>` |
| 30 | BUG-LOCAL-226 (S30 plan §2b audit miss) | FIXED at B4b; entry marked `[FIXED <hash> <date>]` in BUG_LOG.md | `<actual>` | `<Y/N>` |

---

## Documented deviations from plan

| # | Deviation | Sprint location | Rationale |
|--:|---|---|---|
| 1 | `tests/test_dropdown_guardrails.py` listed in CLAUDE.md + parent plan does not exist; substituted `tests/test_workflow_json_guardrails.py` | B0 | File renamed before sprint kickoff; closest match by name + scope. |
| 2 | `tests/v2/test_audio_byte_identical.py` listed in parent plan does not exist; substituted `tests/test_audio_byte_identical.py` | B0 | Path drift before sprint kickoff. |
| 3 | Branch cut from `s29-clean-slate-gate` tip rather than `v2.0-alpha @ HEAD-post-S29-merge` | B0 | S29 not yet merged to v2.0-alpha; code state is equivalent. |
| 4 | B0 narrowed from plan scope (LLM-stack deletion moved to new B4b) | B0 | Sprint plan §2b audit was wrong — `_load_llm` is live via RSS news path. BUG-LOCAL-226 logged. |
| 5 | B1c `_estimate_resident_gb` divides BF16 download size by 2 to match OTR's 8-bit quantization default | B1c | OTR's loader quantizes by default; download size ≠ resident size. Documented in code. |
| 6 | `<add as commits land>` | `<commit>` | `<rationale>` |

---

## Forward work (post-S30, NOT deferred from this sprint)

- **ComfyUI Desktop runtime verification pass** — Jeffrey runs the canonical workflow end-to-end on the 5080. Audio C7 byte-identical confirmed at the real-pipeline level. Validates B3 / B5 wiring at runtime. Not a gate for B8; it's a follow-up operator pass.
- **Manual VRAM soak** via `scripts/vram_profile_slot_swap.py` with Slot1 ≠ Slot2 — confirms `request_slot` slot-transition teardown actually recovers VRAM on the 5080. Not in pytest count.
- **Sprint C** — `meta.story_brief` v2. Opens next per sprint sequencing B → C → A.
- **Audio-intentional sprint** — model-author `generation_config.json` respect for the polish path. Deliberately deferred from S30 to keep audio byte-identity guaranteed across the structural cleanup. Re-derived from scratch with its own design, tests, and baseline-roll discipline.
- **Soak validation of an ungated curated entry** as `vram_fit_tier="PASS"` — gives the gated-without-token error message a confident ungated recommendation. Currently the message admits the gap honestly.

---

## Sources

- `docs/2026-05-14-S30-two-model-selector-sprint-plan.md` — parent sprint plan (the 14-commit playbook).
- `docs/2026-05-14-S30-continuation-plan.md` — fresh-session execution plan for B1d → B8.
- `docs/2026-05-14-S30-B1c-handoff.md` — B0-B1c hand-off doc.
- `BUG_LOG.md` — BUG-LOCAL-226 entry (S30 plan §2b audit miss).
- `ROADMAP.md` — sprint status section.
- `CLAUDE.md` — standing rules, platform pins, test commands, commit-message recipe.
