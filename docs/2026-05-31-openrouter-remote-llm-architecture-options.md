# OpenRouter Remote LLM — Architecture Options (for round-robin)

**2026-05-31 · branch `v2.0-alpha` · DECISION DOC (no code yet — pick the surface, then this becomes the execution plan)**

Purpose: add OpenRouter as an **optional, non-local** provider so the **creative** and **technical** model slots can each select a remote model. This doc lays out the candidate architectures, recommends one, and lists the open questions for the round-robin panel. Jeffrey runs the round-robin.

---

## TL;DR — recommendation

- **The engine is the same in every option** and is contained, but it is **not** a one-line "register a backend" job (my first draft said that — it was wrong; see "Critical findings" below). The live load path calls `load_llm` directly; the `LoaderBackend` / `BACKENDS_BY_KEY` abstraction is **dormant scaffolding** with no production caller. So the engine work is: implement `OpenRouterBackend`, then add a **narrow remote branch in `request_slot`** (and in the generate-fn factory) that routes `loader_backend == "openrouter_http"` rows to it — activating the dormant dispatch table for the remote row only, leaving the local byte-identical path on `load_llm` untouched. The writer, consumers, slot scheduler, and broadcast wiring still do **not** change.
- **The only real decision is the *selection surface*** — how "OpenRouter A / OpenRouter B" appear and how they bind to a real model id.
- **Recommended: Option A — two virtual catalog rows, bound by env vars.** Smallest diff, both slots covered, all guardrails (B6/B7, audio-byte-identical, VRAM) stay green by construction, and it matches your "A/B appear in both dropdowns" mental model exactly. I verified the `openrouter:` id scheme survives `validate_model_id`'s structural reject and is admitted via the curated path. **Option B** (same rows, bound by two config widgets on the writer) is a cheaper upgrade than I first claimed — it does **not** require a B6 change (the writer is exempt from that test) — and is worth it if you want the binding saved *in the workflow JSON*.
- **Two sub-decisions are genuinely open** and are the best things to hand the panel: (1) how remote **technical** calls stay valid JSON, since a remote API can't do the token-level grammar the local path uses; and (2) how the **single-resident model cache** behaves on mixed local+remote slots (a naive remote branch would thrash the local model in/out of VRAM on every slot transition — see C2).

---

## Critical findings from reading the live path (2026-05-31)

You asked me to look at the actual code and pick the best option. Three findings corrected my first draft:

1. **The backend dispatch table is dormant.** `get_backend_for_row` / `BACKENDS_BY_KEY` (`nodes/_otr_model_runtime.py`) are referenced only inside that module and in tests — **never** in the live path. `request_slot` (`nodes/_otr_model_loader.py:712`) calls `load_llm` directly at `:812`. So OpenRouter cannot "just register and dispatch automatically"; it needs an explicit remote branch in `request_slot` (after `validate_model_id` + the cache check, before `resolve_context_cap` / `check_vram_fit` / `auto_download_if_missing` / `load_llm`, all of which assume a real HF model and would choke on a remote id) and a matching branch in the generate-fn factory (`_build_truncating_generate_fn` at `OTR_LedgerScriptWriter.py:586`, `make_generate_fn` / `make_polish_generate_fn` at `_otr_model_loader.py:864/939`). The good news: that branch can call `get_backend_for_row(row).load()` for the remote row, so we *use* the existing abstraction rather than bypassing it — and the local path stays exactly as-is, protecting the audio baseline.
2. **The `openrouter:` id scheme is safe.** `_structural_reject` (`_otr_model_catalog.py:407`) blocks backslashes, leading `/`, `..`, the Windows drive-letter pattern `^[A-Za-z]:`, and `.gguf`/`.bin`. `"openrouter:slot-a"` trips none of them (the drive-letter regex needs a colon at position 1; here it's `p`), and as a curated row it's admitted via Path 1 before the HF-shape check ever runs. So virtual rows validate cleanly with no validator surgery.
3. **B6 is narrower than I assumed.** `test_b6_wiring_guardrails.py` only flags three exact key names (`model_id`, `creative_writing_model`, `technical_model`) as widgets, and it **skips `OTR_LedgerScriptWriter` entirely** (and offers a `NON_LLM_MODEL_WIDGET_OK` opt-out for media nodes). So Option A (no new widgets) is green by construction, and Option B's two new *writer* widgets named off that list (e.g. `openrouter_a_model`) do **not** trip B6 at all. My first-draft "Option B needs a B6 extension" was wrong.

---

## What is being added (scope)

OpenRouter is a cloud HTTP API that proxies many hosted LLMs behind one OpenAI-style `/chat/completions` endpoint and one API key. We want it available as a pick in the two existing model dropdowns on `OTR_LedgerScriptWriter`:

- `creative_writing_model` — narrative passes (outline, cast, dialogue, polish).
- `technical_model` — structured passes (JSON validators, grammar output, reviewer verdicts, critic, news_interpreter).

Your framing: two selectable remote profiles, **"OpenRouter A"** and **"OpenRouter B"**, each appearing as an option in **both** dropdowns. A and B are named handles that resolve to a real OpenRouter model slug (e.g. `anthropic/claude-3.5-sonnet`) plus params.

---

## The shared engine (identical across all options)

This part is fixed regardless of which selection surface we choose. It reuses the project's existing (but currently dormant) backend abstraction rather than bypassing it:

1. **Backend protocol exists but is not yet wired into the live path.** `nodes/_otr_loader_backends.py:47` defines `class LoaderBackend(Protocol)` with `load(repo_id, row) -> dict`, `generate(model, messages, **kwargs) -> str`, `unload(model) -> None`. Adapters are registered in `BACKENDS_BY_KEY` at `nodes/_otr_model_runtime.py:152` (`get_backend_for_row` at `:159`). **However, nothing in production calls `get_backend_for_row`** — the live path is `request_slot` → `load_llm` directly. The adapters are Sprint-D scaffolds that delegate back to `load_llm`. So we are *activating* this abstraction for the remote case, not plugging into a live socket.
2. **New backend.** Add `OpenRouterBackend` (new file `nodes/_otr_openrouter_backend.py`) implementing the three protocol methods:
   - `load()` builds a lightweight HTTP client handle and returns a cache-entry dict matching the local loader's shape (no weights, no tokenizer). It tags the entry as remote (e.g. `cache_entry["provider"] = "openrouter"`) so the generate-fn factory can branch. `unload()` is a no-op.
   - `generate()` POSTs the chat messages to OpenRouter, applies the cost guard, returns the decoded string.
3. **Remote branch in `request_slot`** (`nodes/_otr_model_loader.py:712`). After step 1 (`validate_model_id`) and step 2 (cache hit), insert: if the resolved row's `loader_backend == "openrouter_http"`, route to `get_backend_for_row(row).load(...)` and **skip** steps 3–8 (`resolve_context_cap`, `check_vram_fit`, `auto_download_if_missing`, `unload_llm`, `load_llm`) — every one of those assumes a real HF model on disk/VRAM and would error or waste work on a remote id. Cache and return the remote entry.
4. **Remote branch in the generate-fn factory.** `_build_truncating_generate_fn` (`OTR_LedgerScriptWriter.py:586`) and `make_generate_fn` / `make_polish_generate_fn` (`_otr_model_loader.py:864/939`) currently assume a local model + tokenizer. Add a branch: if `cache_entry["provider"] == "openrouter"`, return a generate fn that calls `OpenRouterBackend.generate` (prompt-budget trimming still applies; no tokenizer needed).
5. **Zero local VRAM for remote.** Because the remote branch never loads weights and never evicts the resident local model (see C2), a remote call uses **no VRAM** and cannot push the 14.5 GB ceiling.

**Net:** the writer's two dropdowns, the slot scheduler, the `technical_model` broadcast socket, and all consumer nodes are untouched. The change is two narrow branches in the loader + one new backend module + catalog rows. The local path stays byte-for-byte as-is.

---

## The real decision — the selection surface

How do "OpenRouter A / OpenRouter B" appear in the dropdowns, and how do they bind to a real model slug?

### Option A — Two virtual catalog rows, env binding **(RECOMMENDED — leanest)**

- Add two virtual `CuratedModel` rows to the catalog: `repo_id="openrouter:slot-a"` / `"openrouter:slot-b"`, `loader_backend="openrouter_http"`, `vram_fit_tier="PASS"`. They appear in **both** dropdowns automatically, because both call the same `_otr_model_catalog.dropdown_choices()` (`OTR_LedgerScriptWriter.py:1532` and `:1546`).
- **Binding lives in env** (mirrors your existing HKCU key pattern — `OPENAI_API_KEY`, `GEMINI_API_KEY`, `HF_TOKEN`):
  - `OPENROUTER_API_KEY` — the key.
  - `OPENROUTER_MODEL_A` / `OPENROUTER_MODEL_B` — the real slugs A and B resolve to.
  - optional `OPENROUTER_A_TEMP`, `..._MAXTOK`, etc.
- **Default OFF:** the two rows are injected into the dropdown **only** when `OPENROUTER_API_KEY` is set (and/or `OTR_ENABLE_OPENROUTER=1`). With it unset, nothing remote appears and the offline baseline is byte-identical.

| | Detail |
|---|---|
| Files touched | catalog (2 rows + `loader_backend` literal `"openrouter_http"` + a `provider` field), new `_otr_openrouter_backend.py`, 2 narrow branches in the loader (`request_slot` + generate-fn factory) |
| Validator | **no change** — curated rows are admitted via Path 1; `openrouter:slot-a` passes `_structural_reject` (verified) |
| New widgets | none |
| New nodes | none |
| Workflow JSON | no node/socket change (dropdown just has two more string values) |
| Guardrails | B6 / B7 green by construction (no new widgets, no forbidden symbols) |
| Pros | smallest diff; both slots covered; reproducible via env; swap what A/B mean without touching the graph |
| Cons | the A→slug binding is in env, not visible in the saved workflow JSON |

### Option B — Two virtual catalog rows, **writer config widgets** (visible binding)

- Same two virtual rows in both dropdowns.
- Binding moves into **two new STRING widgets on the writer** (the *only* node permitted to hold model widgets): `openrouter_a_model`, `openrouter_b_model` (free-text slugs, default `""`). API key stays in env.
- The writer resolves `"openrouter:slot-a"` by reading its own `openrouter_a_model` widget.

| | Detail |
|---|---|
| Files touched | Option A + 2 writer widgets + writer resolve logic + workflow JSON widget defaults |
| New widgets | 2 (on the writer only) |
| Guardrails | **B6 green with no test change** — `test_b6_wiring_guardrails` skips `OTR_LedgerScriptWriter` entirely and only flags the three reserved key names; widgets named e.g. `openrouter_a_model` are invisible to it (verified) |
| Pros | binding saved in the graph → portable + reproducible without env; still no widgets outside the writer |
| Cons | larger surface; widget-drift risk (the project's recurring failure mode); 2 more widget defaults to keep in sync with the workflow JSON |

### Option C — Two companion "OpenRouter Profile" nodes wired into the writer (graph-native, **heaviest** — not recommended)

- New node class `OTR_OpenRouterProfile`, placed twice (A, B), each with model + param widgets, each emitting a STRING profile wired into two new writer inputs.

| | Detail |
|---|---|
| Files touched | new node class + `NODE_CLASS_MAPPINGS` registration + 2 new writer input sockets + workflow JSON nodes + links + every workflow auditor |
| Guardrails | the automated B6 test wouldn't necessarily fire (if the profile node's widgets are named off the reserved list), but this still **violates the intent of Prime Directive 6** — "no node other than the writer exposes a model picker." Would need a documented, audited exception |
| Pros | fully visible in graph; param-rich; literally your "two openrouter nodes" idea |
| Cons | biggest diff and risk; breaks the sole-model-source invariant in spirit; most node/wiring/auditor surface to police; not lean |

### Option D — Direct enumeration, no A/B (simplest concept, cluttered)

- Skip the A/B indirection; add N specific OpenRouter model rows directly (each dropdown entry *is* a real slug).

| | Detail |
|---|---|
| Files touched | same engine as A, minus the indirection |
| Pros | maximally transparent — the entry is the model |
| Cons | dropdown clutter; doesn't match the A/B model; changing the offered set means editing code + re-regressing |

**Recommendation:** ship **Option A**. If the panel weights "binding must live in the saved workflow, not env" highly, upgrade to **Option B**. C and D are documented for completeness.

---

## Open sub-decision — remote **technical** (JSON-strict) calls

The local technical path guarantees valid JSON with token-level grammar (`lmformatenforcer` via `prefix_allowed_tokens_fn`; see `nodes/_otr_constrained_generate.py` and `nodes/_otr_lmfe_compat.py`). A remote HTTP model **cannot** be constrained that way. Because you want OpenRouter selectable on **both** slots, "keep the technical slot entirely local" is off the table — the real question is how hard to backstop a remote technical pick:

- **T1 (recommended).** Use OpenRouter `response_format={type: json_schema}` on schema-capable models, with the project's **existing JSON validator/repair passes** as the backstop. If the chosen remote model doesn't support `json_schema`, fall back to best-effort + repair.
- **T2 (remote-first, local safety valve).** Try remote; on JSON parse failure, auto-retry the *same* call on the local technical model. Strongest integrity, but it needs a local model resident for the retry — which reintroduces the C2 thrash and VRAM cost.
- **T3 (free-form only goes remote).** A remote technical pick serves only the slot's non-grammar calls; the grammar-constrained calls silently run local. Cleanest integrity, but "I picked OpenRouter and half the calls ran local" is surprising behavior to document.

Recommended default: **T1** (it honors "both slots" without forcing a resident local model), with **T2** as the conservative fallback if the panel doesn't trust `response_format` + repair for the structured passes.

---

## Hard constraints (every sprint respects these — non-negotiable)

- **C1 — Audio is king.** Remote is **default-off**, so the byte-identical baseline is always computed on the unchanged local path; `test_audio_byte_identical` stays green. Selecting a remote model is an explicit opt-in that changes script content (as any model swap would) and is never the regression baseline.
- **C2 — VRAM ceiling 14.5 GB + the single-resident cache.** Remote calls use zero local VRAM; the remote branch skips warmup / snapshot / download / vram-fit / load. **Open design point:** `request_slot` keeps at most one resident model and tears it down on every model-id change (`unload_llm` at `:796`). A naive remote branch in a *mixed* run (e.g. creative=local, technical=OpenRouter) would evict and then reload the local model on every slot transition — a real perf cliff. Fix: the remote branch must **leave the resident local model in `LLM_CACHE` untouched** (remote needs no VRAM), so a local↔remote alternation never reloads the local model. Both-remote runs are free of this entirely. Never use `force_vram_offload()` between LLM phases.
- **C3 — Offline-first, no surprise cloud.** No remote call ever fires unless `OPENROUTER_API_KEY` (+ enable flag) is set. Honors the "100% local, offline-first" directive as an explicit, opt-in exception.
- **C4 — Cost guard (paid service).** Hard, configurable per-run token/spend ceiling, conservative default; abort with a clear error *before* exceeding; log spend per call. No unbounded paid calls.
- **C5 — Two-model tag (PD6).** Remote changes the backend behind a slot, not the slot. No new `model_id` widget anywhere; remote ids ride the existing two dropdowns; every LLM call keeps its `# LLM slot: creative|technical` tag. `test_writer_slot_routing` stays green.
- **C6 — B6 / B7 green.** Option A adds no widgets and no forbidden symbols. (Option B requires an audited B6 allowlist extension — see above.)
- **C7 — Cache integrity.** Remote ids are distinct strings (`openrouter:slot-a`), so LLM cache keys differentiate automatically; set each remote row's `context_cap` so `test_llm_cache_mismatch_diagnostics` sees no drift.
- **C8 — No secrets in code/logs.** API key only from env; never logged, never committed. Safe-for-work output; no "dummy" — use "placeholder"/"stub".
- **C9 — Reproducibility.** Stamp `provider` + the resolved remote slug into ledger meta (mirror the existing `meta["creative_model"]` stamp at `OTR_LedgerScriptWriter.py` ~3724-3741) so a run records exactly which remote model produced it.

---

## Sprint outline — recommended Option A (review → code → wire → regress → commit)

Checkbox flips to `[x]` only when: done **and** regression green **and** (if a node was touched) the workflow JSON is re-wired to match. Log bugs to `BUG_LOG.md` first, then pointer here.

### Sprint Status Board

| Sprint | Status | Bug pointers | Notes |
|--------|--------|--------------|-------|
| S0 — Freeze contracts + baseline | NOT STARTED | — | pin backend interface, env names, row schema, default-off gate; capture green baseline |
| S1 — OpenRouter backend engine | NOT STARTED | — | `OpenRouterBackend` + register `openrouter_http`; mocked-HTTP unit tests |
| S2 — Catalog rows + validation + dropdown gating | NOT STARTED | — | virtual A/B rows; `validate_model_id` admit-path; appear only when enabled |
| S3 — Wire remote branch into live path | NOT STARTED | — | `request_slot` + generate-fn branches; skip steps 3–8; don't evict resident local model (C2) |
| S4 — Meta + reproducibility + workflow re-validate | NOT STARTED | — | provider/slug meta stamp; confirm broadcast/consumers carry remote id; auditors pass |
| S5 — Live smoke + cost-guard + docs | NOT STARTED | — | enabled remote smoke + disabled byte-identical confirm; cost-ceiling abort proven |

### Sprint detail

**S0 — Freeze contracts + baseline.** Pin the `OpenRouterBackend` surface (= existing `LoaderBackend`), env var names (`OPENROUTER_API_KEY`, `OPENROUTER_MODEL_A/B`, `OTR_ENABLE_OPENROUTER`), the virtual-row schema, and the default-off gate. Capture a green baseline of all regressions + audio-byte-identical + Bug Bible. Gate: nothing proceeds until baseline green and contracts frozen.

**S1 — Backend engine.** Implement `nodes/_otr_openrouter_backend.py` (`load`/`generate`/`unload`) with timeout, bounded retries, the C4 cost guard, and key-from-env. Register `"openrouter_http"` in `BACKENDS_BY_KEY`. Unit-test with **mocked HTTP** (no network in CI). Regress.

**S2 — Catalog rows + validation + dropdown gating.** Add `"openrouter_http"` to the `loader_backend` Literal (`_otr_model_catalog.py:71`); add the two virtual rows injected only when enabled; add a `validate_model_id` admit-path for `openrouter:*` that bypasses file/VRAM checks (alongside the existing `OTR_MODEL_CATALOG_ALLOW_REMOTE` logic at `:481`). Tests: A/B present when enabled, absent when disabled; `test_model_catalog_scan` green. Regress.

**S3 — Wire the remote branch into the live path.** This is the load-bearing sprint (the dispatch table is dormant, so this is where remote actually becomes reachable). In `request_slot`, after `validate_model_id` + cache-hit, branch on `row.loader_backend == "openrouter_http"`: route to `get_backend_for_row(row).load(...)`, skip steps 3–8, and **do not evict the resident local model** (C2). In the generate-fn factory (`_build_truncating_generate_fn` / `make_generate_fn` / `make_polish_generate_fn`), branch on `cache_entry["provider"] == "openrouter"`. Tests: remote path makes zero CUDA/download calls and leaves a resident local model in place; `test_llm_cache_mismatch_diagnostics` + `test_cache_key_mutations` clean. Regress.

**S4 — Meta + wiring + reproducibility.** Stamp `provider` + resolved slug into ledger meta (mirror `creative_model`). Confirm the `technical_model` broadcast and all consumers carry the remote id unchanged. Re-run `scripts/_audit_workflow_json.py` + `tools/audit_workflow_schema.py` (Option A needs no node/socket change — prove auditors pass). Regress.

**S5 — Live smoke + cost-guard + docs.** One operator smoke run with remote **enabled** (end-to-end + cost-ceiling abort proven) and one with remote **disabled** (byte-identical baseline confirmed unchanged). Update `BUG_LOG.md`, `CLAUDE.md`/README as needed. Final full regress + Bug Bible.

---

## Questions for the round-robin panel

1. **Selection surface — A vs B.** Env binding (lean, A→slug not in graph) vs two writer config widgets (visible/portable, B6 test must be extended). Which wins on reproducibility vs surface?
2. **Technical-slot JSON — T1 vs T2.** Is `response_format=json_schema` + the existing repair net trustworthy enough for the structured passes, or should every grammar-constrained call stay strictly local?
3. **Cost guard.** Default per-run spend/token ceiling and breach behavior — hard abort vs warn-and-continue?
4. **Slot count.** Is A/B (two remote handles) enough, or parametric N?
5. **Streaming.** Match the local path (non-streaming) or stream remote for latency?
6. **Mid-episode failure.** If a remote call errors (network / rate-limit) partway through, abort the run or auto-fall-back to local (ties to T3 and C1)?

---

## Appendix — verified code anchors (as of 2026-05-31)

| Claim | Anchor |
|---|---|
| Backend protocol (`load`/`generate`/`unload`) | `nodes/_otr_loader_backends.py:47` |
| Backend dispatch table + lookup (**dormant** — only callers are this module + tests) | `nodes/_otr_model_runtime.py:152` (`BACKENDS_BY_KEY`), `:159` (`get_backend_for_row`) |
| Live load path (calls `load_llm` directly, not the dispatch table) | `nodes/_otr_model_loader.py:712` (`request_slot`), `:796` (resident-model teardown), `:812` (`load_llm` call) |
| Generate-fn factory (needs remote branch) | `OTR_LedgerScriptWriter.py:586` (`_build_truncating_generate_fn`); `_otr_model_loader.py:864` (`make_generate_fn`), `:939` (`make_polish_generate_fn`) |
| Structural reject (verified `openrouter:` is safe) | `nodes/_otr_model_catalog.py:407` (`_structural_reject`), `:452` (`validate_model_id`, curated Path 1 at `:493`) |
| B6 scope (3 reserved keys; writer exempt; `NON_LLM_MODEL_WIDGET_OK` opt-out) | `tests/test_b6_wiring_guardrails.py:43` (`_MODEL_WIDGET_KEYS`), `:265` (writer skip) |
| Catalog `loader_backend` Literal | `nodes/_otr_model_catalog.py:71` |
| Dropdown builders (feed both slots) | `nodes/_otr_model_catalog.py:361` (`build_dropdown_choices`), `:392` (`dropdown_choices`) |
| Model-id validation + structural reject | `nodes/_otr_model_catalog.py:452` (`validate_model_id`), `:420` (`.gguf`/`.bin` reject) |
| Existing remote-enable env | `nodes/_otr_model_catalog.py:481` (`OTR_MODEL_CATALOG_ALLOW_REMOTE`) |
| Writer's two dropdowns | `nodes/OTR_LedgerScriptWriter.py:1532` (creative), `:1546` (technical) — both `dropdown_choices()` |
| Resolved slot ids + broadcast | `nodes/OTR_LedgerScriptWriter.py:1347-1348`, broadcast `technical_model` (RETURN index 4) |
| Local grammar mechanism | `nodes/_otr_constrained_generate.py`, `nodes/_otr_lmfe_compat.py` |
| Local loader / VRAM / context cap | `nodes/_otr_model_loader.py` (`load_llm`, `request_slot`, `unload_llm`, warmup, `_MODEL_CONTEXT_CAPS`) |
| Guardrail tests | `tests/test_b6_wiring_guardrails.py`, `tests/test_b7_forbidden_sweep.py`, `tests/test_writer_slot_routing.py`, `tests/test_model_catalog_scan.py`, `tests/test_llm_cache_mismatch_diagnostics.py`, `tests/test_cache_key_mutations.py`, `tests/test_audio_byte_identical.py`, `tests/test_workflow_canonical_baseline.py`, `tests/test_workflow_json_guardrails.py` |
| External Bug Bible regression | `C:/Users/jeffr/Documents/ComfyUI/comfyui-custom-node-survival-guide/tests/bug_bible_regression.py` |
| Canonical workflow | `workflows/otr_scifi_16gb_full.json` (writer node id 1) |
