# S31 + S32 — Cowork execution plan

S31 = legacy LLM stack clean break + writer residuals + workflow JSON fixes + CUDA-race fix.
S32 = per-sub-pass routing inside `pick_style` / `lock_cast` / `compose_line`.

One file, two sprints. S31 first, then S32. Branch sequentially.

---

## Sprint roadmap

| Sprint | Branch | Commits | Days | Sessions |
|---|---|--:|--:|--:|
| **S31** | `s31-loader-clean-break` | B0-B8 (9) | ~6 | ~4-5 |
| **S32** | `s32-helper-per-subpass-routing` | B0-B8 (9) | ~4-5 | ~3-4 |

S31 deletes 4 symbols from `nodes/story_orchestrator.py`: `_load_llm` (~600 LOC body), `_unload_llm`, `_LLM_CACHE`, `_generate_with_llm`. Closes BUG-LOCAL-226.

S32 changes 2 helper sub-pass dispatches (`pick_style` pass 2, `lock_cast` schema validation) and adds 1 opt-in (`compose_line` critic).

---

## Hard rules (S31 AND S32)

1. **Audio C7 byte-identical (pytest proxy)** must hold at every commit boundary in default config (`creative_writing_model == technical_model`). Regression → STOP, revert, investigate. Differing-slots config gets its own baseline at S32 B5.
1A. **S31 B4 deletion is non-deferrable.** The 4 legacy symbols (`_load_llm`, `_unload_llm`, `_LLM_CACHE`, `_generate_with_llm`) delete in S31, period. If a session ends mid-B4, the next session resumes B4 — it does NOT advance to B5 with shims still alive, and it does NOT defer deletion to S31.1 or S32. Hard rule #3 (no partial-port states) plus B5's sweep markers structurally enforce this; rule #1A states it explicitly so context-compression cannot lose it. The sprint does not close until B4 has deleted all 4 symbols, the simplified `unload_llm` no longer touches `story_orchestrator`, and the TIMEOUT_RECOVERY CUDA-race is fixed via `invalidate_cache_no_gpu_teardown`. If Cowork proposes deferring deletion for any reason — VRAM concerns, audio regression, time pressure, "let's be safe and ship S31 without B4" — the answer is no. Revert if needed, debug, retry, but do not advance with shims alive.
2. **Runtime audio C7 verification on 5080** strongly recommended after S31 B2 / B4 / B6 and S32 B5 / B6. Proxy is necessary-not-sufficient.
3. **No partial-port / partial-helper states.** Shims in S31 B2 are one-commit-deep, deleted same-sprint at B4. S32 B1 is atomic (signatures + writer wiring).
4. **No legacy back-compat reintroduced.** No `_RENAME_ALIASES`, no soft-landing, no "stamp both legacy + modern" hedges, no `if hasattr(_so, "_LLM_CACHE")` guards post-S31 B4.
5. **One generate surface.** `generate_text` / `generate_with_llm` / any wrapper-by-another-name NOT introduced. Canonical surface: `request_slot(slot, model_id) → make_generate_fn(cache_entry) → fn(messages, *, temperature, max_new_tokens)`. Callers compose chat messages.
6. **Lifecycle helpers are distinct from generate surface.** `load_llm`, `unload_llm`, `request_slot`, `invalidate_cache_no_gpu_teardown` are lifecycle functions, allowed and named for their specific use case. Hard rule #5 covers generate surface only.
7. **Bug Bible regression** 23 / 1 / 2xf at every commit boundary.
8. **Tests written before fixes** for structural defects. Red-on-parent, green-on-fix.
9. **Forbidden-pattern sweep** stays at 0 runtime hits at every commit boundary.
10. **No version-label bumps.** Stay under v2.0-alpha umbrella.
11. **No extra branches.** Every commit lands on the sprint's named branch.

---

## Canonical pytest run

```cmd
cd /d C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe -m pytest "C:/Users/jeffr/Documents/ComfyUI/comfyui-custom-node-survival-guide/tests/bug_bible_regression.py" tests\test_workflow_json_guardrails.py tests\test_workflow_link_target_indexes.py tests\test_core.py tests\test_audio_byte_identical.py tests\test_model_catalog_scan.py tests\test_model_catalog_download.py tests\test_loader_slot_primitives.py tests\test_loader_body_profiles.py tests\test_no_orchestrator_legacy_symbols.py tests\test_writer_input_resolve.py tests\test_fetch_science_news_no_legacy_wrapper.py tests\test_run_with_timeout_safe_invalidation.py tests\test_visual_prompt_coercion_contract.py tests\test_helper_paired_signatures.py tests\test_pick_style_routing.py tests\test_lock_cast_routing.py tests\test_compose_line_routing.py tests\test_writer_paired_wiring.py tests\test_meta_slot_transitions.py -q
```

After every commit:

```cmd
git diff <parent-branch> -- "*.py" | Out-File -Encoding utf8 docs\<sprint>_diff_tmp.txt
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe docs\_s28_forbidden_sweep.py
```

*Keep `Out-File -Encoding utf8`, never `>` — PowerShell `>` writes UTF-16+BOM; sweep classifier sees empty file; vacuous "0 hits."*

Commit message to `.git\COMMIT_EDITMSG` via file tool, then `git commit -F .git\COMMIT_EDITMSG` and `git push origin <branch>`.

**Pytest baselines:**
- S31 baseline (from `s30-two-model-selector` @ B8): **253 / 7 / 2**
- S31 target at B8: **~282 / 7 / 2**
- S32 baseline: ~282 / 7 / 2
- S32 target at B8: **~302 / 7 / 2**

---
---

# SPRINT S31 — legacy LLM stack clean break

**Branch:** `s31-loader-clean-break`. Cut from `s30-two-model-selector` @ B8.
**Headline commit:** **B4**.

## B0 — branch cut + plan landing (~0.25 d)

### Review
Confirm S30 B8 hash matches QA commit table. Pre-sprint grep:

```cmd
findstr /s /n "GemmaHeartbeat TextIteratorStreamer TextStreamer streamer=" nodes\*.py visual\*.py
```

Hits outside `story_orchestrator.py` block sprint start — surface and re-scope.

### Code
Land plan at `docs/2026-05-14-S31-S32-cowork-execution-plan.md`.

### Wire / Pytest
Baseline 253/7/2 recorded.

### Commit subject
`B0: branch cut + S31+S32 Cowork execution plan landing`

---

## B1 — caller-switch pre-work: VRAMContextTest (~0.5 d)

### Review
Pre-grep:
```cmd
findstr /s /n "_SO._load_llm _SO._unload_llm _SO._LLM_CACHE _SO._generate_with_llm" nodes\*.py visual\*.py
```
Expected hits in `vram_context_test.py` only (lines 236, 268). Others fold into B1.

### Code

**`nodes/vram_context_test.py:236`**:
```python
# OLD
            _SO._load_llm(model_id, optimization_profile=optimization_profile)
# NEW
            from . import _otr_model_loader as _OTRML
            _OTRML.request_slot("technical", model_id)
```

**`nodes/vram_context_test.py:266-275`** (canonical pattern, no wrapper):
```python
# OLD
            _ = _SO._generate_with_llm(
                prompt, model_id=model_id, max_new_tokens=max_new_tokens,
                temperature=0.0, top_p=1.0,
                optimization_profile=optimization_profile,
            )
# NEW
            from . import _otr_model_loader as _OTRML
            cache_entry = _OTRML.request_slot("technical", model_id)
            gen_fn = _OTRML.make_generate_fn(cache_entry)
            _ = gen_fn(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_new_tokens=max_new_tokens,
            )
```

Update module docstring (lines 4, 15, 122-123) — drop `_load_llm()` / `_generate_with_llm()` wording. Drop orphan `_SO` import. Keep `NON_LLM_MODEL_WIDGET_OK = True`.

### Wire
None.

### Pytest

| Test | File | Asserts |
|---|---|---|
| `test_vram_context_test_no_direct_load_llm` | `tests/test_no_orchestrator_legacy_symbols.py` (new) | AST scan: 0 runtime `_SO._load_llm` refs |
| `test_vram_context_test_no_direct_generate_with_llm` | same | 0 runtime `_SO._generate_with_llm` refs |
| `test_no_external_caller_of_legacy_symbols` | same | Tree-wide AST scan (exclude `story_orchestrator.py` + `_otr_model_loader.py`): 0 runtime refs to any of 4 legacy symbols |

### Commit gate
3 tests green. 253 existing green. Bug Bible 23/1/2xf. Audio C7 holds. Forbidden sweep clean.

### Commit subject
`B1: caller-switch pre-work — VRAMContextTest off legacy _SO symbols`

---

## B2 — port `_load_llm` body (~1.5 d, HIGH RISK)

### Review
~600 LOC bitsandbytes / NF4 / 8-bit / Standard / Obsidian / Pro profile body moves from `story_orchestrator._load_llm` to `_otr_model_loader.load_llm`.

Inventory before copy:
- `BitsAndBytesConfig` per profile
- `dtype` per profile + device capability
- Attention backend (`sdpa` / `eager` / `flash_attention_2`)
- Special-case model id handling (gated, GGUF rejection)
- `AutoTokenizer.from_pretrained` config
- `AutoModelForCausalLM.from_pretrained` config
- Return: legacy `(model, tokenizer)` tuple → modern cache_entry dict

**CRITICAL: PURE COPY, NOT REWRITE.** Same imports, same order. Same dtype calls. Same kwargs in same positions. Audio C7 is the verification.

### Code

**`nodes/_otr_model_loader.py:load_llm`** — replace wrapper with ported body. Six steps: normalize id → resolve context_cap → profile config → tokenizer load → model load → return cache_entry dict. Raises `ModelLoaderError`.

**`nodes/story_orchestrator.py:_load_llm`** — replace body with ~10-line shim (one-commit-deep, deleted at B4):

```python
def _load_llm(model_id_full, *, device="cuda", optimization_profile="Standard"):
    """S31 B2 shim — body ported to _otr_model_loader.load_llm.
    Exists only B2..B3 so internal orchestrator callers keep
    working until B3/B4. Deleted at B4."""
    from . import _otr_model_loader as _otr_loader_mod
    cache_entry = _otr_loader_mod.load_llm(
        model_id_full, device=device,
        optimization_profile=optimization_profile,
    )
    return cache_entry["model"], cache_entry["tokenizer"]
```

`_LLM_CACHE` dict stays alive through B2-B3 because `_load_llm`'s own cache-hit logic reads it. (S30 B4b already rewrote `_generate_with_llm` to read `cache_entry.get("context_cap")` directly, so that's not a reason — `_load_llm`'s internal caching is.) B4 deletes all four symbols together.

### Wire
None.

### Pytest

| Test | File | Asserts |
|---|---|---|
| `test_load_llm_standard_profile` | `tests/test_loader_body_profiles.py` (new) | cache_entry `quantized=False`, dtype matches Standard |
| `test_load_llm_obsidian_profile` | same | `quantized=True`, BitsAndBytesConfig applied |
| `test_load_llm_8bit_profile` | same | NF4 / 8-bit config differentiation |
| `test_load_llm_returns_cache_entry_dict_shape` | same | All keys: model, tokenizer, model_id, device, quantized, context_cap |
| `test_load_llm_strips_ui_suffix` | same | `[BETA]` suffix → bare id in `cache_entry["model_id"]` |
| `test_orchestrator_load_llm_shim_returns_tuple` | same | `_so._load_llm(...)` still returns tuple |
| `test_request_slot_uses_ported_body` | `tests/test_loader_slot_primitives.py` (extend) | End-to-end through ported body |
| `test_audio_c7_byte_identical_b2` | `tests/test_audio_byte_identical.py` | **Canary** |

### Commit gate
8 tests green. **Audio C7 proxy holds — deciding gate.** Forbidden sweep clean.

**OPERATOR:** runtime 5080 render after B2, bytewise compare against pre-S31 reference.

### Commit subject
`B2: port _load_llm body (~600 LOC) to _otr_model_loader; orchestrator _load_llm becomes thin shim`

---

## B3 — refactor internal callers off `_generate_with_llm` (~0.75 d)

### Review
Per Hard rule #5: NO new wrapper. Internal orchestrator callers switch directly to `request_slot + make_generate_fn`.

Pre-grep:
```cmd
findstr /n "_generate_with_llm" nodes\story_orchestrator.py
```

Known callers: `_fetch_science_news` (RSS rank @ ~64 tokens / temp 0.05, body rerank @ ~8, LTX style brief @ ~80). Surface others.

**Also pre-grep** `_generate_ltx_style_brief` specifically (S30 B4b §Step 4 called it out as potentially deletable post voice-path-cleanbreak P2 / Director deletion):
```cmd
findstr /s /n "_generate_ltx_style_brief" nodes\ visual\ scripts\
```

Branch on result:
- **0 hits outside its own definition** → orphaned; add to B3 deletion list alongside the call-site refactor.
- **Hits exist** → refactor the call site to the canonical `request_slot + make_generate_fn` pattern.

Resolve in B3, not as a residual.

### Code

Replace at each call site:

```python
# OLD
result = _generate_with_llm(
    prompt, model_id=model_id, max_new_tokens=N,
    temperature=T, top_p=P,
    optimization_profile=optimization_profile,
)
# NEW (canonical)
from . import _otr_model_loader as _OTRML
cache_entry = _OTRML.request_slot("technical", model_id)
gen_fn = _OTRML.make_generate_fn(cache_entry)
result = gen_fn(
    messages=[{"role": "user", "content": prompt}],
    temperature=T,
    max_new_tokens=N,
)
```

`make_generate_fn` bakes `top_p=0.92`. RSS callers don't customize — fine. If any caller needs custom top_p, use `_build_truncating_generate_fn(cache_entry, top_p=P, ...)` (writer pattern).

`_generate_with_llm` function definition stays alive but uncalled. B4 deletes.

### Wire
None.

### Pytest

| Test | File | Asserts |
|---|---|---|
| `test_fetch_science_news_uses_request_slot` | `tests/test_fetch_science_news_no_legacy_wrapper.py` (new) | AST scan: `request_slot` present, no `_generate_with_llm` |
| `test_fetch_science_news_news_rank_args` | same | Mock fn, capture args: temperature=0.05, max_new_tokens=64 |
| `test_fetch_science_news_body_rerank_args` | same | max_new_tokens=8 |
| `test_fetch_science_news_style_brief_args` | same | max_new_tokens=80 |
| `test_orchestrator_no_remaining_generate_with_llm_callers` | same | AST scan: 0 internal call sites |

Red-on-parent for last test. Apply refactor, green.

### Commit gate
5 tests green. Audio C7 holds. Forbidden sweep clean.

### Commit subject
`B3: refactor _fetch_science_news + internal callers off _generate_with_llm onto request_slot + make_generate_fn`

---

## B4 — DELETE 4 symbols + simplify `unload_llm` + fix TIMEOUT_RECOVERY CUDA race ⚑ HEADLINE (~1 d)

### Review
B1-B3 left: shims and `_LLM_CACHE` dict. B4 deletes everything legacy AND fixes a separate CUDA-race regression in the timeout recovery path.

**The TIMEOUT_RECOVERY race:** S30 B4b rewired `story_orchestrator._run_with_timeout` to call `_otr_model_loader.unload_llm()` when the worker thread times out. Comment preserved from pre-B4b says `"avoids cudaErrorIllegalAddress from orphan ... worker still on GPU"` — but the new behavior actively *causes* that error. `unload_llm()` calls `model.to("cpu")` and `torch.cuda.empty_cache()` while the orphan worker thread may still be executing CUDA kernels on the cached model. Pre-B4b the path was dict-invalidation-only (safe). B4 reverts to safe semantics via a new lifecycle helper.

**Pre-B4 grep:**
```cmd
findstr /s /n "_load_llm _unload_llm _LLM_CACHE _generate_with_llm" nodes\*.py visual\*.py
```
Anything outside `story_orchestrator.py` (definitions) and `_otr_model_loader.py` (docstring mentions) blocks B4.

### Code

**`nodes/_otr_model_loader.py`** — add new lifecycle helper alongside `unload_llm`:

```python
def invalidate_cache_no_gpu_teardown() -> None:
    """Clear LLM_CACHE dict references WITHOUT touching the GPU.

    Use case: timeout recovery when an orphan worker thread may
    still be executing CUDA kernels on the cached model. Calling
    unload_llm() here would race the active kernel: model.to("cpu")
    moves weights mid-write and torch.cuda.empty_cache() can
    deallocate memory the kernel is still reading from -- both
    trigger cudaErrorIllegalAddress.

    The orphan thread's stack frame holds the model reference and
    the generate loop continues to completion on its own references.
    Once the orphan exits naturally, GC + a subsequent clean
    unload_llm call (when the next request_slot loads a different
    model) handles cleanup safely.

    NOT a general-purpose helper -- only use in code paths where
    GPU teardown is unsafe (timeout recovery, signal handlers).
    """
    LLM_CACHE.clear()
    LLM_CACHE.update({"model_id": None, "slot": None, "cache_entry": None})
```

Add to `__all__`.

**`nodes/_otr_model_loader.py:unload_llm`** — delete lines 246-268 (legacy orchestrator fallback). Simplified body ~40 LOC (down from 90): cache_entry.model.to("cpu") → LLM_CACHE clear+update → gc.collect → empty_cache + ipc_collect + synchronize. Keep cache-identity-preserving pattern.

**`nodes/story_orchestrator.py`** — delete in one commit:
- `_LLM_CACHE` module-level dict
- `_load_llm` function (B2 shim)
- `_unload_llm` function
- `_generate_with_llm` function
- Remove from `__all__` if present

**`nodes/story_orchestrator.py:_run_with_timeout`** — fix TIMEOUT_RECOVERY:

```python
# OLD (B4b state, the CUDA-race regression)
                from . import _otr_model_loader as _otr_loader_mod
                _otr_loader_mod.unload_llm()
                _runtime_log(
                    f"TIMEOUT_RECOVERY: unload_llm() invoked so next "
                    f"phase forces a fresh load (avoids "
                    f"cudaErrorIllegalAddress from orphan {phase_label} "
                    f"worker still on GPU)"
                )

# NEW (S31 B4 fix)
                from . import _otr_model_loader as _otr_loader_mod
                _otr_loader_mod.invalidate_cache_no_gpu_teardown()
                _runtime_log(
                    f"TIMEOUT_RECOVERY: LLM_CACHE invalidated (GPU "
                    f"untouched; orphan {phase_label} worker keeps "
                    f"its model reference until natural completion). "
                    f"Next request_slot forces a fresh load."
                )
```

**`BUG_LOG.md`** — mark BUG-LOCAL-226 as `[FIXED <B4 hash> 2026-05-14]`. File a new BUG-LOCAL-NNN entry for the TIMEOUT_RECOVERY CUDA-race regression with `[FIXED <B4 hash> 2026-05-14]`.

### Wire
None.

### Pytest

| Test | File | Asserts |
|---|---|---|
| `test_orchestrator_no_load_llm_symbol` | `tests/test_no_orchestrator_legacy_symbols.py` (extend) | `not hasattr(_so, "_load_llm")` |
| `test_orchestrator_no_unload_llm_symbol` | same | `not hasattr(_so, "_unload_llm")` |
| `test_orchestrator_no_llm_cache_symbol` | same | `not hasattr(_so, "_LLM_CACHE")` |
| `test_orchestrator_no_generate_with_llm_symbol` | same | `not hasattr(_so, "_generate_with_llm")` |
| `test_unload_llm_no_orchestrator_fallback_block` | `tests/test_loader_slot_primitives.py` (extend) | AST scan: no `story_orchestrator` import, no `_LLM_CACHE` ref in `unload_llm` body |
| `test_invalidate_cache_no_gpu_teardown_clears_dict` | `tests/test_loader_slot_primitives.py` (extend) | After fixture-populated cache, call helper, assert `LLM_CACHE == {"model_id": None, "slot": None, "cache_entry": None}` |
| `test_invalidate_cache_no_gpu_teardown_no_gpu_calls` | `tests/test_loader_slot_primitives.py` (extend) | AST scan of helper body: no `model.to`, no `torch.cuda.empty_cache`, no `torch.cuda.synchronize`, no `gc.collect` |
| `test_run_with_timeout_uses_safe_invalidation` | `tests/test_run_with_timeout_safe_invalidation.py` (new) | AST scan of `_run_with_timeout`: `invalidate_cache_no_gpu_teardown` call present, no `unload_llm` call in timeout-recovery branch |
| `test_audio_c7_byte_identical_b4` | `tests/test_audio_byte_identical.py` | **Canary at deletion** |

### Commit gate
9 tests green. **Audio C7 proxy holds.** Bug Bible 23/1/2xf. Forbidden sweep 0 runtime hits. BUG-LOCAL-226 marked FIXED. New BUG-LOCAL entry for TIMEOUT_RECOVERY also FIXED.

**OPERATOR:** runtime 5080 render after B4 push. Specifically exercise a timeout scenario if possible.

### Commit subject
`B4: delete _load_llm + _unload_llm + _LLM_CACHE + _generate_with_llm; simplify unload_llm; fix TIMEOUT_RECOVERY CUDA race via invalidate_cache_no_gpu_teardown`

---

## B5 — arm forbidden-pattern sweep extinction markers (~0.25 d)

### Code

**`docs/_s28_forbidden_sweep.py`** — add to `forbidden` regex:

```python
    # S31 B5: legacy LLM stack clean break -- 4 deleted symbols
    r"|\b_load_llm\b"
    r"|\b_unload_llm\b"
    r"|\b_LLM_CACHE\b"
    r"|\b_generate_with_llm\b"
    # S31 B5: preemptive lock -- one generate surface (Hard rule #5)
    r"|\bgenerate_text\b"
    r"|\bgenerate_with_llm\b"   # variant without underscore prefix
```

### Wire / Pytest
Manual sweep run. Confirm `docs/2026-05-14-S31-new-forbidden-hits.txt` empty.

### Commit gate
0 runtime hits. Bug Bible 23/1/2xf.

### Commit subject
`B5: arm S31 extinction markers (4 deleted + 2 preemptive)`

---

## B6 — writer + workflow + visual contract residuals (~1 d)

Four fixes bundled.

### Review

**Fix 1 — RSS slot mismatch** (`OTR_LedgerScriptWriter.py:1109`): passes `creative_writing_model`; post-B3 the path uses `request_slot("technical", ...)`. Slot label and resolved id disagree. Differing-slots mode loads creative model for technical-slot work.

**Fix 2 — Standalone self-test drift** (`OTR_LedgerScriptWriter.py:2836`): asserts 11; actual count is 15.

**Fix 3 — Workflow JSON link-row off-by-one** (`workflows/otr_scifi_16gb_full.json`): top-level `links[]` table has 4 rows with wrong `dst_slot` (off-by-one, two are out-of-range). Per-node `inputs[]` arrays are internally consistent so ComfyUI runtime probably routes correctly via the per-node arrays, but the top-level table is wrong, includes out-of-range slot indices, and the existing `test_workflow_json_guardrails` destructures `_dst_slot` without asserting on it.

**Fix 4 — OTR_VisualPromptCoercion model_id contract** (`visual/prompt_coercion.py`): S30 B5 deleted `OTR_VisualLLMSelector`; visual prompt coercion now depends on the writer's `creative_writing_model` broadcast. If the model_id input is unwired in a workflow, current behavior silently falls back to rule-based-only cleanup. Add defensive check that raises `MissingModelInputError` loud (matches cascade's post-S30 B3 pattern via `_otr_model_inputs.require_model`).

**Deferred from B6 (was Fix 3 in draft):** ungated PASS-tier `GatedModelError` recommendation. Catalog has NO ungated PASS-tier entry today — the only PASS-tier entry (Mistral-Nemo) is gated. Recommending a gated model in the gated-error message would tell the user "fix your token, then hit the same gate." Honest-gap message stays. File a `BUG-LOCAL-NNN: GatedModelError ungated recommendation deferred until ungated PASS soak lands`. Re-open when the S30 forward-work "soak validation of an ungated curated entry as vram_fit_tier=PASS" prerequisite completes.

### Code

**`nodes/OTR_LedgerScriptWriter.py:1105-1110`**:

```python
# OLD
        news_article = _fetch_rss_seed_or_die(
            rss_style_slug, creative_writing_model,
        )
# NEW
        # _fetch_science_news routes through request_slot("technical", ...).
        # Pass technical_model so slot label and resolved id agree.
        news_article = _fetch_rss_seed_or_die(
            rss_style_slug, technical_model,
        )
```

**`nodes/OTR_LedgerScriptWriter.py:2836`**:

```python
# OLD
        assert n_optional == 11, (
            f"optional widget count drift: {n_optional} "
            f"(expected 11 after S30 B2a two-widget split)"
        )
# NEW
        assert n_optional == 15, (
            f"optional widget count drift: {n_optional} "
            f"(expected 15: 11 widget-surface + 4 Phase 4 v4 "
            f"sampling knobs)"
        )
```

**`workflows/otr_scifi_16gb_full.json`** — fix 4 link rows (each is +1 above actual input slot):

```json
// OLD                                  // NEW
[14,  11, 0,  3, 2, "AUDIO"]    →    [14,  11, 0,  3, 1, "AUDIO"]
[20,  13, 0,  3, 3, "AUDIO"]    →    [20,  13, 0,  3, 2, "AUDIO"]
[25,  15, 0,  3, 4, "AUDIO"]    →    [25,  15, 0,  3, 3, "AUDIO"]
[105, 14, 1, 12, 4, "AUDIO"]    →    [105, 14, 1, 12, 3, "AUDIO"]
```

**`visual/prompt_coercion.py`** — add defensive check at run-entry of `OTR_VisualPromptCoercion`:

```python
# Inside the node's main run method, before any LLM path branches
from . import _otr_model_inputs as _OTRMI  # or wherever require_model lives
resolved_model_id = _OTRMI.require_model(
    model_id, slot="creative",
    error_hint=(
        "OTR_VisualPromptCoercion requires the writer's "
        "creative_writing_model broadcast output (S30 B5 deleted "
        "the local VisualLLMSelector). Wire writer output 4 "
        "(creative_writing_model) to this node's model_id input. "
        "Unwired = silent fallback to rule-based-only cleanup, "
        "which is not what you want post-S30."
    ),
)
```

Pattern matches `OTR_LedgerFreezeCascade.technical_model` check post-S30 B3.

### Wire
None graph-level for `otr_scifi_16gb_full.json` connections — Fix 3 corrects metadata on existing links, doesn't reroute. **Note:** `OTR_VisualPromptCoercion` is NOT instantiated in `otr_scifi_16gb_full.json` (0 instances confirmed). Fix 4 is a node-contract change that protects future workflows; this workflow needs no edit for that contract. If visual polish should be active in this workflow, separate workflow-edit task (out of scope for B6).

### Pytest

| Test | File | Asserts |
|---|---|---|
| `test_resolve_inputs_rss_uses_technical_model` | `tests/test_writer_input_resolve.py` (new) | Mock `_fetch_rss_seed_or_die`, creative≠technical, arg 2 == technical_model |
| `test_resolve_inputs_rss_default_config_baseline` | same | creative==technical==DEFAULT_LLM, forwarded id == DEFAULT_LLM |
| `test_workflow_link_rows_match_target_input_indexes` | `tests/test_workflow_link_target_indexes.py` (new) | For every node input with `link=L`: find top-level link row `row[0]==L`; assert `row[3]==node["id"]`; assert `row[4]==actual_input_idx`; assert `row[4] < len(target_node["inputs"])` |
| `test_visual_prompt_coercion_missing_model_id_raises_loud` | `tests/test_visual_prompt_coercion_contract.py` (new) | Instantiate node with model_id="" or None, call run, assert `MissingModelInputError` raised with non-empty `error_hint` |
| `test_visual_prompt_coercion_with_wired_model_id_proceeds` | same | Valid model_id → no exception, normal flow |
| (manual) | `python OTR_LedgerScriptWriter.py` | `[2/9] PASS: INPUT_TYPES schema (15 optional widgets...)` |

Red-on-parent for link-row test and visual-prompt-coercion test. Apply, green.

### Commit gate
5 pytest tests green. Pytest ~282 / 7 / 2. Audio C7 holds (default config unchanged). Forbidden sweep clean. Manual self-test 9/9 PASS.

**OPERATOR:** runtime 5080 render after B6 — confirm workflow JSON edits don't break ComfyUI load + queue.

### Commit subject
`B6: writer + workflow + visual residuals — RSS slot fix, self-test drift, workflow JSON link off-by-one, VisualPromptCoercion missing-model-id loud-fail`

---

## B7 — round-robin integration buffer (~0.5 d, variable)

Adjacent findings from any pending review. Folding rules:
- **A — corroborates B1-B6** → already done; skip.
- **B — small fix in clean-break neighborhood** → lands here.
- **C — substantial new contract** → separate sprint plan; do NOT fold.
- **D — flags S32 helper-routing territory** → adds to S32 plan; not in S31.

Empty commit skipped if no findings.

### Commit subject
`B7: round-robin integration — <summary>` or `B7: empty, skipped`

---

## B8 — sprint close (~0.5 d)

### Code
Final QA at `docs/2026-05-14-S31-final-qa-review.md` (mirror S30 format). ROADMAP refresh.

### Wire / Pytest
Full clean `pytest -q`. Confirm ~282 / 7 / 2.

### Commit gate
Audio C7 confirmed at every B1-B6 boundary. **OPERATOR: run post-S31 runtime release gate (next section) before declaring S31 shipped.**

### Commit subject
`B8: Sprint S31 close — legacy LLM stack clean break shipped, TIMEOUT_RECOVERY race fixed, residuals cleared`

---

## S31 acceptance table

| # | Check | Target |
|--:|---|---|
| 1 | Full pytest count | ~282 / 7 / 2 |
| 2 | Bug Bible regression | 23 / 1 / 2 |
| 3 | Audio C7 byte-identical (pytest proxy) | holds B1→B8 |
| 4 | Audio C7 byte-identical (runtime 5080) | confirmed B2, B4, B6 |
| 5 | Forbidden sweep | 0 runtime hits |
| 6-9 | 4 legacy symbols DELETED from `story_orchestrator.py` | ✅ |
| 10 | `_otr_model_loader.unload_llm` legacy fallback block | DELETED |
| 11 | `_otr_model_loader.invalidate_cache_no_gpu_teardown` | EXISTS, dict-only, no GPU calls |
| 12 | `_otr_model_loader.load_llm` owns bitsandbytes body | ✅ |
| 13 | `generate_text` / `generate_with_llm` (no-underscore variants) anywhere | 0 (preemptive lock) |
| 14 | `story_orchestrator._run_with_timeout` uses `invalidate_cache_no_gpu_teardown` | ✅ (CUDA race fixed) |
| 15 | External callers of legacy 4 symbols | 0 |
| 16 | Internal orchestrator callers switched | ✅ |
| 17 | RSS path passes `technical_model` | grep-clean at `_resolve_inputs:1109` |
| 18 | Standalone self-test 9/9 (15 widgets) | PASS |
| 19 | BUG-LOCAL-226 | FIXED at B4 |
| 20 | BUG-LOCAL-NNN (TIMEOUT_RECOVERY) | FIXED at B4 |
| 21 | Workflow JSON link rows match target input indexes | 0 violations |
| 22 | OTR_VisualPromptCoercion raises loud on unwired model_id | ✅ |
| 23 | New S31 extinction markers | 6 |
| 24 | BUG-LOCAL-NNN (ungated GatedModelError recommendation) | FILED, deferred until ungated PASS soak |

---

## S31 post-close runtime release gate (operator, after B8)

```text
Gate 1. Default workflow run: creative == technical == Mistral-Nemo.
        - workflow loads + queues without errors
        - audio output reaches the file
        - ledger meta shows creative == technical

Gate 2. Slot-swap run: creative != technical (e.g. creative=Mistral-Nemo,
        technical=Gemma-2B).
        - workflow loads + queues without errors
        - no OOM during slot transitions
        - ledger meta shows correct distinct values

Gate 3. VRAM recovery (post slot-swap):
        - nvidia-smi shows VRAM back to baseline after run
        - no leaked allocations across swap

Gate 4. RSS/news path attribution:
        - ledger meta confirms RSS rerank invoked TECHNICAL model
          (differing-slots config)

Gate 5. Timeout recovery (synthetic):
        - artificially induce a phase timeout if practical
        - ComfyUI does NOT crash with cudaErrorIllegalAddress
        - next phase loads cleanly after orphan thread completes

Gate 6. Final audio path integrity:
        - rendered episode plays end-to-end
        - audio C7 byte-identical to pre-S31 reference (default config)

Gate 7. Workflow JSON re-save:
        - load workflow in ComfyUI, save without changes, diff
        - 0 unexpected diffs (B6 link-row fix means re-save shouldn't
          touch previously-wrong rows)
```

Any gate failure → P0 BUG_LOG entry + hotfix commit on `s31-loader-clean-break` BEFORE branching S32.

---
---

# SPRINT S32 — dual-LLM completion: per-sub-pass routing

**Branch:** `s32-helper-per-subpass-routing`. Cut from `s31-loader-clean-break` @ B8.
**Headline commit:** **B1** (atomic signature refactor + writer wiring).

## S32 additional hard rules

R1. Default config audio C7 byte-identical holds (canary at every commit).
R2. Differing-slots audio gets its OWN baseline, established at B5 close, stable B5→B8.
R3. Helper signatures break cleanly. `pick_style(generate_fn, ...)` → `pick_style(*, creative_fn, technical_fn, ...)`. No back-compat.
R4. B1 is ATOMIC: signatures + writer wiring same commit (decoupling breaks the build).
R5. VRAM thrash budget: per-beat dispatch in `compose_line` is REJECTED. Critic-via-technical is gated by default-OFF widget.
R6. Slot transition accounting extends `meta` with `slot_calls_by_helper` and `slot_transitions_by_phase`.

## Architectural decisions (settled — no re-debate)

**D1.** `compose_line` critic per-beat dispatch in differing-slots: ~3.3 hr overhead per episode. REJECTED. Critic stays on `creative_fn` by default; new widget `use_technical_critic` (default OFF) opts in. Writer logs one-shot VRAM warning on opt-in + differing-slots.

**D2.** `lock_cast` schema validation: single-attempt technical, fail-fast. No internal retry tier. Writer-side caller triggers creative regen if T returns N/validation-fail. If T can't emit valid Y/N JSON, raise `CastValidationLLMError`.

**D3.** Outline retry: schema validation is pure pydantic (no LLM). The retry prompt is content regeneration → stays creative. No change in S32.

## S32 routing table

**C** = creative slot. **T** = technical slot.

| Phase | Helper | Sub-pass | S31 | S32 | Fires |
|---|---|---|--:|--:|---|
| Style picker | `pick_style` | Pass 1 inventor | C | **C** | once/ep |
| Style picker | `pick_style` | Pass 2 chooser | C | **T** | once/ep |
| News interpreter | `build_news_briefs` | V0 emit | T | T | once/ep |
| News interpreter | `build_news_briefs` | V1 retry | T | T | once/ep |
| News interpreter | `build_news_briefs` | V2 retry | T | T | once/ep |
| News interpreter | `build_news_briefs` | V3 fallback | T | T | once/ep |
| Cast lock | `lock_cast` | Generation | C | **C** | once/ep |
| Cast lock | `lock_cast` | Schema validation | C | **T** | once/ep per attempt |
| Outline | `generate_outline` | Composition | C | C | once/ep |
| Outline | `generate_outline` | Retry on fail | C | C | conditional |
| Composer | `compose_line` | Line composition | C | C | per-beat (~50-200) |
| Composer | `compose_line` | Critic check | C | **C default / T opt-in** | per-beat |
| Composer | `compose_line` | Grammarian fix | C | C | conditional per-beat |
| Polish | `polish_line` | Line polish | C (via for_polish) | C | conditional per-beat |
| Title regen | inline writer | Title generation | C | C | once/ep |

Changes from S31: 2 flips (`pick_style` pass 2 → T; `lock_cast` validation → T) + 1 opt-in (`compose_line` critic).

---

## B0 — branch cut + plan landing (~0.25 d)

Confirm S31 B8 + post-S31 runtime gate passed. ROADMAP refresh.

### Commit subject
`B0: branch cut + S32 helper per-sub-pass routing plan landing`

---

## B1 — helper signatures + writer wiring (ATOMIC) ⚑ HEADLINE (~1 d)

### Review
Pre-grep `pick_style lock_cast compose_line build_news_briefs` outside `OTR_LedgerScriptWriter.py`. Likely none. Surface and fold if any.

### Code

**Signatures** (`_otr_style_picker.py`, `_otr_casting.py`, `_otr_line_composer.py`, `news_interpreter.py`):

```python
# OLD
def pick_style(generate_fn, *, article_text, seed_pool, rng, model_id): ...
def lock_cast(generate_fn, *, brief, num_characters, ...): ...
def compose_line(generate_fn, *, beat, canon_header, cast_rows, ...): ...
def build_news_briefs(generate_fn, *, full_text, headline, ...): ...

# NEW (B1 atomic with writer wiring)
def pick_style(*, creative_fn, technical_fn, article_text, seed_pool, rng,
               model_id_for_meta): ...
def lock_cast(*, creative_fn, technical_fn, brief, num_characters, ...): ...
def compose_line(*, creative_fn, technical_fn, beat, canon_header, cast_rows,
                 use_technical_critic=False, ...): ...
def build_news_briefs(*, creative_fn, technical_fn, full_text, headline, ...):
    # B1: creative_fn unused; future-compatibility only. All sub-passes
    # internally route to technical_fn.
```

B1 routes ALL sub-passes internally through `creative_fn` (or `technical_fn` for `build_news_briefs`). No dispatch yet. Audio C7 holds.

**Writer wiring** (`OTR_LedgerScriptWriter.py`): each helper call site passes `creative_fn=creative_generate_fn`, `technical_fn=technical_generate_fn`. Slot scheduler already provides both.

### Wire
None graph-level.

### Pytest

| Test | File | Asserts |
|---|---|---|
| `test_pick_style_accepts_paired_generators` | `tests/test_helper_paired_signatures.py` (new) | Signature requires `creative_fn` + `technical_fn` kwargs |
| `test_lock_cast_accepts_paired_generators` | same | Same |
| `test_compose_line_accepts_paired_generators` | same | Same + `use_technical_critic` |
| `test_build_news_briefs_accepts_paired_generators` | same | Same (creative_fn unused but accepted) |
| `test_pick_style_internally_uses_creative_fn_default` | same | Distinct mocks; both passes hit creative_fn |
| `test_lock_cast_internally_uses_creative_fn_default` | same | Same |
| `test_compose_line_internally_uses_creative_fn_default` | same | `use_technical_critic=False`, all on creative_fn |
| `test_build_news_briefs_internally_uses_technical_fn` | same | All V0-V3 on technical_fn |
| `test_writer_passes_paired_generators` | `tests/test_writer_paired_wiring.py` (new) | Mock helpers, assert writer hands both fns |
| `test_audio_c7_byte_identical_b1` | `tests/test_audio_byte_identical.py` | Canary |

### Commit gate
10 tests green. Audio C7 holds. Bug Bible 23/1/2xf.

### Commit subject
`B1 ⚑: helper signatures refactored to paired generators + writer wired end-to-end (no dispatch yet)`

---

## B2 — `pick_style` pass 2 → T (~0.5 d)

One-line dispatch change. Update meta: `StylePick.pass1_slot = "creative"`, `pass2_slot = "technical"`.

### Pytest

| Test | File | Asserts |
|---|---|---|
| `test_pick_style_pass1_uses_creative` | `tests/test_pick_style_routing.py` (new) | Pass 1 hits creative_fn |
| `test_pick_style_pass2_uses_technical` | same | Pass 2 hits technical_fn |
| `test_pick_style_meta_records_per_pass_slot` | same | `StylePick.pass1_slot`/`pass2_slot` populated |
| `test_audio_c7_byte_identical_b2` | `tests/test_audio_byte_identical.py` | Default same-model byte-identical |

### Commit subject
`B2: pick_style pass 2 (chooser) dispatches to technical_fn`

---

## B3 — `lock_cast` schema validation → T (~0.5 d)

One-line dispatch change. Add `CastValidationLLMError` raise on malformed T output (per D2).

### Pytest

| Test | File | Asserts |
|---|---|---|
| `test_lock_cast_generation_uses_creative` | `tests/test_lock_cast_routing.py` (new) | Generation hits creative_fn |
| `test_lock_cast_validation_uses_technical` | same | Validation hits technical_fn |
| `test_lock_cast_validation_failfast_no_internal_retry` | same | Malformed T output → `CastValidationLLMError` |
| `test_lock_cast_validation_fail_triggers_writer_regen` | same | Validation N → writer-visible signal |
| `test_audio_c7_byte_identical_b3` | `tests/test_audio_byte_identical.py` | Default holds |

### Commit subject
`B3: lock_cast schema validation dispatches to technical_fn; fail-fast no internal retry`

---

## B4 — `compose_line` critic opt-in widget (~1 d)

### Code

**`OTR_LedgerScriptWriter.py` INPUT_TYPES** — add optional widget (count 15 → 16):

```python
"use_technical_critic": ("BOOLEAN", {
    "default": False,
    "tooltip": (
        "OPT-IN: route compose_line's critic sub-pass to "
        "technical_model. WARNING: differing-slots mode triggers "
        "C->T->C VRAM transitions per voiced beat (~30-60s "
        "overhead per beat on 16GB cards). Default OFF (critic "
        "on creative slot, no transitions)."
    ),
}),
```

**`_otr_line_composer.py:compose_line`** — accept `use_technical_critic: bool = False`; conditionally dispatch critic to technical_fn.

**Writer** — pass `resolved["use_technical_critic"]` into every `compose_line` call. One-shot warning at run start when `use_technical_critic=True AND creative != technical`.

**Standalone self-test bump** `OTR_LedgerScriptWriter.py:2836`: `assert n_optional == 16` with updated message.

### Pytest

| Test | File | Asserts |
|---|---|---|
| `test_compose_line_critic_default_uses_creative` | `tests/test_compose_line_routing.py` (new) | `use_technical_critic=False`, critic on creative_fn |
| `test_compose_line_critic_optin_uses_technical_differing_slots` | same | True + distinct fns → critic on technical_fn |
| `test_compose_line_critic_optin_default_config_no_change` | same | True + same model → byte-identical to default |
| `test_writer_logs_vram_warning_on_optin_differing_slots` | `tests/test_writer_paired_wiring.py` (extend) | Caplog: warning with overhead estimate |
| `test_writer_optional_widget_count_16` | (extend existing) | INPUT_TYPES optional == 16 |
| `test_audio_c7_byte_identical_b4` | `tests/test_audio_byte_identical.py` | Default holds |

### Commit subject
`B4: compose_line critic-via-technical opt-in widget (default OFF) + VRAM warning`

---

## B5 — `build_news_briefs` + outline retry + differing-slots baseline (~0.5 d)

Confirm `build_news_briefs` V0-V3 all on technical_fn. Investigate `_otr_outline.generate_outline` for LLM-side retry (per D3: stays creative; document rationale).

**This commit establishes the differing-slots audio baseline.** End-to-end with creative != technical exercises full routing. Capture audio reference; B6 verifies against this reference.

### Pytest

| Test | File | Asserts |
|---|---|---|
| `test_build_news_briefs_all_retries_use_technical` | `tests/test_helper_paired_signatures.py` (extend) | V0/V1/V2/V3 all on technical_fn |
| `test_outline_retry_uses_creative` | (extend existing) | LLM-side retry (if exists) on creative_fn |
| `test_audio_c7_byte_identical_b5_default` | `tests/test_audio_byte_identical.py` | Default holds |
| `test_audio_differing_slots_baseline_b5` | `tests/test_audio_byte_identical.py` (new fixture) | NEW differing-slots baseline captured |

**OPERATOR:** runtime 5080 render in both default and differing-slots configs.

### Commit subject
`B5: build_news_briefs paired-contract + outline retry + differing-slots audio baseline captured`

---

## B6 — slot transition accounting + meta stamping (~0.5 d)

Extend writer `meta`:
- `meta["slot_calls_by_helper"]`: per-helper per-slot call counts
- `meta["slot_transitions_by_phase"]`: ordered list of `(phase, from_slot, to_slot, model_id_change)`

Helpers return `meta_record` alongside main return. Writer aggregates.

### Pytest

| Test | File | Asserts |
|---|---|---|
| `test_meta_slot_calls_by_helper_shape` | `tests/test_meta_slot_transitions.py` (new) | End-to-end, meta dict shape correct |
| `test_meta_slot_transitions_by_phase_differing_slots` | same | Differing-slots: list includes pick_style C→T at pass 2, lock_cast C→T at validation |
| `test_default_config_zero_transitions` | same | Same model → `meta.slot_transitions == 0` |
| `test_audio_c7_byte_identical_b6_default` | `tests/test_audio_byte_identical.py` | Default holds |
| `test_audio_differing_slots_baseline_b6` | same | B5 differing-slots baseline reproduces |

### Commit subject
`B6: slot transition accounting + per-helper meta stamping`

---

## B7 — round-robin integration buffer (~0.5 d)

Same A/B/C/D rules as S31 B7.

### Commit subject
`B7: round-robin integration — <summary>` or `B7: empty, skipped`

---

## B8 — sprint close (~0.5 d)

Final QA at `docs/<date>-S32-final-qa-review.md`. ROADMAP refresh.

### Commit subject
`B8: Sprint S32 close — dual-LLM completion shipped, per-sub-pass routing live`

---

## S32 acceptance table

| # | Check | Target |
|--:|---|---|
| 1 | Full pytest count | ~302 / 7 / 2 |
| 2 | Bug Bible regression | 23 / 1 / 2 |
| 3 | Audio C7 default (pytest proxy) | holds B1→B8 |
| 4 | Audio differing-slots (pytest proxy) | baseline B5; holds B5→B8 |
| 5 | Audio C7 default (runtime 5080) | confirmed B5, B6 |
| 6 | Audio differing-slots (runtime 5080) | new baseline confirmed B5, B6 |
| 7 | Forbidden sweep | 0 runtime hits |
| 8-11 | Helper signatures (4) accept paired generators | ✅ |
| 12 | pick_style pass 2 → technical_fn | ✅ |
| 13 | lock_cast schema validation → technical_fn | ✅ |
| 14 | compose_line critic-via-technical opt-in available, default OFF | ✅ |
| 15 | build_news_briefs all V0-V3 on technical_fn | ✅ |
| 16 | Writer wires paired generators end-to-end | ✅ |
| 17 | `meta.slot_calls_by_helper` populated | ✅ |
| 18 | `meta.slot_transitions_by_phase` populated | ✅ |
| 19 | Default config slot_transitions == 0 | ✅ |
| 20 | VRAM warning logged on opt-in + differing-slots | ✅ |
| 21 | Writer optional widget count | 16 (was 15 at S31) |

---

## S32 post-close runtime release gate (operator)

Mirror S31 gate, both configs:

```text
Default config gates (1-7 from S31)
Differing-slots config gates:
  + ledger meta shows slot_transitions > 0
  + per-helper slot calls match routing table
  + VRAM warning fires when use_technical_critic=True
  + no OOM during sub-pass dispatch
  + differing-slots audio output matches B5 baseline
```

---

# Post-S32 forward work

## S33 — editor-only cleanup passes (auditors retired)

**Principle:** every cleanup node must actually edit the story. Audit-only nodes that emit reports nobody acts on are pure overhead and get retired. Confirmed by Jeffrey 2026-05-14.

### Restore

| Restored as | What it does | Slot | Default |
|---|---|---|---|
| Extend writer's `enable_polish_pass` widget with `polish_announcer_beats` differentiation | Per-line polish on announcer beats, opt-in separately from character-line polish (closes the S30 Phase 3 deletion gap for the announcer-specific case) | Creative (via `for_polish`) | OFF |

### Retire (delete, arm sweep markers)

| Node / phase | Why retired |
|---|---|
| `OTR_LedgerFreezeCascade` Phase 1 auditor | Audit-only — reports issues but doesn't edit. Drop. |
| `OTR_LedgerFreezeCascade` Phase 9 post-edit auditor | Audit-only — verifies Phase 2 edits landed but doesn't itself edit. If Phase 2 can corrupt the ledger, that's a hard-fail bug at write time, not a soft-check afterward. Drop. |

### Keep as-is

| Node / phase | Why kept |
|---|---|
| `OTR_LedgerFreezeCascade` Phase 2 Script Doctor | Rewrites flagged lines — actual editor. |
| `OTR_LedgerFreezeCascade` Phase 7 audio readiness | Pre-flight check, not story content. Different category from story auditors. |
| `OTR_LedgerFreezeCascade` Phase 8 video readiness | Same — pre-flight. |
| Writer's existing `enable_polish_pass` per-line polish | Already edits via `make_polish_generate_fn`. |

### Permanent retire (already deleted at S30, S33 confirms they stay gone)

- Phase 3 polish (cascade-side variant — writer-side polish covers the same need)
- Phase 4 scene coherence audit
- Phase 4.5 smart suggestion
- Phase 5 voice drift detection
- Phase 6 episode arc Editor notes

### S33 sweep marker additions

After S33 lands, forbidden-pattern sweep gains markers for:
- `OTR_LedgerFreezeCascade` Phase 1 method names (TBD at S33 review — likely `_phase_1_*` or `_auditor_*`)
- `OTR_LedgerFreezeCascade` Phase 9 method names (TBD at S33 review — likely `_phase_9_*` or `_post_edit_auditor_*`)

Net codebase effect: S33 makes the cascade smaller, not larger. One small widget extension (~0.5 d) plus two deletions plus sweep markers. Sprint sized ~2-3 d / 5-6 commits.

## Other deferred items (independent of cleanup philosophy)

- **Soak validation of an ungated curated entry as `vram_fit_tier="PASS"`** — prerequisite for landing the `UNGATED_PASS_RECOMMENDATION` constant in `GatedModelError` (deferred from S31 B6). Catalog currently has zero ungated PASS-tier entries; the three ungated curated models are all WARN-tier. Until this soak completes, `GatedModelError` keeps its honest-gap message (no recommendation). When complete, file the follow-up patch to extend `auto_download_if_missing` raise path with the recommendation.
- **Audio-intentional sprint** — model-author `generation_config.json` respect for polish path. Polish kwargs stop being hardcoded `_POLISH_TOP_P=0.9` / `_POLISH_DO_SAMPLE=True`; instead pull from each model's published `generation_config.json`. Changes WHAT KWARGS polish uses, not WHERE it runs or WHICH MODEL handles it. Own audio C7 baseline-roll. Position: after S33.
- **Loader API consolidation** — split `_otr_model_loader` (~700+ LOC post-S31) into `{__init__.py, bnb_profiles.py, generation.py}`. Post-port hygiene; behavior unchanged.
- **Sprint C: `meta.story_brief` v2** — S30 parent sequencing.
- **Batched-dispatch compose_line refactor** — if D1 opt-in proves limiting in practice, restructure out of per-beat hot path into compose-all-then-critic-batch pipeline.
