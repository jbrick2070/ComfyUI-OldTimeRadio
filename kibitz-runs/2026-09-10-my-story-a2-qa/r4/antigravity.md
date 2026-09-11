VERDICT: build-ready as-is? yes-with-fixes. Core architecture and test boundaries pass cleanly in the ComfyUI runtime environment, but an unhandled NoneType crash in inspect_native_prompt_fit on non-provider prompts and a scalar-only EOS equality bug in constrained generation must be fixed before lock.

MUST-FIX BEFORE BUILD:
1. [contract-inspect-fit] `nodes/_otr_model_loader.py:2320` raises `TypeError: int() argument must be a string, a bytes-like object or a real number, not 'NoneType'` when `max_new_tokens=None` is passed on standard messages.
   - Defect: `inspect_native_prompt_fit` computes `requested = (prepared["context_cap"] if prepared["reserve_remaining"] else max(1, int(max_new_tokens)))`. When inspecting standard messages or text prompts that do not carry `_otr_reserve_remaining_output_capacity` (such as standard prompts routed via `inspect_structured_fit`), `prepared["reserve_remaining"]` is `False`. Passing `max_new_tokens=None` evaluates `int(None)`, throwing an unhandled `TypeError`. In `tests/test_generation_budget.py:112`, this crash was masked because the test fixture wrapped its prompt in `ProviderCapacityMessages`, which sets `reserve_remaining = True`.
   - Concrete fix: In `nodes/_otr_model_loader.py:2319-2321`, fall back to full capacity when `max_new_tokens is None`:
     ```python
     requested = (prepared["context_cap"]
                  if (prepared["reserve_remaining"] or max_new_tokens is None)
                  else max(1, int(max_new_tokens)))
     ```

2. [contract-grammar-lifecycle] `nodes/_otr_constrained_generate.py:374` checks EOS token equivalence assuming `tokenizer.eos_token_id` is an `int`, breaking models with sequence or composite EOS tokens.
   - Defect: Line 374 performs `ended_with_eos = (bool(len(generated_ids)) and tokenizer.eos_token_id is not None and int(generated_ids[-1]) == tokenizer.eos_token_id)`. In HuggingFace Transformers, `eos_token_id` can be a `list[int]` or `set[int]` (e.g. Qwen and Llama families). Comparing `int == list` evaluates to `False`. When generation finishes on an EOS token at or above `effective_max_new_tokens`, `make_constrained_generate_fn` misclassifies the output as unfinished and raises a false `PromptContextOverflowError(phase="output_limit")`.
   - Concrete fix: Mirror the robust multi-token normalization from `nodes/OTR_LedgerScriptWriter.py:1079-1083`:
     ```python
     eos = tokenizer.eos_token_id
     eos_set = {int(v) for v in (eos if isinstance(eos, (list, tuple, set)) else (eos,)) if v is not None}
     ended_with_eos = bool(len(generated_ids)) and int(generated_ids[-1]) in eos_set
     ```

SHOULD-FIX:
1. [contract-inspect-fit] `nodes/_otr_structured_call.py:600` leaves `max_new_tokens` mandatory with no default value.
   - Defect: `def inspect_structured_fit(slot_fn, prompt, schema, *, max_new_tokens: int | None, text_parser=None)` accepts `max_new_tokens: int | None` but lacks a default value, unlike `structured_call` at line 776 which defaults `max_new_tokens: int | None = _STRUCTURED_MAX_NEW_TOKENS`. Callers inspecting a slot without passing an explicit token count will raise `TypeError: inspect_structured_fit() missing 1 required keyword-only argument: 'max_new_tokens'`.
   - Concrete fix: Change signature at `nodes/_otr_structured_call.py:600` to `*, max_new_tokens: int | None = _STRUCTURED_MAX_NEW_TOKENS, text_parser=None`.

2. [contract-prepare-native-prompt] `nodes/_otr_model_loader.py:2392-2412` (`make_generate_fn`) and lines 2600-2620 (`make_polish_generate_fn`) redundantly re-extract message attributes and re-measure tensor dimensions after `prepare_native_prompt`.
   - Defect: Both functions manually inspect `_otr_require_full_output_budget`, `_otr_reserve_remaining_output_capacity`, and `inputs["input_ids"].shape[1]`, duplicating what `prepare_native_prompt` already extracts and packages into `prepared["require_full_output"]`, `prepared["reserve_remaining"]`, and `prepared["prompt_tokens"]`.
   - Concrete fix: Read directly from `prepared` in both functions matching the clean pattern in `nodes/_otr_constrained_generate.py:260-272`.

OPTIONAL / NICE-TO-HAVE:
1. [contract-pin-cache-key] In `nodes/_otr_model_loader.py:214`, `_try_cache_hit_locked` annotates `policy_key: str | None = None`. At runtime, native HF passes a tuple `_hf_key = (_policy.cache_key(), _context_pin)`. Update the type annotation to `policy_key: Any = None` or `tuple | str | None = None` to satisfy static type checkers.

CUT THESE:
1. [contract-load-gate-output-room] `check_context_window` in `nodes/_otr_loader_backends.py:79-85`.
   - Why safe to cut: The function is now a no-op stub (`return None`). Calls inside `nodes/_otr_model_runtime.py` were eliminated. If external nodes do not import this symbol, delete it completely; if preserved for compatibility, keep only a one-line pass.
2. [contract-native-capacity] Module-level `HARD_VRAM_CONTEXT_LIMIT = _hard_vram_context_limit() or DEFAULT_CONTEXT_ESTIMATE` in `nodes/_otr_model_catalog.py:1823`.
   - Why safe to cut: Runtime context resolution reads `_context_pin` dynamically per request. Module-level export is dead code for execution and risks giving consumers the impression that setting the env var after import is ignored.

VERIFY-AT-BUILD checklist:
1. Confirm execution environment: Test execution requires Python environment with PyTorch >= 2.4 (e.g. `C:/Users/jeffr/Documents/ComfyUI/.venv/Scripts/python.exe` with PyTorch 2.10.0+cu130). Running with system Python 3.10 fails due to PyTorch 2.0.1+cpu incompatibility with `transformers`.
2. Confirm GGUF fixture isolation: Verify that the 15 failures in `test_gate_prices_the_policy_context.py`, `test_llm_runtime_policy.py`, and `test_model_catalog_scan.py` are the known catalog fixture/GGUF-disabled items from 2026-09-06 directive and not new regressions.
3. Confirm uncurated first download: Verify that an uncurated model downloaded from HF resolves capacity via `AutoConfig` and loaded decoder config without mutating `max_position_embeddings`.
4. Confirm context pin reactivity: Verify that modifying `OTR_HARD_VRAM_CONTEXT_LIMIT` between runs triggers a cache miss, unloads the resident model via `_self_unload`, and reloads with the updated pin.
5. Confirm minimum output budget: Verify that native generation and Google API routes execute with 1 token remaining capacity when unconstrained, while OpenRouter and Comfy retain explicit floors.

[ASSUMPTION] External callers outside the repository do not rely on `check_context_window` raising `RuntimeError`.
