# Question -- 2026-05-04

# BUG-LOCAL-098: bitsandbytes NF4 silently fails on second `_load_llm` after `_unload_llm`

## Stack

- ComfyUI custom node (Python 3.12, torch 2.10.0+cu130, CUDA 13.0)
- Single RTX 5080 Laptop, 16 GiB VRAM, Blackwell sm_120, Windows
- transformers (modern, supports BitsAndBytesConfig load_in_4bit)
- bitsandbytes (modern, NF4-capable build)
- Mistral-Nemo-Instruct-2407 12B model
- Goal: load via `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")` so the model fits in 16 GiB VRAM

## Symptom

Three back-to-back inferences on the FIRST loaded model work perfectly:
- Allocation: ~7-8 GiB (NF4 size for 12B Mistral-Nemo) ✓
- Weight loading speed: 22 it/s for 363 weights = 16 seconds (slow because bnb is quantizing each weight on the way in) ✓
- Inferences run, no OOM ✓

After the third inference, `_unload_llm()` runs cleanly:
- `model.cpu()` walks weights to RAM
- `del _LLM_CACHE["model"]` + `del _LLM_CACHE["tokenizer"]`
- `gc.collect()` + `torch.cuda.empty_cache()` + `torch.cuda.ipc_collect()`
- `comfy.model_management.unload_all_models()` + `soft_empty_cache()`
- Result: VRAM allocated=0.02 GiB, reserved=0.10 GiB ← clean unload

Then the SECOND `_load_llm()` for the next phase reloads Mistral-Nemo. Same exact code path:
- Same `model_id`
- Same `BitsAndBytesConfig(load_in_4bit=True, ...)` config
- Same canonical snapshot path passed to `from_pretrained` (BUG-LOCAL-085 fix; this part is verified)
- Log says `Enabling 4-bit quantization (NF4)` ← the config IS being passed
- Weight loading speed: **33 it/s for 363 weights = 10 seconds** ← this is the SMOKING GUN. NF4 quantization step is being SKIPPED.
- Allocation: **24+ GiB** ← fp16 size for 12B Mistral-Nemo. The config was silently ignored.
- First inference: torch.OutOfMemoryError because 24 GiB > 16 GiB device limit.

## Diagnosis

bitsandbytes carries module-level state that survives `_unload_llm()`. After the first quantization succeeds, the module's internal `_is_initialized` (or equivalent) flag stays True, the CUDA context references go stale (point at evicted memory), and on the second `from_pretrained` call bitsandbytes short-circuits the quantization init -- presumably checking the stale flag -- and `BitsAndBytesConfig(load_in_4bit=True)` is silently ignored. Result: model loads at fp16, no warning, no error, just OOM at first inference.

The fact that:
1. transformers reports `Enabling 4-bit quantization (NF4)` (config visible)
2. Weight loading speed drops from 22 it/s (quantizing) to 33 it/s (raw load)
3. Allocated VRAM matches fp16 size exactly

...all confirm bitsandbytes is no-op'ing the quantization on second load.

## What changed since v1.7 (which worked)

v1.7 of this codebase ran Mistral-Nemo NF4 reliably across multiple phases without this OOM. Diff vs current is mostly in non-LLM code (visual pipeline, audio compositing). The `_load_llm` / `_unload_llm` shape is functionally identical between v1.7 and HEAD. The known LLM-touching changes since v1.7:
- BUG-LOCAL-085 (commit 56cf493): added `_otr_hf_env.py` resolver that passes a canonical snapshot directory path to `from_pretrained` instead of the model_id. Verified working on the FIRST load.
- A few cache-mismatch field additions (context_cap, budget_profile, model_evicted_to_cpu detection).

The cache_deltas check at the start of `_load_llm` will fire `_unload_llm()` if any field drifted. Most likely the OpenClose path triggers `model_evicted_to_cpu` because accelerate parks weights on CPU under memory pressure between phases.

## Three candidate fix paths -- which is safest?

### Path 1: Module-level reset of bitsandbytes during `_unload_llm`

```python
# In _unload_llm, after the existing cleanup:
import sys
for modname in list(sys.modules.keys()):
    if (modname.startswith("bitsandbytes")
        or modname == "transformers.integrations.bitsandbytes"):
        del sys.modules[modname]
```

Forces re-import on next `_load_llm`. RISKY because torch / accelerate / transformers may hold strong references to bitsandbytes internals (linear layer factories, CUDA stream handles).

### Path 2: Skip `_unload_llm` for same-model_id reloads; just `model.cuda()`

When `cache_deltas == [("model_evicted_to_cpu", ...)]` (only) AND `quantized=True` AND `model_id` matches the cached one, do `_LLM_CACHE["model"].cuda()` to bring weights back instead of unload+reload. Sidesteps the bitsandbytes-state issue entirely because the model is never unloaded. Simplest semantically. Only activates when the eviction was the ONLY mismatch.

### Path 3: Post-load NF4 assertion (tripwire)

After `from_pretrained()`, check `torch.cuda.memory_allocated()` against an expected NF4 ceiling (e.g. <10 GiB for Mistral-Nemo 12B NF4). If higher, raise a loud `RuntimeError("BUG-098: NF4 silently failed; restart ComfyUI")`. This is a TRIPWIRE not a fix; the run still fails, but with a clear error instead of a silent OOM cascade.

## Question

1. Which fix path is safest? Path 2 looks cleanest to me (sidestep the bug rather than try to patch bitsandbytes' state machine). Am I missing something obvious?
2. Is there a 4th path I haven't considered? E.g., is there a transformers / bitsandbytes API call that explicitly resets the quantizer state without re-importing the module?
3. Is the diagnosis correct? Specifically: is the 22-vs-33 it/s delta on weight loading the right read for "quantization no longer applying"? Or is there another explanation for the speed change that wouldn't imply NF4 dropping?
4. v1.7 worked. What might have changed in our environment (transformers / bitsandbytes versions) that makes this fail NOW? The codebase delta isn't compelling -- could it be a library version regression that's unrelated to my code?

Constraints:
- Cannot upgrade torch/CUDA (pinned at 2.10.0+cu130 for Blackwell sm_120 support)
- Cannot break the working FIRST-load NF4 path
- Single 16 GiB GPU; no fallback to fp16 acceptable for 12B models
- Must work with transformers + bitsandbytes (the stack used by every quantized HF model)

Be concrete. Code snippets welcome. If you want to recommend Path 2 with a specific guard condition, write the actual `if` block.
