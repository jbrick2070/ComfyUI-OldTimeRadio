# Synthesis -- 2026-05-04

**Question:** # BUG-LOCAL-098: bitsandbytes NF4 silently fails on second `_load_llm` after `_unload_llm`

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

---

## ChatGPT (gpt-5.5)

Short answer: **use Path 2, but add Path 3 as a mandatory tripwire.** Do **not** do Path 1 except as a last-ditch debug experiment. I would also add a “Path 4” fallback: **run the LLM in a recyclable subprocess** if same-process bitsandbytes reload remains unreliable.

I cannot cite exact file/line numbers because you did not include the repo or the relevant `_load_llm` / `_unload_llm` file, but the change belongs at the point in `_load_llm()` where `cache_deltas` currently causes `_unload_llm()`.

---

## 1. Safest fix path

### Recommendation

Use:

1. **Path 2: avoid unload/reload when the only delta is `model_evicted_to_cpu`**
2. **Path 3: assert that the loaded model is actually bitsandbytes 4-bit**
3. If Path 2 cannot rehydrate cleanly on your installed transformers/bitsandbytes combo, fall back to **process isolation**, not `sys.modules` surgery.

The principle is sound: if the cached model is already the correct model, correct tokenizer, correct quantization mode, correct context profile, etc., and the only mismatch is “weights are currently on CPU”, then **do not destroy the model object and ask bitsandbytes/transformers to reconstruct it**. Keep the existing quantized module graph and move it back to CUDA.

That is the smallest change with the biggest payoff.

---

## Why Path 1 is risky

This is the riskiest option:

```python
import sys
for modname in list(sys.modules.keys()):
    if (modname.startswith("bitsandbytes")
        or modname == "transformers.integrations.bitsandbytes"):
        del sys.modules[modname]
```

Problems:

- Existing model classes may still be live.
- Transformers / accelerate may retain references to classes, hooks, quantizers, or utility functions imported from the old module object.
- On Windows, deleting Python module entries does **not** unload the native DLLs cleanly.
- Reimporting can give you two logically different `bitsandbytes` module objects in the same process.
- Type checks can become weird: an object created by the old `bitsandbytes.nn.Linear4bit` is not necessarily the same class object as the newly imported `bitsandbytes.nn.Linear4bit`.

This kind of fix can appear to work once and then produce nastier nondeterministic failures later. I would not ship it.

---

## Path 2 guard condition

Put the bypass before your existing “cache mismatch means unload” branch.

Something like this:

```python
def _delta_keys(cache_deltas):
    return [d[0] if isinstance(d, tuple) else d for d in cache_deltas]


def _is_bnb_4bit_model(model) -> bool:
    """
    Avoid depending too much on exact bitsandbytes import identity.
    This survives some module reload weirdness and version differences.
    """
    for m in model.modules():
        cls_name = m.__class__.__name__
        mod_name = m.__class__.__module__
        if cls_name == "Linear4bit" and mod_name.startswith("bitsandbytes"):
            return True

    # Secondary check: HF usually sets this.
    if getattr(model, "is_loaded_in_4bit", False):
        return True

    return False


def _cached_model_device_types(model) -> set[str]:
    devices = set()
    for p in model.parameters(recurse=True):
        devices.add(p.device.type)
    for b in model.buffers(recurse=True):
        devices.add(b.device.type)
    return devices


def _try_rehydrate_cached_llm_to_cuda(
    *,
    requested_model_id: str,
    requested_quantized: bool,
    cache_deltas: list,
    device: str = "cuda:0",
) -> bool:
    """
    Return True if handled. Return False if normal load path should continue.
    Raise RuntimeError for suspicious states.
    """
    if not cache_deltas:
        return False

    delta_keys = _delta_keys(cache_deltas)

    if delta_keys != ["model_evicted_to_cpu"]:
        return False

    model = _LLM_CACHE.get("model")
    tokenizer = _LLM_CACHE.get("tokenizer")

    if model is None or tokenizer is None:
        return False

    if _LLM_CACHE.get("model_id") != requested_model_id:
        return False

    if not requested_quantized:
        return False

    if not _LLM_CACHE.get("quantized", False):
        return False

    if not _is_bnb_4bit_model(model):
        raise RuntimeError(
            "BUG-LOCAL-098 guard: cached model claims quantized=True, "
            "but no bitsandbytes Linear4bit modules were found. Refusing reload."
        )

    devices_before = _cached_model_device_types(model)

    # This path should only be for a model parked on CPU.
    # If it is already on CUDA, repair the cache metadata and continue.
    if devices_before == {"cuda"}:
        _LLM_CACHE["model_evicted_to_cpu"] = False
        model.eval()
        return True

    if "cuda" in devices_before and "cpu" in devices_before:
        raise RuntimeError(
            f"BUG-LOCAL-098 guard: cached quantized model is split across devices: "
            f"{devices_before}. Refusing silent repair."
        )

    # Free Comfy-side VRAM before bringing the LLM back.
    try:
        import comfy.model_management as mm
        mm.unload_all_models()
        mm.soft_empty_cache()
    except Exception:
        # Do not make this fatal unless your project expects comfy to exist here.
        pass

    import gc
    import torch

    gc.collect()
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass

    before_gib = torch.cuda.memory_allocated() / 1024**3

    try:
        # Depending on transformers/bitsandbytes versions, either .cuda()
        # or .to("cuda") may be blocked for 4-bit models.
        # Try the least surprising API first.
        model.cuda()
    except Exception as e:
        raise RuntimeError(
            "BUG-LOCAL-098: cached NF4 model existed, but could not be moved "
            "back to CUDA. Do not fall back to from_pretrained() in this process; "
            "that is the path known to silently reload fp16. Use persistent LLM "
            "residency or a recyclable LLM subprocess."
        ) from e

    model.eval()

    after_gib = torch.cuda.memory_allocated() / 1024**3
    delta_gib = after_gib - before_gib

    if not _is_bnb_4bit_model(model):
        raise RuntimeError(
            "BUG-LOCAL-098: after CUDA rehydrate, model no longer appears to be "
            "bitsandbytes 4-bit. Refusing to continue."
        )

    # Mistral-Nemo 12B NF4 should be roughly 7-9 GiB on GPU.
    # Give some margin for buffers/context/library variation.
    if delta_gib > 11.0:
        raise RuntimeError(
            f"BUG-LOCAL-098: CUDA rehydrate consumed {delta_gib:.2f} GiB, "
            "which is too high for Mistral-Nemo 12B NF4. Refusing to continue."
        )

    _LLM_CACHE["model_evicted_to_cpu"] = False
    _LLM_CACHE["device"] = device

    return True
```

Then in `_load_llm()`:

```python
# Existing code probably computes cache_deltas before deciding to unload.
if cache_deltas:
    handled = _try_rehydrate_cached_llm_to_cuda(
        requested_model_id=model_id,
        requested_quantized=quantized,
        cache_deltas=cache_deltas,
        device="cuda:0",
    )

    if handled:
        return _LLM_CACHE["model"], _LLM_CACHE["tokenizer"]

    # Existing behavior for real mismatches:
    _unload_llm()
```

Important behavior: **if rehydrate fails, do not silently fall back to `from_pretrained()` in the same process for this known failure mode.** Raise loudly or use a subprocess fallback.

---

## 2. Add the post-load NF4 assertion anyway

Even if you implement Path 2, you still want a tripwire after every `from_pretrained()`.

Do not rely only on `torch.cuda.memory_allocated()`. Also check the module graph.

Example:

```python
def _assert_bnb_nf4_loaded_or_die(
    model,
    *,
    model_id: str,
    load_vram_delta_gib: float | None = None,
    max_expected_gib: float = 11.0,
):
    linear4_count = 0

    for m in model.modules():
        cls_name = m.__class__.__name__
        mod_name = m.__class__.__module__
        if cls_name == "Linear4bit" and mod_name.startswith("bitsandbytes"):
            linear4_count += 1

    is_loaded_in_4bit = bool(getattr(model, "is_loaded_in_4bit", False))

    if linear4_count == 0 and not is_loaded_in_4bit:
        raise RuntimeError(
            "BUG-LOCAL-098: expected bitsandbytes NF4 model, but the loaded model "
            f"does not contain bitsandbytes Linear4bit modules. model_id={model_id!r}. "
            "Refusing to continue because this is probably an fp16 load."
        )

    if load_vram_delta_gib is not None and load_vram_delta_gib > max_expected_gib:
        raise RuntimeError(
            "BUG-LOCAL-098: expected NF4 load, but CUDA allocation delta was "
            f"{load_vram_delta_gib:.2f} GiB, above expected ceiling "
            f"{max_expected_gib:.2f} GiB for Mistral-Nemo 12B NF4. "
            "Refusing to continue."
        )
```

Use it around `from_pretrained()`:

```python
import torch

before_gib = torch.cuda.memory_allocated() / 1024**3

model = AutoModelForCausalLM.from_pretrained(
    snapshot_path,
    quantization_config=bnb_config,
    device_map={"": 0},
    torch_dtype=torch.bfloat16,
    local_files_only=True,
    # other existing args...
)

after_gib = torch.cuda.memory_allocated() / 1024**3
delta_gib = after_gib - before_gib

_assert_bnb_nf4_loaded_or_die(
    model,
    model_id=model_id,
    load_vram_delta_gib=delta_gib,
    max_expected_gib=11.0,
)
```

For your 14.5 GiB hard ceiling, I would set the assert around **10.5–11.0 GiB** for Mistral-Nemo 12B NF4. Your observed good load is 7–8 GiB, so 11 GiB gives reasonable slack while still catching fp16.

---

## 3. Is there a public bitsandbytes reset API?

To my knowledge: **no, not a supported public API that safely resets all bitsandbytes CUDA / quantizer state inside a live Python process.**

There are internal modules and lazy initialization paths, but I would not build production behavior around them, especially on:

- Windows
- CUDA 13.0
- Blackwell `sm_120`
- Python 3.12
- ComfyUI long-lived process
- strict determinism requirements

`torch.cuda.empty_cache()` also does not reset the CUDA context. It only releases cached blocks from PyTorch’s allocator. It will not reset bitsandbytes extension state.

So if same-process reload is unreliable, the clean reset boundary is the **process**, not `sys.modules`.

---

## 4. A stronger Path 4: recyclable LLM subprocess

If Path 2 is blocked because your installed transformers/bitsandbytes disallows `.cuda()` / `.to()` on 4-bit models, the robust local-only solution is:

> Put the LLM in a dedicated worker process. When you need a true unload/reset, terminate the worker process and spawn a fresh one.

This gives you a real reset of:

- Python module globals
- bitsandbytes native extension state
- CUDA context
- allocator state
- accelerate hooks
- stale device references

This is more plumbing than Path 2, but far safer than deleting `sys.modules`.

High-level shape:

```text
ComfyUI process
  └── LLM worker process
        - imports torch/transformers/bitsandbytes
        - loads Mistral-Nemo NF4
        - runs requested prompts
        - returns text via multiprocessing Pipe/Queue
        - exits when told to unload
```

On Windows you must use spawn-style multiprocessing anyway, which is actually helpful here.

For byte-identical output, send the same deterministic generation config into the worker every time:

```python
do_sample=False
temperature=None
top_p=None
top_k=None
num_beams=1
use_cache=True
```

Or, if you sample intentionally, send fixed seeds and enforce deterministic settings. But for radio-drama text generation, I strongly recommend greedy or otherwise fully deterministic decoding if rule C7 matters.

---

## 5. Is your diagnosis correct?

### Mostly plausible, but I would not overfit to `_is_initialized`

Your conclusion that the second load is not actually NF4 is very likely correct.

However, this specific internal explanation:

> bitsandbytes module-level `_is_initialized` stays True, stale CUDA refs, quantization init short-circuits

is plausible but not proven from the evidence given.

The evidence proves something slightly broader:

- `BitsAndBytesConfig` is visible to your code/logs.
- First load creates a compact 4-bit-ish memory footprint.
- Second load produces an fp16-sized memory footprint.
- Second load does not behave like the first quantizing load.

That means either:

1. transformers did not activate the bitsandbytes quantization path on the second load;
2. transformers activated it but the replacement/conversion did not happen;
3. bitsandbytes/accelerate state caused quantized modules not to be constructed correctly;
4. some interaction with `device_map`, cache state, or local snapshot metadata caused a different load pathway.

The `_is_initialized` stale-state theory is one possible implementation-level cause, but I would avoid putting that as a hard claim in the error message. Use wording like:

> “NF4 quantized load did not materialize; refusing to continue.”

not:

> “bitsandbytes `_is_initialized` stale flag failed.”

---

## 6. Is the 22 it/s vs 33 it/s signal meaningful?

It is a useful smell, but not a proof by itself.

Other possible reasons for the speed difference:

- Windows filesystem cache warmed up.
- Safetensors metadata/pages are hot.
- CUDA kernels/extensions already initialized.
- Python import paths already warm.
- HF/transformers internal caches hot.
- Antivirus/file scanning effects are different on second pass.

So I would not use the progress speed delta as the primary detector.

The stronger detectors are:

1. CUDA allocation delta.
2. Count of `bitsandbytes.nn.Linear4bit` modules.
3. `getattr(model, "is_loaded_in_4bit", False)`.
4. `model.get_memory_footprint()` if available.
5. Parameter/module class inspection.

The speed delta is consistent with NF4 being skipped, but memory footprint plus module inspection is what you should trust.

---

## 7. Why v1.7 may have worked

Yes, this could absolutely be a library/environment regression unrelated to your codebase.

Likely suspects:

### `transformers`

The quantization integration has changed over time. Modern transformers routes quantized loading through quantizer classes and integration layers that have seen churn.

Changes that could matter:

- `BitsAndBytesConfig` handling.
- `device_map` behavior.
- `low_cpu_mem_usage` defaults.
- accelerate integration.
- local path vs model ID resolution.
- quantization config merge behavior.

### `bitsandbytes`

Your stack is unusual and new:

- CUDA 13.0
- Blackwell `sm_120`
- RTX 5080 Laptop
- Windows
- Python 3.12
- torch 2.10.0+cu130

bitsandbytes support for newest CUDA/GPU combinations can be fragile. A modern NF4-capable build may work on first load but still have lifecycle bugs in long-lived processes.

### `accelerate`

Accelerate device dispatch/offload behavior can affect whether a quantized model is considered movable, dispatched, CPU-parked, or tied to hooks.

### ComfyUI memory management

These calls:

```python
comfy.model_management.unload_all_models()
comfy.model_management.soft_empty_cache()
```

are appropriate for Comfy models, but they run inside a long-lived GPU process with its own assumptions. They may indirectly change pressure/timing/lifetime enough to expose a bitsandbytes issue that v1.7 did not.

### BUG-LOCAL-085 canonical snapshot path

You verified the first load works, so I agree this is probably not the primary cause.

Still, one subtle difference is possible: passing a local snapshot path instead of a model ID can affect how transformers resolves config, quantization metadata, trust flags, and cached files. But because first load is correct and second load is not, I would rank this lower.

---

## 8. What I would implement now

### Priority 1: Path 2 narrow bypass

Only bypass unload when all are true:

- cached model exists
- cached tokenizer exists
- requested `model_id` equals cached `model_id`
- requested quantized mode is true
- cached quantized mode is true
- `cache_deltas` contains exactly one key: `model_evicted_to_cpu`
- cached model contains bitsandbytes 4-bit modules
- moving it to CUDA succeeds
- post-move VRAM is within expected NF4 range

No fallback to fp16. No fallback to same-process reload after this known condition.

### Priority 2: Post-load NF4 tripwire

After every `from_pretrained()` expected to be NF4, assert:

- has `Linear4bit`
- has `is_loaded_in_4bit` if available
- VRAM delta under threshold

### Priority 3: Version pin investigation

Recover the v1.7 environment if you can:

```powershell
pip freeze > current-freeze.txt
```

Compare against old:

```text
transformers
accelerate
bitsandbytes
safetensors
huggingface_hub
tokenizers
```

If you do not have the old freeze, create a small test harness that does only:

```python
for i in range(2):
    load NF4
    run tiny deterministic prompt
    unload
```

Then test only combinations of:

- current transformers + current bitsandbytes
- previous transformers + current bitsandbytes
- current transformers + previous bitsandbytes
- previous transformers + previous bitsandbytes

Do not touch torch/CUDA since you are pinned for Blackwell.

---

## Bottom line

Your instinct is right: **Path 2 is the safest fix** because it avoids the broken lifecycle instead of trying to repair bitsandbytes internals.

But ship it with two hard rules:

1. **Only rehydrate the cached model for the exact `model_evicted_to_cpu`-only case.**
2. **Always assert that the model is actually 4-bit after load/rehydrate.**

If that fails because `.cuda()` is unsupported for your quantized model version, the next safest production answer is **LLM subprocess isolation**, not `sys.modules` deletion.

---

## Gemini (gemini-3.1-pro-preview-customtools)

I **PARTIALLY AGREE** with adding the post-load tripwire (Path 3), but I **DISAGREE** with Path 2 because manually calling `.cuda()` on a `bitsandbytes` 4-bit model that was previously pushed to `.cpu()` is highly prone to corrupting the quantization state and breaking ComfyUI's VRAM estimator.

### Factual Errors in ChatGPT's Answer

1. **`model.cuda()` safely rehydrates BNB models:** ChatGPT assumes `model.cuda()` works out-of-the-box for `bitsandbytes` models. It doesn't. `Linear4bit` modules have complex internal states (`quant_state`, `absmax`). While modern `bitsandbytes` allows CPU offloading, a naive `.cuda()` roundtrip outside of `accelerate`'s device placement often results in device mismatch errors during the forward pass.
2. **Bypassing ComfyUI's Memory Manager:** ChatGPT wraps `comfy.model_management.unload_all_models()` in a `try/except/pass`. In a ComfyUI custom node, hiding an 8 GiB LLM allocation from ComfyUI's memory manager guarantees an OOM when your visual pipeline (which assumes that VRAM is free) tries to run.
3. **`torch.cuda.ipc_collect()`:** ChatGPT leaves this in the rehydration block. This is for multiprocess memory sharing. It is irrelevant and potentially destabilizing in a single-process ComfyUI setup.

### Important Omissions

1. **The "Reused Config" Bug (Most Likely Root Cause):** `transformers` mutates the `BitsAndBytesConfig` object during `from_pretrained` (e.g., setting internal flags about quantization status). If your codebase reuses the *same* cached `BitsAndBytesConfig` instance for the second `_load_llm()` call, `transformers` sees the mutated state, assumes quantization is already handled or invalid, and silently skips to an fp16 load.
2. **VRAM Budget Reality:** You have a 14.5 GiB ceiling. Mistral-Nemo NF4 takes ~8 GiB. If you use Path 2 and keep the LLM parked in RAM/VRAM, you only have ~6.5 GiB left for ComfyUI's visual pipeline. You *must* cleanly destroy the LLM to free resources; sidestepping the unload is not a viable architectural choice for a 16 GiB machine running mixed pipelines.

### My Recommendation

Do not hack `sys.modules` (Path 1) and do not attempt to keep the model alive and manually move it (Path 2). Fix the clean reload.

*   **Mandatory Tripwire (Path 3):** Implement ChatGPT's `_assert_bnb_nf4_loaded_or_die`. It is excellent defensive programming. Fail loud, not silently.
*   **Instantiate a Fresh Config:** Ensure `BitsAndBytesConfig` is instantiated *inside* `_load_llm()` immediately before `from_pretrained()`. Do not pass a globally cached config object.
*   **Deep Clean Unload:** Your `_unload_llm()` is leaving dangling references. Do not use `model.cpu()`. Instead, do this:
    ```python
    del _LLM_CACHE["model"]
    del _LLM_CACHE["tokenizer"]
    # Force garbage collection BEFORE empty_cache
    import gc
    gc.collect()
    import torch
    torch.cuda.empty_cache()
    ```
*   **Clear Accelerate State:** Sometimes `accelerate` holds onto device maps. If you still see the bug after ensuring a fresh config and deep clean, add `from accelerate import clear_device_cache; clear_device_cache()` to your unload function.

### Uncertainties to Verify

*   **PyTorch Version:** You mentioned `torch 2.10.0+cu130`. Since PyTorch 2.10 doesn't exist yet (current stables are 2.5/2.6), I assume this is either a typo for `2.1.0` or a specific nightly build required for your Blackwell sm_120 architecture. If it's a nightly, be aware that `bitsandbytes` bindings for unreleased PyTorch versions can have edge-case memory leaks.
*   **ComfyUI Patcher:** I am assuming you are loading the LLM natively via `transformers` rather than wrapping it in a ComfyUI `ModelPatcher`. If you *are* using a `ModelPatcher`, ComfyUI might be intercepting the load process and stripping the `Linear4bit` classes.

---

## NVIDIA ()



---

## To decide (Claude / human)

### Where they agree (ship now)

1. **Path 1 (sys.modules delete) is unsafe.** Both reject. Live class refs, native DLL handles on Windows, module identity drift -> nondeterministic later failures.
2. **Path 3 (post-load NF4 tripwire) is mandatory.** Both call for the same shape: `Linear4bit` module count > 0, `is_loaded_in_4bit` True, CUDA allocation delta < ~10-11 GiB. Fail loud, not silent.
3. **22-vs-33 it/s signal is suggestive but not conclusive.** Stronger detectors: VRAM allocation delta + module-class inspection + `is_loaded_in_4bit`. Speed delta could also be FS-cache warmup, cuDNN init caching, AV scanning differences.

### Where they disagree (defer until tested)

| | ChatGPT | Gemini |
|---|---|---|
| Path 2 (`.cuda()` rehydrate of cached model) | Yes with strong guards | No -- `Linear4bit` `quant_state`/`absmax` corruption risk on naive `.cuda()` roundtrip |
| Root cause hypothesis | bitsandbytes module-level state survives unload | "Reused config" -- transformers mutates BitsAndBytesConfig instance during from_pretrained |
| `model.cpu()` in `_unload_llm` | Doesn't comment | Drop it |
| `torch.cuda.ipc_collect()` | Include in rehydrate | Reject in single-process ComfyUI |

### Why I'm NOT shipping Path 2 tonight

- Gemini's pushback on `.cuda()` rehydrate is specific: `Linear4bit` has internal `quant_state` (absmax tensors, scaling factors) that are easy to corrupt on a naive device move. Failure mode would be silent corruption of generated text or a forward-pass device mismatch -- possibly worse than the current OOM.
- Need an isolated test harness to validate `.cuda()` works on a CPU-parked NF4 Mistral-Nemo in our specific stack (torch 2.10 nightly, bnb-for-CUDA-13, Blackwell sm_120) before shipping into the live pipeline.
- ALSO can't drop `model.cpu()` from `_unload_llm` (Gemini's other suggestion) -- that re-breaks BUG-LOCAL-073 (synchronize-before-cpu hardening for abandoned worker threads from `_run_with_timeout`).

### Verified facts

- `BitsAndBytesConfig` IS instantiated fresh in `_load_llm()` (verified by reading lines 2133-2143 of story_orchestrator.py: `from transformers import BitsAndBytesConfig; quant_config = BitsAndBytesConfig(...)` is inside the function body, not module-level). So Gemini's "reused config" hypothesis is unlikely for THIS codebase. But worth pinning with a comment so future refactors don't regress.
- `torch 2.10.0+cu130` is a real Blackwell-pinned build, not a typo. Gemini's uncertainty about this is from training cutoff -- it exists.

### Final grounded recommendation -- ship tonight

1. **Path 3 tripwire** -- post-`from_pretrained` assertion. Both LLMs agree.
   - Count `Linear4bit` modules with class module starting with `bitsandbytes`
   - Check `is_loaded_in_4bit` attribute
   - Compare CUDA allocation delta against threshold (10.5 GiB for Mistral-Nemo 12B NF4, leaving 4 GiB slack on the 14.5 GiB ceiling)
   - Raise `RuntimeError` with `BUG-LOCAL-098` reference and "restart ComfyUI" guidance
2. **`accelerate.clear_device_cache()` in `_unload_llm`** -- Gemini's specific suggestion. Defensive; no-op if accelerate not present.
3. **Fresh-config comment pin** -- add an explicit `# BUG-LOCAL-098: instantiate BitsAndBytesConfig fresh per-load; do NOT cache the instance` comment above the existing `quant_config = BitsAndBytesConfig(...)` so a future "optimization" doesn't move it to module scope.

### Defer to tomorrow (with isolated test harness)

- Build `scripts/test_nf4_reload.py` -- a 3-iteration NF4 load+inference+unload loop in isolation. Reproduce the bug outside ComfyUI.
- Test Path 2 (`.cuda()` rehydrate) against the harness.
- If Path 2 corrupts (Gemini's prediction), test Path 4 (subprocess isolation per ChatGPT recommendation).
- `pip freeze` snapshot, compare against archived v1.7 freeze if available.

### Tonight's ship checklist

- [ ] Add `_assert_bnb_nf4_loaded_or_die()` helper in story_orchestrator.py
- [ ] Wire it after `from_pretrained()` with VRAM-delta measurement
- [ ] Add `accelerate.clear_device_cache()` (try/except) to `_unload_llm`
- [ ] Add fresh-config comment above existing `BitsAndBytesConfig(...)` instantiation
- [ ] Tests: source-code regression guards for the three pieces
- [ ] BUG_LOG.md entry update -- BUG-098 moves from `[DIAGNOSED, FIX PENDING]` to `[FIX PARTIAL: tripwire + accelerate clear; Path 2 deferred]`
- [ ] Commit + push
