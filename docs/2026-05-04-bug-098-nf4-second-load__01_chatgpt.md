# Round A -- ChatGPT (gpt-5.5) elapsed=128.6s

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
