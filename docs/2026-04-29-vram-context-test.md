# VRAM vs Context-Length Measurement

**Goal:** decide per-model `context_cap` values in
`nodes/story_orchestrator.py::_MODEL_CONTEXT_CAPS` based on hard
data instead of guesswork.

**Hardware:** RTX 5080 Laptop, 16 GB VRAM, Blackwell sm_120, Windows
11, torch 2.10.0+cu130, bitsandbytes 4-bit NF4 quantization.

**Constraint:** OTR must keep peak VRAM <= 14.5 GB during the LLM
phase to leave headroom for FLUX (~6 GB), HuMo (~16 GB but loaded
sequentially after LLM unload), and Bark (~3 GB).

**Method:** `scripts/vram_context_test.py` loads each model in 4-bit
NF4, runs a single 16-token generation at progressively longer
prompt lengths (2K -> 32K tokens), records THREE separate
memory numbers per measurement:

- **`VRAM nvml`** -- `pynvml.nvmlDeviceGetMemoryInfo(...).used`,
  the same number `nvidia-smi` reports. Includes PyTorch's
  allocator + CUDA driver overhead + any other GPU process
  active on the device. **This is the cap-tuning truth.**

- **`VRAM torch`** -- `torch.cuda.max_memory_allocated()`.
  PyTorch caching allocator only, this process only. Useful for
  understanding "ours vs driver" but undercounts the full device
  picture by ~500 MB-1 GB worth of CUDA overhead.

- **`CPU RAM`** -- `psutil.Process().memory_info().rss`. This
  process's host-side resident-set size. **SEPARATE memory
  space from VRAM.** Reported only for visibility (4-bit NF4
  load can briefly spike CPU RAM during the conversion step)
  and **never mixed** into VRAM cap decisions.

Stops probing a given model when the first OOM hits; the next
model is measured from a clean cache.

**How to run:**

```cmd
:: Stop ComfyUI Desktop first to free VRAM
cd /d C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe scripts\vram_context_test.py
```

Each run appends a `## Run <timestamp>` section below with a results
table. Compare runs to see how VRAM scales with context length per
model.

## Decision rule

For each model, set `_MODEL_CONTEXT_CAPS[model_id]` to the largest
context length where the **`VRAM nvml`** column stays under
**12 GB** (leaves 4 GB headroom on a 16 GB Blackwell for KV cache
growth during 1024-token generation, plus co-resident
FLUX/Bark/HuMo overhead). Round down to nearest 1024.

**Use the `VRAM nvml` column for the decision** -- it's the
authoritative GPU memory number. The `VRAM torch` column is
informative but undercounts by ~500 MB-1 GB driver overhead.
The `CPU RAM` column is host-side and **does NOT count toward
VRAM** -- it exists in a completely separate memory space and
is reported only so we can sanity-check that the host has
enough RAM to load + quantize the model.

If a model OOMs at ALL probed lengths, leave its cap at the current
default (8192) and flag for review.

## Optional dependencies for full measurement

The script gracefully degrades if either is missing, but
gets the most informative output when both are installed:

```cmd
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe -m pip install nvidia-ml-py psutil
```

- **`nvidia-ml-py`** (imports as `pynvml`) -- enables the
  `VRAM nvml` column. Without it, only the PyTorch number is
  reported, which undercounts driver overhead.
- **`psutil`** -- enables the `CPU RAM` column. Without it,
  CPU host-side memory is unobservable but doesn't affect
  GPU measurements.

## Initial cap values (pre-measurement, conservative)

| Model | Cap | Rationale |
|---|---:|---|
| mistralai/Mistral-Nemo-Instruct-2407 | 16384 | 12B base, validated at 8K, doubled |
| google/gemma-4-E2B-it | 16384 | effective ~2B, plenty of headroom |
| google/gemma-4-E4B-it | 16384 | effective ~4B, similar to Mistral 12B in active params |
| Qwen/Qwen2.5-14B-Instruct | 12288 | 14B base, smaller cap to stay safe |
| Nitral-AI/Captain-Eris_Violet-V0.420-12B | 12288 | 12B RP fine-tune, EXPERIMENTAL |
| inflatebot/MN-12B-Mag-Mell-R1 | 12288 | 12B RP fine-tune, EXPERIMENTAL |
| google/gemma-2-2b-it | 8192 | LEGACY, removed from dropdown 2026-04-29 (BUG-110) |
| google/gemma-2-9b-it | 8192 | LEGACY, same removal |
| (anything else, default) | 8192 | conservative for unknown models |

## Results

(empty -- waiting for first measurement run)

