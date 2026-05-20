# Round A -- ChatGPT (gpt-5.5) elapsed=176.6s

## Short verdict

Most likely: **P1-ish — a native loader/version-path defect in the current ComfyUI / torch 2.10.0+cu130 / safetensors 0.7.0 / Blackwell path, triggered while resolving the fp4-mixed Gemma text encoder through `CLIPType.LTXV`.**

But I would phrase it slightly broader than your P1:

> The strongest suspect is the current ComfyUI `load_clip` + quantized LTXV text-encoder path, not the workflow wiring and not ordinary Python-level VRAM exhaustion.

The **46 GB LTX checkpoint path is still a real suspect**, but I would not describe it as “mmap exceeds the 16 GB VRAM ceiling.” The stack is in:

- `nodes_lt_audio.py:203` — `comfy.sd.load_clip(...)`
- `comfy/sd.py:1241` — inside `load_clip`
- `comfy/utils.py:136` — inside `load_torch_file`
- `torch/storage.py:468` — `__getitem__`
- then native Windows access violation

That points to **native storage / mmap / tensor materialization**, not a normal CUDA OOM and not directly “VRAM ceiling exceeded.”

Your proposed cold-launch LTX-only graph is a good test, but it disambiguates **intrinsic loader/file/version bug vs co-resident FLUX/Whisper/offloader interaction**. It does **not** cleanly disambiguate P1 vs P2, because it still asks `load_clip` to walk both:

```python
ckpt_paths=[gemma_3_12B_fp4_mixed, ltx_2.3_22b_dev]
```

So the cheapest P1-vs-P2 diagnostic is slightly different.

---

## 1. P1 vs P2: which is more likely?

### My call: P1 is more likely, but not overwhelmingly

Given the hard constraints:

- Windows 10
- RTX 5080 Laptop / Blackwell `sm_120`
- torch `2.10.0+cu130`
- safetensors `0.7.0`
- ComfyUI Desktop `0.21.1`
- fp4-mixed Gemma 3 12B
- log lines immediately before death:

```text
Found quantization metadata version 1
Using MixedPrecisionOps for text encoder
```

I would rank the candidates roughly:

1. **P1 / quantized LTXV text-encoder load path regression** — most likely.
2. **P2 / 46 GB checkpoint walking / host mmap / large-file interaction** — plausible.
3. **Co-resident FLUX/offloader/host-commit interaction** — plausible if LTX-only passes.
4. **Driver / CUDA / Desktop auto-update regression** — plausible secondary cause, likely expressed through #1 or #3.

The fact that the **same workflow and same files worked five days earlier** is important. That lowers the probability of a purely static “46 GB safetensors cannot be handled” explanation. If the files are byte-identical and the hardware is the same, then something in the execution environment changed: ComfyUI Desktop bundle, ComfyUI core, `nodes_lt_audio.py`, `comfy.sd.load_clip`, torch, safetensors, CUDA runtime, driver, or model-management behavior.

So I would not say:

> “The 46 GB checkpoint is inherently too large.”

I would say:

> “The current `load_clip` path may now be touching the 46 GB checkpoint in a way the previous working build did not, or it may now be resolving the fp4-mixed Gemma storage through a changed quantization path.”

That makes P1 more likely as stated, but P2 remains close enough that you should identify **which file is being touched at the crash point** before filing upstream.

---

## 2. The cheapest diagnostic to disambiguate P1 vs P2

Your cold-launch LTX-only graph is useful, but the cheapest P1/P2 discriminator is:

> **Run fresh-process, single-file probes against each safetensors file using the same Python environment.**

Do not load both files in the same process. A fatal access violation will kill the interpreter, so each probe must be a separate process.

### Probe A: Gemma file only

Run from the ComfyUI environment:

```powershell
$env:PYTHONFAULTHANDLER="1"
cd path\to\ComfyUI

python .\probe_safetensors_touch.py "path\to\models\text_encoders\gemma_3_12B_it_fp4_mixed.safetensors"
```

### Probe B: LTX 46 GB checkpoint only

```powershell
$env:PYTHONFAULTHANDLER="1"
cd path\to\ComfyUI

python .\probe_safetensors_touch.py "path\to\models\checkpoints\ltx-2.3-22b-dev.safetensors"
```

Minimal probe:

```python
# probe_safetensors_touch.py
import sys
import torch
import safetensors
from safetensors import safe_open

p = sys.argv[1]

print("file:", p, flush=True)
print("torch:", torch.__version__, "cuda:", torch.version.cuda, flush=True)
print("safetensors:", safetensors.__version__, flush=True)

with safe_open(p, framework="pt", device="cpu") as f:
    meta = f.metadata()
    keys = list(f.keys())

    print("metadata:", meta, flush=True)
    print("num_keys:", len(keys), flush=True)
    print("first_keys:", keys[:5], flush=True)
    print("last_keys:", keys[-5:], flush=True)

    # Touch a few tensors, including first/middle/last.
    # This catches obvious storage/mmap/page indexing faults without
    # forcing a full model execution.
    sample = []
    if keys:
        sample.append(keys[0])
        sample.append(keys[len(keys) // 2])
        sample.append(keys[-1])

    for k in dict.fromkeys(sample):
        print("touch:", k, flush=True)
        t = f.get_tensor(k)
        print("  shape:", tuple(t.shape), "dtype:", t.dtype, "device:", t.device, flush=True)

        if t.numel() > 0:
            flat = t.reshape(-1)
            # Touch both ends of the storage.
            _ = flat[0].item()
            _ = flat[-1].item()

print("OK", flush=True)
```

Interpretation:

| Result | Meaning |
|---|---|
| Gemma-only probe crashes | Strongly supports P1-ish storage/quantized-file/load-path issue. |
| LTX 46 GB-only probe crashes | Strongly supports P2-ish large-file/mmap/storage issue. |
| Both single-file probes pass, but `LTXAVTextEncoderLoader` crashes | The problem is likely in `comfy.sd.load_clip` combining/detecting the two paths, or in the quant-aware Comfy reconstruction path rather than raw safetensors. |
| Both pass and LTX-only graph passes | The problem is likely co-residence/offloader/host-commit state from prior FLUX/Whisper loads. |

If you want to get even closer to the actual stack frame at `comfy/utils.py:136`, use a Comfy-level single-file probe too:

```python
# probe_comfy_load_torch_file.py
import sys
import torch
import comfy.utils

p = sys.argv[1]
print("loading:", p, flush=True)

sd = comfy.utils.load_torch_file(
    p,
    device=torch.device("cpu")
)

print("OK num_keys:", len(sd), flush=True)
print("first_keys:", list(sd.keys())[:5], flush=True)
```

But caution: probing the 46 GB checkpoint this way may be much heavier than the `safe_open` touch probe. Use it only if the first probe is inconclusive.

---

## 3. Other root causes you should keep on the board

Yes. The top-level failure class is “native extension died while materializing storage.” P1 and P2 are good candidates, but I would also track these:

### A. ComfyUI Desktop / ComfyUI core regression

Very plausible.

You already noted that the working tag was:

```text
v2.0-alpha-cleanbreak
commit 1aed66d
2026-05-12
```

and the current crash is under Desktop `0.21.1`. If `nodes_lt_audio.py`, `comfy.sd.load_clip`, model detection, or quantization handling changed between those dates, that is a prime suspect.

The critical call site is your provided:

```python
# comfy_extras/nodes_lt_audio.py:203 in your build
clip = comfy.sd.load_clip(
    ckpt_paths=[clip_path1, clip_path2],
    embedding_directory=folder_paths.get_folder_paths("embeddings"),
    clip_type=clip_type,
    model_options=model_options)
```

If the old working build did not pass both paths, or handled LTXV CLIP detection differently, that would explain a lot.

### B. Host RAM / pagefile / Windows commit pressure

Do not focus only on VRAM.

The process may have:

- FLUX model manager state
- Whisper large v3 fp16 already loaded
- Gemma 9.45 GB safetensors
- LTX 46.15 GB safetensors
- mmap views / CPU tensors / metadata / temporary tensors
- CUDA allocations/offload state

A Windows access violation in a native extension can be the ugly failure mode of memory pressure or invalid mmap access, not a clean Python `MemoryError`.

### C. CUDA driver / CUDA 13 / Blackwell path

Also plausible, especially because `sm_120` is new and torch `2.10.0+cu130` is not a boring mature stack. If the crash only happens with `device="default"` and not with `device="cpu"`, that shifts suspicion toward CUDA/offloader/device-placement rather than raw safetensors.

The `LTXAVTextEncoderLoader` already exposes this diagnostic:

```python
if device == "cpu":
    model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")
```

So another cheap diagnostic is:

> Duplicate the minimal LTX-only test and set the node widget `device` to `"cpu"`.

Interpretation:

| `device="cpu"` result | Meaning |
|---|---|
| CPU mode also crashes | More likely raw safetensors / Comfy load path / host mmap. |
| CPU mode passes, default crashes | More likely CUDA/offloader/device-placement/quant-kernel interaction. |

This does not separate P1 from P2 as cleanly as per-file probing, but it separates CPU-file-path from CUDA/device-path quickly.

### D. File corruption or sparse/partial download

Less likely because the workflow worked earlier, but still cheap to exclude.

Compute SHA256 for both large files and record them in the bug report:

```powershell
Get-FileHash "path\to\gemma_3_12B_it_fp4_mixed.safetensors" -Algorithm SHA256
Get-FileHash "path\to\ltx-2.3-22b-dev.safetensors" -Algorithm SHA256
```

### E. Antivirus / filter driver / controlled folder access

Usually not my first suspect, but Windows mmap + very large model files + native storage faults can interact badly with file filter drivers. If the per-file LTX probe dies inconsistently, check this.

---

## 4. If the LTX-only isolation test loads cleanly

Then the loader/files/version triple is probably not sufficient by itself. The next most targeted test is **two-prompt same-process co-residence testing**, not a full episode.

The important trick: do not try to force ordering inside one giant graph. Use the same ComfyUI process and queue tiny prompts sequentially.

### Test matrix

Restart ComfyUI between each condition.

#### Baseline

1. Cold start ComfyUI.
2. Run minimal LTX-only loader graph.
3. Confirm clean.

#### Whisper co-residence

1. Cold start.
2. Prompt 1: load `whisper_large_v3_fp16.safetensors` with the same `AudioEncoderLoader` path used by the real workflow. Use the smallest trivial consumer needed to force it to load.
3. Prompt 2, same process: run minimal LTX-only graph.

If this crashes, Whisper residency/host-memory pressure is involved.

#### FLUX co-residence

1. Cold start.
2. Prompt 1: load FLUX through the same loader path as the episode workflow. Do a tiny forced use, e.g. very small latent and 1 sampling step, just enough to force real model load.
3. Prompt 2, same process: run minimal LTX-only graph.

If this crashes, FLUX/model-manager/offloader state is involved.

#### FLUX + Whisper

1. Cold start.
2. Prompt 1: minimal FLUX load/use.
3. Prompt 2: minimal Whisper load/use.
4. Prompt 3: minimal LTX-only graph.

If only the combined case crashes, it is likely aggregate host commit / offloader state / cache residency.

### Very useful extra discriminator

If ComfyUI exposes the usual free/unload API in your Desktop build, test:

1. Cold start.
2. Minimal FLUX load.
3. Call Comfy’s unload/free endpoint or UI unload action.
4. Run LTX-only graph.

If unloading models before LTX makes the crash disappear, the smallest production fix is likely **execution ordering / explicit pre-LTX unload**, not model replacement or low-level VRAM optimization.

That aligns with your “smallest change, largest payoff” constraint.

---

## 5. If the LTX-only isolation test crashes with the same signature

Then prepare an upstream issue as a minimal native-crash repro.

Maintainers will not want a full radio-drama graph. Give them the smallest shape that reaches:

```python
LTXAVTextEncoderLoader.execute(
    text_encoder="gemma_3_12B_it_fp4_mixed.safetensors",
    ckpt_name="ltx-2.3-22b-dev.safetensors",
    device="default",
)
```

The issue should include:

### Environment block

- Windows 10 build
- GPU: RTX 5080 Laptop, 16 GB VRAM, Blackwell `sm_120`
- NVIDIA driver version
- ComfyUI Desktop `0.21.1`
- actual ComfyUI git commit if available
- Python `3.12.11`
- torch `2.10.0+cu130`
- CUDA runtime from `torch.version.cuda`
- safetensors `0.7.0`
- VRAM state is not the primary repro if cold-load crashes

### Exact stack

Include your stack exactly, especially:

```text
File "...\torch\storage.py", line 468 in __getitem__
File "...\ComfyUI\comfy\utils.py", line 136 in load_torch_file
File "...\ComfyUI\comfy\sd.py", line 1241 in load_clip
File "...\ComfyUI\comfy_extras\nodes_lt_audio.py", line 203 in execute
```

### Exact model files

Include:

- filenames
- byte sizes
- SHA256 hashes
- download source / repo
- safetensors metadata if possible

For example, attach output from the probe:

```text
metadata: ...
num_keys: ...
first_keys: ...
last_keys: ...
```

### Minimal repro workflow or script

Best upstream repro shape:

```python
import torch
import folder_paths
import comfy.sd
from comfy_extras.nodes_lt_audio import LTXAVTextEncoderLoader

print("torch", torch.__version__, torch.version.cuda)

out = LTXAVTextEncoderLoader.execute(
    text_encoder="gemma_3_12B_it_fp4_mixed.safetensors",
    ckpt_name="ltx-2.3-22b-dev.safetensors",
    device="default",
)

print("loaded", out)
```

Run it from the ComfyUI root with the models in the normal Comfy folders.

Also provide a `"cpu"` variant:

```python
out = LTXAVTextEncoderLoader.execute(
    text_encoder="gemma_3_12B_it_fp4_mixed.safetensors",
    ckpt_name="ltx-2.3-22b-dev.safetensors",
    device="cpu",
)
```

That single comparison is very valuable to maintainers.

---

## Bottom line

I would not spend time on low-level VRAM tuning yet.

Do this order:

1. **Cold LTX-only graph** — separates intrinsic crash from co-residence.
2. **Fresh-process per-file safetensors probes** — separates Gemma/fp4 path from 46 GB checkpoint path.
3. **`device="cpu"` LTX-only variant** — separates CUDA/offloader path from CPU storage path.
4. If LTX-only passes, do **same-process sequential co-residence tests**: Whisper → LTX, FLUX → LTX, FLUX+Whisper → LTX.
5. If LTX-only crashes, file upstream with the minimal `LTXAVTextEncoderLoader.execute()` repro, exact versions, hashes, and stack.

My current prior: **P1 / current quantized LTXV CLIP loader path regression is the leading suspect**, with **P2 large-checkpoint mmap/materialization** close enough that you should identify the crashing file before making the upstream report.
