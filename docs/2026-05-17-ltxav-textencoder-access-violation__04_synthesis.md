# Synthesis -- 2026-05-17

**Question:** # LTXAVTextEncoderLoader Access Violation Diagnosis

## Question

What is the most likely root cause of a `Windows fatal exception: access
violation` thrown by `LTXAVTextEncoderLoader.execute()` while loading
Gemma 3 12B + LTX 2.3 22B via `comfy.sd.load_clip`, and what is the cheapest
diagnostic to disambiguate between the two top candidates?

## Hard facts

* Platform: Windows 10, RTX 5080 Laptop, 16 GB VRAM, Blackwell sm_120.
* Software: ComfyUI Desktop 0.21.1, Python 3.12.11 (uv-managed), torch
  2.10.0+cu130, safetensors 0.7.0.
* The crashing node is `LTXAVTextEncoderLoader` (lives in
  `comfy_extras/nodes_lt_audio.py` despite the filename -- "lt" is "LTX-V",
  not "load_torch"). Source:

  ```python
  @classmethod
  def execute(cls, text_encoder, ckpt_name, device="default"):
      clip_type = comfy.sd.CLIPType.LTXV
      clip_path1 = folder_paths.get_full_path_or_raise(
          "text_encoders", text_encoder)
      clip_path2 = folder_paths.get_full_path_or_raise(
          "checkpoints", ckpt_name)
      model_options = {}
      if device == "cpu":
          model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")
      clip = comfy.sd.load_clip(
          ckpt_paths=[clip_path1, clip_path2],
          embedding_directory=folder_paths.get_folder_paths("embeddings"),
          clip_type=clip_type,
          model_options=model_options)
      return io.NodeOutput(clip)
  ```

  Line 203 of `nodes_lt_audio.py` (the line in the crash stack frame) is
  the `clip = comfy.sd.load_clip(...)` call.

* Widgets passed (workflow node id 57):
  * `text_encoder`: `gemma_3_12B_it_fp4_mixed.safetensors`  (file size:
    9,447,702,218 bytes = 9.45 GB on disk)
  * `ckpt_name`: `ltx-2.3-22b-dev.safetensors`              (file size:
    46,149,344,974 bytes = 46.15 GB on disk)
  * `device`: `"default"`

* Crash stack (most-recent-first):

  ```
  Windows fatal exception: access violation
  Stack (most recent call first):
    File "...\torch\storage.py", line 468 in __getitem__
    File "...\ComfyUI\comfy\utils.py", line 136 in load_torch_file
    File "...\ComfyUI\comfy\sd.py", line 1241 in load_clip
    File "...\ComfyUI\comfy_extras\nodes_lt_audio.py", line 203 in execute
    File "...\comfy_api\latest\_io.py", line 1833 in EXECUTE_NORMALIZED
    File "...\comfy_api\internal\__init__.py", line 149 in wrapped_func
    File "...\ComfyUI\execution.py", line 297 in process_inputs
    ... [normal execute() dispatch]
  ```

* Pre-load GPU state (from the comfy log just before crash):
  * FLUX loaded "full load: True", 22,700.13 MB resident (on a 16 GB
    physical card -- dynamic offloader in effect).
  * AudioEncoderLoader had already loaded `whisper_large_v3_fp16.safetensors`
    earlier in the run (the file is canonical Comfy-Org/HuMo_ComfyUI byte-
    identical, 3 GB, encoder + decoder both present).
  * `Found quantization metadata version 1` + `Using MixedPrecisionOps for
    text encoder` lines appear immediately before the access violation,
    consistent with fp4_mixed quant being applied on the Gemma 3 12B
    weights.

* Same workflow JSON ran end-to-end successfully at tag
  `v2.0-alpha-cleanbreak` (commit `1aed66d`, 2026-05-12). The
  `nodes_lt_audio.py` / `LTXAVTextEncoderLoader` class is bundled with
  ComfyUI Desktop and may have been added/changed in a version bump
  between 2026-05-12 and 2026-05-17.

* `comfy.sd.load_clip(ckpt_paths=[gemma_3_12B_fp4_mixed, ltx_2.3_22b_dev],
  clip_type=CLIPType.LTXV, ...)` is being asked to walk a 9.45 GB fp4-
  mixed encoder file AND a 46.15 GB diffusion checkpoint and reconstruct
  a CLIP-shape text encoder out of their combined weights.

## Top two candidate root causes

**(P1) Quantization-aware load path defect.** safetensors 0.7.0 + torch
2.10.0+cu130 + comfy.sd.load_clip on a fp4_mixed quantized text encoder
where `Found quantization metadata version 1` triggers a code path that
mis-allocates tensor storage. The Blackwell sm_120 architecture is new
enough that path may not be widely exercised. Storage `__getitem__`
indexing into wrong-shape pages reads unmapped memory -> access
violation.

**(P2) 46 GB checkpoint mmap / sharded-load issue.** `comfy.sd.load_clip`
called with two ckpt_paths walks both files. The 46.15 GB
`ltx-2.3-22b-dev.safetensors` is far above what CLIP loaders typically
handle. Either an `int32` offset overflow inside safetensors header /
load_torch_file or an mmap window that exceeds the 16 GB VRAM ceiling
causes storage indexing into invalid memory.

## Cheapest disambiguating test

A cold-launch isolation run with ONLY the `LTXAVTextEncoderLoader` node +
a trivial `CLIPTextEncode` consumer (force the CLIP to actually resolve)
will tell us:

* Crash with same signature -> bug is intrinsic to the loader / file /
  version triple. Co-residence with FLUX is not the cause. Repro for
  upstream issue.
* Clean load -> bug is co-residence + dynamic offloader fragmentation.
  Fix lives in HuMo's pre-LTX unload sequence or workflow execution
  order, NOT in the loader. Workflow JSON correctly wires LTX but the
  graph schedules LTX after FLUX without unloading.

## What I want from the round-robin

1. Among (P1) and (P2), which is more likely given the four hard
   constraints (ComfyUI 0.21.1, torch 2.10.0+cu130, safetensors 0.7.0,
   Blackwell sm_120) and the fact the same workflow + files worked five
   days ago at `v2.0-alpha-cleanbreak` on the same machine?
2. Are there other root cause candidates I haven't considered (CUDA
   driver regression, ComfyUI Desktop version bump between 2026-05-12
   and 2026-05-17, etc.)?
3. If the isolation test in Track 1 produces a clean load, what's the
   most targeted second test to isolate the co-residence interaction
   without spinning a full episode? (Some incremental adds of FLUX,
   Whisper, etc. before LTX, in a small graph, to find the exact
   trigger.)
4. If the isolation test crashes with the same signature, what's the
   minimal upstream-issue repro shape that ComfyUI maintainers would
   accept?

## Hard stops (already locked)

* No model swap.
* No workflow data edit.
* No version bump until root cause is locked.
* No `wrong_model` failure class in the harness.
* No code change to `scripts/otr_api.py` or harness beyond commit
  `0facea7` (worker process-death detection).

---

## ChatGPT (gpt-5.5)

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

---

## Gemini (gemini-3.1-pro-preview-customtools)

### 1. Verdict

**PARTIALLY AGREE:** ChatGPT’s isolation methodology is sound, but it misdiagnoses the most likely root cause by failing to recognize that a Windows access violation during a `safetensors` read is the classic signature of system pagefile exhaustion, not a quantization-path defect.

### 2. Factual Errors in the ChatGPT Answer

*   **Misinterpretation of the stack trace:** ChatGPT claims the crash at `torch/storage.py`, line 468 (`__getitem__`) points to a "native storage / mmap / tensor materialization" defect (P1). This is incorrect. On Windows, when `safetensors` memory-maps a file, PyTorch tensors act as pointers to virtual memory. If the OS cannot page in the data—usually because the system commit limit is reached and the pagefile cannot expand—Windows throws a `0xC0000005 Access Violation`. This is a system-level Out-Of-Memory (OOM) error, not a code defect.
*   **Ineffective P2 Probe:** ChatGPT suggests using `safetensors.safe_open` as a lightweight probe. This will not accurately reproduce the crash because it does not replicate the cumulative memory pressure of your workflow, nor does it necessarily use the exact same memory-mapping flags as `comfy.utils.load_torch_file` (line 136).

### 3. Important Things That Were Omitted

*   **The Commit Charge Math:** You are running a massive stack on a laptop. FLUX (22.7 GB resident) + Whisper (3 GB) + Gemma (9.45 GB) + LTX (46.15 GB) = **~81.3 GB of committed memory**. 
*   **The "Worked 5 Days Ago" Variable:** What changes in 5 days on a solo developer's workstation running heavy generative workflows? **C: drive free space.** Windows dynamically expands the pagefile to accommodate massive memory-mapped files. If your C: drive has filled up with generated radio-drama assets over the last 5 days, the pagefile can no longer expand to 81+ GB, resulting in the exact access violation you are seeing.
*   **VRAM vs. System RAM Distinction:** You have a 14.5 GB VRAM ceiling, but this crash is happening in host memory (System RAM + Pagefile) during the `load_torch_file` phase, before the weights are even moved to the GPU.

### 4. My Recommendation

Skip the single-file Python probes. They are a waste of time for this specific stack trace. Do this instead:

1.  **Check your C: drive free space immediately.** You need at least 80–100 GB of free space for the Windows pagefile to back an 81 GB commit charge. If you have less than this, clear space and run the workflow again. This is the highest-probability root cause.
2.  **Run your proposed cold-launch isolation test.** (LTX node + trivial consumer). If it passes, P1 (quantization defect) is completely ruled out, and the issue is definitively cumulative memory pressure (FLUX + Whisper + LTX).
3.  **Implement a pre-LTX unload.** If the isolation test passes, the cheapest fix that requires no low-level VRAM hacking is to ensure FLUX and Whisper are purged from host memory before LTX loads. Since you cannot edit the workflow data, you must enforce this via ComfyUI's execution model (e.g., using a node that forces garbage collection/unloading of prior models, or splitting the API calls).
4.  **Verify Windows Pagefile Settings.** Ensure your virtual memory is set to "Automatically manage paging file size for all drives." A hardcoded pagefile size will instantly kill this workflow.

### 5. Uncertainties to Flag

*   **ComfyUI Desktop Version Bump:** I am uncertain if the update to ComfyUI Desktop 0.21.1 between 2026-05-12 and 2026-05-17 changed how `comfy.sd.load_clip` handles memory mapping for `CLIPType.LTXV`. If the new version attempts to load the 46 GB file directly into RAM instead of mmap-ing it, it will immediately crash a 16 GB system.
*   **Blackwell sm_120 fp4 kernels:** While I strongly suspect pagefile exhaustion, Blackwell is bleeding-edge. If clearing disk space and isolating the node still results in an access violation on `device="default"` but passes on `device="cpu"`, then ChatGPT's P1 theory regarding a cu130/fp4_mixed kernel bug becomes the primary suspect.

---

## NVIDIA (nvidia/llama-3.3-nemotron-super-49b-v1.5)

### 1. **Agreements Between ChatGPT and Gemini**  
**Concur:**  
- **Cold-launch isolation test** is critical to distinguish between intrinsic loader/file defects and co-residence/memory pressure issues.  
- **System resource constraints** (host RAM, pagefile) are a plausible root cause given the combined 80+ GB memory demand.  

**Flaw They Missed:**  
Both underemphasize the **impact of ComfyUI Desktop version bump (0.21.1)** between the working commit (`1aed66d`, 2026-05-12) and the current failure. A regression in `comfy.sd.load_clip` or memory-mapping logic could explain why the same files worked before.  

---

### 2. **Disagreements and Third Path**  
**P1 (Quantization Defect) vs. Pagefile Exhaustion (Gemini's View):**  
- **Gemini is more convincing** here. The crash occurs in `torch.storage.py` during `__getitem__`, which is a classic Windows access violation from **pagefile exhaustion** when memory-mapped files cannot be paged in.  
- **ChatGPT's P1 theory** (quantization path defect) is less likely unless the newer ComfyUI version introduced a bug in handling `fp4_mixed` metadata. However, Gemini correctly notes that the crash is a system-level OOM, not a tensor-shape misallocation.  

**Third Path:**  
- **Hybrid Cause**: The 2026-05-17 ComfyUI update might have changed how `load_clip` memory-maps large files (e.g., loading more into RAM upfront), exacerbating pagefile pressure. This combines Gemini’s pagefile theory with ChatGPT’s version-regression concern.  

---

### 3. **Factual Errors in Prior Answers**  
- **ChatGPT’s P2 Misstatement**:  
  - Claimed the 46 GB checkpoint could cause VRAM ceiling exceedance. **Error**: The crash occurs in *host* memory (`load_torch_file`), not VRAM. VRAM limits are enforced later during GPU allocation.  
- **Gemini’s Overconfidence**:  
  - Asserts the crash is *definitely* pagefile exhaustion without considering that the ComfyUI update might have altered memory-mapping behavior (e.g., eager loading of weights).  

---

### 4. **Important Omissions by Both**  
1. **ComfyUI Version-Specific Changes**:  
   - No analysis of what changed in `comfy.sd.load_clip` or `nodes_lt_audio.py` between `v2.0-alpha-cleanbreak` (working) and Desktop 0.21.1 (failing).  
2. **Model Unloading Behavior**:  
   - FLUX and Whisper may not be fully unloaded before LTX loads, even if the workflow orders them sequentially. ComfyUI’s model management logic could retain memory.  
3. **CUDA Driver Version**:  
   - Not mentioned in hard facts. A driver update between 2026-05-12 and 2026-05-17 could introduce compatibility issues with Blackwell `sm_120`.  

---

### 5. **Prioritized Recommendations**  
1. **Check Disk Space & Pagefile Settings** (Gemini’s Priority):  
   - Verify ≥80 GB free on C:.  
   - Ensure pagefile is auto-managed or manually sized to accommodate 80–100 GB.  

2. **Run Cold-Launch Isolation Test** (ChatGPT’s Priority):  
   - If LTX-only fails, focus on loader/file defects (e.g., test with `device="cpu"` to isolate CUDA).  
   - If LTX-only passes, proceed to co-residence tests.  

3. **Test Sequential Model Loading with Explicit Unloading**:  
   - Load FLUX → unload → load Whisper → unload → load LTX.  
   - Use ComfyUI’s API/UI to force unloads between steps.  

4. **Verify Model File Integrity & Version Changes**:  
   - Compute SHA256 hashes for both models.  
   - Compare `comfy.sd.load_clip` and `nodes_lt_audio.py` between the working commit (`1aed66d`) and current ComfyUI 0.21.1.  

5. **Test with `device="cpu"`**:  
   - If CPU mode passes, the issue is likely CUDA/offloader-related (e.g., Blackwell-specific kernel bug in torch 2.10.0+cu130).  

---

### 6. **Uncertainties Requiring Verification**  
1. **ComfyUI 0.21.1 Changelog**:  
   - Did `load_clip` or LTXV handling change in a way that affects memory mapping (e.g., eager vs. lazy loading)?  
2. **CUDA Driver Version**:  
   - Was there a driver update between 2026-05-12 and 2026-05-17 that could interact poorly with Blackwell?  
3. **Exact Memory Pressure**:  
   - Use Windows Performance Monitor to track **commit charge** and **pagefile usage** during the crash.  

--- 

### Final Verdict  
**Most Likely Root Cause**:  
- **Pagefile exhaustion due to cumulative memory pressure** from FLUX (22.7 GB) + Whisper (3 GB) + LTX (46 GB) + Gemma (9.45 GB), exacerbated by potential changes in ComfyUI 0.21.1’s memory-mapping behavior.  

**Cheapest Disambiguation**:  
1. Clear disk space → retest.  
2. If unchanged, run LTX-only isolation test.  
3. If LTX-only passes, test sequential loading with explicit unloads.  

**If Isolation Test Fails**:  
- File upstream issue with minimal `LTXAVTextEncoderLoader.execute()` repro, hashes, and version details. Prioritize testing `device="cpu"` to isolate CUDA.

---

## To decide (Claude / human)

- [ ] All three agree:
- [ ] Two-vs-one splits:
- [ ] Facts to verify:
- [ ] Final grounded recommendation:
