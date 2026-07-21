# Run Gemma 4 12B GGUF in OTR

> **Optional peer, not the canonical OTR writer.** The saved workflow uses
> `google/gemma-4-12b-it` through the offline Transformers/HF NF4 lane with
> lm-format-enforcer. The instructions below apply only when a user explicitly
> chooses the separate llama.cpp/GGUF backend.

OTR's writer dropdown also exposes Gemma 4 12B as:

```text
unsloth/gemma-4-12b-it-GGUF
```

This is a native in-process GGUF lane. It uses `llama-cpp-python` from the
ComfyUI venv and loads a local `.gguf` file directly. It does not use Ollama,
does not start `llama-server`, and does not talk to port 8080.

## Model File

Download the Q8_0 weight from Hugging Face:

```text
unsloth/gemma-4-12b-it-GGUF/gemma-4-12b-it-Q8_0.gguf
```

Place it here:

```powershell
C:\ComfyUI-Models\LLM\converted\gemma-4-12b-it\gemma-4-12b-it-Q8_0.gguf
```

Override the location only when needed:

```powershell
$env:GEMMA4_12B_GGUF_PATH = 'D:\models\gemma-4-12b-it-Q8_0.gguf'
```

## Runtime

The ComfyUI venv must import `llama_cpp`. Use a CUDA-enabled
`llama-cpp-python` build for this Windows/Python/CUDA stack; a CPU-only wheel
will not make the 12B Q8_0 writer lane usable.

Known-good install for this ComfyUI venv:

```powershell
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe -m pip install --only-binary=:all: --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 llama-cpp-python==0.3.33
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe -m pip install --only-binary=:all: nvidia-cuda-runtime-cu12==12.4.127 nvidia-cublas-cu12==12.4.5.8
```

The CUDA 12 runtime packages supply the DLLs expected by the
`llama-cpp-python` CUDA wheel even when the ComfyUI torch build is newer.

Readiness check:

```powershell
cd C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe docs\gemma4\gemma4_test.py
```

## OTR Dropdown

Set the writer model slot to:

```text
unsloth/gemma-4-12b-it-GGUF
```

The old temporary handle is not supported. Use the row above.

## VRAM Knobs

Q8_0 is a real local residency path. The selector unloads any other resident
writer LLM before loading the GGUF, reuses it while selected, and closes it on
slot transition.

Useful environment variables:

```powershell
$env:GEMMA4_12B_N_CTX = '8192'
$env:GEMMA4_12B_N_GPU_LAYERS = '-1'
$env:GEMMA4_12B_MAX_NEW_TOKENS = '512'
```

If a mixed audio/video render runs out of memory after writing, switch away
from the 12B row or lower `GEMMA4_12B_N_CTX` before the video-heavy leg.
