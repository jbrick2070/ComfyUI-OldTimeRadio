# Gemma 4 12B GGUF Native Writer Lane

> **Optional peer, not the canonical OTR writer.** The saved workflow now uses
> `google/gemma-4-12b-it` through the fully local Transformers/HF NF4 lane so
> lm-format-enforcer can hard-constrain structured JSON. This page remains for
> users who deliberately select the separate llama.cpp/GGUF backend.

OTR also exposes this Gemma 4 12B GGUF row in the writer dropdown:

- `unsloth/gemma-4-12b-it-GGUF`

This lane runs in-process through `llama-cpp-python`. It does not use
Ollama, does not start a sidecar, and does not talk to a localhost server.

## Model File

Download this file from `unsloth/gemma-4-12b-it-GGUF`:

- `gemma-4-12b-it-Q8_0.gguf`

Default path:

```powershell
C:\ComfyUI-Models\LLM\converted\gemma-4-12b-it\gemma-4-12b-it-Q8_0.gguf
```

Override only when needed:

```powershell
$env:GEMMA4_12B_GGUF_PATH = 'D:\models\gemma-4-12b-it-Q8_0.gguf'
```

## Runtime Binding

The ComfyUI Python environment must be able to import `llama_cpp`.
Use a CUDA-enabled `llama-cpp-python` build for this host; a CPU-only
wheel will load but will not be useful for the 12B Q8_0 writer lane.

Known-good Windows install for this ComfyUI venv:

```powershell
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe -m pip install --only-binary=:all: --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 llama-cpp-python==0.3.33
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe -m pip install --only-binary=:all: nvidia-cuda-runtime-cu12==12.4.127 nvidia-cublas-cu12==12.4.5.8
```

The second line provides the CUDA 12 DLLs required by the available
`llama-cpp-python` CUDA wheel. OTR preloads those DLLs before importing
`llama_cpp`; it still runs entirely in-process and does not start any server.

Quick readiness probe:

```powershell
cd C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe docs\gemma4\gemma4_test.py
```

The script prints a readiness dict before it loads the model.

## VRAM

Q8_0 is a real local residency path. The selector treats it like a local
LLM, not like OpenRouter or Comfy Credits:

- a different resident writer LLM is unloaded before the GGUF loads;
- the GGUF cache entry is reused while selected;
- switching away closes the llama-cpp model and clears CUDA cache.

Useful knobs:

```powershell
$env:GEMMA4_12B_N_CTX = '8192'
$env:GEMMA4_12B_N_GPU_LAYERS = '-1'
$env:GEMMA4_12B_MAX_NEW_TOKENS = '512'
```
