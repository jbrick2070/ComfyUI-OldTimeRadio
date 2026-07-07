# Gemma 4 Local OpenAI-Compatible Lane

OTR has one Gemma 4 12B HTTP lane:

- `local_gemma4_12b`: backed by any external local OpenAI-compatible server.

The old `google/gemma-4-12b-it` sidecar route is removed. Stale saved pins are
rejected before the generic HuggingFace admit path so OTR does not attempt a
bad in-process transformers load.

Mistral-Nemo remains the default writer. `local_gemma4_12b` is visible in the
writer dropdown next to the other Gemma rows.

## External 12B Server

Start your local server outside ComfyUI, then launch ComfyUI with:

```powershell
$env:GEMMA4_12B_BASE_URL = 'http://127.0.0.1:8080/v1'
$env:GEMMA4_12B_MODEL_ID = 'ggml-org/gemma-4-12B-it-GGUF:Q4_K_M'
```

Then choose `local_gemma4_12b` in the writer model dropdown. If those defaults
already match your server, no environment variables are required.

The backend does not start llama-server, LiteRT-LM, or any other daemon. If the
endpoint is down, it fails closed with a named local OpenAI error.

## Memory Rule

The 12B lane should be external to the ComfyUI process on the 16 GB RTX 5080
laptop. It uses zero ComfyUI-process VRAM, and the writer never assumes a
Comfy-native 12B safetensor exists.

Do not put a nonexistent file such as `gemma4_12b_it_fp8_scaled.safetensors`
into `models/text_encoders` and expect OTR to use it. Native Comfy Gemma 4
work should be detect-and-test only, with the known Comfy-packaged E2B/E4B
files treated separately from the 12B writer lane.

## Smoke Probe

The code-level health helper is:

```python
from nodes import _otr_local_openai_backend as lob
print(lob.validate_local_openai_server())
```

It tries `/v1/models` when available and always sends a tiny
`/v1/chat/completions` prompt. It never starts or manages a server.
