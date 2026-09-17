#!/bin/bash
set -euo pipefail
echo "=== writer backend ==="
grep -E 'Loading LLM|Selector|gguf|quantized|gemma-4|HF_HOME|llama.cpp|GGUF' /workspace/otr-config/comfy_8188.log | tail -n 40
echo "=== treatment ==="
grep -E 'my_story_treatment|GenerationDegeneracy|liveness' /workspace/otr-config/comfy_8188.log | tail -n 20
