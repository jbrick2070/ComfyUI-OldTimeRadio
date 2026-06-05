<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: no. The plan fails to account for weight key prefixes in multimodal architectures and lacks a concrete VRAM eviction protocol for the 16GB constraint.

MUST-FIX BEFORE BUILD:
1. [Candidate Paths - B] Config rename ignores weight prefixes. You cannot simply rename `model_type` from `gemma4_unified` to `gemma4`. Multimodal wrapper models in transformers use prefixed state_dict keys (e.g., `text_model.model.layers...` or `language_model...`). Loading this directly into a `gemma4` text-only class will fail due to missing keys.
   *Fix:* Write an offline conversion script that extracts the `text_config` block into a new root `config.json`, changes its `model_type` to `gemma4` (or whatever 5.5.0 expects), and rewrites the `.safetensors` files to strip the multimodal prefix from all text tower weights.
2. [Candidate Paths - C] VRAM Coexistence OOM. A 16GB GPU cannot concurrently hold the main OTR pipeline models (Mistral-Nemo + audio) and an 8GB NF4 Gemma-4 sidecar. 
   *Fix:* The IPC protocol must include explicit VRAM handoffs. The main process must move its LLM/audio weights to system RAM, signal the sidecar to load Gemma into VRAM, wait for the generation response, signal the sidecar to evict Gemma, and then reload its own models.
3. [Candidate Paths - A] Invalid dependency target. The grounding explicitly shows `"transformers_version": "5.10.0.dev0"`. This is an unreleased, unstable development branch. You cannot safely in-place upgrade a protected, offline-first production stack to a `.dev0` build.
   *Fix:* Discard Path A entirely. Do not touch the main venv's transformers version.

SHOULD-FIX:
1. [Candidate Paths - D] GGUF/llama.cpp is superior for the VRAM constraint. Instead of a heavy HuggingFace sidecar (Path C) that requires manual VRAM eviction, llama.cpp handles memory mapping and partial GPU offload natively. 
   *Fix:* Prioritize Path D. Check if llama.cpp supports the text-only Gemma 4 architecture. If yes, convert the extracted text weights to GGUF and run a lightweight llama.cpp server sidecar.

OPTIONAL / NICE-TO-HAVE:
- If using Path C (sidecar), use a tiny FastAPI server with standard OpenAI-compatible endpoints. This makes the IPC transport trivial (HTTP) and allows easy swapping of the sidecar backend (HF vs vLLM vs llama.cpp).

CUT THESE (over-engineering):
1. [Candidate Paths - A] In-place upgrade. Safe to cut because upgrading to a dev build violates the "protected stack" constraint and introduces massive regression risk for zero guaranteed payoff.

[ASSUMPTION] I am assuming the `.safetensors` checkpoint for this unified model prefixes the text tower weights (e.g., with `text_model.`), which is standard HuggingFace practice for multimodal architectures. If they are not prefixed, Path B (with just a config extraction) becomes much easier, but you must verify the keys first.