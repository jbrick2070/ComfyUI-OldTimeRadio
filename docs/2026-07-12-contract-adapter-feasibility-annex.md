# Feasibility annex: can contract adapters be trained on this box?
2026-07-12. Companion to docs/2026-07-12-contract-adapter-problem-statement.md
and kibitz-runs/2026-07-12-contract-adapter/r1/final.md.
Question researched: is the adapter idea possible at all, and which models
that fit the RTX 5080 Laptop (16 GB, Blackwell sm_120) can be trained this
way easily?

## Answer

YES. The proposal is the textbook QLoRA use case, and every base model on
the curated catalog except one trains comfortably on this card. The
runtime story is even better than the problem statement hoped: the
project's own 4-bit loader (bnb_nf4) is EXACTLY the base configuration
QLoRA adapters are trained against and served on, so "which adapter
format/runtime can coexist with the current 4-bit model loader" (open
question 1) has a native answer -- a PEFT LoRA on the NF4 base, no new
loader, ~100-300 MB VRAM over the resident base, far inside the 14.5 GB
ceiling.

## Per-base verdict (curated catalog, `nodes/_otr_model_catalog.py`)

| Base (exact row) | Params | QLoRA train VRAM | Verdict on 16 GB |
| --- | --- | --- | --- |
| google/gemma-4-E2B-it (matformer) | ~2B eff. | a few GB | EASY |
| google/gemma-4-E4B-it (matformer) | ~4B eff. | <12 GB (Unsloth-documented) | EASY |
| gemma-2-2b-it (technical slot) | 2B | a few GB | EASY |
| mistralai/Mistral-Nemo-Instruct-2407 (DEFAULT_LLM, creative slot -> serves P5) | 12B | ~10-13 GB at 2-4k ctx | EASY-to-COMFORTABLE (its NF4 base already runs at 7.74 GiB on this exact GPU, `scripts/otr_gemma4_doctor.py:12`) |
| Qwen2.5-14B-Instruct | 14B | ~13-15 GB, batch 1 + grad ckpt + short ctx | TIGHT BUT FITS |
| Gemma-4-12B Q8_0 GGUF peer | 12B | train on the HF checkpoint (~10-13 GB), convert | FITS (see GGUF lane below) |

Not "easily": 26-30B-MoE-class (e.g. Gemma-4 26B MoE, Qwen3-30B-A3B) sit at
~16-17.5 GB even with Unsloth -- over this card's real headroom once the
desktop's ~1.5 GB baseline is counted. Out of scope.

Contract adapters are also the CHEAP end of fine-tuning: LoRA rank 8-16 on
a few hundred to a few thousand accepted-artifact examples is minutes to a
few hours per run on a 12B base at short context, not a multi-day job.

## Runtime coexistence, per loader backend

- **transformers + bnb_nf4 (`llm_quant_policy` default, OTR_LedgerScriptWriter.py:1243):**
  native. `PeftModel.from_pretrained(cache_entry["model"], adapter_dir)` is
  the QLoRA serving configuration; adapters CANNOT be merged into 4-bit
  weights, which is a feature here -- the adapter stays a separate,
  versioned, independently removable artifact (guardrail 5 satisfied by
  construction). Unload = drop the PEFT wrapper, base state restored.
- **GGUF lane (`_otr_gguf_backend.py`, in-process llama-cpp-python):**
  supported via a second artifact form: train the PEFT LoRA against the HF
  checkpoint, run `convert_lora_to_gguf.py`, load next to the Q8_0 base
  with `lora_path` / `--lora`. Hot-swappable, no merge, works on a
  quantized base (post-2024 llama.cpp LoRA refactor). Small quality caveat:
  adapter trained against fp16/NF4 weights applied on a Q8_0 base is a
  minor mismatch -- acceptable, but evaluate on the GGUF lane specifically
  before promoting there.
- **OpenRouter lane (`_otr_openrouter_backend.py`):** no adapter possible;
  out of scope by definition.

## Training stack on this machine

- **peft >= 0.18.0** is the floor for the transformers v5 line this venv
  runs (transformers 5.5) -- older PEFT is incompatible. QLoRA
  (bitsandbytes 4-bit + LoRA) is a supported first-class path in
  PEFT/TRL.
- **bitsandbytes** ships cu128 wheels that cover Blackwell; NF4 inference
  is already proven in production on this GPU, so the bnb kernel story is
  known-good locally.
- **Unsloth** officially supports Blackwell RTX 50-series (needs
  triton >= 3.3.1, CUDA 12.8 builds). Its own guidance for Blackwell on
  Windows: WSL2 is the supported path; native Windows fights triton.
  Decision: train in WSL2 with Unsloth for the 2x speed / lower VRAM, or
  train natively with plain PEFT + TRL (no triton dependency) and accept
  slower runs. Either satisfies the "separate pinned trainer environment"
  rule -- the repo's requirements.txt has no peft/trl and must stay that
  way; runtime adapter deps lazy-import only after adapter selection.
- Torch 2.10/cu128 on the box already matches the cu128 requirement.

## Sequencing reminder (from the r1 kibitz verdict)

Trainability was never the blocker. Before any training run: prospective
telemetry capture (the current `_call` journal is not training-fidelity),
residual-error census after current hardening, and a constrained-decoding
control arm (lm-format-enforcer already wired for the HF lane at
`OTR_LedgerScriptWriter.py:4815`; `json_schema` response_format on the GGUF
lane) -- the adapter is only worth training for the defect classes that
survive both. See kibitz-runs/2026-07-12-contract-adapter/r1/final.md.

## Sources

- Unsloth requirements / VRAM tables: https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements
- Unsloth on Blackwell RTX 50-series: https://unsloth.ai/docs/blog/fine-tuning-llms-with-blackwell-rtx-50-series-and-unsloth
- Unsloth Windows install (WSL2 guidance): https://unsloth.ai/docs/get-started/install/windows-installation
- NVIDIA: training LLMs on Blackwell desktops with Unsloth: https://developer.nvidia.com/blog/train-an-llm-on-an-nvidia-blackwell-desktop-with-unsloth-and-scale-it/
- Gemma 3n (E2B/E4B matformer) fine-tuning, <12 GB QLoRA: https://unsloth.ai/docs/models/tutorials/gemma-3-how-to-run-and-fine-tune/gemma-3n-how-to-run-and-fine-tune and https://unsloth.ai/blog/gemma-3n
- PEFT >= 0.18 requirement for transformers v5: https://github.com/huggingface/peft/releases
- PEFT quantization (QLoRA) developer guide: https://huggingface.co/docs/peft/en/developer_guides/quantization
- bitsandbytes integrations (transformers/PEFT/TRL): https://huggingface.co/docs/bitsandbytes/main/integrations
- llama.cpp LoRA-on-GGUF (convert_lora_to_gguf, --lora on quantized base): https://github.com/ggml-org/llama.cpp/blob/master/convert_lora_to_gguf.py and https://huggingface.co/blog/ngxson/gguf-my-lora
- VRAM sizing overview (full/LoRA/QLoRA by model size): https://www.spheron.network/blog/gpu-vram-requirements-fine-tune-llm-2026/
