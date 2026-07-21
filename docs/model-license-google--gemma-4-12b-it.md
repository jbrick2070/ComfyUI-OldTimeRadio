---
repo_id: google/gemma-4-12b-it
license: apache_2_0
license_audit_status: mit_equivalent
verdict_date: 2026-06-03
audit_method: hf_model_card_read_plus_license_file
reviewer: cowork_audit_2026-06-03
runtime_verification_date: 2026-07-20
---

# Gemma-4-12b-it license audit

## Verdict

Apache 2.0 -- MIT-equivalent permissive for the OTR catalog's purposes.
The Gemma 4 family (E2B / E4B / 12B / 26B A4B / 31B) ships under Apache
2.0, confirmed on the official Google HuggingFace repo and the Gemma 4
license document. Free for commercial and non-commercial use,
fine-tuning, modification, and redistribution, with no revenue ceiling
and no use-restriction appendix. Catalog `license_audit_status` is
`mit_equivalent`.

## Source

HuggingFace model card: https://huggingface.co/google/gemma-4-12b-it
  -- declares `License: apache-2.0`.
Gemma 4 license document: https://ai.google.dev/gemma/docs/gemma_4_license
Gemma 4 12B announcement (2026-06-03):
  https://developers.googleblog.com/gemma-4-12b-the-developer-guide/

## OTR disposition

- Restored 2026-07-20 as the canonical creative + technical writer on the
  in-process Transformers/HF lane. The official `Gemma4Unified` implementation
  in Transformers 5.10.4 loads the existing safetensors in NF4 at 7.15 GiB
  allocated / 7.29 GiB peak on the 16 GB RTX 5080 and produces coherent prose.
- Structured SciFi passes bind their exact Pydantic schema to
  lm-format-enforcer's `prefix_allowed_tokens_fn`, making invalid JSON tokens
  unsampleable on this lane. The independent GGUF row remains available but is
  not the canonical selection.
- `prompt_profile = modern`. Eligible for the creative slot AND the
  technical slot. Apache 2.0 satisfies the Sprint D / D3 creative-binding gate.

## Notes

Ungated download on HuggingFace (the audited model-info `gated` flag is False),
so the catalog correctly uses `requires_auth = false`. Runtime inference is
fully offline from:

`C:\ComfyUI-Models\huggingface\hub\models--google--gemma-4-12b-it`

The cache has a complete weighted snapshot plus a newer metadata-only snapshot.
OTR selects the weighted revision for model/config coherence and attaches the
newer local `chat_template.jinja` only when needed. No overlay, LoRA, Ollama,
llama.cpp, sidecar, HTTP server, or port is involved. The shared ComfyUI venv
must use `transformers>=5.10.4`; the historical Transformers 5.5 text-tower
remap is intentionally retired because it generated only `"."`.
