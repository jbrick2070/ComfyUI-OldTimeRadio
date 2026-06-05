---
repo_id: google/gemma-4-12b-it
license: apache_2_0
license_audit_status: mit_equivalent
verdict_date: 2026-06-03
audit_method: hf_model_card_read_plus_license_file
reviewer: cowork_audit_2026-06-03
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

- Added 2026-06-03 as a CANDIDATE writer model (12B class, same tier as
  the Mistral-Nemo default) for soak evaluation -- the E2B / E4B picks are
  too small for the strict structured passes. Not bound as the default
  workflow writer until a soak proves it.
- `prompt_profile = modern`. Eligible for the creative slot AND the
  technical slot. Apache 2.0 satisfies the Sprint D / D3 creative-binding
  gate, so default-binding is allowed if a soak promotes it.

## Notes

Ungated download on HuggingFace (the model_info gated flag reads False;
`requires_auth: true` in the catalog row passes the token harmlessly).
Encoder-free multimodal, used text-only by the OTR writer (same
`transformers_multimodal_text_only` backend as gemma-4-E4B). ~23.9 GB
safetensors; NF4-quantized at load to ~8 GB on the RTX 5080.
