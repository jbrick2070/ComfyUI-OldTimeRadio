---
repo_id: google/gemma-4-E2B-it
license: apache_2_0
license_audit_status: mit_equivalent
verdict_date: 2026-05-21
audit_method: hf_model_card_read_plus_license_file
reviewer: cowork_reaudit_2026-05-21
---

# Gemma-4-E2B-it license audit

## Verdict

Apache 2.0 -- MIT-equivalent permissive for the OTR catalog's purposes.
The HuggingFace model card declares `License: apache-2.0` and links a
Gemma-4-specific license document. Free for commercial and
non-commercial use, fine-tuning, modification, and redistribution, with
no revenue ceiling and no use-restriction appendix. Catalog
`license_audit_status` is `mit_equivalent`.

## Re-audit note (2026-05-21)

This row previously carried `license: gated_terms` /
`license_audit_status: pending`, written 2026-05-16 on the assumption
that Gemma 4 inherited the older Google "Gemma Terms of Use" -- the
restricted custom license used by Gemma 1 / 2 / 3, which carries a
Prohibited Use Policy and a downstream flow-down obligation. That
assumption was wrong. The Gemma 4 family ships under Apache 2.0,
confirmed on the official Google HuggingFace repo. The 2026-05-16 audit
cited the generic `ai.google.dev/gemma/terms` page rather than the
Gemma-4-specific license document. Corrected here to match the model
card.

## Source

HuggingFace model card: https://huggingface.co/google/gemma-4-E2B-it
  -- declares `License: apache-2.0`.
Gemma 4 license document: https://ai.google.dev/gemma/docs/gemma_4_license

## OTR disposition

- Compact multimodal-text-only option; the `TEST_TECHNICAL_LLM`
  catalog constant. Not bound in the shipped default workflow JSON.
- `prompt_profile = modern`. Eligible for the creative slot AND the
  technical slot.
- Default workflow JSON binding allowed without further review --
  Apache 2.0 satisfies the Sprint D / D3 creative-binding gate.

## Notes

The HuggingFace repo is gated for download access (accept-terms
click-through; `requires_auth: true` in the catalog row). That gate is
a download-flow artifact and is orthogonal to the Apache 2.0 license,
which imposes no use restriction relevant to OTR. Same handling as the
Mistral-Nemo row. The whole Gemma 4 family (E2B / E4B / 26B A4B / 31B)
ships under the same Apache 2.0 license.
