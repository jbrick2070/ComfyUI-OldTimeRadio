---
repo_id: google/gemma-4-E2B-it
license: gated_terms
license_audit_status: pending
verdict_date: 2026-05-16
audit_method: hf_model_card_read_plus_license_file
reviewer: cowork_d0b_pending_g2
---

# Gemma-4-E2B-it license audit

## Verdict

Gemma Terms of Use (Google) -- NOT MIT-equivalent. Permits commercial use
with restrictions enumerated in the Gemma Prohibited Use Policy. Requires
acceptance of click-through terms on HuggingFace (`requires_auth: true`).
Catalog `license_audit_status` is `pending` until operator G2 review
confirms the OTR use-case (offline non-distribution-of-weights) sits
inside the permitted envelope.

## Source

HuggingFace model card: https://huggingface.co/google/gemma-4-E2B-it
Google Gemma Terms of Use: https://ai.google.dev/gemma/terms
Gemma Prohibited Use Policy: https://ai.google.dev/gemma/prohibited_use_policy

## OTR disposition

- Compact multimodal-text-only technical-slot option.
- `prompt_profile = modern`. Eligible for technical slot only at present.
- Default workflow JSON binding allowed for technical slot (matches
  current Sprint C usage). Operator G2 review may flip
  `license_audit_status` to `mit_equivalent` if the Gemma Terms are
  determined OTR-compatible at the use-case level.

## Notes

The Gemma family is permissive in practice for the OTR workflow (local
inference, no weight redistribution, no model-output filtering of the
prohibited categories required) but the formal license is not Apache 2.0
or MIT. Pending verdict is the conservative position until operator G2
confirms.
