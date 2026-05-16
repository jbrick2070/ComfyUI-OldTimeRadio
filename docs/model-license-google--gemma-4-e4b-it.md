---
repo_id: google/gemma-4-E4B-it
license: gated_terms
license_audit_status: pending
verdict_date: 2026-05-16
audit_method: hf_model_card_read_plus_license_file
reviewer: cowork_d0b_pending_g2
---

# Gemma-4-E4B-it license audit

## Verdict

Same as Gemma-4-E2B-it (Google Gemma Terms of Use). Gated for download
acceptance; license itself is not MIT-equivalent but permissive in
practice for the OTR offline use-case. Catalog `license_audit_status`
is `pending` until operator G2 confirms.

## Source

HuggingFace model card: https://huggingface.co/google/gemma-4-E4B-it
Google Gemma Terms of Use: https://ai.google.dev/gemma/terms
Gemma Prohibited Use Policy: https://ai.google.dev/gemma/prohibited_use_policy

## OTR disposition

- Slightly larger technical-slot option than E2B-it.
- `prompt_profile = modern`. Eligible for technical slot only.
- Default workflow JSON binding allowed for technical slot.

## Notes

Operator G2 review applies to both Gemma rows at once (same license).
Flipping the verdict on E2B-it should flip E4B-it identically.
