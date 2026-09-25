---
repo_id: mistralai/Mistral-Nemo-Instruct-2407
license: apache_2_0
license_audit_status: mit_equivalent
verdict_date: 2026-05-16
audit_method: hf_model_card_read_plus_license_file
reviewer: cowork_d0b_pending_g2
---

# Mistral-Nemo-Instruct-2407 license audit

## Verdict

Apache 2.0 -- MIT-equivalent permissive for the OTR catalog's purposes.
Free for commercial and non-commercial use. No revenue ceiling, no AUP
restriction beyond standard prohibitions. Catalog `license_audit_status`
flips to `mit_equivalent`.

## Source

HuggingFace model card: https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407
Mistral AI repositories ship Mistral-Nemo under Apache 2.0 per Mistral's
public statement at release. Gated on the hub for download bookkeeping
(`requires_auth: true` in the catalog row) but the license itself imposes
no use restriction relevant to OTR.

## OTR disposition

- Audio C7 byte-identical baseline. Default for both writer slots.
- Default workflow JSON binding allowed without review.
- May be used as creative slot OR technical slot.
- `prompt_profile = modern`. Reflection pass runs against this row.

## Notes

The gated nature on HuggingFace is a download-flow artifact (accept-terms
click-through). The license itself is Apache 2.0 and does not gate use.
G2 operator confirmation: this disposition is the catalog default-spine
assumption; reviewer should spot-check the Mistral AI announcement page
and the HF model card LICENSE file before D1a opens.
