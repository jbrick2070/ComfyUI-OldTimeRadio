---
repo_id: inflatebot/MN-12B-Mag-Mell-R1
license: community
license_audit_status: pending
verdict_date: 2026-05-16
audit_method: hf_model_card_read_plus_license_file
reviewer: cowork_d0b_pending_g2
---

# MN-12B-Mag-Mell-R1 license audit

## Verdict

Community finetune of Mistral-Nemo (the `MN-12B` prefix is the
convention). Inherits Apache 2.0 from upstream in principle but the
model card license declaration must be confirmed at G2. Catalog
`license_audit_status` is `pending`.

## Source

HuggingFace model card: https://huggingface.co/inflatebot/MN-12B-Mag-Mell-R1
Base model: Mistral-Nemo (Apache 2.0).

## OTR disposition

- Ungated community model. `vram_fit_tier = WARN` -- 12B at the edge.
- `prompt_profile = modern`. Default workflow JSON binding NOT
  recommended.
- Available for experimentation; not part of default-ship surface.

## Notes

Same posture as Captain-Eris row. G2 review applies the same standard:
explicit license line on the model card -> flip to `mit_equivalent`;
absent -> hold at `pending`.
