---
repo_id: Nitral-AI/Captain-Eris_Violet-V0.420-12B
license: community
license_audit_status: pending
verdict_date: 2026-05-16
audit_method: hf_model_card_read_plus_license_file
reviewer: cowork_d0b_pending_g2
---

# Captain-Eris_Violet-V0.420-12B license audit

## Verdict

Community finetune; ungated. License posture inherits from upstream
base model (Mistral-Nemo derivative -- Apache 2.0) but the model card
does not always pin a license explicitly. Catalog
`license_audit_status` is `pending` until operator G2 inspects the
model card and confirms whether to flip to `mit_equivalent` or hold
at `pending` / `research_lane`.

## Source

HuggingFace model card: https://huggingface.co/Nitral-AI/Captain-Eris_Violet-V0.420-12B
Base model: Mistral-Nemo (Apache 2.0).

## OTR disposition

- Ungated community model. `vram_fit_tier = WARN` -- 12B at the edge,
  not soak-tested.
- `prompt_profile = modern`. Default workflow JSON binding NOT
  recommended.
- Available for operators who want to experiment; not part of the
  default-ship surface.

## Notes

Community finetunes often inherit Apache 2.0 from upstream Mistral but
the formal license declaration is sometimes missing from the model
card. G2 review should spot-check the model card README for an
explicit license line; if absent, hold at `pending`.
