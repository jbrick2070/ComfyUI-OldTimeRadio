---
repo_id: Qwen/Qwen2.5-14B-Instruct
license: apache_2_0
license_audit_status: mit_equivalent
verdict_date: 2026-05-16
audit_method: hf_model_card_read_plus_license_file
reviewer: cowork_d0b_pending_g2
---

# Qwen2.5-14B-Instruct license audit

## Verdict

Apache 2.0 -- MIT-equivalent permissive. Free for commercial use, no
revenue ceiling, no AUP clause beyond the Apache 2.0 standard
prohibitions. Catalog `license_audit_status` is `mit_equivalent`.

## Source

HuggingFace model card: https://huggingface.co/Qwen/Qwen2.5-14B-Instruct
Qwen2.5 series license per Alibaba release: Apache 2.0 for the
non-72B-Chat variants. The 14B-Instruct variant is Apache 2.0; the 72B
variant carries a separate Qwen Research License (not on the OTR
catalog). Ungated; no acceptance click-through required.

## OTR disposition

- Ungated ungated; 14B at FP16 requires quantization or offload to fit
  the 16 GB ceiling. Catalog notes mark this row `vram_fit_tier = WARN`.
- `prompt_profile = modern`. Eligible for either slot once VRAM
  approach is validated.
- Default workflow JSON binding NOT recommended until soak-tested
  (catalog notes say "Available for users with bigger rigs; NOT
  advertised in gated-error recovery hint").

## Notes

License posture is clean; VRAM posture is the gate. Sprint D scope does
NOT include moving Qwen out of WARN tier -- that is downstream Sprint A
or Sprint G work.
