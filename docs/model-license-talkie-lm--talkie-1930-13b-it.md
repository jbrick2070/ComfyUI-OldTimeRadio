---
repo_id: talkie-lm/talkie-1930-13b-it
license: non_commercial
license_audit_status: research_lane
verdict_date: 2026-05-16
audit_method: hf_model_card_read_plus_license_file
reviewer: cowork_d0b_pending_g2
---

# talkie-1930-13b-it license audit

## Verdict

PENDING DEFINITIVE READ. Default placeholder posture per v3 plan: treat
as `non_commercial` + `research_lane` until operator G2 review of the
HuggingFace model card and any LICENSE file confirms the actual terms.
Historical-text-trained models commonly ship under CC-BY-NC, CC-BY-SA,
or research-only terms; OTR core stays MIT and any non-permissive
upstream component must enter the catalog at `research_lane` with
default-workflow binding blocked.

## Source

HuggingFace model card: https://huggingface.co/talkie-lm/talkie-1930-13b-it
LICENSE file in repo (if present).
Underlying base model: pending identification at G2 (talkie-lm
ecosystem documentation needed).

## OTR disposition

- Period-trained 13B model at GPTQ int4 quantization.
- `prompt_profile = otr_1940s_v1`. Catalog-selectable but NOT eligible
  for default workflow JSON binding until `license_audit_status` flips
  to `mit_equivalent`.
- Era mismatch caveat: training corpus is pre-1930; OTR period system
  prompt targets 1938-1952. Modern news with post-1952 references
  produces era-anachronistic dialogue. Documented as `research_lane`
  caveat.
- D4 runtime gates: VRAM peak, determinism xfail (GPTQ split-K
  nondeterminism), diction guard, modern-news warning.

## G2 review checklist

1. Read https://huggingface.co/talkie-lm/talkie-1930-13b-it model card
2. Look for explicit license line in README.md or LICENSE file
3. Identify the upstream base model (Mistral / Llama / Qwen variant)
4. Determine commercial-use posture; revenue ceiling if any; AUP clause
5. Flip `license` enum + `license_audit_status` to actual verdict
6. Update this file's verdict_date and audit_method fields
7. Commit verdict ON sprint-d-period-llm BEFORE D1a opens

## Notes

This audit file is structured as a placeholder so the D0b framework
exercises against a real disk artifact. The fields are not yet truth;
G2 makes them truth. D1a reads this file's license / license_audit_status
fields into the catalog row, so flipping these fields is the canonical
way to flip the row.
