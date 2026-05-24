---
repo_id: google/gemma-2-2b-it
license: gated_terms
license_audit_status: research_lane
verdict_date: 2026-05-24
audit_method: hf_model_card_read_plus_license_file
reviewer: cowork_bug_local_262_2026-05-24
---

# Gemma-2-2b-it license audit

## Verdict

Restricted Gemma Terms of Use -- `gated_terms` / `research_lane` for the
OTR catalog. The HuggingFace model card declares `License: gemma` and
links the Google "Gemma Terms of Use" plus a Prohibited Use Policy. This
is a custom Google license, NOT Apache 2.0. It permits commercial and
non-commercial use, fine-tuning, and redistribution, but carries a
downstream flow-down obligation (every redistributor must pass the
terms + Prohibited Use Policy along) and a use-restriction appendix.
Catalog `license_audit_status` is `research_lane`: the model is usable
for OTR's internal technical-slot generation but is NOT bound in the
shipped default-workflow creative-binding JSON without further review.

## Distinction from the Gemma 4 family

Gemma 1 / 2 / 3 ship under the restricted Gemma Terms of Use. Gemma 4
(E2B / E4B / 26B A4B / 31B) ships under Apache 2.0 -- see
`docs/model-license-google--gemma-4-e2b-it.md`. The two families are NOT
interchangeable for licensing purposes. gemma-2-2b-it is a Gemma 2 row
and inherits the restricted license.

## Source

HuggingFace model card: https://huggingface.co/google/gemma-2-2b-it
  -- declares `License: gemma`.
Gemma Terms of Use: https://ai.google.dev/gemma/terms
Gemma Prohibited Use Policy: https://ai.google.dev/gemma/prohibited_use_policy

## OTR disposition

- Smallest curated technical-slot pick: 2B parameters, NF4-quantized,
  tiny VRAM footprint. Targeted at the writer's `technical_model` slot
  (JSON validators, GBNF grammar output, reviewer verdicts, the style
  picker's chooser pass).
- `prompt_profile = modern`.
- Research-lane: technical-slot use only. NOT eligible for binding into
  the default-workflow creative slot without an operator review of the
  Gemma Terms of Use flow-down obligation.

## Notes

The HuggingFace repo is gated for download access (accept-terms
click-through; `requires_auth: true` in the catalog row).

Gemma-2 has no system role by design -- its chat template contains a
literal `raise_exception("System role not supported")`. The OTR writer
generate path folds any system message into the first user turn via
`normalize_messages_for_tokenizer` (BUG-LOCAL-262) so the row is a
clean technical-slot pick. This is a generation-path detail, orthogonal
to the license verdict.
