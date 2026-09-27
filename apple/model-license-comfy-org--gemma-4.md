---
repo_id: Comfy-Org/gemma-4
license: apache_2_0
license_audit_status: mit_equivalent
verdict_date: 2026-09-26
audit_method: hf_model_api_card_license_read
reviewer: cowork_2026-09-26
---

# Comfy-Org/gemma-4 license audit

## Verdict

Apache 2.0 -- MIT-equivalent permissive for the OTR catalog's purposes.
The Hugging Face model API for `Comfy-Org/gemma-4` reports card license
`apache-2.0` (tag `license:apache-2.0`) and `gated: false`, read without a
token on 2026-09-26 at revision `63d0f7c476756b88910170c1df75e2384ea1af31`,
the revision OTR pins. The repo holds Comfy-Org's ComfyUI conversions of
Google's Gemma 4 weights, which Google ships under Apache 2.0 (see
`model-license-google--gemma-4-e2b-it.md`). Catalog `license_audit_status`
is `mit_equivalent`.

## Source

Hugging Face model API: https://huggingface.co/api/models/Comfy-Org/gemma-4
  -- `cardData.license = apache-2.0`, `gated = false`.
Upstream model license: https://ai.google.dev/gemma/docs/gemma_4_license

## OTR disposition

- The catalog row `comfy_native:gemma4-e2b-it-int8-convrot` (plan row 0n)
  names this repo as its `hf_repo_id`: the one file it loads,
  `text_encoders/gemma4_e2b_it_int8_convrot.safetensors`, comes from here,
  pinned by revision, size and sha256 in `_otr_visual_assets._PINNED_SOURCES`.
- `prompt_profile = modern`. Eligible for the creative slot and the
  technical slot. Not selected by any shipped workflow yet.

## Notes

Ungated: the file downloads without a token or a licence click. The
repo also carries other Gemma 4 conversions (E4B, 12B, bf16 builds);
this audit covers the repo, and the catalog uses one file from it.
