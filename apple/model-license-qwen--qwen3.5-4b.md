---
repo_id: Qwen/Qwen3.5-4B
license: apache_2_0
license_audit_status: mit_equivalent
verdict_date: 2026-09-06
audit_method: hf_hub_metadata_read
reviewer: cowork_4060_2026-09-06
---

# Qwen3.5-4B license audit

## Verdict

Apache 2.0 -- MIT-equivalent permissive for the OTR catalog's purposes.
Free for commercial and non-commercial use, fine-tuning, modification
and redistribution, with no revenue ceiling and no use-restriction
appendix. Catalog `license_audit_status` is `mit_equivalent`, matching
the Gemma 4 and Mistral-Nemo rows.

## Source

Hugging Face Hub metadata read, 2026-09-06, anonymous:
https://huggingface.co/Qwen/Qwen3.5-4B

Exact fields returned:

- `License: apache-2.0` (metadata field AND the `license:apache-2.0` tag)
- Author: Qwen
- Parameters: 4659.9M
- Architecture: `qwen3_5`; model class `AutoModelForMultimodalLM`
- Library: transformers; `safetensors` tag present
- Task: image-text-to-text

No download gate or accept-terms interstitial was reported for the repo,
which is consistent with the catalog row's `requires_auth: false`. This
matches the earlier campaign finding recorded in `4060_DRILL_LOG.md`
Step99 (official metadata public / ungated / Apache 2.0).

## OTR disposition

- Curated writer row, `loader_backend = transformers_multimodal_text_only`.
  The repo is a multimodal checkpoint used in TEXT-ONLY mode, the same
  handling as the Gemma 4 E2B / E4B rows.
- `prompt_profile = modern`. Eligible for the creative slot AND the
  technical slot.
- Apache 2.0 satisfies the SprintD / D3 creative-binding gate, so a
  default workflow JSON binding is permitted without further license
  review. No such binding is made by this audit.

## Notes

The license verdict is INDEPENDENT of runtime qualification. As of this
audit the row carries `vram_fit_tier = "WARN"` and has NOT been
soak-tested: speed, VRAM residency, prose, constrained JSON and a whole
episode result on the 8GB RTX4060 all remain NOT TESTED. Per the
operator ruling of 2026-09-06 (see `LLM_PREFLIGHT_GUIDE.md` Gate2) a
WARN row stays visible in the dropdown and auto-downloads when selected;
this file records the license only, and must not be read as a fit or
performance clearance.

The `approx_safetensors_gb = 8.68` figure in the catalog row is the
official shard total (9,319,828,096 bytes) -- disk, not VRAM resident.
