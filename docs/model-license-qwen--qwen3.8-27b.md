---
repo_id: Qwen/Qwen3.8-27B
license: apache_2_0
license_audit_status: mit_equivalent
verdict_date: 2026-09-21
audit_method: hf_hub_metadata_read
reviewer: cowork_5080_2026-09-21
---

# Qwen3.8-27B license audit

## Verdict

Apache 2.0 -- MIT-equivalent permissive for the OTR catalog's purposes.
Free for commercial and non-commercial use, fine-tuning, modification and
redistribution, with no revenue ceiling and no use-restriction appendix.
Catalog `license_audit_status` is `mit_equivalent`, matching the Qwen3.5-4B,
Gemma 4 and Mistral-Nemo rows.

## Source

Hugging Face Hub metadata read, 2026-09-21, anonymous:
https://huggingface.co/Qwen/Qwen3.8-27B

Exact fields returned:

- `License: apache-2.0` (metadata field AND the `license:apache-2.0` tag)
- Author: Qwen
- Parameters: 27781.4M
- Architecture: `qwen3_5`; model class `AutoModelForMultimodalLM`
- Library: transformers; `safetensors` tag present
- Task: image-text-to-text

No download gate or accept-terms interstitial was reported, and an
anonymous read of the config and the 18-shard file list succeeded from the
pod with no token, which is what the row's `requires_auth: false` records.
This is the same posture as its 4B sibling rather than an inference from it.

## OTR disposition

- Curated writer row, `loader_backend = transformers_multimodal_text_only`,
  `text_only_load = native_text_decoder`. A multimodal checkpoint driven in
  TEXT-ONLY mode, the same handling as the Qwen3.5-4B row.
- `prompt_profile = modern`. Eligible for the creative slot AND the
  technical slot.
- Apache 2.0 satisfies the SprintD / D3 creative-binding gate, so a default
  workflow JSON binding would be permitted without further license review.
  **No such binding is made, here or anywhere.** The row is selectable in
  every graph and selected by none of them.

## Notes

The license verdict is INDEPENDENT of runtime qualification, and the two
must not be read together. What HAS been measured, on a RunPod RTX 5090 on
2026-09-21:

- loads under `bnb_nf4` in 84 s, 17.7 GiB resident / 17.8 GiB peak on a
  33.7 GiB card
- free-form prose at 22.5 tok/s
- LMFE-constrained JSON parsed clean at 27.6 tok/s
- `config.text_config.model_type` is `qwen3_5_text`, so the checkpoint
  genuinely splits and the text-decoder-only load is not an assumption

What has NOT been measured: any published episode. The row carries
`vram_fit_tier = "WARN"` for exactly that reason, and per the operator
ruling of 2026-09-06 a WARN row stays visible in the dropdown and
auto-downloads when selected. This file records the license only and must
not be read as a fit or performance clearance.

The `approx_safetensors_gb = 51.75` figure in the catalog row is the
official shard total (55,563,006,776 bytes across 18 shards) expressed in
GiB -- disk, not VRAM resident. The NF4 load measured above is roughly a
third of it.
