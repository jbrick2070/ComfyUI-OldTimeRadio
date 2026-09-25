---
repo_id: unsloth/Llama-3.2-3B-Instruct
license: community
license_audit_status: research_lane
verdict_date: 2026-09-06
audit_method: hf_hub_metadata_read
reviewer: cowork_4060_2026-09-06
---

# Llama-3.2-3B-Instruct (unsloth mirror) license audit

## Verdict

**NOT permissive. Llama 3.2 Community License, not Apache 2.0 or MIT.**
Catalog `license` is `community` and `license_audit_status` is
`research_lane` -- deliberately NOT `mit_equivalent`, which every other
ungated row in this catalog carries. Do not copy those rows' status onto
this one.

## Source

Hugging Face Hub metadata read, 2026-09-06, anonymous:
https://huggingface.co/unsloth/Llama-3.2-3B-Instruct

Exact fields returned:

- License: `llama3.2` (tag `license:llama3.2`)
- Author: unsloth; `base_model: meta-llama/Llama-3.2-3B-Instruct`
- Parameters: 3212.7M (3,212,749,824)
- Model class `AutoModelForCausalLM`, architecture `llama`
- Library transformers; `safetensors` tag present
- Downloads 3.9M

## Gating

**Ungated on THIS mirror; the upstream Meta repo is NOT.**
`meta-llama/Llama-3.2-3B-Instruct` refuses anonymous filesystem reads
(`HF_FS_ACCESS_DENIED`), while this mirror served `config.json` and
`model.safetensors.index.json` on an identical anonymous call, and its
repo metadata carries no gated status. That is why the mirror is the
curated row and the upstream repo is not.

**An ungated mirror does not relicense the weights.** Download access and
licence are different questions and only the first is settled by the
mirror. Anyone shipping this commercially is bound by Meta's terms, not
by unsloth's redistribution.

## Why this row exists

Every other curated row needs 4-bit quantization to fit an 8 GB card, and
bitsandbytes NF4 is a compiled-CUDA path. On a host where bitsandbytes is
unavailable -- AMD/ROCm in particular -- those rows must load unquantized,
and none of them fits:

```
Qwen/Qwen3.5-4B        8.06 GiB bf16 text-only   over
google/gemma-4-E2B-it  ~8.65 GiB bf16            over
google/gemma-4-E4B-it   8.38 GiB bf16            over
google/gemma-4-12b-it  ~24 GB bf16               far over
google/gemma-2-2b-it   ~5.2 GiB bf16             fits, but GATED
unsloth/Llama-3.2-3B    5.98 GiB bf16            FITS, ungated
```

This is the only row that fits an 8 GB card unquantized without a token.

## OTR disposition

- Curated writer row, `loader_backend = transformers_safetensors`, plain
  llama, no `trust_remote_code`.
- `vram_fit_tier = "WARN"`: it has never been soak-tested, and no AMD
  hardware has run it. Per the operator ruling of 2026-09-06 a WARN row
  stays visible in the dropdown and auto-downloads when selected.
- `prompt_profile = modern`. Eligible for either slot.
- **NOT bound in any default workflow JSON.** `research_lane` status means
  a default creative binding needs a further licence decision that this
  audit does not grant.

## Notes

The catalog badge UNDERSTATES this row. `vram_badge_for` halves
`approx_safetensors_gb` on the assumption of a 4-bit load, so the picker
shows roughly 3.2 GB while an unquantized load really costs 5.98 GiB --
on precisely the hosts this row exists to serve. The heuristic is wrong
here in the safe direction for NVIDIA and the unsafe direction for AMD.

Nothing about this row is qualified on hardware. Speed, memory, prose,
constrained JSON and any episode result are all NOT TESTED.
