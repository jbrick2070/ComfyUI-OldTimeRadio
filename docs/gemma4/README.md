# Run Gemma 4 12B in OTR

The only Gemma 4 12B writer is the official Hugging Face checkpoint:

```text
google/gemma-4-12b-it
```

Set **both** writer slots to that id. NF4 is baked into the pick -- there
is no other 12B variant and you do not also change Quant. Canonical can
stay Qwen; switching the two writer widgets to this row loads NF4 even
if Quant still says `none`.

This is the Transformers / bitsandbytes NF4 lane. It auto-downloads an
ungated snapshot into the canonical HF cache. It does not use Ollama,
llama.cpp, a sidecar, or a port.

16 GB NVIDIA graphs already save this pick with Quant `bnb_nf4`. Canonical
still saves Qwen 3.5 4B + Quant `none` until you change the two writer
widgets (or apply an `otr_16gb_*` profile).

## VRAM

NF4 measured at 7.15 GiB allocated / 7.29 GiB peak on the 16 GB RTX 5080,
including coherent prose and LMFE-constrained JSON. Full precision is not
the 16 GB load.

If Quant is left at `none`, the run stops before load with a mismatch
naming `bnb_nf4`. Change Quant. Do not raise the VRAM ceiling to paper over
it.
