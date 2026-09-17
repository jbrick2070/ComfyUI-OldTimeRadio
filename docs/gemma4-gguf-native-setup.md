# Gemma 4 12B writer -- GGUF lane retired

The writer GGUF identity is gone. There is no `unsloth/gemma-4-12b-it-GGUF`
row in the COMBO, and `validate_model_id` / `request_slot` reject it.

Use:

```text
google/gemma-4-12b-it
```

NF4 is baked into that pick. Details: [gemma4/README.md](gemma4/README.md).

Video and image GGUF UNets stay: Foley, mime, Klein, LTX, and Wan. Those
are engine weight files, not writer models, and they are not this page.
