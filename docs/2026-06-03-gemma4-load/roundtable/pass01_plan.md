# gemma-4-12b -- build plan: text-only load on the EXISTING transformers 5.5

## Verdict (grounded)
No sidecar, no upgrade. transformers 5.5 already models the `gemma4` text
architecture; the checkpoint's text tower loads into 5.5 `Gemma4ForCausalLM` with
0 unexpected keys / 0 shape issues / 1 tied-head artifact. Load text-only on the
existing stack.

## Why each rejected
- **A (upgrade to 5.10):** 5.10 is `.dev0`, unreleased; installing into the
  protected venv is the forbidden regression vector. Not needed anyway.
- **Sidecar (C):** unnecessary given the in-place load works; keep ONLY as the
  fallback if the GPU smoke shows garbage.
- **D GGUF / E vLLM:** no gemma4_unified support; more surface, no gain.

## The load recipe (text-only, NF4, transformers 5.5)
1. Detect the model: repo_id `google/gemma-4-12b-it` (or config `model_type ==
   "gemma4_unified"`).
2. Build the config from the checkpoint's `text_config` block (NOT the unified
   top-level): `Gemma4TextConfig(**text_config)` (drop the `gemma4_unified_text`
   `model_type` label). Keep `tie_word_embeddings=True`.
3. Load weights text-only: take `model.language_model.*` tensors, remap prefix
   `model.language_model. -> model.`, DROP the 11 vision/audio embedder tensors
   (`model.embed_vision.*`, `model.embed_audio.*`, `model.vision_embedder.*`).
   Load into `Gemma4ForCausalLM(config)` with NF4 (bitsandbytes -- proven on this
   stack). The single absent `lm_head.weight` is filled by tying.
   - Cleanest mechanism (no venv mutation): runtime registration in OTR's loader
     -- `AutoConfig.register` / `AutoModelForCausalLM.register` is NOT even needed
     since `Gemma4ForCausalLM` is already importable; just instantiate it directly
     and `load_state_dict(strict=False)` the remapped text tensors, or write a
     thin `from_pretrained` shim that points at an OVERLAY config (never mutate
     the cached `config.json`).
4. Cap context: `text_config.max_position_embeddings` is 131072 -> set an OTR
   input/`max_new_tokens` budget; do not default to 131k KV (would blow VRAM).
5. EOS: top-level `eos_token_id=[1,106]` vs text `eos_token_id=1` -- use the
   text model's stop ids for generation.

## OTR wiring (where it lands)
- `nodes/_otr_model_catalog.py`: the gemma-4-12b row stays. Change
  `loader_backend` to a new value (e.g. `transformers_gemma4_text_only`) OR keep
  `transformers_multimodal_text_only` and branch inside the loader on
  `model_type == "gemma4_unified"`.
- The loader (`nodes/_otr_model_loader.py` / the Selector `load_llm` path): add
  the gemma4-text branch (config-from-text_config + prefix-remap + NF4 +
  tie-head). This is the only real code change. PD6: no new model-pick widget
  (the writer already routes model_id); PD3: no INPUT_TYPES change if the row's
  surface is unchanged -- verify the workflow JSON still validates.
- Keep `vram_fit_tier`: re-confirm PASS after the smoke (NF4 ~8 GB < 14.5).

## The one operator gate (GPU; offline-unprovable)
Real load + 20-token greedy decode on a fixed prompt -> confirm SANE text (not
garbage / no multimodal-sentinel spam), confirm peak VRAM < 14.5 GB, confirm EOS
terminates. If sane: ship it as a writer option. If garbage: the RoPE
"proportional" / tied-head / NF4 numerics differ -> fall back to the sidecar (C)
with the serialized VRAM handoff (main evicts via OTR's existing Zero-Prime
eviction -> sidecar loads -> frees -> main reloads).

## Invariants
Stack untouched (no transformers/torch change) - offline-first (local snapshot,
local_files_only) - VRAM (NF4 + context cap < 14.5) - PD3 (workflow JSON
re-validate) - PD6 (no new model widget) - audio-king N/A.

## First step when building
Wire the loader branch behind a flag, then hand Jeffrey the one GPU smoke
(load + 20-token decode) as the go/no-go before promoting gemma-4-12b to a
selectable writer.
