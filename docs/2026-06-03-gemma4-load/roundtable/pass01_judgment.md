# gemma4-load roundtable -- pass01 judgment (Claude is the judge)

Panel: 6/6 usable (opus-4.8, sonnet-4.6, gpt-5.5, gemini-3.1-pro, grok-4.3,
deepseek-v4-pro) at max_tokens 8000. Spend ~$0.75.

## Unanimous panel position
- Path A (in-place upgrade) is effectively DEAD: `config.json` says
  `transformers_version "5.10.0.dev0"` -- a dev/unreleased build, no stable pin,
  installing it into the protected venv is the forbidden regression vector.
- Path B (rename/alias to 5.5 `gemma4`) is THE pivot -- could avoid a sidecar --
  but must be settled by an OFFLINE weight-key + config-signature diff, never
  guessed. Sonnet's sharpest risk: if 5.5's config swallowed the new fields as
  `**kwargs`, the model would load but run WRONG attention/RoPE = garbage text,
  worse than a clean crash.
- Path C (sidecar) is the robust answer; its real problem is VRAM coexistence on
  ONE 16 GB GPU -> needs a strict serialized handoff (main unloads -> sidecar
  loads -> frees -> main reloads), spawn-on-demand, minimal transport.
- Paths D (GGUF/llama.cpp) and E (vLLM) cut: no gemma4_unified support for a
  5.10-dev arch. Sequencing: ship the mistral-nemo fallback first.

## Grounded findings (the judge's verification -- this is what decides it)
I ran the panel's demanded offline checks against the real checkpoint + the
installed transformers 5.5 (`_otr_gemma_bcheck.py`, `_otr_gemma_keymatch.py`):

1. **5.5's `Gemma4TextConfig` ACCEPTS every "new" text field** (global_head_dim,
   num_global_key_value_heads, rope_parameters, layer_types,
   use_bidirectional_attention, attention_k_eq_v, vocab_size_per_layer_input,
   hidden_size_per_layer_input, enable_moe_block, num_kv_shared_layers) --
   **0 swallowed as kwargs**. => Sonnet's silent-wrong-arch risk is RULED OUT;
   transformers 5.5 already models this text architecture.
2. **Weight keys map cleanly.** Checkpoint = 677 tensors, all under `model.`:
   666 text (`model.language_model.*`) + 11 thin vision/audio embedders
   (`model.embed_vision/embed_audio/vision_embedder.*`). Mapping
   `model.language_model.* -> model.*` against a real 5.5 `Gemma4ForCausalLM`
   (built on meta from the checkpoint's text_config): **0 UNEXPECTED keys, 0
   shape issues**, and exactly **1 MISSING: `lm_head.weight`** -- which is the
   TIED head (`tie_word_embeddings: true`), auto-resolved by tying to
   `embed_tokens.weight`. Not a real gap.
3. **NF4 works on this exact stack** -- mistral-nemo loaded 4-bit NF4 at 7.74 GiB
   in the 2026-06-03 soak. The panel flagged "does bnb NF4 work on sm_120/Win?"
   as an unknown; it is PROVEN yes. The ~8 GB NF4 footprint fits 16 GB.

## Verdict: the SIDECAR IS NOT NEEDED.
The text tower is load-compatible with the installed transformers 5.5. A
text-only load (build a Gemma4TextConfig from the checkpoint's text_config, load
the prefix-stripped `model.language_model.*` weights into 5.5
`Gemma4ForCausalLM`, tie the head, drop vision/audio embedders, NF4) runs on the
existing stack with ZERO upgrade and ZERO sidecar -- the protected venv is
byte-untouched. C (sidecar) demotes to a FALLBACK only if the one remaining gate
fails.

## The one remaining gate (operator/GPU; not provable offline)
A real load + a short greedy-decode smoke to confirm SANE text (not garbage):
key-compatibility is proven, but RoPE "proportional" runtime behavior + tied-head
+ NF4 numerics should be eyeballed once on a 20-token generation. Cap context
(text config max_position_embeddings is 131072 -> set an OTR cap; do not default
to 131k KV).

## Convergence
The panel converged on "verify B before trusting"; the judge ran the verification
and it PASSED at the key/config level. The decision is resolved (pending the GPU
sanity smoke). No second paid pass warranted. CONVERGED.
