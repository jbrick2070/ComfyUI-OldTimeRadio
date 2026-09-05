Pull note: the Shell tool refused every command this session, so I could not run the section-0B fetch/pull. Everything below is read off the working tree as-is.

## Verdict: diagnosis confirmed, but the framing "confirm the think_policy" points at the wrong lane

The GO_FORWARD/HANDOFF suspicion (thinking model cycling under the structured multipass) is correct. What is wrong is where to look: `think_policy` exists **only on GGUFRow** (`_otr_gguf_backend.py:191`, `376`), and it is already set right there (`qwen3_no_think`, `/no_think` applied on every call at `1502`). Leg C did not run that row.

What leg C actually ran:

- `Qwen/Qwen3-8B` is not a curated row anywhere in `nodes/` — the only Qwen3-8B row in code is `unsloth/Qwen3-8B-GGUF`. The dropdown label comes from `_otr_model_catalog.build_dropdown_choices` line 786: an **uncurated HF-cache hit** admitted because its `config.json` says `*ForCausalLM`. It loads on the plain **transformers** lane (weights at `C:\ComfyUI-Models\huggingface\hub\models--Qwen--Qwen3-8B\...`).
- That snapshot's `tokenizer_config.json` carries `enable_thinking` in the chat template (verified: 1 hit). Qwen3's template defaults thinking **on** unless `apply_chat_template(..., enable_thinking=False)` is passed.
- Every transformers-lane render site passes only `tokenize=False, add_generation_prompt=True` and nothing else: `OTR_LedgerScriptWriter.py:869`, `_otr_model_loader.py:1899` and `2108`, `_otr_constrained_generate.py:256`. `enable_thinking` appears in exactly one `.py` in the repo — the GGUF backend. The transformers lane has no think-policy concept at all, curated or not.

Why that produces a *verbatim cycle* rather than a `<think>` preamble: the dossier pass is schema-bound (LMFE prefix fn is built whenever `schema_model` is set, `OTR_LedgerScriptWriter.py:837-846`). The grammar's first legal token is `{`; a thinking-mode Qwen3 has nearly all its mass on `<think>`. Masked off its manifold, it samples a degenerate tail — the exact mechanism the GGUF backend documents at `1494-1496` ("the JSON grammar blocks the `<think>` the model still opens"). The retry ladder then *lowers* temperature 0.3 → 0.15 → 0.1, which is the regime Qwen's own card warns causes endless repetition in thinking mode. Three attempts, three cycles, guard fires (`MIN_CYCLE_TOKENS=48` × 3 repeats — not a false positive: gemma and Mistral ran the identical pass clean in legs A/B).

Leg 0 passing is consistent, not contradictory: a free 40-token probe just emits `<think>...` and looks alive.

So: it is not a bad model, a bad guard, or a bad `think_policy`. It is a **lane gap** — Qwen3 reasoning suppression was root-fixed on 2026-07-16 for GGUF only, and the transformers lane was never given the equivalent.

## Fix shape

One idiom already exists for exactly this class: the BUG-LOCAL-262 capability probe (`tokenizer_supports_system_role`, cached on `cache_entry["_system_role_supported"]`). Mirror it.

1. **Probe, don't catalog.** In `_otr_loader_backends.py`, add `tokenizer_supports_thinking_switch(tokenizer)`: render a 2-message probe with and without `enable_thinking=False`; supported iff neither raises *and* the renders differ. Cache on `cache_entry["_thinking_switch_supported"]`. This is a tokenizer property, so it covers the uncurated row (which has no catalog entry to hang a policy on) and any future hybrid-thinking model without a registry edit.
2. **One render helper, four call sites.** Add `render_chat_prompt(cache_entry, tokenizer, messages)` that does system-role normalization + `enable_thinking=False` when supported, and route `OTR_LedgerScriptWriter.py:869`, `_otr_model_loader.py:1899`/`2108`, and `_otr_constrained_generate.py:256` through it. Four hand-copied kwargs is how the guard itself was missed on two routes in August; one helper is the shape that can't drift.
3. **Strip the empty envelope on free-form output.** `enable_thinking=False` puts `<think>\n\n</think>\n\n` in the *prompt*, so normally nothing leaks — but lift `_strip_leading_think_envelope` out of `_otr_gguf_backend.py` into a leaf module both lanes import, and apply it on unconstrained transformers output with the same fail-loud rule (wrapper-only reply raises, never returns empty).
4. **Leave alone:** the decode guard thresholds, the retry ladder's temperature descent, and the GGUF row. None of them is the defect.
5. **Prove it the only way the charter accepts:** rerun leg C on `scifi_news_pro` with `Qwen/Qwen3-8B` in the creative slot; pass = episode in `otr/obs/`. Then leg D (GGUF sibling) still needs its 16 GB `Q4_K_M` profile — that blocker is unchanged and separate.

Kibitz call: this is a new capability on the shared writer path (both boxes call these render sites), so under the 08-17 test it is a **YES** — arc before code, plus the 0B before/after receipt showing gemma/Mistral prompts render byte-identical (their templates have no `enable_thinking`, so the probe returns unsupported and the kwarg is never passed).
