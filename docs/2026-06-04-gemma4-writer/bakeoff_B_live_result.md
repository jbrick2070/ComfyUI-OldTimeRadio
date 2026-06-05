# Live bake-off result -- B/A/telemetry vs gemma full-writer viability

**Date:** 2026-06-04 | **Lane:** gemma-4-12b-it Q4_K_M via Ollama `/v1`
(`OPENROUTER_BASE_URL=localhost:11434/v1`, A default-off). ComfyUI headless,
writer pruned to the node-7 closure (audio/video never load).

## PROVEN LIVE: B + telemetry

Two gemma runs both cleared the style picker that **hard-aborted last session**
(63-vs-5). Console:

```
[OTR_StylePicker] inventor attempt 1/3 OK: ['stratosphere_monitoring_alert',
  'haze_emergency_dispatch','toxic_data_extraction','smog_survival_log',
  'planetary_breath_analysis'] (valid=5 distinct=5 truncated=0)
[OTR_StylePicker] chooser picked 'smog_survival_log'
```

`meta.style_pick` stamped live: `chosen`, 5 `candidates`, `valid_count=5`,
`distinct_count=5`, `truncated_count=0`, `pass1_attempts` (1 and 2 across runs),
`model_id=openrouter:slot-a`, **`model_slug` resolved to
`hf.co/unsloth/gemma-4-12b-it-GGUF:Q4_K_M`** (telemetry slug-resolution works
end-to-end). B's parser net + the telemetry are confirmed on real gemma output.

## gemma is NOT a viable full writer yet -- downstream token-ceiling truncation

The run then **halts at `build_news_briefs`** (news_interpreter structured JSON):

```
[OpenRouter] ...gemma... hit finish_reason=length -- output truncated at the token ceiling
[OTR_StructuredCall] 'build_news_briefs' attempt 1 failed: no decodable top-level JSON ... char 0
  ... attempt 2 (structural retry @0.35) failed ... attempt 3 (typed repair @0.10) failed
[OTR_LedgerScriptWriter] news_interpreter FAILED ... news_briefs_required=True -- HALTING
```

**Root cause:** EVERY gemma call hits `finish_reason=length`. gemma is verbose
(near-certainly spending its token budget on a `<think>` preamble before the
JSON; the `<think>` strip then leaves an incomplete/empty body -> `char 0`).
Short style descriptors fit under the cap; the larger 4-brief JSON does not.

This is **not** a regression and **not** a B/A defect -- B/A did their job
(gemma got further than ever). It is gemma's own verbosity vs the remote output
cap, exactly the class the exact-count audit flagged as repair-ladder-guarded
(the ladder fired and still couldn't recover an output that never completes).

## Verdict (task 8) + next step (task 5)

- **Mistral stays the shipping default.** Per the plan: gemma does not clearly
  beat mistral -- it cannot finish an episode. Stop here on adoption.
- **To make gemma viable, the next experiment is task 5 + token budget:**
  1. `enable_thinking:false` so gemma goes straight to JSON (stops burning the
     budget on reasoning). Needs a server that honors it (llama.cpp llama-server
     `--chat-template-kwargs`, or Ollama's `think` param if its `/v1` exposes it).
  2. Raise the remote output cap (`OPENROUTER_MIN_OUTPUT_TOKENS` / slot
     max-tokens) so a completed brief-JSON fits.
  3. Re-run; if gemma reaches STORY_READY, then grade narrative vs mistral.
- A's GBNF live proof still pending a grammar-capable server (llama-server not
  installed; LM Studio present but grammar-field support unverified).
