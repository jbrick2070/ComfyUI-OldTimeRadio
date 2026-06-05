# OTR Writer-Model Bake-Off -- gemma-4-12b vs mistral-nemo

**Date:** 2026-06-04
**Branch:** v2.0-alpha
**Question:** Can `gemma-4-12b` (GGUF, via Ollama through OTR's OpenRouter lane)
beat the incumbent default writer `mistralai/Mistral-Nemo-Instruct-2407` on story
quality? If so, promote it to the out-of-box `DEFAULT_LLM`.

**Verdict (TL;DR): No. mistral-nemo stays the default.** gemma-4-12b cannot
complete an episode through OTR's writer -- it aborts at the style-picker's
exact-count structured pass -- while mistral-nemo produces a full story on the
identical harness. `DEFAULT_LLM` is unchanged (`mistralai/Mistral-Nemo-Instruct-2407`,
`nodes/_otr_model_catalog.py:32`).

## Method

- **Rubric:** 7 axes x 0-5 = 35 (news-grounding, dramatic arc, voice
  distinctness, dialogue quality, coherence, payoff, few-AI-tells), graded from
  the pre-audio writer/freeze ledger -- same rubric as
  `docs/2026-05-31-otr-story-quality-comparison.md`.
- **Harness (identical for both runs):** canonical
  `workflows/otr_scifi_16gb_full.json`, `target_words=320`, fixed
  `OTR_CAST_SEED=11` / `OTR_STYLE_SEED=11`, real RSS (`news_interpreter` ON, no
  baked premise). Pruned to the writer -> freeze -> cast-lock path and cancelled
  the instant the story finalized, so the audio sub-DAG never loaded (driver:
  `scripts/_otr_bakeoff_run.py`).
- **gemma-4 transport:** Ollama (`hf.co/unsloth/gemma-4-12b-it-GGUF:Q4_K_M`) on
  `:11434`, reached via OTR's OpenRouter lane pointed at Ollama's OpenAI endpoint
  (`OPENROUTER_BASE_URL=http://localhost:11434/v1`,
  `OTR_OPENROUTER_SLOT_A_DEFAULT=<gemma slug>`, both writer slots ->
  `openrouter:slot-a`).
- **mistral transport:** local transformers, 4-bit NF4,
  `HF_HOME=C:\ComfyUI-Models\huggingface`.

## De-risk: `<think>` / harmony-channel strip (shipped this session)

gemma-4 is a thinking-mode model; its `<think>...</think>` (and OpenAI-harmony
`<|channel|>...`) scaffolding lands inline in the response and would break the
writer's structured (JSON/GBNF) passes. Added a strip at the single response
chokepoint (`_extract_text` in `nodes/_otr_openrouter_backend.py`):

- Removes balanced `<think>...</think>`, a dangling `</think>` (Ollama pre-fills
  the open tag in its chat template, so the completion carries only the close),
  and harmony channel markers. Strict no-op when absent -- mistral/claude output
  is byte-for-byte unaffected.
- **10 new unit tests** in `tests/test_openrouter_backend.py`. Full suite
  **3705 passed / 12 skipped / 0 failed**; Bug Bible 0-fail.
- **It worked:** in the gemma run the early structured passes (`NewsCuration`,
  `NewsCurationDeep`) parsed cleanly -- the abort below is *not* a tag-leak
  failure.

## Result -- gemma-4-12b: ABORT (not viable as-is)

The writer aborted at the **style-picker "inventor" pass**
(`nodes/_otr_style_picker.py`):

```
inventor failed after 3 attempts; errors:
  ['parse failed: inventor returned 63 parseable lines, need exactly 5: ...']
```

The inventor asks the LLM for **exactly 5** distinct snake_case style
descriptors. gemma-4-12b returned **63** valid-format lines -- it ignored the
count constraint on all 3 attempts, so the writer fail-closed (no broken episode
shipped).

- This is **instruction non-compliance on an exact-count structured pass**, not
  a transport or tag-strip problem (the earlier structured passes succeeded).
- Run status: FAILED before any story lines were produced -> **not scorable** on
  the 7-axis rubric.

## Result -- mistral-nemo: complete story

Produced **"Serum's Gambit"** (style `clinical_trial_triumph`; news: ScienceDaily
"undruggable pancreatic-cancer target cracked, nearly doubles survival"), 18
ledger lines, on the identical harness.

| Axis | Score | Note |
|------|-------|------|
| News-grounding | 2.5 | uses the medical/serum frame but drifts to a black-market-serum subplot rather than the actual breakthrough |
| Dramatic arc | 3.0 | setup -> complication -> cliffhanger; OTR critic called it "uneven" |
| Voice distinctness | 3.5 | Erin (reckless idealist) vs Creed (cautious/institutional) read distinctly; clean attribution |
| Dialogue quality | 3.0 | some strong lines; 3 flat/expository lines flagged by the critic (b002/b005/b010) |
| Coherence | 3.5 | plot holds; a few vague threads (Spender, "the arrangement") |
| Payoff | 2.5 | ends on a hook ("...something you should know about those results"); no reversal landed |
| Few AI-tells | 3.0 | a few cliches ("the numbers don't lie", "burn... like wildfire") |
| **Total** | **~21 / 35 (60%)** | one mid draw; `freeze_verdict=needs_full_rerun`, corroborated by the OTR critic |

Context: this is a single run. The documented **best-local mistral baseline is
27/35 (77%)** ("Doorway to Anomaly",
`docs/2026-05-31-otr-story-quality-comparison.md`), with run-to-run variance
18.5-27/35. Even a below-average mistral draw **completes** -- which is the
decisive contrast against gemma-4's abort.

## Decision

- **Keep `DEFAULT_LLM = "mistralai/Mistral-Nemo-Instruct-2407"`** -- already the
  catalog default (`nodes/_otr_model_catalog.py:32`). No code change required.
- **gemma-4-12b is not viable as the writer as-is.** The blocker is exact-count
  compliance on the style-inventor (and, by extension, any other exact-count or
  strict structured pass).

## Path forward (if gemma-4 is revisited)

1. **Constrain the inventor output.** Ollama supports grammar/`format`
   constrained decoding; emit the 5 descriptors under a GBNF / JSON-array schema
   via the OpenRouter lane (which already passes `response_format` on technical
   passes) so the count is forced, not trusted.
2. **Or add post-parse leniency** to `_parse_inventor_output`: when more than N
   grammar-valid lines return, take the first N distinct. Small, model-agnostic
   robustness win -- but a writer change with its own review, out of scope here.
3. **Or prompt-harden** the inventor for thinking models (least reliable).
4. **Round 2 (optional, per handoff):** Qwen3.6-35B-A3B (Apache-2.0 MoE) once a
   GGUF + llama.cpp / LM Studio endpoint is staged. Never put 2-3B-class models
   in the writer lineup (BUG-LOCAL-308).

## Caveats

- gemma aborted before producing a story, so a head-to-head *story* comparison
  was not possible -- the result is binary (completes vs aborts), which is itself
  the answer.
- Real-RSS news differed slightly between the two back-to-back runs; gemma never
  reached the dialogue stage, so this did not affect the verdict.
- The mistral score is a single same-day draw; the cited 27/35 baseline stands as
  the best-local reference.

## Reproduce

- Driver: `scripts/_otr_bakeoff_run.py <model> <tag> 320` (machine-local;
  `scripts/_*.py` is gitignored).
- gemma: `... openrouter:slot-a gemma4 320` (ComfyUI launched with the Ollama
  env). mistral: `... mistralai/Mistral-Nemo-Instruct-2407 mistral 320` (ComfyUI
  launched with `HF_HOME=C:\ComfyUI-Models\huggingface`).
- Captured ledgers: `_otr_bakeoff_gemma4.json` (FAIL),
  `_otr_bakeoff_mistral.json` (STORY_READY).
