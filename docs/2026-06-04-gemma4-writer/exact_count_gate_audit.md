# Exact-count / shape gate audit (gemma overgeneration blast radius)

**Date:** 2026-06-04 | **Context:** B (parser net) + A (GBNF) close the style-picker
exactly-5 gate. This audit answers: would gemma's overgeneration break any *other*
strict gate after the style-picker (the "pass style-picker, break the next gate" risk)?

**Method:** grepped `nodes/*.py` for pydantic count bounds (`min_length`/`max_length`/
`min_items`/`max_items`), `len(...) == N` checks, count-raise language
(`exactly`/`must have N`/`expected N`), and the cast-contract raise sites.

## Finding: the style-picker was the unique vulnerability

The style-picker `_parse_inventor_output` was a **line-based parser with a hard
exactly-5 count raise and retries that each hard-failed** — no repair ladder. That
shape is what gemma's 63-vs-5 overgeneration broke. B (skip-and-take-first-5) + A
(GBNF decode cap) close it. No other gate shares that shape.

## Every other count/shape gate is already tolerant

| Gate (file) | Kind | Why overgeneration is safe |
|---|---|---|
| `StylePick.candidates` min=max=5 (`_otr_style_picker`) | exact | **Fixed by B/A** (parser truncates; contract intact) |
| `StylePick.seed_sample` / `article_hash` (`_otr_style_picker`) | exact | Python-generated (rng.sample / SHA256) — not LLM output |
| Cast count 1-6 (`_otr_outline`, `_otr_stage1_plan`, `_otr_casting`) | range | `structured_call` JSON-schema validate + bounded repair (smoke test asserts "10 names rejected") |
| Beats 4-32, key_terms 1-N, outline str fields (`_otr_outline`, `news_interpreter`) | range/cap | `structured_call` repair ladder re-prompts on violation |
| Stage1 / slot-drama / story-brief fields | range/cap | `structured_call` repair ladder |
| `max_attempts must be >= 1` (news/casting/outline/line_composer) | precond | Python parameter, not LLM output |
| Cast-contract raises (`_otr_cast_contract`) | structural | TypeError/FileNotFound on dir/dict shape — not counts |
| "exactly one mp4 per episode" (`_otr_paths`, upscale) | output | filesystem contract, not LLM output |
| Chooser pick (`_otr_style_picker` `_run_chooser`) | exact-match | BUG-295 fallback to first candidate — never hard-aborts |
| Title parse (`OTR_LedgerScriptWriter`) | line parse | falls back to "" on no parseable TITLE — never hard-aborts |

## Conclusion

**No other gate needs B-style treatment.** The hard line-parse-then-raise-on-count
pattern existed only in the style-picker. Everywhere else, an overgenerating model is
either repaired (the `structured_call` JSON + bounded-repair ladder), fed Python-
generated data, or hits a non-LLM contract. Gemma's overgeneration is recovered, not
hard-aborted, at every downstream gate.

**Live confirmation (gate F):** the 6-pt conformance harness still runs gemma end-to-
end through the writer to confirm no *runtime* surprise beyond this static read.
