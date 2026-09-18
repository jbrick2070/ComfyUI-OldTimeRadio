# Judgment -- Shakespeare and Public Domain on a non-English row

Driver and sole judge: Claude (Fable 5.1). Contrarian: Grok 4.6 via
cursor-agent (`cursor-grok-4.6-high-fast`), briefed to refute the anchor.
One round. Every claim below was grounded against the real files before it
was folded in; the anchor is left as written (the pre-panel record).

## What the contrarian got right (grounded, folded in)

1. **`passage_text` re-render is a no-op.** `project_payload` already ran
   inside `_resolve_inputs` (`_otr_writer_inputs.py:590-614`); no production
   reader of `plan.passage_text` exists after that. The translated text must
   instead reach `generate_outline(verbatim_texts=...)` at
   `OTR_LedgerScriptWriter.py:4656`, because Stage 3 pastes those texts as
   "SPOKEN WORDS (fixed, verbatim from the source)" (`_otr_outline.py:1219-
   1235`) to derive intent and mood. English texts there against a native
   performance would mis-describe every beat to Ghost and the voice engine.
   `news_article` / the interpreter stay English -- the announcer chrome is
   already row-owned.
2. **The spoken English that actually reaches the microphone is the coda
   fact.** `compose_news_coda` (`_otr_line_composer.py:1784-1842`) authors the
   bridge natively and appends `provenance_coda_line` unchanged under
   `PROTECTED_FACT_COMPONENT_FLAG`; the clean stage skips it
   (`_otr_ledger_clean.py:1904-1917`). Its source is `spoken_coda_line`
   (`_otr_provenance.py:129-179`) with three English template sets
   (`_CODA_BY_STATUS`, `_NAMED_CODA_BY_STATUS`, `_LICENSED_NAMED_CODA`).
   The anchor's "seven rows x three templates" named the wrong strings.
3. **D must follow the `spoken_chrome` precedent, not add unused keys.**
   `work_frame_sentence` (`_otr_line_composer.py:1711-1720`) already reads
   the registry's `spoken` block per row (`config/episode_languages.json`,
   `spoken.*`). The coda templates and My Story's `attribution_sentence`
   (`_otr_story_input.py:471-481`, plus `ANONYMOUS_ATTRIBUTION`) join that
   block. `WORK_FRAME_SENTENCE` (public API) is untouched.
4. **Speakers are never returned by the model.** The executor (`:5606-5610`)
   and the outline compare speakers by exact string; a `MACBETH` ->
   `Macbeth` rename would fail loud. The call returns TEXTS in order; the
   plan's speakers are copied.
5. **Size for the 8 GB floor.** `otr_4060_floor.json:38` is `gguf_n_ctx:
   2048` (offload profile 4096). Scenes carry `recommended_word_budget: 300`.
   Batch entries in order under a source-word budget (about 120 words per
   call), with `max_new_tokens` sized to the batch -- never the outline's
   150. Do not call `context_cap_for` at `:3769` (cold slot).
6. **Guard on the empty instruction, not on the stamp.** English STAMPS
   `en` (`:3698`); only `native_authoring_instruction` is empty for it.
7. **The admission text is part of the change.** `check_source_bank_admission`
   (`_otr_episode_languages.py:503-524`) and the writer comment at
   `:3329-3334` still say "a verbatim lane cannot also be a translation
   lane"; lifting the JSON exclusions alone leaves the old law in the only
   gate. Rewrite both; the error text becomes the readiness path for a bank
   that has no translation machinery, not a fidelity refusal.
8. **`non_verbatim_credit_line` keys on `startswith("adapted from")`**
   (`_otr_verbatim_lane.py:85-86`). Any registry template for the printed
   credit must keep that coupling or the helper must stop string-matching.
9. **Zhu Shenghao died in 1944**, so the inventory's "before 1944" sentence
   was wrong; life+80 still clears (2024). Fixed in the inventory.

## What the contrarian got wrong or that does not bite

- The cast-coverage graft (`_otr_cast_coverage_repair.py:40-47, 101-110`)
  composes a NEW row from an English `first_speech` source block -- but that
  composer call has carried `language_instruction` since `bf484094`, so the
  row is authored natively, and it is deliberately NOT verbatim-flagged
  because it is composed. Not a translation defect.
- No verbatim TEXT check exists after the executor (contrarian's own answer
  1): flags only. A collides with nothing there.

## The settled design (CODE row)

**Mechanism (Shakespeare):** `_otr_verbatim_translation.translate_entries(
entries, *, language_instruction, creative_fn, max_source_words=120)`:
batches consecutive entries under the word budget; each call is one
`structured_call` (schema: list of `text`, same length as the batch) with the
row's `writer_instruction` leading the system message and the batch as
`speaker: text` rows in the user message; speakers are copied from the plan;
structural validation only (count, non-empty); two attempts per batch, then
`RuntimeError` -- loud, before any audio spend. `max_new_tokens` is
explicit per batch. Called in the writer right after the plan is read
(`:3769`) and only when `_EPLANG.native_authoring_instruction(meta)` is
non-empty; the result replaces `_verbatim_plan.entries` so both the executor
(`:5612`) and `verbatim_texts` (`:4656`) see the translation. Receipt:
`meta["verbatim_passage"]["translation"] = {iso, model_id, batches,
attempts, source_sha256, text_sha256}`; `raw_sha256` (English file) stays.
`_otr_verbatim_lane` remains model-free.

**Spoken and printed Python English (D), all lanes:** extend each row's
`spoken` block with `coda_by_status`, `coda_named_by_status`,
`coda_licensed_named`, `attribution_named`, `attribution_anonymous`; make
`spoken_coda_line` and `attribution_sentence` take `episode_meta` and read
`spoken_chrome`. English rows carry today's exact strings (byte-identical).
Printed credit lines (`printed_credit_line`, `credits_source_line`,
`ANONYMOUS_CREDIT`, `writer_tail` "Story generation models used:") get
`credits` block templates in the same change, keeping the `adapted from`
prefix coupling intact. `noncommercial_notice` is an operator warning and
stays English.

**Public Domain:** no new mechanism -- the composer seam is native; with D
in place, lift the exclusion.

**Config:** `source_bank_exclusions` -> `[]` on all seven rows. Admission
docstring, error text and writer comment rewritten.

**Tests:** batching under the budget; structural failure retries once then
raises; English row makes no call and the plan object is unchanged; executor
writes translated text with `VERBATIM_SOURCE_FLAG`; `verbatim_texts` at the
outline call site are the translated ones (source inspection); receipt
carries both hashes; every registry row validates the new spoken/credits
keys; English `spoken_coda_line` / `attribution_sentence` outputs are
byte-identical to today's; admission passes for both banks on every row and
the old refusal test flips; `apple/MULTILINGUAL.md` sentence on the two
banks updated.

**Vendored public-domain translations (C)** stay a separate data row: when
a vendored (play, scene, iso) exists the plan step selects from it and the
model call is skipped; French (Hugo) and Italian (Rusconi) first. Needs the
operator's scope word and the Gemini deep-research return.

## Roster

Round 1: Grok 4.6 (cursor-agent) REFUTE on the anchor -- ten claims, seven
grounded and folded, two non-issues, one factual fix. No second round: each
claim collapsed into a concrete change. Driver grounded every citation
against the files at `d380c999`.
