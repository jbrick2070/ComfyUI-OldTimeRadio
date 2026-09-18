# Driver anchor -- Shakespeare and Public Domain on a non-English row

Driver: Claude (Fable 5.1), sole judge. Contrarian: Grok via cursor-agent.
Written BEFORE the panel, against the real files at `d380c999`.

## The ruling this serves (2026-09-18, standing rulings)

Every lane is eligible for the episode language. For the two fidelity banks
the author's own passages are English and no model authors them, so
"eligible" means TRANSLATION of the verbatim passage. Operator: *"if it is an
English source, translation is necessary ... I said I didn't want translate
but we have to"*, and *"somewhere there is a public domain Shakespeare
translation for each, but how to find it easily."*

## What the code actually does today (grounded)

- `config/episode_languages.json`: all seven non-English rows carry
  `source_bank_exclusions: ["shakespeare", "public_domain"]`. The gate is
  `_otr_episode_languages.check_source_bank_admission`, called once at
  `OTR_LedgerScriptWriter.py:3341`, before any model spend.
- `nodes/story_packs/banks.json:152`: `shakespeare` is the ONLY bank with
  `verbatim_passage: true`. `public_domain` is a prose adaptation lane
  (`style_pool_class: adaptation`, no verbatim gate).
- Shakespeare path: `_otr_verbatim_lane.plan_verbatim_passage` (no model)
  cuts one seeded window of consecutive speeches into `entries`
  (`speaker`, `text`, `speech_index`, `chunk_ordinal`, `chunk_count`) and a
  body-free receipt with `raw_sha256`. The writer reads the plan at
  `OTR_LedgerScriptWriter.py:3769` (AFTER the language stamp at `:3698`), seats
  the passage speakers as cast (`:4095`), builds the outline from the plan
  order, and THE EXECUTOR at `:5591-5614` writes `_entry.text` straight into
  each character beat with `VERBATIM_SOURCE_FLAG`. That flag protects the row
  from every later rewriter (`_otr_ledger_clean.PROTECTED_ROW_FLAGS`, and
  `row_is_verbatim` readers in freeze, cleanup, voice, coverage).
  `project_payload` hands `passage_text` to the interpreter and the outline
  seed text (`Excerpt (a verbatim passage, performed as written)`).
- Public Domain path: `_otr_public_domain_sources` (interpreter v3) turns a
  Gutenberg / Standard Ebooks / fixture text into briefs; lines are then
  COMPOSED by the per-line composer with source grounding. That seam already
  receives the native instruction as of `bf484094`. Provenance is stamped by
  `_otr_provenance` and the announcer speaks a Python-owned English credit
  line (`_otr_provenance.py:100,113`: "Tonight's tale was adapted from ...").
- Loader shape: `_otr_shakespeare_sources._speaker_from_line` (~352) accepts
  `NAME:` (ASCII colon) or a Folger ALL-CAPS line only.

## The three answers

### A. Translate the planned passage, keep the executor (DRIVER'S PICK)

One new step, one owner: after the plan is read (`:3769`) and the language is
stamped, on a non-English row call the creative slot ONCE with the whole
passage as `speaker: text` rows and a row-owned instruction, returning the
same number of rows with the same speakers in the same order, text in the
episode language. Build a new plan whose `entries` carry the translated text
and whose `passage_text` is re-rendered from them. The executor at `:5612`
is untouched; the rows keep `VERBATIM_SOURCE_FLAG` so no rewriter paraphrases
the translation either.

- Validation is STRUCTURAL only: row count, speaker sequence and non-empty
  text. No prose judgement (story quality is closed). Two attempts, then
  fail LOUD: an English verbatim row on a native episode is exactly the
  defect, so a silent fallback is not an option; a raise before any audio
  spend is the honest outcome.
- Receipt: `meta["verbatim_passage"]["translation"] = {iso, model_id,
  source_sha256 (English cut), text_sha256 (translated), attempts}`. The
  English `raw_sha256` stays -- provenance of the source is unchanged.
- Credits: `printed_credit_line` / `_otr_provenance` English templates get a
  row-owned counterpart (see D below).
- English and Off: the step is skipped on `iso == en`; byte-identical.
- Coverage: all seven languages, all fourteen scenes, today.
- Cost: one creative call per episode, passage-sized (a scene window, not a
  play).

### B. Perform at the line seam, relax the verbatim gate for non-English

Route non-English Shakespeare through the composer with `source_block` and
drop `VERBATIM_SOURCE_FLAG`. REJECTED: it un-verbatims the lane -- the
executor exists precisely because the composer paraphrased ("this blade high
between us" in a scene with no blade, 2026-09-11) -- and it re-exposes the
rows to every rewriter. Translation-per-line also loses the passage's
cross-line coherence.

### C. Vendor a public-domain translation as the source text

The lane stays verbatim in the truest sense: it performs a real translator's
words. Inventory (`pd_translation_inventory.md`, this folder): French (Hugo,
d.1873) and Italian (Rusconi, d.1889) cover all ten plays on Wikisource with
validated text; Spanish covers five (Menendez y Pelayo, Marquez, Moratin);
Portuguese has Hamlet clean and Midsummer as OCR; Mandarin three plays (Zhu
Shenghao, traditional characters, two label styles); Japanese one play
(Tsubouchi, old kana -- a G2P risk for `misaki[ja]`); Hindi one poor scan.
Every named translator died before 1944, so life+80 (Spain) is clear.

- NOT the primary mechanism: four of seven languages cannot be covered, and
  the loader needs a per-language label normaliser (`NAME.` / `Nome.` same
  line / full-width `：` / full-width space / `--`) or a one-time
  normalisation to `NAME:` at vendoring time (cheaper).
- A GOOD SECOND LAYER under A: when a vendored translation exists for
  (play, scene, iso), the plan step selects from it and skips the model
  call; the receipt says which. French and Italian first. This is a data
  row (vendor + normalise + `raw_sha256` receipts), separate from A, and it
  needs the operator's word on scope (10 plays x 2 languages to start).
- Licence: the vendored English is Folger CC BY-NC; the translations are PD.
  The credit line must name the translator.

### D. The finding neither option fixes on its own: Python-owned spoken English

Both fidelity lanes, and My Story, SPEAK Python-authored English sentences
on a native episode: `_otr_provenance` credit lines ("Tonight's tale was
adapted from {work_title}, by {author}"), `printed_credit_line` /
`non_verbatim_credit_line` ("freely adapted from ..."), and My Story's
`attribution_sentence` fallback. Python has no authority to translate, so
these belong in the language REGISTRY as row-owned authored strings
(`config/episode_languages.json`, beside `writer_instruction` /
`title_instruction`), one template per row per sentence, English rows
unchanged. Seven rows x three templates is authored data, not a model pass.
This is in scope for the row because Public Domain cannot ship natively
without it.

## Public Domain specifically

Lift the exclusion (one config edit) once D is in place. The composer seam is
already native (`bf484094`). Verify: source grounding and `provenance_normalize`
are name/identity checks, not English-text checks -- grep confirms no gate
compares composed text against the English source verbatim.
`_otr_name_authority` is "correctly inert" on these banks (its own docstring).

## Where the model call lives (A)

`_otr_verbatim_lane` is documented "nothing here calls a model" -- keep it so.
Put `translate_verbatim_plan(plan, *, language_instruction, creative_fn,
technical_fn?)` in a new small module `_otr_verbatim_translation.py`
(structured_call, schema = list of {speaker, text}), called from the writer
right after `:3769` under `if _language_prompt_lead:`. The existing
`_language_prompt_lead` (`:4799`) is computed later; compute the instruction
once with `_EPLANG.native_authoring_instruction(meta)` at the call site.

## Tests A needs

- Structural: same count, same speakers, non-empty; a mismatched reply is
  retried once then raises `RuntimeError` (loud).
- English row: plan object identity unchanged, no call made.
- Executor test: translated entries reach the beats with
  `VERBATIM_SOURCE_FLAG`; receipt carries both hashes.
- Writer wiring: `inspect.getsource` asserts the call site under the
  language guard.
- D: every row in the registry validates the three templates; English rows
  produce the exact current strings.
- Admission: after the exclusion is lifted, `check_source_bank_admission`
  passes for the two banks on every row; the old refusal test flips.

## What I am NOT proposing

- No language detector, no back-translation check, no prose quality gate.
- No change to the passage selector, the seed, or the topology.
- No translation of speaker NAMES (identity keys; the cast roster stays
  the source's).
- No Off-row change; `Off` is not a registry language.

## Questions for the contrarian

1. Is there a gate that compares Shakespeare row text against the English
   source after the executor (a verbatim CHECK, not just the flag)? If so,
   A breaks it and I missed it.
2. Does `project_payload` / the interpreter / the outline seed need the
   translated `passage_text`, or should the announcer framing be built from
   the English and authored natively anyway (it already is)?
3. Is one whole-passage call safe under the 8 GB writer's context budget
   (`_creative_context_cap_fn`), or must A chunk by speech?
4. Anything in D I under-scoped: other Python-owned SPOKEN English strings on
   a native episode (grep announcer templates, credits, music-cue rows).
