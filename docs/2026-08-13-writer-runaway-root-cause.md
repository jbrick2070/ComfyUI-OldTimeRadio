# The writer decode runaway -- root cause and candidate fixes

**Status: IMPLEMENTED AND VERIFIED LOCALLY.** Started as a diagnosis by the
RENDER window 2026-08-13 while running the 45-word render gate, then became the
fix recorded in section 7. **The r4 panel caught this file claiming three
different lifecycle states at once** ("no code was changed" / "not yet adopted"
/ "SHIPPED"), which is exactly the kind of drift that costs the next window a
re-grounding pass -- so there is ONE state line and this is it. Section 5 is a
decision record (adopted / rejected and why), not an open menu.

The in-decode HALT is a separate, already-settled coder item
(`docs/2026-08-13-codex-consult-indecode-halt.md`) and is NOT part of this
change; it bounds the COST of a runaway, while this bounds the MECHANISM. This
document asks the question that halt does not answer -- **why does the model run
away at all**.

**SCOPE, stated plainly:** this fix covers the `scifi_codex` lane, which is the
execution lane of the `scifi_news` bank ALONE. Five other runnable banks
(`media_archive`, `shakespeare`, `public_domain`, `original` on the writer's
inline body; `scifi_news_pro` on the `_otr_scifi_fable2` markup lane) decode a
DIFFERENT shape and are not covered here. "No runaway on any bank" is NOT
established by this change and must not be reported as such.

## 1. The signature, observed live twice

The UCLA-marmot leg (2026-08-13 ~23:00, historical) and the `fastwan_8gb`
render-gate leg (2026-08-13 10:44, this window) produced the same shape:

```
marmot   P3 attempt 1  t=0.720  -> OUTPUT_TRUNCATED  14191 tok  1268 s
         P3 attempt 2  t=0.320  -> OUTPUT_TRUNCATED  14191 tok  1296 s
         P3 attempt 3  t=0.100  -> complete draft in 68 s
fastwan  P3 attempt 1  t=0.720  -> OUTPUT_TRUNCATED  13912 tok (2472-tok prompt)
         P3 attempt 2  t=0.320  -> COMPLETE parseable draft; failed only
                                   draft.cast_coverage (3/4, missing announcer)
         P3 attempt 3  t=0.100  -> typed repair; leg continued to P5
```

The runaway text is a verbatim paragraph cycle. Live specimen frozen at
`tmp/_runaway_specimen_20260813_fastwan_P3.log`; the cycling unit is roughly
120 tokens of visual-prompt boilerplate ("...the lab's environment is one of
controlled chaos, where order and disorder coexist in a delicate balance ...
a row of servers humming softly in the corner...").

**The ladder self-heals, but it no longer self-heals forever.** Neither leg was
terminal on its own. **CORRECTED 2026-08-13 by the r1 panel:** an earlier draft
of this document called the defect "cost only" and cited an unbounded
validation loop as an invariant to protect. Both halves were wrong.
`MAX_CANDIDATE_CYCLES: int = 3` (`nodes/_otr_scifi_codex.py:28-45`) is an
OPERATOR RULING made the same day, and it explicitly caps the unbounded reroll
that `016ad146` introduced -- the operator chose a hard bound for ALL cycles,
with the trade-off stated. So enough runaway decodes now KILL a leg rather than
merely delaying it, and the invariant to protect is the THREE-CYCLE CEILING,
not an unbounded loop.

## 2. Verified facts (file:line, checked on the real Windows files)

**F1. The generation kwargs.** `nodes/OTR_LedgerScriptWriter.py:950-970`
assembles exactly: `do_sample=True`, `temperature`, `top_p`,
`max_new_tokens`, `pad_token_id`, then conditionally `min_p` (only when > 0),
conditionally `repetition_penalty` (only when != 1.0), and
`prefix_allowed_tokens_fn` + `num_beams=1` when constrained. Nothing else.
`stopping_criteria` sits behind `if stop:` at `:978` and the structured passes
never pass `stop`.

**F2. `no_repeat_ngram_size` does not exist in this repo.** A grep for
`no_repeat_ngram|frequency_penalty|presence_penalty` across `nodes/` returns
ZERO matches. The single hard anti-loop mechanism transformers offers is not
merely unset -- it is absent from the codebase.

**F3. The armed levers.** `min_p` default 0.05
(`OTR_LedgerScriptWriter.py:2983`), `repetition_penalty` default 1.03 with the
widget clamped to a 1.2 MAXIMUM (`:3001`). `top_p` 0.95 for the "balanced"
creativity map. P3's base temperature is 0.72
(`nodes/_otr_scifi_codex.py:3264`).

**F4. The unbounded fields.** `RadioScoreDraftShotV4.description` and
`.visual_prompt`, and `RadioScoreDraftSceneV4.env` and `.description`, are bare
`Field(min_length=1)` with NO `max_length`
(`nodes/_otr_scifi_codex.py:427-434`). The sibling evidence models directly
above them all carry explicit caps -- `MAX_QUOTE_CHARS` at `:186`,
`MAX_CLAIM_CHARS` at `:240`, `MAX_ENTITY_NAME_CHARS` at `:251`. The asymmetry
is the seam: the arrays are bounded, the strings are not.

**F5. The budget arithmetic is a consequence, not a cause.** 2,472-token prompt
+ 13,912 output = 16,384 = the Mistral-Nemo context cap. OUTPUT_TRUNCATED is
where the budget ran out, not a limit the decode was pushing against.

## 3. The measured mechanism -- a CPU-only probe, no GPU, no model load

Running the real transformers logits processors in the real order
`generate()` assembles them (repetition penalty -> temperature -> top_k ->
top_p -> min_p), against a synthetic step emulating the locked state: the loop
continuation token holding the mass, the closing quote -- the only
schema-legal exit from an unbounded JSON string -- in the tail.

```
  raw p(quote)  |  LIVE stack   |  rep=1.20 (max)  |  n-gram ban(4)
       0.0300   |  0.000000000  |    0.000000000  |    1.000000000
       0.0100   |  0.000000000  |    0.000000000  |    1.000000000
       0.0050   |  0.000000000  |    0.000000000  |    1.000000000
       0.0010   |  0.000000000  |    0.000000000  |    1.000000000
       0.0001   |  0.000000000  |    0.000000000  |    1.000000000

repetition_penalty=1.03 applied to the loop token:
   seen  1x -> -0.001064
   seen 60x -> -0.001064      frequency-aware: False
```

Three results, each load-bearing:

1. **Escape probability is exactly zero under the live stack**, even when the
   model wants the quote 3% of the time. Once one token holds >= 95% of the
   mass, `top_p=0.95` truncates the nucleus to that single token and sampling
   is effectively greedy; `min_p=0.05` scales its threshold off the top token
   and cuts the rest. The levers meant to add variety are what seal the exit.
2. **The maximum the widget even allows (1.2) does not change that** -- still
   exactly zero. This is a mechanism problem, not a tuning problem.
3. **HF's `repetition_penalty` is not frequency-aware.** A token emitted 60
   times is penalised identically to one emitted once. That is the mechanical
   reason it cannot break a loop no matter how long the loop runs.

**Stated condition, sharpened after r1.** The zero holds while the loop is
LOCKED -- the continuation token sitting above the top_p nucleus threshold.
**This measures the LOCK-IN MECHANISM, not the trigger.** The specimen is
heartbeat text and carries no logits, quote rank, or allowed-token telemetry,
so verbatim repetition is evidence of a lock, not proof that any particular
token held >= 95%. What is proven: once a loop is locked, the configured stack
leaves no exit. What is inferred: that the observed repetition IS such a lock.
Proving the trigger would need a captured prefix replayed through the real
model with the processed distribution recorded -- worth doing, not needed to
justify bounding the string.

## 4. What the constraint machinery does and does not do

lmfe (`_otr_constrained_generate.py:116-140`, bound at
`_otr_scifi_codex.py:2020`) masks EOS until the JSON document is complete --
correct, and it removes the exit that ends most unconstrained degenerate loops.
The closing quote remains legal at every step, so this is NOT an lmfe defect:
the quote is reachable, it is simply never sampled. The interaction is what
matters -- the constraint removes one exit and the samplers price the other at
zero.

## 5. Candidate fixes -- to be pressure-tested, NOT yet adopted

**C1. `no_repeat_ngram_size` (probe says it restores escape to 1.0).**
THE OPEN RISK, and the reason this document exists rather than a patch: a
global n-gram ban over a JSON document forbids legitimately repeated
structural text. JSON keys repeat by design -- `"shot_index"`, `"fact_ids"`,
`"arc_phase"` appear in every element. If the ban makes a schema-required token
illegal while lmfe simultaneously masks everything else, the intersection could
be EMPTY, and an empty allowed set is a far worse failure than a slow decode.
The probe used a synthetic vocabulary and did NOT test that interaction.
Unresolved: does the ban apply to the prompt as well as the completion, and
what n makes structural repetition safe while still catching a ~120-token
prose cycle?

**C2. `max_length` on the four unbounded string fields (F4).** Gives lmfe a
bound to enforce, so the string must terminate. Consistent with what the
evidence models already do. Risk: a cap that is too tight becomes a quality
gate on prose, which THE LAW forbids -- it must be a structural ceiling far
above any natural value, never a length target.

**C3. Raise `repetition_penalty`. REFUTED as a standalone fix** by the probe:
the widget's own 1.2 maximum still yields zero escape.

**C4. Lower the temperature. REFUTED as a reliable fix.** The marmot leg ran
away at 0.320; this leg escaped at 0.320. Contributing, not curative.

**C5. The in-decode halt.** Already designed and settled elsewhere. It bounds
the COST and does not touch the cause; it remains wanted either way.

## 6. Invariants any fix must not break

* **THE LAW:** an audit may improve a story, never fail one for length,
  language, style or quality. No fix may become a writer veto.
* **No word-count chasing:** output capacity is never tied to `target_words`;
  the PBUG-20260729-02 ruling explicitly forbids capping the budget to the word
  target.
* **The THREE-CYCLE CEILING stays** (`MAX_CANDIDATE_CYCLES = 3`). Corrected
  from an earlier draft of this line, which said the opposite and named the
  `016ad146` unbounded reroll as the invariant; the operator ruling of
  2026-08-13 replaced that reroll with the hard bound.
* **Ledger completeness:** no pass may be removed or short-circuited without
  every field it wrote getting a new owner.

## 6A. HOW THE CEILING VALUE WAS MEASURED (provenance)

The 6,000-char ceiling is not a guess, and the numbers quoted in the code
comments come from here rather than from memory. Method, so it can be re-run:

* Walk every `*.json` under `C:\Users\jeffr\Documents\ComfyUI\output\otr\episodes`
  (4,608 files parsed), recursing into every nested dict/list.
* Collect `len()` of every string held under the model-authored keys `title`,
  `premise`, `setting`, `env`, `description`, `visual_prompt`, `intent`,
  `arc_phase`, `generation_prompt`, `text` -- 8,119 strings in total.
* Result: means 8-177 chars, p95s 12-281, and these maxima --
  `title` 50, `arc_phase` 87, `intent` 194, `generation_prompt` 341,
  `description` 795, `visual_prompt` 958, `env` 1,149, `setting` 735,
  `text` 4,426, `premise` 4,549.
* The two outliers (`premise` 4,549 and `text` 4,426 against p95s of 281 and
  234) may themselves be degenerate output rather than authorship, which is an
  argument for a ceiling ABOVE them rather than between them.

The ceiling sits above every one of those, so no episode this project has
produced would have touched it. **A first draft sized each field separately
(title 240, arc_phase 400, intent 800, generation_prompt 1500) and the suite
rejected it** -- `test_p3_score_draft_preserves_arbitrarily_long_authored_fields`
builds ~2,400-char fields and asserts they round-trip. That test was right and
the tight ceilings were wrong: differentiated bounds buy nothing when the goal
is termination, and every extra bound is another chance to bind real writing.

Cross-model check (CPU, tokenizers only): 6,000 chars is 1,189-1,212 tokens
under every cached local writer -- Mistral-Nemo, gemma-2-2b, gemma-4-E2B/E4B,
gemma-4-12b, Captain-Eris-12B -- i.e. within 2% across all of them. That is
7.4% of Mistral's 16,384 window and 14.8% of the 8,192 window every other local
model carries. The fix is not Mistral-shaped.

## 7. SHIPPED 2026-08-13 -- what was actually built

Adopted: structural ceilings on every model-authored string in the P3 draft
AND on the P5 line text, plus `_assert_authored_text_within_bounds`, which
rerolls on an exact-ceiling hit so a forced-shut string can never ship as
clean-parsing text cut off mid-word. The surface receipt now reports
`structural_ceilings` and every ceiling value.

NOT adopted: the n-gram ban (C1 -- global, includes the prompt in the n-gram
history on decoder-only models, and unproven against the real lmfe
intersection), repetition-penalty tuning (C3 -- measured inert), and
temperature tuning (C4 -- not curative). The in-decode halt (C5) is untouched
and remains a separate settled item; it bounds cost, while this change removes
the mechanism.
