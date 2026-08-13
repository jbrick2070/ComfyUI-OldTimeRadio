# Cross-bank runaway exposure -- what `832eaf6b` does and does NOT cover

Companion to `docs/2026-08-13-writer-runaway-root-cause.md`. That document
diagnoses and fixes the runaway on the `scifi_codex` lane; this one answers the
operator's question -- *"are we sure there is no runaway on any media bank?"* --
and the answer is **no, and the biggest exposure is on a different lane with a
different mechanism than the one we just fixed.**

Produced by an external audit lane, then GROUNDED against the real files by the
driver. Every claim below that this document asserts was checked; the two that
corrected the driver's own prior statements are called out as corrections.

## 1. THE CORRECTION THAT MATTERS MOST: only `scifi_codex` binds LMFE

The measured mechanism in the root-cause document -- lm-format-enforcer masks
EOS until the JSON document is complete, so the closing quote is the only exit
and the samplers price it at zero -- **is specific to the `scifi_codex` lane.**

`_bind_local_slot_schema` is defined and called ONLY in
`nodes/_otr_scifi_codex.py` (`:1300`, `:2204`, `:2206`). The writer merely
EXPOSES the capability (`OTR_LedgerScriptWriter.py:740` attaches
`_otr_bind_schema` to local-transformers closures); nothing else consumes it.
`structured_call()` adds a TEXTUAL schema instruction and validates AFTER the
slot returns -- it does not bind a grammar.

Consequences, and they are not small:

* Every other lane's "JSON" is **post-validated, not constrained**. EOS is never
  grammar-masked there, so the specific trap we fixed does not exist on them.
* **Adding `Field(max_length=...)` to those models would not stop a decode.**
  Without token-selection binding, a ceiling is only a post-hoc validation
  failure -- the runaway still happens, then the artifact is thrown away. On
  those lanes a length cap is worth having as finite-surface hygiene, but it is
  NOT the fix it was on `scifi_codex`.

## 2. The lane map, by generation shape

| bank -> pipeline | shape | full-window runaway? | `832eaf6b` helps? |
|---|---|---|---|
| `scifi_news` -> `scifi_news_circuit` | LMFE-CONSTRAINED JSON | fixed | **this is the fix** |
| `scifi_news_pro` -> `scifi_news_pro_multipass` | post-validated JSON (P0/P1/P2/P2c/P4) + **raw markup P3** | **YES -- critical** | no |
| `media_archive` -> `legacy_many_pass` | post-validated JSON + inline paths | no full-window reservation | no |
| `shakespeare` -> `legacy_many_pass_adapt` | same inline shape | no full-window reservation | no |
| `public_domain` -> `legacy_many_pass_adapt` | same inline shape | no full-window reservation | no |
| `original` -> `original_multi_pass` | post-validated JSON front + shared inline | no full-window reservation (largest single request 1,400) | no |

## 3. `scifi_news_pro` is the critical exposure

Its P1/P2/P2c/P4 passes and its raw P3 markup call all use
`ProviderCapacityMessages` with `max_new_tokens=None`, which for local Mistral
resolves to `context_cap - prompt_tokens` -- the same "reserve the whole
remaining window" shape that produced the 13,912-token decode we just fixed.

The **raw P3 markup call is the worst of them** because it has no exit at all:

* full remaining capacity,
* `max_new_tokens=None`,
* no `stop` passed -- and the writer installs `StoppingCriteria` only under
  `if stop:` (`OTR_LedgerScriptWriter.py:972`),
* no schema binding, so no grammar end-condition either,
* parsed only AFTER generation returns.

The prompt's literal `END.` instruction is not a runtime stop condition. So a
markup repetition loop has exactly two exits: the model emitting EOS on its own,
or the output ceiling. And a ceiling hit raises `PromptContextOverflowError`,
which `_run_markup_ladder()` does not catch -- **so a true capacity runaway
aborts the pass on attempt one rather than rerolling.**

## 4. CORRECTION -- the 2026-08-12 fable2 death was NOT a capacity runaway

The driver previously described that incident as evidence a markup attempt can
decode to the ceiling. That was wrong. `PBUG-20260812-03`
(`docs/PROD_BUG_LOG.md:3932`) is a `viz_green` leg that failed in **3.0 minutes**
after **four COMPLETED but malformed** markup attempts -- `UNKNOWN_SPEAKER:
*SFX` / `SKELETON_BREAK`, a repair-rule gap, not a token-ceiling event. The
four-rung ladder applies when a COMPLETED response is parser-rejected; it is not
the runaway path.

So the P3 markup capacity exposure in section 3 is a REAL STRUCTURAL exposure
that has **not yet been observed live**. Stating it as observed would repeat the
admission-rule violation this project already has scar tissue for.

## 5. THE SETTLED IN-DECODE HALT HAS A GAP, and the coder window must see it

`docs/2026-08-13-codex-consult-indecode-halt.md:172` installs the guard on
`schema_model is not None`, reasoning that this covers P0/P1/P2/P3/P5 because
`_invoke_codex_structured_once` binds every local slot to its result schema.

That reasoning is correct **and it is exactly why the halt would miss the worst
path in the repo.** The fable2 P3 markup call passes `schema_model=None`. A
schema-gated guard installs on the lane that now ALSO has structural ceilings,
and does not install on the lane that has full remaining capacity, no schema, no
stop, and no rerollable disposition on a ceiling hit.

**The halt's installation condition should be liveness-based, not
schema-based.** A degeneracy guard is a liveness contract; it belongs on every
local `generate()` call, whether or not a grammar is bound.

## 6. Priority

1. **`scifi_news_pro` -- critical.** Highest-value next fix: a **schema-less
   markup liveness guard** that terminates a degenerate raw decode well before
   the context ceiling and returns a REROLLABLE outcome to the markup ladder. It
   must cover `schema_model=None` explicitly, and per the halt design it must
   raise a rerollable capacity phase rather than becoming a writer veto.
2. **`original`** -- bounded, but the largest inline request (1,400 x 3).
3. **`media_archive` = `shakespeare` = `public_domain`** -- bounded and equal on
   current evidence; a stricter ordering among the three would be an assumption.

Uncapped authored strings exist across all of these lanes and are worth closing
as finite-surface hygiene -- but per section 1, on a post-validated lane that is
hygiene, not a runaway cure.
