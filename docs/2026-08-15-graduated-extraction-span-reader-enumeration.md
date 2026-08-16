> **CLOSED-BY-RIP 2026-08-16 (PBUG-20260816-01).** This document's own scope
> line reads "the work below is `scifi_news` only", and every span reader it
> enumerates lived in `_otr_scifi_codex.py` / `_otr_scifi_source_repair.py`,
> both deleted with the lane. NOTHING here is owed. Kept as the record of
> what the relaxed-extraction contract would have required.

# The graduated extraction contract -- every span reader, and what each one does when the spans are gone

**Operator ruling, 2026-08-14:** *"If it fails once on extraction, we relax the
extraction requirements on the second pass -- it just has to get the gist of the
story and populate the coda."*

`docs/GO_FORWARD_PLAN.md` attaches one non-negotiable condition to that ruling:

> **STAMP WHICH CONTRACT PRODUCED THE INDEX.** A relaxed extraction is no longer
> span-proven, so nothing downstream may claim verified provenance for it.
> Enumerate every span reader and give each a defined behaviour BEFORE writing
> the relaxed pass.

This file is that enumeration. It was built by reading the real Windows files at
`ea5fb495`, not from memory. **Written before the relaxed pass, which is the
whole point** -- the standing rule is that a pass may not leave a ledger field
unowned, and this ruling makes a field *conditionally absent*, which is the same
hazard wearing a different hat.

---

## 0. The premise correction: this is a ONE-LANE change, not two

The plan says *"Applies to BOTH news lanes."* Grounded against the tree, that is
not right, and acting on it would have produced a relaxed pass for a lane that
has no strict contract to relax.

* **`scifi_news` (codex, `_otr_scifi_codex.py`)** -- P0 emits `FactIndexV4`, and
  every fact, entity and number carries `SourceSpanV4` rows whose literal
  identity `payload[field][start:end] == quote` is enforced by
  `_validate_fact_index` (`:1746`). This is the lane with the strict contract,
  and the measured `quote_not_literal` failure is its failure.
* **`scifi_news_pro` (fable2, `_otr_scifi_fable2.py`)** -- P0 emits `DossierLLM`,
  and `_make_dossier_validator` (`:802`) **returns `None` unconditionally**
  (`:812-820`). Unverifiable entities and numbers are not refused; they are
  DROPPED after the call by `_filter_dossier_entities` (`:844`), delete-only, on
  the reasoning that "no structural retry can fix knowledge". There are no spans
  anywhere in the pro lane.

So the pro lane **already ships the graduated behaviour**, in its strongest
form: it never fails an extraction for unverifiability at all. The work below is
`scifi_news` only. The pro lane needs no change and gets none.

Its delete-only filter is, however, the precedent the relaxed pass copies -- see
section 3.

## 1. Every reader of a span, and its defined behaviour

`source_spans` / `source_span` appear in exactly two modules. Nothing else in
`nodes/`, and nothing in `scripts/` (the citation audit
`scripts/audit_spoken_citations.py` contains no reference to spans or to the
fact index at all -- it reads the spoken coda receipt, which is a different
field with a different owner).

| # | Reader | Site | What it reads | Behaviour under a RELAXED index |
|---:|---|---|---|---|
| 1 | `_validate_fact_index` | `_otr_scifi_codex.py:1746-1799` | every span; literal identity, allowed field, `0 <= start < end <= len` | Takes an explicit `contract`. Under `strict` it is byte-for-byte what it is today, including "must contain at least one source span" (`:1780`, `:1793`). Under `relaxed` the span clauses are SKIPPED -- ids, duplicate ids, `number.fact_id` resolution and the `payload_sha256` binding are still enforced. |
| 2 | `_span_ok` | `:1714` | one span vs the payload | Never called under `relaxed`. It is the literal-identity predicate itself; there is nothing for it to decide. |
| 3 | `_span_mismatch` | `:1738` | one span, for the error text | Never called under `relaxed`. |
| 4 | `_rebase_p0_index` | `:1802-1831` | `span["start"]/["end"]` for `field == "full_text"`, shifted by the window offset | Unchanged and correct as written: it iterates the span lists, so an empty list is a no-op. `number["source_span"]` is the one site that assumes a mapping -- it becomes a guarded rebase, because a relaxed number has no span to move. |
| 5 | `_merge_p0_indices` | `:1895-2017` | `_span_identity` inside the fact, entity and number dedupe keys | The keys still work: a relaxed fact keys on `(claim_identity, ())`, a relaxed entity on `(name_identity, ())`, so distinct claims and names stay distinct. `_span_identity(None)` for a spanless number is defined to `()`. The final `_validate_fact_index` call at `:2009` inherits the merged contract (see section 2). |
| 6 | `repair_literal_source_metadata` | `_otr_scifi_source_repair.py:79-340`, sole caller `_otr_scifi_codex.py:4119` | drops non-literal spans, then fails rows to `no_literal_source_spans_remain` | **Strict path only.** It is the deterministic repair wired into the STRICT ladder; the relaxed pass does not ask for spans, so there is nothing for it to repair. Unwired from the relaxed attempt, not modified. |
| 7 | P0's own schema contract | `schema_shape_instruction(...)`, budget probe at `:3423` | `min_length=1` on `FactV4.source_spans` / `EntityV4.source_spans`, required `NumberV4.source_span` | **This is why the strict schema is not touched.** The local backend decodes under a GRAMMAR derived from the schema, so `min_length=1` does not merely instruct the model, it constrains generation. Relaxing the shared type would silently relax the strict pass's grammar. The relaxed attempt gets its own smaller schema instead. |
| 8 | `lane_meta["fact_index"]` | `:4412` | the whole index, into durable meta | Carries the index as it actually is -- spans present under strict, absent under relaxed -- beside the new contract stamp (section 2). Nothing is synthesised into it. |
| 9 | P0 call journal | `:4161-4177` | `source_window` offsets, per accepted call | Unchanged. Window offsets are a property of the WINDOW, not of the spans, so they remain true under either contract. |

### Readers that turn out not to be span readers

Worth stating, because the plan's own worry was that something downstream would
quietly claim proof it no longer has:

* **`_compact_p0_fact_context` (`:1490`, feeds P3)** -- `fact_id` and `claim`
  only. No spans.
* **`_script_artifact_inputs` (`:2116`, feeds every P5 dialogue and review job)**
  -- `fact_id`, `claim`, `tone`. No spans.
* **`_news_coda_source_anchors` (`:3545`, feeds P6, THE CODA)** -- `entity.name`,
  `number.verbatim`, `fact.numeric_tokens`. **No spans**, exactly as the operator's
  ruling predicted. The coda's grounding check
  (`_names_a_source_anchor`, `:3577`) is a word-boundary match of those verbatim
  strings against the coda text, and it is unaffected by a relaxed index.

That is the load-bearing finding of this enumeration: **the coda -- the surface
the operator named when he asked for the relaxation -- never reads a span.** The
relaxed pass can populate it exactly as well as the strict one, which is what
makes the ruling sound rather than merely lenient.

## 2. The stamp, and the weakest-link rule the ruling does not state

P0 does not run once. `run_scifi_codex_episode` (`:4259-4276`) runs
`_invoke_p0_window` once per source window and merges the results, so **one
episode can mix a strict window with a relaxed one.**

* Each window carries its own contract.
* `_merge_p0_indices` takes the **weakest link**: any relaxed window makes the
  merged index `relaxed`. A part-proven index is not a proven index.
* The merged contract is stamped in two places -- `lane_meta["fact_index_contract"]`
  (durable, beside the index it describes) and the P0 journal receipt (per
  window, so an audit can see WHICH window relaxed and why).
* The stamp is **never a model-authored field**. It is not added to
  `FactIndexV4`, because a field in the schema is a field the model is asked to
  fill, and a model must not be able to declare its own extraction proven.

## 3. What "relaxed" is allowed to mean

Relaxed is a SMALLER JOB, not a suspended check. It drops one requirement --
transcribe literal quotes and their offsets -- and keeps everything the coda and
the drama actually consume. The measured failure is `quote_not_literal`, which
is a TRANSCRIPTION defect, not a comprehension one.

| | strict (attempt 1) | relaxed (attempt 2) |
|---|---|---|
| fact claims | yes | yes |
| entity names, number verbatims | yes | yes |
| `source_spans` with literal quotes and offsets | required | **not asked for** |
| ids unique, `number.fact_id` resolves, payload digest binding | yes | yes |
| entity names / number verbatims corroborated in the window text | via the span | **delete-only filter** |

The last row is the one that keeps relaxed honest. Without it, "relaxed" would
mean "may invent", which no rule in this project permits. Entity names and
number verbatims are exactly the copy-checkable strings, and they are exactly
what the coda speaks -- so the relaxed pass corroborates them against the window
evidence and DROPS the ones it cannot find, the same delete-only shape the pro
lane has shipped since 2026-07-10 (`_filter_dossier_entities`). Dropping only
ever shrinks what the coda may claim. Fact CLAIMS are kept as written: a claim
is a paraphrase, there is no substring check that could judge one, and the model
was shown nothing but the window's own text.

## 4. Order of work

1. This enumeration. (Done -- the plan requires it before any code.)
2. `_validate_fact_index` takes `contract`; `strict` is byte-identical to today.
3. The relaxed schema + the corroborate-and-drop filter.
4. `_invoke_p0_window`: strict ladder first; on `CodexPassError`, one relaxed
   attempt; return the index AND its contract.
5. Merge to the weakest link; stamp; receipt.
6. Tests: strict unchanged, relaxed accepted, mixed-window merge degrades,
   a relaxed index still yields coda anchors, and an invented entity is dropped.
