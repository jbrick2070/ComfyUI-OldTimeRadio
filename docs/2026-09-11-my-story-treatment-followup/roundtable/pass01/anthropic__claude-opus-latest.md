<!-- requested_model: ~anthropic/claude-opus-latest | resolved_model: anthropic/claude-opus-5 -->

## Critique of Root's diagnosis and proposed discriminator

### 1. The proposed discriminator is confounded and cannot arbitrate the three named hypotheses

The brief asks for ONE diagnostic that distinguishes **prompt contract vs native model/sampling vs transport**. Root's candidate pair is "a narrowly specified prompt-only contract change vs unchanged-code different installed model family."

- **Arm B (different model family) changes many variables at once**: weights, tokenizer, vocab, chat template, and the EOS union (248044/248046 is Qwen-specific, per the `fit` block in the P0 rewrite receipt). It also changes the LMFE `JsonSchemaParser`/`TokenEnforcer` token mapping (helper cached at 130-159). A pass or fail in that arm is uninterpretable, and it directly strains the contract clause "may change only explicitly and record the changed variable."
- **Arm A cannot distinguish anything against an n=1 success baseline.** Root concedes success02 vs failure03 "could be stochastic." A single prompt-changed pass is therefore indistinguishable from attempt02's luck. Root has written a fix and labelled it a discriminator.
- **Neither arm touches sampling or the constrained-decode seam**, i.e. the arm that the evidence most implicates (below). "Transport" is not addressed by either arm at all, despite being in the question.

### 2. Unexplained transport-adjacent evidence is dropped, not dispositioned

"Server selectively stopped, port8000 empty, desktopVRAM 2083MiB" after a 408.62s terminal error is the only transport-flavoured observation in the record, and the diagnosis neither explains it nor excludes it. Asserting "no cancellation" is not the same as excluding a mid-run server-side event. If the transport arm is being retired, retire it with the receipt that does so.

### 3. Grounding gap: the audit's central claims are unverifiable in this pass

Root's exculpatory audit cites `_otr_my_story.py` 499-500, `OTR_LedgerScriptWriter` 744-764, native callback 1067-1098, helper 130-159, and `ProviderCapacityMessages` behaviour. **None of those files/regions are in the supplied excerpts** except `_otr_my_story.py`, and in the supplied copy the `StoryTreatment` bind is at `_pass_treatment` (`bind_schema(StoryTreatment)`), not any line 499-500 I can confirm. Likewise the exact-cycle guard, guard windows (512/448), cycle lengths (147/50/48) and the retry-routing behaviour live in `_otr_structured_call` / the guard module, which are absent. "No stale cursor found" and "no demonstrated new routing bug" are therefore assertions, not shown results. Declare that dependency instead of letting it read as settled.

### 4. Retry-routing narrative is internally inconsistent on the supplied evidence

`_full_artifact_repair` builds a repair turn from `failed_output`. All three failed attempts record `raw_output: ""` with text only in `raw_completion`. Root states attempt2 "uses original prompt at 0.5, not prior failed text" — consistent with a non-PostValidationError halt bypassing the repair factory — but then states attempt3 was a "typed full repair." Those cannot both be true unless the guard exception carries `raw_completion` into repair on the third rung only. That routing asymmetry is unexplained and is itself a candidate defect; it also means the ladder spent two of three rungs re-rolling the same prompt, which is exactly the "low information" behaviour Root deprecates in reruns.

### 5. Demonstrated code defects Root does not claim

- **Double injection of the imperative source into the P1 prompt.** `_call` prepends `_SOURCE.raw_source_block(bundle.fields)` to the last user message; `_pass_treatment` *also* embeds the full interpretation (`json.dumps(..., indent=2)`), whose `requirements[*].text` paraphrase the same `plot` field. Attempt01's logline is a near-verbatim recitation of the `plot` field, repeated in cycles. This is a concrete, code-level copy attractor on the same string the failure occupies. Root's diagnosis nowhere addresses it — and it is prompt-contract evidence stronger than "add be concise."
- **Forensic telemetry blinds the one field you need.** The `finally` block parses only `attempt["raw_output"]` for `proposed_acts`/`proposed_characters`; failed attempts have `""` there, so both counts stay `null` in the ledger while the near-complete attempt3 text sits unparsed in `raw_completion`. Confirmed in the supplied ledger.
- **Unvalidated `cast_plan.requested` drift.** The prompt supplies "characters requested: 2"; P0 returned `requested: 3` and it is stamped into `story["cast_plan"]` beside `counts.requested_characters: 2`. Nothing reconciles a known integer against the model's echo. Minor, but it shows P0 already paraphrasing/inflating on this exact input — relevant conditioning evidence Root files as "P0 succeeds."
- **No sampling seed.** `_resolve_seed()` seeds only the Python `rng` (voices, style roll). Nothing in the excerpts pins decode entropy. So *no* arm of *any* experiment here is reproducible, which is the real reason a rerun is low information — not luck-seeking per se.

### 6. Guard-limitation framing is used to close a question it only opens

Root is right that reintroducing a string/word/time ceiling would not prove semantic nontermination, and right to forbid it. But the same paragraph converts "the only detector is exact-cycle" into "the guard is fine." The consequence is that the *first* ~2700 tokens of varying-noun degeneration are invisible, so the observed cycle length (147/50/48) is a property of the detector, not of the failure. Any inference that leans on those numbers (e.g. "structural, not entropic") is unsupported.

### 7. Contract risk in Arm A

A "logline is one sentence" tightening targets output length on an optional metadata field. The schema and treatment system *already* say "one sentence." Tightening the same instruction risks (a) no measurable change, and (b) drifting toward a content gate on optional metadata, which the contract forbids. If the prompt arm is kept, the changed variable must be structural (e.g. removing the duplicated raw-source prepend for P1 only), not stylistic.

### 8. Better single arm, one recorded variable

The evidence — degeneration confined to *inside an open JSON string*, three times, at three temperatures, never at EOS or length cap — points at the interaction of the enforcer mask with `top_p 0.95 / min_p 0.05 / repetition 1.03` (Root's own note mentions a "min_p fallback" path immediately before `generate`). The highest-information single change: **re-run the same qualifying full-API attempt with P1 decode set to greedy (sampling off), everything else held and recorded.**

- Loops persist at greedy → entropy is excluded; prompt contract or enforcer conditioning owns it, and the duplicated-source hypothesis (§5) becomes the next target.
- Terminates → sampling/mask interaction owns it, and the model-family swap is never needed.

That is one variable, cheap, reproducible, stays inside the existing three-retry budget, adds no verifier call, and — unlike either of Root's arms — actually splits two of the three named hypotheses.