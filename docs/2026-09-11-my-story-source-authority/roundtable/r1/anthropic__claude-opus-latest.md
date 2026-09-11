<!-- requested_model: ~anthropic/claude-opus-latest | resolved_model: anthropic/claude-opus-5 -->

# Adversarial review — R1 anchor ("source priority and the final act's endpoint")

## 1. The verdict precedes the grounding it is "subject to"

"Root proposal / verdict" states **CONDITIONAL GO … subject to R1-R4 grounding**, while the R1 questions at the end are still open ("Is this the narrowest coherent correction? Which proposed scope clause is actually needed?"). Issuing a GO and then asking reviewers whether the change is necessary anchors the review. Either the R1 question set is real (then the status is *undecided*), or the GO is real (then R1 is theater). Pick one.

Related: the doc itself concedes "They do not prove a prompt-only change cures model judgment," and "Reject changes that merely … move a model decision into a refusal" — yet all three clauses are prompt-wording changes whose whole mechanism is model judgment. The plan's own epistemics contradict its disposition.

## 2. Clause 1 targets a pass that did not fail

"Live evidence": **P0 marked explicitly requested girlfriend mention** — i.e. the interpret pass *recorded* the requirement. The loss happened downstream (P1 omitted; P2 omitted). So sharpening required/preferred prose in `my_story_interpret_system` addresses an artifact that already contained the fact.

Worse, `Requirement.strength` (`_otr_my_story.py:122`) has **no shown consumer**. `_pass_treatment` (:509) hands the model `json.dumps(interp.model_dump())` wholesale; `_pass_act` (:587) receives **only** treatment JSON + cast + `plan` — requirements/`named_cast` never reach the act author at all. Redefining `strength` semantics in P0 changes a field nothing enforces. Clause 1 is the least grounded of the three and should be dropped from R2 scope unless a reviewer can point at the code path that reads `strength`.

## 3. The likely structural defect is untouched: requirements do not transit to the act pass

Given :587, if `treatment.acts[n].turns` / `ending_state` omit a required person, the act author's only access to that person is the raw block prepended in `_call` (:375) while the *directive* part of its prompt actively says "where it should leave the story: `plan.ending_state`". The narrowest correction is therefore plausibly **treatment-side** (a coherence obligation between `acts[-1].ending_state` and `ending`, and/or passing required requirements into the act prompt), not act-side ending surfacing. `_make_treatment_validator(act_count)` appears (from the repair text at :509) to police act *count*, not plan/ending coherence. Clause 2 patches the symptom in the last consumer while leaving the plan that caused it authoritative. R1 should compel a comparison of these two intervention points before GO.

## 4. Clause 2's "raw source above derived plans" is already true — which weakens its own thesis

`_call` (:375) already prepends `_SOURCE.raw_source_block(...)` to the top of the final user message for **every** pass, and the doc says "Full raw source reached every author/corrector." So "raw listener source above derived plans" is not a change; it is the status quo that failed. Claiming it as remedy #2 is either a no-op or an undisclosed reordering of existing content. State precisely what bytes move.

## 5. Clause 2 has a concrete implementation trap in `_pass_act`

At :587, `is_last` is only rendered into the prompt **inside the `if must_speak:` branch** ("This is the LAST act, so they must speak here."). If all cast have spoken, `unheard == ""` and the model is never told it is writing the last act. Any "final act must land `treatment.ending`" clause must be emitted unconditionally, not appended to `unheard`. Also note `is_last` is currently narrowed further in `post_validator=_make_act_validator(treatment, plan.n, must_speak if is_last else ())` — the validator is cast-only, so the new ending obligation would be unvalidated instruction text. That is acceptable only if the plan says so out loud; it currently implies an "obligation."

Additional ambiguity: "Earlier acts follow their local endpoint; do not force premature endings" plus "raw source above derived plans" gives act 1 of 3 two competing authorities with no stated tie-break when `plan.ending_state` contradicts the source. Expect drift in *earlier* acts — an untested regression surface, and there is no pre/post comparison corpus.

## 6. Clause 3 commands the spoken corrector to do something its schema cannot do

`rewrite_spoken_from_source` (:337) validates via `_apply_spoken_edits`, and the instruction it already carries is explicit: edits are **replacement text at `line_id` + optional `start_char`/`end_char`**, "Never change speakers, order or ids." There is no insertion primitive. Telling it that "the complete spoken ledger cannot defer an ending beyond the story" (Clause 3) therefore either

* produces edits that `validate` rejects → burns the 2-attempt budget (`attempt_limit = min(SOURCE_REWRITE_ATTEMPTS, …)`, :125) and yields `unresolved`, or
* pressures it to overwrite unrelated tail lines into a synthesized ending — a destructive change on a ledger whose conservation guarantees ("nine rows through 38 calls", `PROTECTED_FACT_COMPONENT_FLAG`) are load-bearing.

The observed `edits: []` at 1595/131072 tokens is equally consistent with *correct abstention given the edit shape*. This is the single most dangerous item in the plan and it is presented as the safest.

## 7. Clause 3's "does it need a new parameter?" is already answered by the code

`rewrite_story_source` (:125) already accepts `instruction=""`, and `rewrite_spoken_from_source` (:337) already uses it. What's missing is that **`_call` (:375) never passes `instruction`**. So:

* No new parameter/schema/call is needed to scope the generic clause per artifact — contradicting the plan's hedged "decide whether … a small existing-owner scope argument is necessary."
* But the deferral rule ("a partial act may defer; the sole/final act cannot") needs `is_last` and `len(treatment.acts)`. Inside `_call` you have `pass_id` (`"act_%d" % plan.n` → gives n, **not N**) and `bundle`. So the real change is threading one string (or `is_last`) from `_pass_act` through `_call`'s `**kwargs`. R1 should require the plan to name that exact seam rather than leaving it as an R2/R3 "decide whether."

## 8. `author_context` asymmetry — an unexamined transport discrepancy

At :375, `author_context` is snapshotted **before** the raw-source prepend, and that pre-prepend copy is what goes to `rewrite_story_source(..., author_context=author_context)`. The corrector therefore sees an authoring context that appears *not* to have included the source block. The doc asserts "No code/decoder/source transport bug was found by root/Terra in the retained-result path." This is at minimum a semantic-fidelity discrepancy in the corrector's inputs, and it is exactly the kind of thing a "prompt ambiguity" narrative would miss. It should be explicitly checked or acknowledged, not covered by a blanket no-bug claim (absence of found bug ≠ absence of bug).

## 9. No validation path exists before the one shot that matters

"No GPU during review/coding" + "fresh one-run qualification" + n=1 failure sample. That means a judgment-dependent prompt edit will be shipped with **zero** empirical evidence, evaluated by **one** sample, against a base rate of 0/6 qualifications. Statistically this run cannot distinguish the fix from sampling variance in either direction. Missing artifacts:

* a falsification criterion (what result would mean "the ambiguity theory was wrong"?);
* a rollback/stop rule at attempt 7;
* an offline replay harness over the retained `pairlock_06_*` raw fields/receipts (CPU-only prompt-diff inspection is possible even without generation, and cached raw outputs permit re-validation of the corrector's edit shape — cheap, in-scope, and absent from the plan).

## 10. Regression risk is asserted, not measured

"Finished Sonnet QA + full regression" cannot detect **over-correction**: stronger required/ending obligations plus a rescoped corrector will raise false-positive rewrites on stories that previously passed. `MUST PRESERVE: … No forced edit when already correct` is a wish, not a mechanism — `applied = accepted != original` (:375) records it after the fact but nothing prevents it. Ask for a small held-out set of previously-acceptable episodes and a change-rate bound, or drop the "no forced edit" guarantee.

## 11. Over-engineering / scope smells

* The **MUST PRESERVE** and **OUT OF SCOPE** lists are longer than the proposal. Several entries ("no RSS" — undefined; "truthful provider/OOM/cancel"; "existing ledger freeze/publication") are unrelated to a prompt-wording change and are unfalsifiable review instructions. They inflate reviewer load without constraining the actual diff.
* The other-bank citations are decorative: Sci-Fi's `priced_ending`, Public Domain's `outline_macro`/`line_composer`, Shakespeare's `coda` are *different pipelines with different ownership* — the doc even says "Reuse that existing ownership pattern, not their unrelated schemas." What remains after that subtraction is "put the ending near the output target," i.e. §5. Precedent is thin; don't let it substitute for evidence.
* Three clauses bundled as one "narrow" change. They have independent evidence strength (weak / medium / dangerous). They should be severable so Clause 1 can be dropped and Clause 3 gated on the insertion-capability question without blocking Clause 2.

## 12. Independence claim

"Opus and Gemini API reviewers are the available independent families … Do not invent Cursor consensus" is sound discipline. But "Terra's independent trace" is asserted with no statement of what Terra had access to; if Terra read the same anchor doc, its agreement is correlated, not independent. Say so.

---

## Recommended R1 answers

1. **Not the narrowest.** Narrowest defensible change = (a) scope the generic corrector clause per pass via the **already-existing `instruction=` parameter**, wired from `_pass_act`/`_pass_treatment` through `_call` (:375); (b) make the last-act ending obligation unconditional in `_pass_act` (:587), outside the `must_speak` branch. Everything else is unproven.
2. **Clause 1: cut** until a consumer of `Requirement.strength` is exhibited.
3. **Clause 3 spoken half: block** until someone answers whether `_apply_spoken_edits` can express an ending at all; if it cannot, an "ending cannot be deferred" instruction to that owner is a defect, not a fix.
4. **Require** treatment-side coherence analysis (`acts[-1].ending_state` vs `ending`) as the competing hypothesis before accepting the act-side fix.
5. **Require** an offline replay/prompt-diff artifact and an explicit stop rule before the seventh run; a single qualification run is not a test.