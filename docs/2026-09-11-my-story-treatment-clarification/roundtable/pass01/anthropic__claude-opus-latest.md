<!-- requested_model: ~anthropic/claude-opus-latest | resolved_model: anthropic/claude-opus-5 -->

## Demonstrated defects (from the artifacts, not inference)

**1. Attempts 2 and 3 are, for thousands of characters, byte-identical — including identical token counts (1568/1568) and identical cycle lengths (259/259).** The plan states P1 runs at temperatures .5 and .1, unseeded, fresh server. Two independent stochastic draws at *different* temperatures do not agree verbatim for ~1500 tokens. This is the strongest evidence in the packet and the plan does not mention it. Candidate explanations, all consequential:

- `do_sample`/temperature is not actually reaching `generate` for P1 retries (effectively greedy both times) — which would also make "Prior Opus proposed greedy sampling" a no-op already in force;
- attempt 3's prompt is identical to attempt 2's, i.e. the retry ladder does **not** feed the prior failed completion into P1 (contrast attempt 1→2, which clearly *did* change behaviour);
- the retry re-uses cached state/output rather than regenerating.

The plan's claim that "the real-ladder regression proves latest captured completion handoff and actual replacement" is scoped to the *final repair call*, not to P1's three-attempt ladder. Extending it there is an unsupported causal claim, and the 2/3 identity is direct counter-evidence.

**2. The instruction-leakage diagnosis generalises from one attempt out of three.** Only attempt 1 copies `(No quotation marks)`. Attempts 2 and 3 do something different and worse: they serialise the *entire* schema, contract name (`OTR_SCHEMA_CONTRACT_V1`), act counts, tone notes and safety constraints into the `logline` string, then degenerate into a synonym chain. A title-field clarification cannot touch that. Root already concedes it is "not a proven cure"; the artifact shows it is not even the dominant mode. Anything more than a one-line hygiene fix here is misdirected effort.

**3. The failure is unterminated-string, not unterminated-object.** In all three attempts the model never emits the closing quote of the first prose field. So the constrained decoder is operating exactly as designed (a JSON string may contain arbitrary text) and cannot help. The prior audit finding that "closing quote token1 is admitted at all three 03 decoded endpoints" is consistent with this and, as the plan says, proves nothing about *why* the model doesn't pick it. Correctly stated; don't build on it.

**4. The prompt's "Schema:" block is not valid JSON** — every value is English prose (`one sentence`, `concrete place`, `array of objects, each:`; note the stray semicolons). The model is asked for "one JSON object only" while being shown a prose-valued pseudo-object. Attempts 2/3 are literally the model continuing that prose register inside a string. This is a real, visible defect, and it makes Gemini's valid-JSON-example proposal a better-motivated candidate than the title wording. It is still unproven.

## Method gaps

- Four runs each changed something (EOS repair, source/voice fixes, completion handoff). The plan admits "no four-run causal experiment" but then reasons from the 3/4 failure rate as if the arm were constant. It isn't; 3/4 is a rate across four different builds.
- No pre-change baseline rate for the same graph/source, so no effect size any single next run could detect. Any one-shot outcome after any change is uninterpretable — the same objection correctly raised against Opus and Gemini applies to the title fix.
- "Existing full repair owner" is an empty code block. I cannot review the repair owner, and no conclusion about handoff should be drawn from an absent file.

## On the model-family swap

Reject it as the next action. With the attempt-2/3 identity unexplained, a different model family confounds the arm you are trying to isolate and costs a full canonical run per data point. It is a fallback, not a discriminator.

## ONE next action

Do not change P1's prompt yet. Run one bounded instrumented replay of the existing saved workflow that records, per P1 attempt, the exact rendered prompt hash and the effective sampling kwargs actually passed to `generate` (temperature, `do_sample`, top_p, min_p, repetition_penalty). Purpose: decide whether attempts 2 and 3 were genuinely distinct draws with distinct inputs. If they were not, the "three attempts" budget is currently worth one attempt, and that — not title wording — is the recurring P1 failure.