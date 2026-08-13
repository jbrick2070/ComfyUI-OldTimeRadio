# The in-decode halt for the writer runaway -- Codex consult, 2026-08-13

Ran BEFORE any code was written, because the halt touches every
authoring pass. It changed four things about the proposed shape.
The design it settled is summarised in GO_FORWARD_PLAN item 2; this
is the full text, and the summary is not a substitute for it.

Driver: Claude (Cowork) wrote the anchor against the real Windows
files first and remains the judge; every claim below was ground-
checked before anything was folded in. Reviewer: Codex CLI, which
the operator's 2026-08-11 routing makes the consult of record for a
quandary in place of a full kibitz arc.

---

## Part 1 -- the anchor, as sent

# Codex consult -- designing the in-decode halt for a writer runaway

Real Windows repo: `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio`.
Read the actual files. Cite file:line. Be adversarial about my framing.

## The failure

A local transformers `model.generate()` writer pass runs away: it enters an
anaphoric closing cadence and cannot leave it. Captured evidence from a live leg:

```
RUNAWAY EVIDENCE (55577 chars, 13828 tokens, ended_with_eos=False)
TAIL: ... Let the echoes echo. Let the truth prevail. Let the future be ours to
      forge, hand in hand, echo by echo. Welcome, dear listener, to Echoes of
      Error. Let the echoes resound. Let the truth triumph...
```

Mixed structure: verbatim anchors ("echo by echo") recur exactly, but they are
separated by ~25 tokens of slot-substituted variation
(inspire/echo/resound, unite/prevail/triumph, shape/forge/build).
`repetition_penalty=1.03` and `min_p=0.05` were BOTH active and did not stop it.

Two occurrences in one night, on DIFFERENT passes (P1 and P3) and different
source material, so it is not pass-specific or premise-specific. One ran 21
minutes and 14,191 tokens; the ladder self-healed on the next rung.

## Hard constraints from the operator, which a fix may not violate

1. **The guard may NEVER read `target_words`.** Word-count chasing is forbidden
   by standing directive, and capping the output budget to the word target is
   explicitly forbidden by a prior ruling on this same bug (PBUG-20260729-02).
2. **The writer must never VETO.** Operator ruling: "the writer should never
   veto, the writers should keep on passing in a loop to agents to clean up the
   ledger." So the halt MUST raise a phase-carrying, REROLLABLE capacity error
   -- if it raises anything terminal it silently becomes the writer veto the
   directive forbids.
3. The outer validation loop is unbounded ON PURPOSE and must stay that way --
   that is how one bank recovers from an announcer-coverage weakness.

## What the repo already has

* `nodes/OTR_LedgerScriptWriter.py` around :977 has a `stopping_criteria` block
  that is behind `if stop:`, and `invoke_structured_slot` never passes `stop`.
  So the criteria path exists and is dead.
* A `TextIteratorStreamer` heartbeat has already landed, so a runaway is now
  VISIBLE in the log. The HALTING half does not exist.
* The capacity raise already attaches `raw_completion` and now logs head+tail.
* `_otr_structured_call.py` consumes a capacity error INSIDE the ladder (around
  :1075) and only propagates it to the cycle loop if it is the LAST rung.
* `PromptContextOverflowError` is the existing phase-carrying rerollable error.
  `_otr_model_loader.py:1305` raises a BARE `ModelLoaderError` with no phase for
  the same condition, and tests `>=` where the writer tests `==`.

## The design I am considering, which I want you to break

A `StoppingCriteria` subclass with TWO signals, primary first:

1. **PRIMARY -- open-JSON-string token counter.** These passes decode a JSON
   artifact whose ARRAYS are schema-bounded but whose STRING fields have no max.
   So a 13,828-token run is provably stuck inside ONE unclosed JSON string.
   Track the currently-open string and halt past a threshold (~2,000 tokens
   inside one string). Indifferent to what the prose is doing.
2. **SECONDARY -- n-gram self-similarity window.** Catches verbatim anchors, but
   fires late here because the anchors are ~25 tokens apart.

On halt: raise the phase-carrying rerollable capacity error so the ladder's next
rung retries at a lower temperature, exactly as it already does for a real
capacity failure.

## What I need answered, grounded in the files

1. **Is the open-string counter actually implementable at the token level?**
   `StoppingCriteria.__call__` gets `input_ids`. Do I have to decode
   incrementally to track JSON string state, and what does that cost per token
   on a 14k-token decode? Is there a cheaper equivalent signal? Is there a
   correctness trap with escaped quotes, or with the tokenizer splitting a quote
   into a multi-character token?
2. **Where EXACTLY does the criterion belong** so that all of P0-P5 get it
   without six edits, and so a future pass cannot silently opt out? Trace
   `invoke_structured_slot` and the `if stop:` block, and say whether wiring it
   there is right or whether it belongs deeper.
3. **Does raising from inside a StoppingCriteria even work?** My reading is that
   `generate()` STOPS and RETURNS normally on a criteria hit -- it does not
   propagate an exception -- so the halt must be detected AFTER `generate()`
   returns and converted into the rerollable error there. Confirm or refute, and
   say how the caller distinguishes "halted by degeneracy" from "hit the token
   ceiling" from "finished cleanly", given `ended_with_eos` is already computed.
4. **Will this be mistaken for a capacity failure in a way that matters?** If it
   raises the same error type as a genuine context overflow, the log will say
   the wrong cause. Should it be a distinct phase/subtype, and does anything
   downstream branch on that phase?
5. **What is the threshold argument?** I want a number defensible without
   reading `target_words`. The healthy history on this box is P3 base calls at
   60/75/65 s and the runaway was 20x that. What signal would you lead with, and
   what would you set the open-string bound to so a LEGITIMATE long string field
   is never cut?
6. **What breaks?** Every test that exercises the writer decode path, anything
   asserting the `if stop:` block stays dead, and any fixture that would now
   halt.

If the whole shape is wrong, say so and say what the right one is.

---

## Part 2 -- Codex's review, verbatim

## Verdict

The overall shape is sound: an always-on, local-Transformers decode guard should stop generation normally, then the transport should convert that stop into a distinct, phase-carrying rerollable error.

I would change four parts of the proposal:

1. Do not reuse the existing opt-in `stop` path.
2. Do not throw from inside `StoppingCriteria`.
3. Do not label degeneracy as `output_limit`.
4. Do not ship the n-gram detector as an independent hard halt until it is calibrated against healthy artifacts.

There is also a direct conflict in the repo: the outer candidate loop is not currently unbounded.

### 1. Open-string tracking is implementable

`StoppingCriteria.__call__` receives the entire token sequence after each new token. The existing substring criterion already remembers `prompt_len`, slices only newly generated IDs, decodes that increment, and maintains state across calls ([OTR_LedgerScriptWriter.py:415](C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/OTR_LedgerScriptWriter.py:415), [OTR_LedgerScriptWriter.py:444](C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/OTR_LedgerScriptWriter.py:444)). That is the correct complexity: O(generated tokens), not repeatedly decoding the full growing suffix at O(n²).

Maintain these lexical fields:

- `in_string`
- `escaped`
- `open_string_tokens`
- `hit`
- `reason`
- `hit_at_generated_token`

Scan every character in each newly decoded token fragment:

- If `escaped`, clear it.
- Else if inside a string and the character is `\`, set `escaped`.
- Else if the character is `"`, toggle string state.
- Reset the open-string count when the string closes.
- Evaluate the threshold only after scanning the complete token fragment.

That handles escaped quotes and runs of backslashes across token boundaries. A quote embedded in a multi-character token is not a problem because you scan the decoded fragment character-by-character. Do not assume a quote has its own token ID.

The real tokenizer trap is that decoding isolated tokens is not formally compositional for every tokenizer. Use `clean_up_tokenization_spaces=False`, decode only unseen IDs, and add tests against every locally supported tokenizer family. A lazy token-ID-to-decoded-fragment cache will make repetition especially cheap.

I would not reach into LM Format Enforcer’s private parser state. The writer obtains its schema parser and prefix function at [OTR_LedgerScriptWriter.py:864](C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/OTR_LedgerScriptWriter.py:864), but depending on LMFE’s internal parser stacks would couple the guard to third-party internals.

One correction: “13,828 total tokens proves one string consumed 13,828 tokens” is not logically true merely because arrays are bounded. Several finite, unbounded string fields could share that total. The captured prefix/tail strongly indicates one cadence-locked field, but the new counter—not the aggregate token count—is what will prove its exact length.

### 2. Exact placement

Do not put the criterion in `invoke_structured_slot`. That function has no tokenizer, model, prompt length, or `generate()` call; it merely dispatches the callable ([\_otr_structured_call.py:676](C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_structured_call.py:676), [\_otr_structured_call.py:705](C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_structured_call.py:705)).

Put it in `_build_truncating_generate_fn`, unconditionally for every local schema-bound invocation:

- `schema_model is not None` installs the correctness guard.
- `stop` merely appends the optional substring criterion.
- Failure to construct the degeneracy guard must fail loudly. The existing `except Exception: ... stop-strings disabled` behavior is appropriate for optional quality stops, not for a liveness contract ([OTR_LedgerScriptWriter.py:972](C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/OTR_LedgerScriptWriter.py:972)).

This covers the Sci-Fi passes because `_invoke_codex_structured_once` binds every local slot to its exact result schema ([\_otr_scifi_codex.py:2020](C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_scifi_codex.py:2020)), and the scheduler rebuilds `_build_truncating_generate_fn` with that schema ([OTR_LedgerScriptWriter.py:714](C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/OTR_LedgerScriptWriter.py:714)). P0, P1, P2, P3, and P5 all traverse that shared route ([\_otr_scifi_codex.py:3054](C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_scifi_codex.py:3054), [\_otr_scifi_codex.py:3207](C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_scifi_codex.py:3207), [\_otr_scifi_codex.py:3222](C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_scifi_codex.py:3222), [\_otr_scifi_codex.py:3250](C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_scifi_codex.py:3250), [\_otr_scifi_codex.py:3280](C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_scifi_codex.py:3280)). There is no current P4 authoring pass.

For a durable root fix, factor the guard into a leaf module and install it in the other local Transformers transports too. `_otr_model_loader.make_generate_fn` still has an independent `model.generate()` path and still raises a bare `ModelLoaderError` after output exhaustion ([\_otr_model_loader.py:1321](C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_model_loader.py:1321), [\_otr_model_loader.py:1340](C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_model_loader.py:1340)). Your cited line has shifted because the working tree already contains an uncommitted pre-call phase fix; the post-generation mismatch remains.

### 3. Stopping semantics

Your preferred interpretation is correct for a criterion that returns true: `generate()` returns normally.

In installed Transformers 5.10.4, the generated token is appended and streamed first, then `stopping_criteria(...)` updates `unfinished_sequences`; `streamer.end()` and the normal return occur afterward ([utils.py:2845](C:/Users/jeffr/Documents/ComfyUI/.venv/Lib/site-packages/transformers/generation/utils.py:2845), [utils.py:2850](C:/Users/jeffr/Documents/ComfyUI/.venv/Lib/site-packages/transformers/generation/utils.py:2850)). `StoppingCriteriaList` simply ORs each result ([stopping_criteria.py:495](C:/Users/jeffr/Documents/ComfyUI/.venv/Lib/site-packages/transformers/generation/stopping_criteria.py:495)).

An exception deliberately raised by a criterion would currently propagate—there is no catch around that call—but it is the wrong mechanism:

- It skips the writer’s decode/evidence construction.
- It skips Transformers’ `streamer.end()`.
- It enters the wrapper as an unclassified exception.
- It cannot attach the full returned token sequence cleanly.

Use a latched criterion:

```text
criterion detects -> criterion.hit = True -> return True
generate returns -> decode once -> inspect criterion.hit -> raise typed error
```

Classification must use authoritative signals, in this order:

1. `guard.hit`: degeneracy halt, regardless of generated length.
2. `generated_tokens >= effective_max_new_tokens and not ended_with_eos`: capacity ceiling.
3. `ended_with_eos`: clean model termination.
4. Otherwise: another normal stopping criterion, such as a supplied stop substring.

Do not infer degeneracy merely from `generated_tokens < effective_max_new_tokens`; the existing optional substring stop can also return early. The guard’s `reason` is the discriminator.

### 4. Use a distinct subtype and phase

Do not emit `PromptContextOverflowError(phase="output_limit")`. That would make diagnostics lie: the model did not exhaust capacity; the transport deliberately halted a detected decode pathology.

I recommend:

- `GenerationDegeneracyError`
- phase `decode_degeneracy`
- fields such as `halt_reason`, `open_string_tokens`, `generated_tokens`, `raw_completion`, and repetition statistics

Add it to the shared caught-error family and make `is_rerollable_capacity_error`—or preferably a renamed `is_rerollable_generation_error`—accept both `output_limit` and `decode_degeneracy`.

Downstream behavior is predicate-driven, not phase-string-driven. The ladder catches the shared tuple and asks the predicate ([\_otr_structured_call.py:207](C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_structured_call.py:207)); the lower-temperature rung uses that same predicate ([\_otr_structured_call.py:1066](C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_structured_call.py:1066)); the candidate loop does likewise ([\_otr_scifi_codex.py:1847](C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_scifi_codex.py:1847)). The phase vocabulary and predicate currently live centrally in [\_otr_generation_budget.py:43](C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_generation_budget.py:43) and [\_otr_generation_budget.py:119](C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_generation_budget.py:119).

One downstream message must change: every rerollable capacity error is currently summarized as “model output ended at the provider capacity limit” ([\_otr_scifi_codex.py:1879](C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_scifi_codex.py:1879)). That would be false for a guard halt.

### 5. Threshold

Lead with open-string tokens, not wall clock. Wall time changes with model, quantization, offload, prompt length, and machine load. Token count is deterministic and portable.

I would use 2,048 open-string tokens as the provisional hard bound. At the observed roughly 11 tokens/second, it halts near three minutes instead of twenty-one. More importantly, P3’s own contract says fields should end when their sentence is done and that available capacity is a ceiling rather than a target ([\_otr_scifi_codex.py:517](C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_scifi_codex.py:517)). A 2,048-token `title`, `intent`, `description`, or `visual_prompt` is already outside that pass contract. P5’s unit is one spoken line ([\_otr_scifi_codex.py:644](C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_scifi_codex.py:644)).

But “a legitimate long string is never cut” cannot be proven while the schema permits unbounded strings. Any finite limit excludes something the formal schema accepts. The defensible calibration is:

```text
bound = max(2048, 4 × maximum healthy per-string token count)
```

Measure that with the exact production tokenizer across accepted P0/P1/P2/P3/P5 artifacts, without consulting `target_words`.

For the n-gram signal, I would initially record telemetry only. After calibration, make it an accelerator restricted to the currently open value string—for example, a minimum 512-token string plus repeated six-token shingles across consecutive windows. An independent n-gram hard halt risks rejecting intentional refrains, parallel rhetoric, and repeated schema keys.

### 6. What breaks or needs coverage

There is no visible repository test asserting that the `if stop:` path remains dead. The important additions are:

- Lexer tests for escaped quotes, odd/even backslashes across token boundaries, and quotes embedded in multi-character tokens.
- A normal long-but-closed string that crosses the threshold token count and closes in the same emitted token: it must not halt.
- A fake `generate()` that actually calls the criterion token-by-token, returns early, and proves the transport raises only afterward.
- Separate assertions for `decode_degeneracy`, `output_limit`, EOS, and optional substring stop.
- Ladder proof that degeneracy on the base rung reaches the 0.32 structural retry, matching the existing output-limit test at [test_a4_capacity_phase_advances_the_ladder.py:81](C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/tests/test_a4_capacity_phase_advances_the_ladder.py:81).
- Parameterized proof that every actual pass—P0/P1/P2/P3/P5—receives the guard through schema binding.
- A guard preventing any import or access to `target_words`.
- Transport parity for `_otr_model_loader`, including `>=`, EOS detection, raw evidence, subtype, and phase.
- Healthy-corpus false-positive tests before enabling the n-gram hard halt.

Finally, your third hard constraint contradicts the actual tree. The code currently fixes `MAX_CANDIDATE_CYCLES = 3` and explicitly kills the pass after three candidate ladders ([\_otr_scifi_codex.py:28](C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_scifi_codex.py:28), [\_otr_scifi_codex.py:2323](C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_scifi_codex.py:2323)); a regression test requires that bounded behavior ([test_scifi_candidate_liveness.py:52](C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/tests/test_scifi_candidate_liveness.py:52)). Meanwhile, `GO_FORWARD_PLAN.md` still says the loop is unbounded and must remain so ([GO_FORWARD_PLAN.md:606](C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/docs/GO_FORWARD_PLAN.md:606)).

Therefore an implementation cannot truthfully “leave the unbounded loop unchanged.” If your present instruction supersedes the August 13 three-cycle ruling, restoring the unbounded validation loop and replacing that test must be an explicit part of the same change.
