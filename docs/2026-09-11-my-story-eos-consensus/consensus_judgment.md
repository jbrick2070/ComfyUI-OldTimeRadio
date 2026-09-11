# Native EOS repair: Opus and Cursor consensus

Jeffrey requested Opus and Cursor consensus on the live My Story follow-up.
This is a scoped R4 continuation; the completed R1-R3 campaign was not repeated.
Codex wrote the grounded anchor first and remains the sole coder and judge.

## Decision

Complete the final regression/Bible and finished-code QA, commit and push the
corrected candidate, then run ONE fresh full canonical 5080 pair-lock attempt.
Use the same source fields, Qwen3.5-4B pair, NF4/SDPA profile and sampling controls
as attempt 01. No evidence currently justifies changing sampling, quantization,
models or the correction budget. Inspect the actual source, ledger and pixels;
publication alone is insufficient. Preserve failed attempts in the denominator.
Mac/4060 remain held. RunPod remains blocked by missing authenticated access.

Both independent reviewers recommend this sequence. Their proposed code issues
were checked against the actual source and installed dependencies, as below.
The EOS contract mismatch is fixed in code; a live cure is not yet established.
P1 repetition inside open JSON strings remains a distinct observation.

Runtime-control correction after independent pre-run audit: the review question
used the factory's generic top-p 0.92 in its sampling discussion. The actual
archived canonical request has creativity=balanced, which resolves temperature
0.85 and top-p 0.95, with min_p 0.05 and repetition_penalty 1.03. The retry keeps
that actual request; it does not apply the generic defaults. The historical
`otr_w45_still_pan` profile name does not impose a word target. Explicit Qwen
overrides replace that profile's Mistral-Nemo defaults. Before submission, compare
the entire new API prompt structurally with archived attempt 01; a run-label
changes console text only and should not change any graph/input field.

## Actual reviewers and coverage

- Cursor: `cursor-grok-4.6-high`, local CLI, read-only ask mode. Written review:
  `../../kibitz-runs/2026-09-11-my-story-eos-consensus/r4/cursor.md`.
  It reviewed the initial five-file snapshot. Later deltas were lazy base/polish
  boundary inspection, the actual RNG retry test, and writer token diagnostics.
- Local Claude Opus returned a weekly-limit message, not a review. The first
  API fallback returned empty content at its output limit, also not a review.
- Successful API Opus: requested `~anthropic/claude-opus-latest`, resolved
  `anthropic/claude-opus-5`, low reasoning. Review and provider usage are in
  `roundtable/pass02/`. It received the revised six-file diff, source excerpts
  and root anchor. Subsequent deltas strengthen its requested three-way negative
  control and suppress false truncation diagnostics on clean exact-capacity EOS.
- Actual Sonnet QA: `anthropic/claude-sonnet-5`, low reasoning, via the requested
  latest alias. `roundtable/sonnet_final/` reviewed the finished six-file candidate;
  `roundtable/sonnet_delta/` reviewed the final diagnostic correction and complete
  writer classification block. Its final delta has no remaining must-fix.

The successful Opus call reports $0.192185. The failed API call's harness
discarded its usage, so its printed $0.0000 is not evidence of zero charge.
Local CLI dollar usage was not returned. Sonnet reports $0.100754 plus $0.022520;
reported successful API review spend totals $0.315459. Failed-call usage is unknown.

## Grounded judgments

### Cursor

1. **CONFIRMED:** lock the next action to the corrected candidate after QA,
   with unchanged sampling and model/profile controls. GO_FORWARD now says so.
   Known constrained-sampling lock-in is a plausible mechanism, not a measured
   cause of this particular P1 failure; no logits were saved.
2. **CONFIRMED:** all five live halts were `verbatim_cycle` with
   `open_string_tokens=None`. Provider-capacity messages disable the string-size
   tracker; the observed open strings are visible in generated text. PBUG wording
   now makes that distinction explicit. No size cap is re-enabled.
3. **ACCEPTED:** record actual final token, effective EOS IDs and completion
   membership at the existing writer decode owner. This log precedes and does
   not override guard classification. Transport tests verify the fields.
4. **CONFIRMED TEST LIMIT:** the factory stubs plus actual EosTokenCriteria prove
   kwargs/classification, not live model behavior. Installed Transformers list
   wiring was additionally read; fresh canonical qualification remains required.
5. **ACCEPTED:** remove contradictory sequencing and state current RunPod auth
   blockage. Keep the original failed commit distinct from the retry candidate.
6. **UNVERIFIABLE UNTIL LIVE:** an aligned EOS set need not prevent all legal
   trailing whitespace. Do not infer which hidden token attempt 01 emitted.

### Opus

1. **MISREAD: empty output indexing at a zero allowance.**
   `nodes/_otr_generation_budget.py:214` normalizes requested/minimum to at least
   one, refuses insufficient room before generation, and returns a positive
   allowance. Base/polish inspect the final token only when length reaches that
   allowance. Empty output cannot reach this branch. No defensive dead branch
   or new refusal was added.
2. **MISREAD: explicit `eos_token_id=None` restores model defaults.** Installed
   Transformers 5.10.4 `generation/utils.py:1743` first inherits defaults, then
   `:1747` applies explicit kwargs. `configuration_utils.py:1267` uses setattr
   for explicit None; `utils.py:1986` preserves None and `:1321` omits the EOS
   criterion. Omission inherits; explicit None disables. Also, any valid
   configured EOS makes the shared resolver nonempty. Opus's proposed omission
   would introduce the fallback behavior it claimed to prevent. No EOS-empty
   rejection gate was added. Dewey independently checked the installed source.
3. **PARTLY ACCEPTED:** the negative control now compares all three RNG draws,
   including first versus third. Four float32 draws make accidental collision
   negligible; this is not the deterministic formatting brittleness of the old
   line-distance pin. The alleged cross-test `_min_p_unsupported` state leak is
   **MISREAD**: it is allocated inside each factory invocation at writer line 878.
   The test creates a fresh factory and verifies the attempted min_p arguments.
4. **NO CHANGE: collection padding.** Lists preserve explicit order; sets are
   sorted for deterministic normalization. Padding remains one valid scalar.
   Its exact set-order test intentionally verifies this normalization contract.
5. **MISREAD: ordinary EOS cache thrashing.** Explicit IDs come from the same
   resolver used by the default path. Unchanged owners produce identical tuples;
   changed EOS configuration correctly invalidates tokenizer preprocessing.
   No evidence establishes an unnecessary rebuild in this implementation.
6. **ALREADY COVERED:** the writer log includes generated count, so an absent
   final token is not presented without sequence-size context. Guard priority
   remains unchanged.
7. **CONFIRMED:** EOS alignment is not a demonstrated cure for P1. Both the
   ledger evidence and PBUG retain the 400.67-second repetition failure, with
   no media and no cancellation claim. Model/NF4 causality is unproven.
8. **REJECTED OPTIONAL CUTS:** retain EOS IDs in durable source fit receipts
   (`_otr_story_source.py` owns their persistence) and resolver nonmutation
   tests. The decoder log and LMFE tests exercise different owners. A hypothetical
   falsy custom text-config or need for full token dumps is not evidenced by the
   loaded Qwen configuration and is not added to this repair.
9. **VERIFIED:** installed `generate` prepares config/special-token tensors,
   builds EosTokenCriteria from the list, and supplies it to decoding. LMFE's
   installed multi-EOS admission is exercised by real-library tests. The live
   expected union is 248044 plus 248046; its actual receipt is still required.

### Sonnet finished-code QA and final delta

1. **MISREAD: empty resolution erases a valid internal default.** The review
   eventually reads explicit None semantics correctly, but assumes a valid
   configured EOS could coexist with an empty resolver result. The resolver
   consumes that exact configuration first. No such model configuration is
   present in the installed Qwen snapshot. Empty supported EOS means no supported
   EOS was available; it does not discard a valid terminator. Generic malformed
   future configurations do not justify a new rejection gate in this repair.
2. **CONFIRMED REVIEW-PACKET GAP:** the initial excerpt did not show the complete
   writer capacity branch. Root read the complete file and supplied that branch
   to the delta review. The existing raise already requires `not ended_with_eos`.
3. **ROOT FIX, ACCEPTED BY FINAL SONNET:** the preceding diagnostic block could
   still print OUTPUT_CAP/OUTPUT_TRUNCATED for an EOS at exact capacity. Its
   condition now also requires `not ended_with_eos`. Existing return/failure
   outcomes, guard priority, sampling and retries are unchanged. Tests assert
   clean EOS evidence and absence of both misleading diagnostics. The final
   Sonnet review finds no remaining must-fix in this delta.
4. **UNCHANGED VERIFICATION LIMIT:** real native EOS identity and source/ledger
   recovery must be checked on the next canonical attempt. No live success is
   claimed by either the code tests or the reviews.

## Final scope and evidence

The six code/test owners and hashes are in `roundtable/snapshot_final.json`.
Shared resolver, all four native generators, real LMFE cache/completion and
existing writer diagnostics are wired without a new node or widget. The real
canonical JSON remains unchanged: 23 nodes, 63 links, 37 writer widget slots.
Full-suite and controlled Bible comparisons live alongside the retained failed
attempt in `../2026-09-11-my-story-5080-qualification/`.

The first full regression caught an eager ordinary-path last-token read and a
line-distance test pin. Both were corrected; their initial failed comparison is
preserved. Final qualification must report the rerun rather than relabeling the
initial result. No quarantines or inherited-file deletions are authorized.
