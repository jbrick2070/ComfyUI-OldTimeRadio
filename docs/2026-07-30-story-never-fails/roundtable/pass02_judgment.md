# Round 2 Judgment — Coding Plan

## Verdict

NO on Revision 1 as written; YES on the narrowed implementation after the
corrections below. The panel converged on the same two real gaps as the Codex
anchor: typed retry ownership and an executable overlap/rebase/merge contract.

## Grounded findings accepted

- Move the fresh-candidate loop inside `invoke_codex_structured`, where
  `StructuredCallFailedError` is still typed. Do not catch a generic
  `CodexPassError` outside.
- Keep the public helper finite by default; the canonical runner opts into
  fresh candidates.
- Compile and post-validate P3 inside its candidate campaign. Authored graph
  defects reject that P3 candidate; a post-acceptance compiler invariant remains
  a permanent code/configuration error.
- Make the overlap algorithm prove positive progress and complete coverage.
- Rebase every nested `full_text` span, including number evidence, using the
  explicit span field and window offset.
- Specify duplicate-bundle ownership, number transfer/remapping, stable order,
  and entity/number caps.
- Use one P5 campaign, not a second nested retry system.
- Cut Pro complete-source work from the canonical chunk. Shared acquisition may
  land now; the Pro dossier adapter remains a separately grounded follow-up.
- Define exact future-only source receipt keys and malformed RSS alternative
  behavior.

## Panel claims corrected or rejected

- `p0_source_char_budget()` already computes a concrete budget from the measured
  context/output contract; no guessed 4,000-character constant is needed.
- `SourceSpanV4` already has a `field` discriminator. Facts and entities hold
  `source_spans`; numbers hold `source_span`.
- Every accepted local `FactIndexV4` has a nonempty tone and at least one fact
  by schema.
- `_validate_fact_index` exists and already validates global literal spans,
  number references, the allowed-field set, and expected A0 digest.
- Discarding the middle of a long article or adding a fatal fixed window/cycle
  count directly contradicts the operator's complete-source and candidate-not-
  episode rulings. Operational cost is surfaced and cancellable, not hidden by
  another slice.
- No `CampaignState` object is needed. Accepted P0/P1/P2/P3 values already live
  in sequential local variables; the current invocation alone loops until it
  returns.
- The installed Comfy interrupt inherits `BaseException`, so existing
  `except Exception` blocks do not swallow it. New polls must preserve that.
- RSS alternatives are block-aware stripped before their lengths are compared.

## Final coding rulings

1. First model attempt stays prompt-byte-identical. Retry metadata appears only
   after typed candidate exhaustion.
2. Recover only JSON decode, Pydantic schema, post-validation, and explicitly
   rerollable output-limit exhaustion. Prompt-no-room, provider/config/runtime,
   and deterministic errors remain loud.
3. P5 explicit spoken-safety findings join the P5 post-validator so the normal
   repair ladder/fresh-candidate campaign owns them before ledger assembly.
   The existing terminal cleanup remains a defense and should see an already
   clean candidate.
4. Overlap is 239 characters within the existing measured allowance. If a
   sentence cut would leave no positive step beyond the overlap, use the hard
   allowance cut.
5. Deterministic merge traverses windows and local rows stably. The first
   duplicate fact owns the canonical bundle; validated number evidence from
   later exact duplicate bundles may be transferred and deduplicated onto it.
6. Registry RSS remains bounded by the 2 MiB decoded-fetch owner, expressed as
   at most 2 MiB characters for selected text and a worst-case UTF-8 serialized
   envelope of four times that plus fixed framing. Pinned custom premises keep
   48,000 bytes.
7. The common fetcher may compare RSS and linked article bodies only for the
   existing five-candidate shortlist. This closes the 300-character teaser
   hole without widening the feed crawl.

## Round 2 spend

- Three-model pass: approximately USD 0.1347; GPT exhausted hidden-reasoning
  output space.
- GPT-only medium-reasoning completion: approximately USD 0.1807.
- Total completed Round 2 spend: approximately USD 0.3154.
- Completed campaign total through Round 2: approximately USD 0.8578.
