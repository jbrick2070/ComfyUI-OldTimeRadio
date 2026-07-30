# Round 1 Judgment — High-Level Arc

## Verdict

NO on `pass00_plan.md`. The direction is approved, but the first draft was too
broad and imposed a stronger story/ledger fidelity contract than the operator
wants. The build must separate three concerns:

1. retain and inspect the complete available selected source within the
   existing security envelope;
2. let the fictional writer freely re-author failed drafts until it emits a
   downstream-safe candidate;
3. assemble, freeze, save, and reopen the production ledger once from the final
   accepted candidate.

## Grounded panel findings accepted

- Cut the subjective story-quality review. It could create a taste-based
  episode-killing loop and is not needed for ledger correctness.
- Define complete source operationally as the selected static fetched
  response/container and supported text elements, not pagination, client-side
  rendering, or inaccessible paywalled content.
- Define the normalized A0 byte envelope separately from the 2 MiB decoded
  network envelope.
- A multi-window FactIndex merge must deduplicate rows, rebase `full_text`
  coordinates onto A0, revalidate against A0, renumber accepted facts, and
  remap every `NumberV4.fact_id`.
- Treat `scifi_news_pro` as a separate adapter. It does not share the canonical
  FactIndex schema.
- Record future-only source route, RSS alternative index/count, body size, and
  body digest.
- Do not route post-assembly deterministic serializer/freeze/save errors back
  into a creative writer. Catch model-owned defects before ledger mutation.
- Keep the final line/hash/authorship proof. It protects the final accepted
  ledger from corruption; it does not require fidelity to a rejected draft.

## Panel findings corrected or rejected

- The DeepSeek claim that sentence-boundary windowing does not exist is false.
  `p0_source_chunks` searches the latter half of each bounded window for
  sentence endings before falling back to a hard character boundary.
- Gemini's P3/P5 ownership claim was based on an imprecise grounding sentence.
  Production code shows P3 authors the score and closed line graph; P5 is the
  sole spoken-text writer. Spoken-surface and safety re-authoring belongs to P5
  while the accepted P3 graph remains valid.
- A durable cross-process checkpoint store, retry backoff subsystem, and
  heartbeat service are not required for this coding slice. The host already
  owns execution and cancellation. A bounded in-memory attempt journal can be
  stamped into the final ledger after acceptance.
- The writer does not need a factual reconstruction of the article's plot.
  The operator explicitly grants fictional latitude. Every source window must
  be considered for factual evidence and the coda; P1-P5 may invent and rewrite
  the drama.
- Repository gates remain in the execution checklist because project policy
  requires them, even though they are not product architecture.

## Operator clarifications incorporated

- Earlier drafts are disposable. The writer may replace their plot, people,
  events, and wording.
- The final production ledger is the downstream authority. It must be
  internally executable and safe; it need not preserve an abandoned story
  draft or match article plot semantics.
- Complete source access constrains real factual claims and the factual coda,
  not fictional drama.
- Additional passes are ordinary writer repair, not a degradation mode or a
  canned fallback.

## Round 1 spend

- Initial oversized pass: approximately USD 0.2782. Only Gemini returned text,
  and that review was discarded because it reviewed historical grounding as if
  it were the proposal.
- Focused three-model pass: approximately USD 0.2642.
- Total completed Round 1 spend: approximately USD 0.5424.
