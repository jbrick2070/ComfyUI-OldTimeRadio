# Codex Grounded Anchor — Round 1

## Verdict

NO. The operator direction is correct, but the draft is not yet safe to
implement. It combines source completeness, evidence aggregation, story
quality, and episode liveness without sufficiently precise ownership and
termination contracts.

## Grounded observations

1. The fetch layer currently loses available article text in several distinct
   ways: `_fetch_full_article` returns only the first 12,000 characters,
   `_fetch_single_feed` examines only the first RSS `content` value,
   `_resolve_body` does not scrape a linked article when inline content exceeds
   300 characters, and the final shortlist is capped before body resolution.
   These are source-acquisition concerns and should be fixed independently from
   model retry policy.
2. `_otr_scifi_p0_contract.py` already has lossless sentence-boundary source
   chunking, but `_otr_scifi_codex.py` has no caller, A0 offset rebaser,
   dependency-preserving merger, or complete-source receipt for it.
3. `validate_payload_envelope` rejects payloads above 48,000 bytes before P0.
   Increasing one ceiling is not enough: complete-source handling must avoid
   silently turning the model context limit into a different head slice.
4. The shared structured-call ladder is deliberately finite and has callers
   outside this story lane. Open-ended episode persistence must be opt-in and
   lane-local.
5. A structurally accepted P5 story is already protected by exact hashes in
   the tail finalizer, but safety cleanup, gap audit, freeze, and ledger
   persistence can still raise. The design must distinguish authored-field
   defects from deterministic configuration, cancellation, security, and I/O
   failures.
6. The bounded FactIndex has cross-row dependencies: numbers reference fact
   IDs. A merge that selects candidate IDs must define how it selects,
   deduplicates, renumbers, and rewrites references in Python without changing
   source claims, verbatim quotes, or coordinates.
7. `scifi_news_pro` has a separate 3,600-character digest and a different
   writer architecture. It shares the operator goal but cannot be declared
   solved merely by adapting the canonical `scifi_news` implementation.

## Must fix before implementation

1. Define episode acceptance precisely. A model candidate remains finite. A
   recoverable model-output failure retires that candidate, not the episode.
   The outer campaign creates a fresh, model-visible nonce, polls cancellation,
   and never leaks a partial artifact. There is no fatal fixed candidate count.
2. Define the non-recoverable set narrowly: cancellation, unavailable or
   invalid source/configuration/model assets, source-security refusal, and
   durable atomic-I/O failure. A malformed model response, exhausted output
   budget, safety wording defect, or authored ledger gap is recoverable.
3. Separate immutable complete-source authority from bounded model views. A0
   owns the digest and global coordinates. Every source character must be
   covered by an offset-receipted window; every accepted quote must be validated
   locally, rebased, and validated again against A0.
4. Specify a deterministic aggregate algorithm. It must preserve literals and
   coordinates, deduplicate candidates, keep number-to-fact references valid,
   prevent first-window bias, and prove that beginning, middle, and tail
   evidence competed for the bounded final dossier.
5. Preserve the best complete valid story while repair continues. Do not let a
   subjective quality reviewer become a new episode-killing gate. Quality can
   rank complete valid candidates or request another candidate; it cannot make
   a valid story disappear or create endless taste-based rejection.
6. Route authored defects to their single owning pass and mechanically rebuild
   the ledger only from accepted immutable artifacts. Failed ledger candidates
   must not mutate the accepted story or the on-disk canonical ledger.
7. Split delivery into reviewable chunks: common full-source acquisition;
   canonical `scifi_news` complete-source P0; lane-local liveness and
   story-preserving ledger repair; then the separately verified
   `scifi_news_pro` adapter.

## Cut or defer

- A universal `retry_until_valid` switch whose exception taxonomy is not
  explicit.
- A prose-quality gate that can reject the only complete valid story.
- Claiming both RSS runners are fixed by identical mechanics.
- Persisting the complete article in downstream output solely for observability
  unless a real consumer requires that storage.
- Any deterministic canned-story fallback. Deterministic mechanics may repair
  serialization and references, but story prose remains source-grounded author
  output.

## Required proof

- Full source survives beyond 12,000 characters and above the former 48,000-byte
  lane ceiling while the 2 MiB fetch-security bound remains intact.
- Every window is covered exactly, offsets rebase to A0, and tail evidence can
  win the bounded dossier.
- Multiple recoverable failures can precede success without rerunning accepted
  earlier passes; cancellation and permanent failures still exit promptly.
- The best valid story remains byte-identical across failed safety/ledger repair
  candidates and saved/reopened ledger audits.
- Frozen ledgers and the canonical workflow remain byte-identical.
