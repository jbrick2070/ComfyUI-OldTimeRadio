# Pass 09 judgment -- canonical ledger text-metric ownership

## Verdict

Proceed to live requalification. The live `scifi_news` warning was a confirmed
cross-bank ownership defect: the stored row count followed the producer regex,
while freeze and role aggregates used whitespace splitting. The root fix now
gives every durable text/count surface one lexical owner and preserves the raw
Phase-0 diagnosis before self-healing the final state.

Antigravity remained review-only on exact `gemini-3.5-flash-high`. Sol grounded
the review against the real Windows files and was the sole driver, coder, and
judge.

## Grounded findings

Accepted:

- `production_ledger.stamp_word_counts` disagreed with the row producer.
- all six banks share that aggregate/freeze tail and require the same coverage;
- `_recompute_totals` must restamp every row before rolling up aggregates;
- `_otr_ledger_freeze._check_per_line_invariants` must consume the same helper.

Corrected or expanded by Sol:

- live `l003.word_count=21` was canonical; the split-derived value 20 was the
  false result;
- the sibling surface also included readiness, scrub, reviewer, genre/outro,
  writer cleanup, anti-loop, radio editor, story-spine, reroll, and Scifi Codex
  durable mutations;
- a new dependency-free `_otr_text_metrics.py` leaf is safer than placing the
  helper in the higher-level scrub module suggested by the reviewer;
- the final refresh belongs after every permitted text mutator and before
  Phase 10, while Phase 0 must retain the incoming producer diagnosis.

Discarded:

- canonical story budgets must not count `text_for_tts`; that field is a
  delivery projection, not content ownership;
- row identity already hashes raw canonical text and invalidates punctuation
  changes; derived-count refresh must not rebuild authorship hashes or seals;
- the review's double-ASCII-hyphen example contradicted its em-dash wording.
  The grounded contract keeps ASCII hyphens intra-word and treats en/em dashes
  as punctuation boundaries, matching the live sentence.

## Implemented root fix

- one stdlib-only helper owns straight/smart apostrophes, ASCII hyphens, en/em
  dash boundaries, character counts, row restamping, and atomic text updates;
- every confirmed production ledger-text writer uses the atomic setter;
- every save re-derives row, cast, scene, root, and character/announcer totals,
  including clearing aggregates that no longer own rows;
- the freeze cascade performs a count-only refresh at the final text boundary;
- an AST regression guard rejects new direct production `row['text']` writes.

No canonical workflow edit was needed: this is shared Python ownership behind
the already-wired six-bank tail.

## Verification receipts

- affected and sibling focused suites: 509 passed, then 293 passed;
- ownership guard: 3 passed;
- Phase-0/Phase-10 plus retired-label guard: 70 passed;
- full project suite: 8,315 passed, 33 skipped, 1 expected failure;
- Bug Bible standard gate: 17 passed, 20 skipped, 3 expected failures;
- BUG-12.67 executable OTR gate: 1 passed;
- canonical workflow gate: 95 passed, 2 skipped;
- JSON round-trip and link referential-integrity audit: green (23 nodes,
  58 links, no duplicate node IDs or invalid link endpoints);
- canonical workflow SHA-256 remained
  `f9d9c2c3a101ec607c9658456f6e191a164d8214be7b6d560bc68975d0511e9a`.

Live qualification remains the final gate. Any new live functional defect
again stops the run and triggers a targeted exact-Flash-High sibling audit
before qualification continues.
