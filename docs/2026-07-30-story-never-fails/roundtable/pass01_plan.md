# Story-First Writer Liveness and Complete-Source Plan — Revision 1

## 0. Acceptance contract

The final production ledger is the executable authority for downstream
consumers. Earlier story drafts are working material and may be replaced.

- P1-P5 may freely invent and rewrite SFW characters, places, events, conflict,
  dialogue, and dramatic structure.
- The complete selected article is the authority only for real science claims,
  literal evidence, and the factual closing note.
- A writer candidate is accepted when it compiles against the accepted graph,
  its spoken text is safe and nonempty, and Python can assemble one internally
  coherent ledger from it.
- Requested word count, prose taste, article-plot fidelity, and fidelity to a
  rejected draft are not acceptance gates.
- Final line text, line hashes, TTS projection, authorship receipt, saved
  ledger, and reopened ledger still agree for the accepted candidate. This is a
  corruption proof, not a prior-draft fidelity rule.

## 1. Operational definition of complete source

“Complete” means all supported text in the selected static source that OTR
successfully acquired within the existing fetch contract:

- an RSS `content[*].value` fragment after block-aware extraction; or
- paragraphs and h2/h3 headings in the first selected article container of the
  bounded static HTML response.

It does not claim to recover paywalled, paginated, script-rendered, image-only,
table-only, or otherwise unavailable material.

The existing network contract remains unchanged: HTTPS only, public addresses,
bounded redirects and deadlines, accepted MIME types, and at most 2 MiB of
decoded response bytes.

The normalized canonical source has a separate explicit envelope:

- registry-owned RSS `full_text` UTF-8 bytes must not exceed 2 MiB;
- its complete serialized seven-field normalized A0 payload must not exceed
  2 MiB plus 128 KiB of framing;
- the existing 48,000-byte serialized limit remains unchanged for
  `operator_pinned` custom premises;
- exceeding either bound is a source/configuration refusal, never a silent
  trim.

## 2. Common RSS and article acquisition

1. Extract every `entry.content[*].value` with
   `_extract_rss_fragment_text`.
2. Choose the longest nonempty extracted RSS alternative, with stable
   first-alternative tie breaking. Record the chosen zero-based alternative
   index and total alternative count.
3. For each candidate in the existing bounded body-resolution shortlist,
   attempt the linked static article fetch even when RSS content exceeds 300
   characters.
4. Choose the longest cleaned body among RSS content, linked article, and
   summary. Ties prefer RSS, then linked article, then summary.
5. Never discard a nonempty shorter RSS body merely because a linked fetch is
   unavailable or below a quality floor.
6. Remove the 12,000-character article slice only after the canonical lane can
   consume whole sources in windows.
7. Replace the body reranker's first-800-character view with a bounded
   head/middle/tail preview. The complete chosen body remains untouched.
8. Keep the current five-candidate network shortlist for this build. “Best”
   means the strongest candidate selected within that explicit bounded
   shortlist; widening the feed crawl is a separate performance/product
   decision.

Stamp the future-only live ledger source receipt with:

- selected body route;
- RSS alternative index/count when applicable;
- selected clean body characters and UTF-8 bytes;
- SHA-256 of that exact selected clean body (the later A0 normalization and
  seven-field digest remain separate receipts);
- existing headline/source/url/date/selection time.

Do not rewrite old ledgers or snapshots.

## 3. Canonical whole-source P0

Normalized A0 remains immutable and owns the sole final digest and coordinate
system.

### 3.1 Map

1. For RSS, make `full_text` the canonical chunk-owned evidence field. Retain
   non-duplicate headline and summary framing; do not let derived `seed_text`
   hide or duplicate the article body. Preserve the existing pinned-premise
   projection behavior.
2. Use `p0_source_char_budget` and adapt `p0_source_chunks` to overlap adjacent
   `full_text` windows by `MAX_QUOTE_CHARS - 1`. A legal 240-character literal
   quote must be fully visible in at least one window even when it crosses the
   nominal cut.
3. Every `(offset, window_payload)` runs one finite structured P0 candidate
   ladder.
4. Validate each accepted window result against that exact window.
5. For a `full_text` span only, add the window offset to `start` and `end`.
   Other source fields retain their A0 coordinates.
6. Replace the window result's digest with the immutable A0 digest and validate
   the rebased result again against A0 before it can enter aggregation.
7. Record window start/end, overlap, local result hash, rebased result hash, and
   acceptance cycle. The union of windows must cover every A0 `full_text`
   character; no gap is allowed.

### 3.2 Merge

Use a pure deterministic merge. A second model selector would introduce
another failure surface where no authorship is needed.

- Treat a fact and its eligible number rows as one window-owned bundle keyed by
  `(window_ordinal, old_fact_id)`.
- Deduplicate facts by normalized claim plus exact rebased literal-span
  identity; entities by normalized name plus exact rebased literal-span
  identity; and numbers by verbatim value, exact rebased span identity, and
  parent fact bundle.
- Preserve each local P0 result's ordering: the evidence model has already
  ranked the strongest rows inside that window.
- When the number of nonempty windows is within a schema cap, take one row from
  each window, then fill remaining slots round-robin from those windows.
- When nonempty windows exceed a schema cap, choose evenly spaced window
  ordinals including the first and last and take the first surviving row from
  each selected window. This gives beginning, middle, and ending evidence a
  deterministic share without head bias.
- Copy the selected immutable rows, assign `F01..F06`, `E01..E04`, and
  `N01..N04`, retain numbers only for selected fact bundles, and remap every
  selected number's parent `fact_id` through the new fact-ID map.
- Use the first accepted nonempty local tone, assign the immutable A0 digest,
  and run `_validate_fact_index` against A0 one final time.

The final FactIndex remains bounded, but every source window was read by P0 and
the deterministic policy includes article-tail evidence without adding another
LLM gate.

No separate subjective story-quality reviewer is added. The existing creative
seams use the resulting article-wide factual choices as inspiration, and the
operator's fictional latitude remains intact.

## 4. Fresh model candidates until valid

Keep `_otr_structured_call.structured_call` finite and unchanged for all other
callers. Add an opt-in canonical-lane outer campaign around
`invoke_codex_structured`; use it for each P0 window and for P1, P2, P3, and P5.
An accepted upstream pass remains fixed while only the current pass creates a
fresh candidate. Add one additional P5-owned loop around post-authoring safety
cleanup.

Each campaign cycle:

1. Poll ComfyUI cancellation.
2. Create a unique model-visible cycle number and nonce.
3. Invoke the current pass's existing complete finite structured ladder with
   its immutable accepted inputs and bounded deduplicated findings from the
   previous rejected cycle.
4. Accept only the pass's existing schema and post-validator result.
5. If the finite ladder raises typed `StructuredCallFailedError`, discard that
   candidate and begin a fresh cycle. Any other exception exits immediately.
6. For P5, additionally compile against the accepted P3 score, require the
   exact line-ID and spoken-surface contracts, run technical safety cleanup,
   and rescan it. A residual authored-text safety defect starts another P5
   candidate with the complete bounded finding list.
7. On success, return the accepted artifact. A new P5 candidate may replace all
   fictional prose.

Recoverable cycle failures are narrowly typed:

- exhausted JSON/schema/post-validation/output-limit ladder at any P0-P5
  model-owned pass;
- missing, duplicate, extra, or malformed P5 line rows;
- empty, action-only, label-prefixed, production-markup, or unsafe spoken
  text;
- a safety-cleanup residual attributable to authored text.

Do not retry:

- cancellation;
- missing/invalid prompt pack, schema, model, or backend configuration;
- a provider/runtime outage rather than a returned bad candidate;
- ambiguous or invalid accepted P3 graph/compiler state;
- source-security refusal;
- ledger serializer, voice configuration, atomic save, reopen, disk, hash, or
  freeze-proof failure.

There is no fatal fixed outer model-candidate count. Individual candidates and
feedback prompts remain bounded. The final ledger records per-pass cycle totals
and a bounded hash/error summary; raw rejected drafts are not persisted.

## 5. Ledger boundary

Do not ask an LLM to patch the production ledger and do not use generic cleanup
to mint/deduplicate Sci-Fi line IDs.

1. Generate and validate P5 candidates before `_assemble_ledger` mutates the
   ledger.
2. Once P5 succeeds, assemble from the accepted P2/P3/P5 artifacts exactly
   once.
3. Stamp delivery telemetry and authorship from that final candidate.
4. Run the existing cleanup, gap audit, freeze, atomic save, reopen, and exact
   line/hash proof.
5. Warnings remain warnings. Deterministic hard errors remain loud because
   another fictional rewrite cannot repair code, configuration, voice stock,
   or disk integrity.

The ledger need not semantically match the premise, article plot, or rejected
drafts. Its own IDs, references, cast, roles, spoken lines, TTS projections,
clips, timing writeback surfaces, hashes, and serialization must be internally
coherent.

## 6. `scifi_news_pro` boundary

The common body acquisition and provenance improvements apply immediately
because the Pro runner shares the fetcher.

Complete-source dossier aggregation and open-ended markup-writer candidates
require a separate Pro adapter:

- window its 3,600-character dossier input without changing its schema;
- validate/filter each local dossier against its source window;
- deduplicate and consolidate the bounded dossier fields with explicit caps;
- give later factual-read passes only admitted dossier facts/numbers/entities;
- wrap its complete finite markup ladder in fresh outer writer candidates,
  while leaving assembly/save failures loud.

This adapter is a separate implementation chunk with separate tests. The
canonical `scifi_news` chunk must not claim to solve it.

## 7. Tests and mutations

### Common source

- content alternative 0 is a teaser and alternative 1 contains a tail sentinel;
- stable first-alternative tie;
- linked article wins when longer; RSS wins when longer or on a tie;
- a failed linked fetch preserves short nonempty RSS content;
- text beyond 12,000 characters survives;
- head/middle/tail preview contains a tail sentinel;
- source receipt route/index/count/character/byte/hash fields are exact.

### Canonical P0

- a normalized payload above the old 48,000-byte cap is admitted within the new
  envelope;
- overlapping windows cover the complete body without gaps and a legal
  boundary-spanning 240-character quote is fully visible;
- local spans rebase to literal A0 spans;
- duplicate rows collapse;
- deterministic over-cap selection includes beginning, middle, and final
  windows;
- fact renumbering remaps every number reference;
- final digest is A0 and final `_validate_fact_index` passes;
- mutations that skip a window, omit rebasing, retain the old cap, break
  overlap, break number remapping, or restore the 12,000-character slice make
  tests fail.

### Model and P5 writer liveness

- two exhausted/malformed candidates followed by success do not kill the
  episode;
- every cycle has a different model-visible nonce and receives all bounded
  findings;
- a failed current pass does not regenerate accepted upstream artifacts;
- a safety residual triggers another P5 candidate;
- a replacement candidate may use entirely different fictional prose;
- cancellation escapes immediately;
- configuration, provider outage, invalid graph, save, reopen, and proof errors
  are not retried;
- `_assemble_ledger` runs once, after candidate acceptance;
- final ledger hashes match the accepted final candidate.

### Gates

Focused tests and firing mutations, full Windows suite with repo-local temp
root, read-only Bug Bible, variants, UTF-8/no-BOM/nonzero/AST/diff hygiene,
canonical workflow hash, exact-path commits, immediate pushes, and final
`HEAD == origin/v2.0-alpha`.

No GPU campaign, headless render, Window B, degrade mode, workflow edit, frozen
artifact migration, or survival-guide mutation.
