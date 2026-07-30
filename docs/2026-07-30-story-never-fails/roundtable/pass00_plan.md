# Story-First, Complete-Source, Ledger-Until-Valid Plan

## 0. Operator ruling

The story is the primary artifact. The writer may re-author the story and
rebuild the ledger as many times as necessary, but it must not terminate an
episode merely because an authored JSON, graph, spoken row, or ledger candidate
failed validation. Keep working until both the story and ledger are valid.
Prefer the highest-quality source-grounded story available. Read enough of the
selected RSS article to understand the complete story, not an arbitrary head
slice.

This supersedes the finite-then-fail assumptions in the 2026-07-29 writer plan,
including its rejection of an indefinite repair loop. It does not authorize
swallowing ComfyUI cancellation, missing models, invalid configuration,
security refusals, or irrecoverable filesystem errors.

## 1. Existing barriers

1. `story_orchestrator._fetch_full_article` returns `text[:12000]`.
2. `_resolve_body` accepts any inline RSS body over 300 characters without
   checking whether the linked article is more complete.
3. `validate_payload_envelope` rejects a normalized source payload over 48,000
   bytes.
4. `scifi_news` sends the whole P0 evidence projection through one local model
   context. `p0_source_char_budget` and `p0_source_chunks` exist but have no
   caller, merger, deduper, or offset rebaser.
5. `scifi_news_pro` truncates its source digest to 3,600 characters.
6. `structured_call` has a deliberate finite ladder. The lane converts
   exhaustion to a fatal pass error.
7. Post-P5 safety cleanup and ledger/freeze validation can still terminate
   after a structurally valid story candidate exists.

## 2. Source acquisition

Remove narrative character slices, not safety bounds.

- Keep the existing HTTPS, redirect, deadline, MIME, address, and 2 MiB decoded
  network bounds.
- Extract every paragraph and heading from the selected article body; remove
  the 12,000-character return slice.
- Extract every RSS `content` candidate and retain the longest nonempty text,
  with stable first-candidate tie breaking.
- For every shortlisted story with a link, attempt the bounded article scrape
  even when inline RSS text exceeds 300 characters. Choose the richest clean
  body among linked article, inline RSS content, and summary, and receipt its
  source and character count.
- Body reranking may use bounded head/middle/tail samples plus total body
  length, but the selected full body remains immutable and reaches the story
  lane.

## 3. Complete-source evidence

The full normalized source remains A0 and owns the sole digest/coordinate
system. A model context window may see a window; the system must still read
every A0 character.

- Raise the 48,000-byte payload admission ceiling only to the common bounded
  fetch envelope; do not remove bounds.
- Partition only `full_text` on sentence boundaries using the existing
  `p0_source_char_budget` and `p0_source_chunks`.
- Run P0 independently on every window. Validate each candidate against that
  exact window, then rebase `full_text` offsets onto A0 and validate again
  against the full immutable source.
- Hierarchically merge adjacent accepted dossiers so every article window
  competes for the final bounded FactIndex. A merge model returns candidate IDs
  only; Python copies and renumbers already accepted rows. It may not rewrite
  claims, quotes, entities, numbers, or coordinates.
- The final FactIndex remains bounded at 6 facts, 4 entities, and 4 numbers, but
  its candidates came from the entire article. The merge should reward causal
  coverage, human stakes, findings, consequences, and coverage of the article
  ending rather than merely keeping the earliest rows.
- Give `scifi_news_pro` the same complete-source window/merge discipline for
  its dossier instead of the 3,600-character digest.

## 4. Retry until valid

Retry at the owning pass, preserving every earlier accepted artifact.

- Add an opt-in `retry_until_valid` mode to the Sci-Fi lane wrapper, not to all
  generic structured callers.
- A complete finite ladder remains one repair cycle. On recoverable
  parse/schema/post-validation/output-limit exhaustion, start another cycle
  with the original source plus bounded rejection feedback from the latest
  candidate. Never recursively embed prior prompts.
- Poll ComfyUI cancellation before every new cycle and let its real exception
  propagate untouched.
- Do not retry permanent prompt-no-room arithmetic, missing model/backend,
  invalid bank/pack/schema configuration, source-security refusal, or durable
  I/O corruption. These are not story-authoring failures.
- Record every cycle, owner, backend, rejected-candidate hash, rejection class,
  and final accepted cycle in the live ledger. Save progress receipts between
  cycles without accepting a partial story or ledger.
- P0 retries P0 only; P1 retries P1; P2 retries P2; P3 retries P3; P5 retries
  P5. Earlier accepted source facts, dramatic question, cast, and score remain
  unchanged while the current owner repairs its artifact.

## 5. Story quality and ledger integrity

- Never replace a source-grounded story with canned prose or a generic
  deterministic utterance merely to make the ledger pass.
- P5 safety or spoken-surface rejection returns to P5 with all findings in one
  bounded request. A rejected candidate is retained only as a repair input and
  hash receipt, never as ledger authority.
- Once a story candidate is structurally valid, assemble a fresh ledger from
  that immutable candidate. If a gap audit identifies an authored field defect,
  route the complete finding to the pass that owns that field and rebuild from
  the resulting accepted artifact. Pure ledger serialization/path/save defects
  remain technical errors rather than creative rewrites.
- Add a story-quality review that can request a source-grounded P5 rewrite for
  concrete craft defects. It cannot reject for word count, vocabulary taste,
  or genre preference. The last structurally valid candidate remains available
  while a better rewrite is attempted; only a validated improvement becomes
  story authority.
- The accepted story text, its ledger rows, line hashes, authorship receipt,
  saved/reopened ledger, captions, and TTS projection must all agree exactly.

## 6. Scope and wiring

- Apply the common fetch/body-selection fix to clients sharing
  `_fetch_science_news`.
- Apply complete-source and retry-until-valid behavior to both shipped RSS
  story runners: `scifi_news_circuit` and `scifi_news_pro_multipass`.
- No node, widget, link, input, output, or canonical workflow JSON change is
  required. If implementation proves otherwise, update and validate
  `workflows/otr_canonical.json` in the same change.
- Frozen ledgers, snapshots, hashes, and fixtures are never migrated or
  re-pinned.

## 7. Verification

1. Full-body extraction proves text beyond character 12,000 survives.
2. Multiple RSS content candidates prove the richest complete body wins.
3. A linked article longer than acceptable inline RSS proves the linked body
   wins; the converse proves inline RSS remains when richer.
4. Multi-window P0 proves a decisive fact in the final article window can win
   the final index and keeps literal A0 coordinates.
5. Merge mutation tests catch skipped windows, missing offset rebasing,
   rewritten candidate rows, broken number-to-fact references, and first-window
   bias.
6. Retry tests fail twice then succeed, proving accepted earlier artifacts are
   not regenerated and cancellation escapes immediately.
7. Safety/ledger rejection returns to the owning story pass and never accepts a
   partial ledger.
8. Run focused tests, full Windows suite, read-only Bug Bible, variants,
   UTF-8/no-BOM/nonzero/AST/diff hygiene, canonical workflow hash, exact-path
   commits, immediate pushes, and final HEAD == origin.

## 8. Explicit exclusions

- No GPU campaign or headless render in the coding chunk.
- No frozen-artifact migration.
- No security-bound relaxation.
- No silent fallback to another story pipeline.
- No unbounded prompt growth: the number of cycles may be open-ended, but every
  individual prompt, source window, output, and repair handoff remains bounded.
