# Story Liveness and Complete RSS Source - Revision 3

## Contract

- The final ledger is the audible downstream authority.
- Abandoned fictional drafts have no fidelity claim.
- Factual evidence is drawn from the complete selected static source.
- JSON/schema/postvalidation/output-limit defects retire a candidate, not the
  episode.
- Cancellation and deterministic source, security, configuration, provider,
  filesystem, freeze, and proof failures remain loud.
- Assembly, authorship stamping, save, reopen, and proof run once.

## Common source acquisition

In `story_orchestrator.py`:

- inspect every mapping row in a list-valued RSS `content` collection;
- choose the longest block-aware extracted alternative, earliest on ties;
- retain its raw index and the raw list count;
- fetch the linked article for every member of the existing five-candidate
  shortlist;
- choose the longest of RSS, article, and summary, with tie order RSS, article,
  summary;
- preserve all extracted article text within the existing 2 MiB secure-fetch
  bound;
- rerank with a deterministic head/middle/tail preview of at most 800 chars;
- stamp route, RSS index/count, exact chars, UTF-8 bytes, and exact body SHA-256
  under `meta.news_seed`.

Summary extraction and derived `seed_text` stay unchanged.

## A0 admission and P0 windows

- Operator-pinned A0 keeps the 48,000-byte cap and seed-first projection.
- RSS A0 allows up to 2 MiB normalized `full_text` characters and the ratified
  serialized envelope bound; it is never sliced.
- RSS projection is full-text-first. Its field allowlist is computed once.
- `p0_source_chunks(..., overlap_chars=0)` remains backward-compatible.
- Canonical production uses `MAX_QUOTE_CHARS - 1` overlap.
- The source budget reserves the maximum bounded outer-retry receipt.
- Each window validates locally, rebases both bounds of full-text spans, then
  validates against complete A0.

The deterministic merge namespaces local IDs by window, deduplicates exact
fact/entity/number bundles, samples surviving windows evenly when caps require
it, assigns contiguous IDs, remaps number parents, keeps the first tone and A0
digest, and clears the complete-A0 validator.

## Fresh candidate campaign

`_invoke_codex_structured_once` owns one finite existing ladder and one journal
entry. `invoke_codex_structured(..., retry_until_valid=False)` is the
compatibility wrapper. Canonical P0/P1/P2/P3/P5 opt in.

Only these exhausted terminal causes start a fresh complete candidate:

- JSON decode error;
- Pydantic validation error;
- structured postvalidation error;
- rerollable `output_limit` capacity error.

Cycle one is byte-identical to the old prompt. Later cycles add one small
`writer_retry` mapping with cycle, unique UUID nonce, error type, collapsed
bounded rejection, and an instruction that the prior candidate is abandoned.
Rejected raw output is never included.

There is no fatal fixed outer count. Comfy cancellation is polled at cycle and
model-call boundaries and escapes by identity.

## Final story and ledger

P5 reports graph, roster, markup, empty/audibility, and explicit spoken-safety
findings through the same candidate campaign. After acceptance:

1. run the existing safety cleanup as defense;
2. canonicalize spoken text;
3. validate graph and safety against that exact canonical representation;
4. assemble once;
5. stamp delivery and authorship once;
6. freeze, save, reopen, and prove exact final hashes normally.

No model patches an assembled ledger, and no rejected prose enters metadata.

## Scope and proof

No workflow, node, widget, link, registry, pipeline, pack, schema, or frozen
artifact changes. Pro whole-source support remains a separate follow-up.

Focused tests and firing mutations cover source choice, full tails, receipts,
window union/overlap/rebase/merge, unlimited recoverable cycles, permanent
failure classification, cancellation, unsafe-candidate replacement, divergent
fiction acceptance, and assemble-once identity. Finish with the full Windows
suite, read-only Bug Bible, variants, hygiene, workflow hash, exact-path
commit/push, and HEAD/origin equality.
