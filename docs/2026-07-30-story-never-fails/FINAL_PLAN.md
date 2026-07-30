# Story Never Fails / Complete RSS Source - Final Plan

Date: 2026-07-30

## Operator contract

The final production ledger is the audible downstream authority. It must be
structurally valid, internally consistent, safe for every downstream consumer,
and exactly hash/proof coherent. It does not owe semantic fidelity to an
abandoned fictional draft or to the factual article's plot.

The selected article is a factual springboard. Characters, events, dialogue,
and fictional story structure may be replaced completely. Facts represented
as facts and the factual coda remain grounded in the source evidence index.

Recoverable model-output defects retire a candidate, not the episode. There is
no fatal fixed outer candidate count. Cancellation stops the campaign.
Deterministic source/security/configuration/provider/I/O/compiler/invariant/
freeze/proof failures remain loud.

## Complete selected source

`story_orchestrator.py` examines every mapping row in list-valued RSS
`content`, extracts each with the block-aware helper, and keeps the longest
nonempty result; ties keep the earliest zero-based raw index. Malformed rows
still count. A non-list top-level value yields no alternatives.

For each member of the existing five-candidate shortlist, it fetches the
linked article even when RSS text is longer than the former 300-character
shortcut. Selection compares already-clean RSS, URL article, and summary by
character count; ties prefer RSS, URL, summary. Summary route is
`summary_fallback` when a URL exists and `summary_only` otherwise.

The local 12,000-character article slice is removed. The underlying secure
transport remains HTTPS-only and capped at 2 MiB decoded bytes.

Body reranking remains at most 800 characters. Long text uses two five-
character `" ... "` separators and splits the remaining 790 characters into
head 263, centered middle 263, and tail 264.

`meta.news_seed` keeps existing fields and adds:

- `body_source`
- `rss_content_index`
- `rss_content_count`
- `body_bytes_utf8`
- `body_sha256`

The hash is lowercase SHA-256 of the exact selected body encoded as UTF-8.
`body_chars` counts that exact body's Python characters. Summary extraction
and derived `seed_text` are unchanged. `_fetch_rss_seed_or_die` still emits
exactly the public seven string fields. The receipt travels beside that
payload, is validated against the selected body, and is promoted only after
the writer creates the actual episode ledger; it is never stamped into the
temporary pre-ledger singleton that sourcing replaces.

## A0 and complete-source P0

Operator-pinned normalized A0 keeps the 48,000-byte serialized limit. RSS
allows at most 2,097,152 normalized `full_text` characters and at most
8,519,680 serialized normalized A0 UTF-8 bytes. Overage fails loudly; no slice
is taken.

RSS evidence projection is `full_text`, `seed_text`, `headline`, `summary`;
operator-pinned projection keeps historical seed-first order. Exact substring
aliases are omitted. The resulting immutable field allowlist is reused by
every window, repair, local validator, rebase, merge, and final validator.

`p0_source_chunks` keeps its compatible `overlap_chars=0` default. Canonical
RSS uses `MAX_QUOTE_CHARS - 1 == 239`. The exact window-body allowance is the
source budget minus all non-body frame characters. A latter-half sentence
boundary may shorten a window only when the result remains longer than the
overlap. Otherwise the hard end wins. Each next start is `end - overlap`, must
advance, begins at complete-A0 coordinate zero, and the final end equals the
complete body length. The union has no gaps and any 240-character quote is
fully contained in at least one window.

Each window:

1. carries the complete normalized A0 digest;
2. runs one finite P0 candidate ladder/campaign;
3. validates every fact/entity/number span against local text without global
   coordinate relocation;
4. deep-copies and adds the window offset to both bounds of only
   `field == "full_text"` spans;
5. validates those exact coordinates against complete A0 without relocation.

The merge traverses window order then local row order. Fact identity is
normalized claim plus exact span bundle. Entity identity is normalized name
plus exact span bundle. Number identity is exact verbatim/span plus canonical
fact-bundle identity. Local parents are namespaced by
`(window_ordinal, local_fact_id)`.

When surviving windows fit the cap, selection takes one row per window then
fills round-robin. Otherwise it chooses positions
`floor(i * (count - 1) / (limit - 1))`, including both ends. Limits remain six
facts, four entities, and four numbers. IDs are rewritten contiguously and
number parents remapped. Tone is the first accepted local tone; digest is A0's
digest; the merged artifact clears the complete-A0 validator.

P0 sizing reserves 1,024 tokens for a later retry mapping.

## Fresh model-authored candidates

`_invoke_codex_structured_once` owns one finite existing ladder and one journal
entry. `invoke_codex_structured(..., retry_until_valid=False)` is compatible by
default. Canonical P0, P1, P2, P3, and P5 opt in.

Only exhausted terminal causes of these exact kinds start a new candidate:

- `json.JSONDecodeError`
- Pydantic `ValidationError`
- structured `PostValidationError`
- a generation-capacity error for which
  `is_rerollable_capacity_error(error)` is true (`output_limit`)

Every other failure remains loud. Cycle one has no `writer_retry`. Later
cycles add cycle number, a unique 32-hex UUID nonce, error type, a
bounded ASCII rejection summary, and a fixed instruction that the prior
candidate is abandoned and a fresh complete object is required. Pydantic
errors carry only error counts and error codes, never `input_value` text.
Rejected raw output is never included in a fresh prompt or durable journal.

Every outer cycle has its own journal entry. Failed entries contain hashes and
bounded error receipts but no accepted artifact. Only the winning cycle stores
the accepted object.

Comfy cancellation is polled before each cycle and before/after model calls.
Only missing-Comfy `ModuleNotFoundError` is caught around import. No layer
catches `BaseException`.

## P5 and ledger finality

The P5 post-validator projects every line as `line_id`, `speaker_role`, `skip`,
and `text`. It aggregates graph, roster, ID coverage, markup, empty/audibility,
and every explicit spoken-safety finding on the raw compiled draft. It then
canonicalizes spoken text and validates that exact candidate again before
acceptance, so a defect exposed by cleanup retires the candidate instead of
killing the episode afterward. A fresh candidate may be wholly different
fiction.

After a clean candidate:

1. the accepted artifact is already the canonical spoken representation;
2. the existing safety cleanup runs as a defense, makes no model call on
   clean text, and preserves that artifact identity when nothing changes;
3. graph and spoken safety validate the exact post-defense representation;
4. the ledger is assembled once;
5. delivery and authorship are stamped once;
6. freeze, save, reopen, and final line/hash proof run normally.

No model patches an assembled ledger. Rejected prose never enters ledger
metadata. Exact final line and hash checks remain corruption guards.

## Scope and closeout

No public workflow, node, widget, link, registry, schema, prompt-pack, frozen
ledger, or snapshot rebaseline. `workflows/otr_canonical.json` remains
byte-identical.

The common fetcher supplies the complete selected body to clients that share
it. `scifi_news_pro_multipass` still applies its separate 3,600-character
dossier cap; Pro whole-source support remains an explicit follow-up.

Close with focused firing mutations, the full Windows suite using a repo-local
temp root, read-only Bug Bible, variants, UTF-8/no-BOM/nonzero/AST/diff
hygiene, exact workflow hash, exact-path commit and push, and HEAD/origin
equality. Do not run a GPU campaign, headless render, Window B, degrade mode, or
modify the survival-guide repository.
