# Codex Grounded Anchor — Round 3

## Verdict

YES-WITH-FIXES. Revision 2 requires no LiteGraph, node, widget, registry, or
public payload-schema change. The wiring risk is accidental widening through
shared fetch dictionaries and helper defaults.

## Grounded wiring facts

1. `workflows/otr_canonical.json` already selects `scifi_news`, which routes to
   `scifi_news_circuit`. No new input or output is needed.
2. The public source payload remains the same seven strings:
   headline, summary, full_text, source, date, link, and seed_text.
3. RSS alternative index/count and body route/hash are private fetch metadata
   and additive `meta.news_seed` fields. They must not become node inputs,
   widgets, or required source-payload keys.
4. `retry_until_valid` is a Python-only helper argument with a false default.
   Only the canonical runner opts in; no story-pipeline registry field is
   added.
5. Window overlap, rebasing, and merge are private P0 mechanics. Existing
   `FactIndexV4` and downstream P1/P3/P5 artifact shapes remain unchanged.
6. No new prompt seam is needed because merge is deterministic. Therefore
   `story_packs/pipelines.json`, pack allowlists, and `scifi_news.json` remain
   unchanged.
7. The shared fetcher changes the selected `full_text` supplied to every client
   using `_fetch_science_news`. `scifi_news_pro` still applies its existing
   3,600-character digest cap, so this chunk must not claim Pro whole-source
   support.

## Must pin in implementation

- Keep internal fetch metadata underscore-prefixed until it is copied into
  additive ledger metadata. `_fetch_rss_seed_or_die` must continue constructing
  only the seven-key payload.
- Preserve summary extraction and derived `seed_text` behavior. Only selected
  `full_text` and source receipts change.
- Keep helper defaults backward-compatible:
  `overlap_chars=0`, `retry_until_valid=False`.
- Do not add a required ledger top-level key. New receipts live under existing
  `meta.news_seed` and `meta.scifi_codex` mappings.
- Do not edit `banks.json`, `pipelines.json`, source packs, INPUT_TYPES,
  RETURN_TYPES, workflow nodes, links, or positional widgets.
- Run the workflow hash check and variants check as firing guards.
- Test both direct helper callers and the real nested feed/P0 production
  invocation so an unwired implementation cannot pass.

## No rebaseline

Existing frozen ledgers, snapshots, fixture hashes, and workflow hashes are not
re-pinned. New source/body digests and coordinates apply only to future live
RSS runs.
