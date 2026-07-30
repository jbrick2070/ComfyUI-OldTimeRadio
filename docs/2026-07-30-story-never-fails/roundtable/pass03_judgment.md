# Round 3 Judgment - Wiring and Persistence

## Decision

ACCEPT WITH CORRECTIONS.

The two successful reviewers agreed that no workflow, node, widget, registry,
pack, or public seven-field source-payload change is required. DeepSeek did not
return usable content before its output limit, so it contributes no finding to
this round.

## Accepted corrections

1. Canonicalize spoken text before the final post-cleanup graph and safety
   validation. The exact bytes assembled into the ledger must be the bytes that
   passed.
2. Compute the authoritative P0 field allowlist once from the complete A0
   projection. Every window, repair, local validator, rebase, and merge uses
   that same allowlist; windows are never independently de-aliased.
3. Reserve the maximum outer-retry mapping in P0's source-window budget.
   Cycle one must leave enough room for cycle two's bounded rejection receipt.
4. Namespace number-parent lookup by `(window_ordinal, local_fact_id)` and map
   it through a concrete canonical fact-bundle identity.
5. Add the window offset to both `start` and `end`, and only when
   `field == "full_text"`.
6. Treat a non-list top-level RSS `content` value as no alternatives:
   `("", None, 0)`. Malformed rows inside a list still count toward raw count.
7. Define summary routes from URL presence: a summary selected for a candidate
   with a URL is `summary_fallback`; without a URL it is `summary_only`.
8. Catch only missing-Comfy `ModuleNotFoundError` around the lazy import.
   Polling itself is outside that catch. Installed Comfy interruption inherits
   `BaseException`, and no layer catches `BaseException`.
9. Give every outer candidate cycle its own journal entry. Failed entries have
   hashes and bounded rejection metadata but no accepted artifact; only the
   accepted cycle contains the accepted object.
10. When overlap deduplication leaves a sampled window without a surviving
    row, select deterministically from the surviving-window collection itself.

## Rejected or already satisfied findings

- `frame_chars` is already computed locally by `p0_source_chunks`; it is not an
  undefined variable.
- A fixed outer retry ceiling conflicts with the operator's explicit ruling.
  Finite work remains inside each candidate; cancellation is the exit from a
  persistently bad model-output campaign.
- The five article fetches already run through a bounded five-worker executor,
  preserving shortlist order through `executor.map`.
- Full-A0 validation after rebasing already rejects any coordinate or field
  drift; no second coordinate search is permitted during rebase.

## Final wiring boundary

Private fetch metadata stays underscore-prefixed until copied into additive
`meta.news_seed` receipt fields. `_fetch_rss_seed_or_die` continues to emit
exactly seven strings. Whole-source P0 and fresh-candidate recovery are enabled
only by the canonical `scifi_news_circuit` runner. `scifi_news_pro_multipass`
still has its separate 3,600-character dossier cap and is not claimed solved.
