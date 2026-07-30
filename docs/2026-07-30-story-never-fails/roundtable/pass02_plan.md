# Story Liveness and Complete RSS Source — Revision 2

## 0. Final behavior

- The final production ledger is the downstream authority.
- Rejected fictional drafts have no preservation claim. A fresh candidate may
  replace all invented plot, characters, events, and dialogue.
- Real factual claims and the factual coda use a bounded FactIndex derived from
  every window of the complete selected static source.
- Model-output defects retire a candidate, not the episode.
- Cancellation and deterministic source/configuration/provider/graph/compiler/
  voice/filesystem/freeze/hash failures remain loud.
- The ledger is assembled and saved once from accepted artifacts.

## 1. Common source acquisition

### `nodes/story_orchestrator.py`

Add these private helpers:

- `_select_rss_content(entry) -> (text, raw_index_or_none, raw_count)`
- `_select_news_body(candidate, fetched_body) -> (text, route)`
- `_body_rerank_preview(text, limit=800) -> str`

Rules:

1. `raw_count` is the length of the raw `content` list. Iterate it by raw
   zero-based index. A null, non-mapping, missing, non-string, or extraction-
   failing value counts toward `raw_count` but contributes empty text.
2. Extract each usable value with `_extract_rss_fragment_text`. Choose the
   longest nonempty extracted text; a tie keeps the earliest raw index.
3. `_fetch_single_feed` stores `rss_full`, `_rss_content_index`, and
   `_rss_content_count`.
4. `_resolve_body` fetches the linked static article for each member of the
   existing five-candidate shortlist, even when RSS text exceeds 300
   characters. It does not widen that shortlist.
5. Compare the already-clean RSS text, already-clean article text, and existing
   summary text by character count. Tie priority is RSS, article, summary.
   A failed linked fetch never deletes nonempty RSS.
6. `_fetch_full_article` returns all extracted p/h2/h3 text from the selected
   container; remove only `[:12000]`. Keep the 2 MiB network seam unchanged.
7. `_body_rerank_preview` preserves the old 800-character maximum. Short text
   is unchanged. Long text gets deterministic head/middle/tail slices after
   subtracting two fixed separator lengths from the budget.
8. Stamp `meta.news_seed` with the existing keys plus:

```text
body_source: "rss_full" | "url_scrape" | "summary_fallback" | "summary_only"
rss_content_index: int | null
rss_content_count: int
body_bytes_utf8: int
body_sha256: 64 lowercase hex
```

`body_chars` remains the number of characters in the exact selected clean body.
`body_sha256` hashes that exact body. It is intentionally distinct from the
later normalized seven-field A0 digest.

## 2. Payload admission and P0 projection

### `nodes/_otr_scifi_codex.py`

`validate_payload_envelope` already derives the policy from
`resolved["seed_source"]`:

- `custom_premise` / `operator_pinned`: retain the 48,000-byte serialized cap;
- `rss_fetch` / `rss`: allow at most 2 MiB characters in normalized
  `full_text`, and at most `(4 * 2 MiB) + 128 KiB` in serialized normalized A0.

Any overage raises `CodexPayloadOversizeError`; never slice.

For `_p0_evidence_projection`:

- RSS order is `full_text`, `seed_text`, `headline`, `summary`, retaining each
  value only when it is not already an exact substring of a retained value.
  This guarantees the chunk-owned body cannot be hidden by an alias.
- Operator-pinned order stays byte-for-byte as it is now.

## 3. Overlapping source windows

### `nodes/_otr_scifi_p0_contract.py`

Extend:

```python
p0_source_chunks(payload, *, budget_chars, overlap_chars=0)
```

Keep `overlap_chars=0` backward-compatible.

Production uses `MAX_QUOTE_CHARS - 1`:

1. `allowance = budget_chars - frame_chars`.
2. Require `0 <= overlap_chars < allowance`.
3. `start` begins at zero; `hard_end = min(len(body), start + allowance)`.
4. A latter-half sentence boundary may shorten `end`.
5. If `end - start <= overlap_chars`, ignore the sentence cut and use
   `hard_end`.
6. Emit `(start, payload_with_body[start:end])`.
7. If `end == len(body)`, stop. Otherwise the next start is
   `end - overlap_chars`; assert it is greater than the previous start.

The first start is zero, the final end is `len(body)`, every step is positive,
the union has no gap, and no window exceeds the measured allowance.

## 4. Windowed P0 and deterministic merge

Add pure helpers in `nodes/_otr_scifi_codex.py`:

- `_rebase_p0_index(index, *, full_text_offset, a0_digest)`
- `_merge_p0_indices(indices, *, a0_payload, allowed_source_fields, a0_digest)`
- `_evenly_spaced_indices(count, limit)`

Factor the existing P0 prompt/repair closures into one
`_invoke_p0_window(...)` so the prompt evidence, allowed fields, literal
deterministic repair, and local validator all use the same window.

For each window:

1. Run P0 with the normal P0 pass ID and finite internal ladder.
2. Validate against its local payload and immutable A0 digest.
3. Deep-copy and add the window start only to spans whose
   `SourceSpanV4.field == "full_text"`:
   - every `FactV4.source_spans` member;
   - every `EntityV4.source_spans` member;
   - every `NumberV4.source_span`.
4. Validate the rebased result against full A0. Never search globally for a
   quote during rebasing; the window occurrence owns its coordinates.

Merge traversal is window order, then local row order:

1. Fact duplicate identity is
   `(whitespace-collapsed-casefolded claim, field, start, end, quote)`.
2. Entity duplicate identity is
   `(whitespace-collapsed-casefolded name, field, start, end, quote)`.
3. Number duplicate identity is
   `(verbatim, field, start, end, quote, canonical fact-bundle identity)`.
4. First duplicate fact owns the canonical bundle. Later exact duplicate
   bundles do not add a fact, but their locally validated numbers may transfer
   to that canonical bundle and are deduplicated.
5. Fact selection:
   - if nonempty window count is at most six, take one surviving fact per
     window, then fill round-robin in window order;
   - otherwise choose six unique indices
     `floor(i * (count - 1) / 5)` for `i=0..5` and take each chosen window's
     first surviving fact.
6. Entity selection uses the same policy with cap four.
7. Numbers are traversed by selected fact order then stable candidate order;
   retain at most four, only when their canonical parent fact was selected.
8. Assign contiguous F/E/N IDs. Rewrite every retained number's `fact_id`
   through the canonical fact-ID map.
9. Tone is the first local accepted tone. Local schema guarantees it is
   nonempty.
10. Assign A0 digest and run final `_validate_fact_index` against full A0.

The short-source path still produces one P0 call and the same final artifact.

## 5. Typed fresh-candidate campaign

Refactor current `invoke_codex_structured` into:

- `_invoke_codex_structured_once(...)`: exactly today's bounded ladder and
  journal behavior, but re-raises raw `StructuredCallFailedError` after marking
  the candidate failed;
- `invoke_codex_structured(..., retry_until_valid=False)`: compatibility
  wrapper.

The canonical P0 windows, P1, P2, P3, and P5 pass
`retry_until_valid=True`.

A failure is recoverable only when `StructuredCallFailedError.last_error` is:

- `json.JSONDecodeError`;
- Pydantic `ValidationError`;
- `_otr_structured_call.PostValidationError`;
- a capacity error for which
  `_otr_generation_budget.is_rerollable_capacity_error(...)` is true.

Everything else is converted through the existing permanent error behavior.
In particular, prompt-no-room, pack/configuration, provider/auth/network,
Python invariant, and raw type/value failures do not loop.

Candidate loop:

1. Poll Comfy interruption before each cycle and before/after each primary or
   alternate model invocation. Catch only missing-Comfy `ModuleNotFoundError`.
   Never catch `BaseException`.
2. Cycle one passes the original artifact inputs byte-for-byte.
3. After recoverable exhaustion, cycle N adds one bounded `writer_retry`
   mapping to a copy of the original artifact inputs:

```text
cycle: int >= 2
nonce: UUID hex
previous_rejection_type: nonempty string
previous_rejection: whitespace-collapsed, maximum 1200 characters
instruction: prior candidate is abandoned; return a fresh complete object
```

4. Never include rejected raw output or earlier prompts.
5. Each call-journal entry records cycle, nonce when present, status, attempt
   hashes, and terminal disposition. Only the accepted entry contains the
   accepted artifact.
6. There is no fatal fixed outer cycle count.

P3 graph compile/validation remains inside its post-validator; authored graph
defects are recoverable candidate failures. A graph that somehow drifts after
acceptance is a permanent invariant failure.

## 6. P5 safety and final ledger

Extend P5's compiled-candidate validator and raw finding collector to report
every explicit `scan_spoken_ledger` safety hit along with all graph/markup
findings. Safety language therefore enters the same typed P5 repair/fresh-
candidate campaign before acceptance.

The existing `_apply_script_safety_cleanup` remains a defense after P5 returns.
For an accepted clean candidate it performs no model call and no text change.
Revalidate graph and spoken safety after it. Any residual at that point is a
permanent invariant/cleanup failure, not a second retry loop.

Then:

1. canonicalize spoken text;
2. `_assemble_ledger` once;
3. stamp word delivery and authorship once;
4. retain exact final line hashes;
5. run cleanup/freeze/save/reopen/proof normally.

No LLM patches the production ledger. No rejected story text enters ledger
metadata. No retry wraps assembly, voice configuration, save, reopen, freeze,
or proof.

## 7. Scope

The common acquisition change affects clients sharing `_fetch_science_news`.
The canonical whole-source/retry implementation applies to
`scifi_news_circuit`.

`scifi_news_pro_multipass` receives the complete selected body from the common
fetcher but retains its existing 3,600-character dossier behavior in this
chunk. A separate Pro dossier/markup design is required before claiming
whole-source Pro support.

No node, widget, link, workflow schema, frozen artifact, GPU campaign, headless
render, Window B, degrade mode, or survival-guide mutation.

## 8. Proof

Focused tests and firing mutations cover:

- malformed/multiple RSS alternatives and stable longest selection;
- RSS/article/summary route and tie policy;
- failed scrape preservation, 12,000+ tail, 800-char preview, exact receipt;
- RSS above old 48 KB accepted; pinned source above it rejected; new RSS bounds;
- overlap progress/coverage/quote containment and duplicate-later occurrence;
- local/global span validation, deterministic caps/dedupe/selection, number
  transfer/remapping, tail inclusion, final A0 digest;
- five recoverable cycles then success, first prompt identity, unique nonces,
  bounded feedback, output-limit retry, prompt-no-room/provider/config refusal,
  cancellation identity, and no rejected raw leakage;
- unsafe P5 candidate repaired/rerolled before acceptance;
- unrelated replacement fiction accepted when mechanically valid;
- accepted upstream passes unchanged and `_assemble_ledger` called once.

Then full Windows suite with repo-local temp root, read-only Bug Bible, variants,
UTF-8/no-BOM/nonzero/AST/diff hygiene, canonical workflow hash, exact-path
commit/push, and final HEAD == origin.
