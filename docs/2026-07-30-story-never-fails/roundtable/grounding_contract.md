# Focused Grounding Contract

The document under review is the current `passNN_plan.md`. Historical plans are
not proposals. This pack contains the narrow production facts the panel should
check.

## Common RSS acquisition

`nodes/story_orchestrator.py::_fetch_full_article` extracts the first matching
article container's paragraphs and headings, collapses whitespace, and then
clips:

```python
content_tags = body.find_all(["p", "h2", "h3"])
text = " ".join(tag.get_text(" ", strip=True) for tag in content_tags)
text = re.sub(r'\s+', ' ', text).strip()
return text[:12000]
```

The network seam that supplies the HTML remains HTTPS-only and caps the decoded
response at 2 MiB:

```python
# nodes/_otr_feed_fetch.py
ALLOWED_SCHEME = "https"
MAX_DECODED_BYTES = 2 * 1024 * 1024
```

`_fetch_single_feed` currently reads only the first inline RSS content value:

```python
content_candidates = entry.get("content", [])
rss_full = ""
if content_candidates:
    rss_full = _extract_rss_fragment_text(
        content_candidates[0].get("value", "")
    )
```

The body resolver treats 300 characters as enough to skip the linked article:

```python
rss_full = out.get("rss_full")
has_acceptable_inline_rss = bool(rss_full) and len(rss_full) > 300
if has_acceptable_inline_rss:
    out["full_text"] = rss_full
    out["_body_source"] = "rss_full"
elif out.get("link"):
    fetched = _fetch_full_article(out["link"], timeout=5)
    if fetched and len(fetched) > 300:
        out["full_text"] = fetched
        out["_body_source"] = "url_scrape"
    else:
        out["full_text"] = out.get("summary", "")
```

Only the first five headline-ranked candidates reach body resolution, and body
reranking sees only the first 800 characters. The 400-character floor is a
preference: when no candidate reaches it, the richest real candidate is already
selected instead of killing the episode.

The live ledger already receives `headline`, `source`, `url`, `date`,
`body_chars`, and `selected_at` under `meta.news_seed`. It does not currently
receive the selected body route, RSS alternative index/count, or body digest.

## Canonical Sci-Fi lane source authority

`nodes/_otr_scifi_codex.py::validate_payload_envelope` normalizes the
span-bearing fields first, then rejects the complete seven-field source payload
above 48,000 serialized UTF-8 bytes:

```python
for _field in _SPAN_SOURCE_FIELDS:
    clean[_field] = _normalize_span_source_text(clean[_field])
...
if len(json.dumps(clean, ensure_ascii=False).encode("utf-8")) > 48_000:
    raise CodexPayloadOversizeError(...)
...
source_digest=_digest(clean)
```

The resulting normalized A0 payload and digest are the sole coordinate
authority. P0 receives a lossless de-aliased view of only legal span fields.

`nodes/_otr_scifi_p0_contract.py` already contains unused whole-body windowing.
It explicitly returns `(offset, payload)` pairs and never trims:

```python
def p0_source_chunks(payload, *, budget_chars):
    ...
    body = str(payload.get("full_text") or "")
    ...
    while offset < len(body):
        window = body[offset:offset + allowance]
        if offset + allowance < len(body):
            boundary = max(window.rfind(mark) for mark in _SENTENCE_END)
            if boundary > allowance // 2:
                window = window[: boundary + 1]
        fitted = dict(payload)
        fitted["full_text"] = window
        windows.append((offset, fitted))
        offset += len(window)
    return windows
```

There is no production caller, global offset rebaser, deduper, or merger.

The accepted P0 schema is bounded:

```python
class FactIndexV4(_Strict):
    facts: list[FactV4] = Field(min_length=1, max_length=6)
    entities: list[EntityV4] = Field(max_length=4)
    numbers: list[NumberV4] = Field(max_length=4)
    tone: str
    payload_sha256: str
```

Every fact/entity span is literal. Every number references an accepted
`fact_id`. P3 authors the score and closed line graph from compact P0 facts and
tone. P5 is the sole spoken-text writer: it authors one text row for every
accepted graph line. Neither receives source spans or the complete article.

## Candidate lifecycle

`nodes/_otr_structured_call.py::structured_call` is deliberately a finite
per-candidate ladder. `nodes/_otr_scifi_codex.py::invoke_codex_structured`
converts ladder exhaustion into a fatal lane error:

```python
except StructuredCallFailedError as exc:
    ...
    raise CodexPassError(f"{pass_id} failed: {exc}") from exc
except Exception as exc:
    ...
    raise CodexPassError(f"{pass_id} failed: {exc}") from exc
```

P0, P1, P2, P3, and P5 are called sequentially once. P5's compact draft
validator reports all known graph and spoken-markup findings in one repair
message. After a structurally accepted P5 result, safety cleanup can still
raise on residual spoken findings. P5 therefore owns a rewritten spoken-story
candidate; P3 is replayed only if its accepted graph itself is defective.

The project doctrine already distinguishes candidate exhaustion from episode
exhaustion: keep each candidate finite; on recoverable model-output exhaustion,
retire the candidate and start a fresh one with a model-visible nonce. Preserve
accepted earlier artifacts. Poll ComfyUI cancellation. Do not impose a fatal
fixed outer candidate count.

Permanent failures remain loud: cancellation, missing/invalid prompt pack or
model configuration, source-security refusal, and durable atomic filesystem
failure. Parse, schema, output-budget, authored graph, and spoken-surface
defects are recoverable candidate failures.

The operator explicitly grants fictional latitude. The complete article is
evidence for real science claims and the factual coda, not a requirement that
the invented plot reproduce the article. P1-P5 may freely invent and rewrite
SFW people, places, events, conflict, dialogue, and dramatic structure. The
final acceptance contract is a downstream-safe, internally coherent ledger; it
is not article-plot fidelity or fidelity to an abandoned earlier draft.

## Ledger integrity

`_assemble_ledger` mechanically builds the production ledger from accepted P2,
P3, and P5 artifacts and records exact accepted line text plus SHA-256 hashes.
`_CodexTailFinalizer` verifies the in-memory and saved ledger against those
hashes. Those checks remain useful corruption guards for the final accepted
candidate, but repair may freely replace earlier story drafts. A failed repair
candidate must never replace the canonical on-disk ledger.

Warnings do not block. Hard gap/freeze errors and reopen/save failures currently
raise. The design must not pretend that retrying creative prose repairs a
deterministic serialization, configuration, or disk defect.

## Separate `scifi_news_pro` runner

`nodes/_otr_scifi_fable2.py` is a different writer architecture. Its digest
builder clips the source view to the first 3,600 characters. It shares the
common RSS fetcher but not the canonical Codex FactIndex pipeline. Any claim
that this runner has complete-source support needs its own window/merge adapter
and tests; changing the canonical lane alone does not prove it.

## Non-negotiable scope

- Existing frozen ledgers and snapshots remain byte-identical.
- The 2 MiB HTTPS/network-security envelope remains.
- No GPU campaign or headless render is authorized in this coding slice.
- There is no node, widget, link, or workflow schema change in the proposed
  internal implementation. `workflows/otr_canonical.json` must remain
  byte-identical unless that fact changes.
