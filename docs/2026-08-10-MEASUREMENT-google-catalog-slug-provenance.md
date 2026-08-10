# MEASUREMENT -- live Google catalog fetch for the slug-provenance build

**Date of fetch: 2026-08-10.** This is step 4 of
`docs/2026-08-09-BUILD-SPEC-slug-provenance-non-video.md` ("Live catalog run ->
human-reviewed dates"), performed and written down so the next coder window can
build steps 1-5 as ONE atomic commit with real dates in hand instead of pausing
mid-build to go fetch them.

**It is a MEASUREMENT, not a verdict.** The spec is explicit that the verifier
never writes dates back, because *a date is a human claim that someone looked*.
This file records what was seen; committing any of it as a `verified_on` date is
the human step, and it stays the human step.

## How it was fetched

The pack's own `nodes/_otr_google_api/client.py` (`get_json` + `resolve_api_key`),
so header, base override and error classification stayed single-sourced. Key
travelled as the `x-goog-api-key` header, never a query string. `pageSize=200`
held across pagination.

**The listing is COMPLETE.** One page, terminal, no `nextPageToken`. Per the
spec's rule that only a terminal page may authorize a "missing" verdict, this
fetch is eligible to support one -- and it produced none.

**52 unique model ids** after stripping exactly one `models/` prefix.

## Result: every shipped Google id is present

| Lane | Shipped id | In catalog |
|---|---|---|
| `eng_google_tts` | `gemini-2.5-flash-preview-tts` | yes |
| `eng_google_tts` | `gemini-2.5-pro-preview-tts` | yes |
| `eng_google_tts` | `gemini-3.1-flash-tts-preview` | yes |
| `eng_google_lyria` | `lyria-3-clip-preview` | yes |
| `eng_google_image` | `gemini-3-pro-image` | yes |
| `eng_google_image` | `gemini-3.1-flash-image` | yes |
| `eng_google_image` | `gemini-3.1-flash-lite-image` | yes |

Zero shipped concrete ids missing from a completed catalog, so the verifier's
exit-2 condition does not fire against this data.

## The escalated id -- catalog presence CHANGES NOTHING, and this is why

`gemini-3.1-flash-image-preview` **is present in the 2026-08-10 catalog.**

It still ships `unverified`, exactly as spec section 0A requires, and this
measurement must not be read as closing that escalation:

* **The measured endpoint is not the runtime endpoint.** Runtime sends this id to
  a Vertex proxy -- `ApiEndpoint(path=f"/proxy/vertexai/gemini/{model_id}")`,
  `nodes/_otr_shared/cloud_media_invoke.py:510` -- not to the catalog endpoint
  fetched here. Catalog presence proves the id is *listed*; it cannot prove that
  *route* works. Dating it off this fetch would be exactly the evidence-shaped
  field this whole chunk exists to kill.
* **It does bear on one disputed claim.** Codex reported a public Google shutdown
  of this id on 2026-06-25 with `gemini-3.1-flash-image` as the replacement, and
  the spec recorded that as UNVERIFIED from this box. The id is still listed
  forty-six days after that date, which is evidence against a completed
  *catalog* removal. It says nothing either way about the proxy route.
* **The settlement the spec proposed is unchanged and still the cheapest one:**
  one still through the Nano Banana lane. It either renders or the proxy rejects
  the id. That is a render decision and it is the operator's call, because
  repointing the selector at the stable twin would change which model renders
  stills, and the recipes are not on the table.

## What this unblocks, and what it does not

**Unblocked:** spec steps 1-5 can now be built as one commit. The preview rule
("a `preview` slug may not rest at `unverified`") is satisfiable for all four
shipped preview-labelled ids -- three TTS plus Lyria -- because each was seen in
a completed direct-authority listing on 2026-08-10, which is what
`catalog_listed` means.

**Not unblocked:** the named exception list still has exactly one entry, and it is
still `gemini-3.1-flash-image-preview`, for the reason above.

## Raw data

The full 52-id listing is in the session scratchpad
(`google_catalog.json`) and is deliberately NOT committed -- it is a dated
snapshot of someone else's service, and the seven rows that matter to this pack
are transcribed in the table above. Re-fetch rather than trust a stale copy.
