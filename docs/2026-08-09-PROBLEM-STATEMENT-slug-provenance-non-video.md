# PROBLEM STATEMENT -- slug provenance, NON-VIDEO lanes

**Date:** 2026-08-09
**Queue position:** GO_FORWARD 6-STATUS follow-ups (the queue's numbered rows are
all operator-blocked; this is the topmost unblocked item)
**Scope ruling (operator, 2026-08-09):** *"full four round but no video related
items"* / *"dont touch any video stuff"*. **Every video lane is OUT OF SCOPE.**

---

## 1. The defect class

`tencent/hy3:free` sat in the OpenRouter dropdown until its promo ended and the
slug stopped resolving. The shape: **a concrete version pin, no date, and
nothing able to notice it went stale.**

Chunk A (2026-08-07) fixed one lane. `ab76f6bc` (2026-08-09) extended coverage to
five more via `nodes/_otr_slug_provenance.py` + `tests/test_slug_provenance.py`,
which require an ENTRY for every shipped concrete slug -- either a real ISO date
or an explicit `UNVERIFIED` marker whose lane names the authority that could
settle it.

`ab76f6bc` deliberately did **not** invent dates. 21 of 27 entries are honestly
`UNVERIFIED`, because stamping today's date on slugs nobody checked would
manufacture BUG-12.86 at scale -- *a field that reads as evidence and is not*,
and worse than no date because it looks settled.

**That was the right call then. It is no longer the only option available**, and
that is what this chunk acts on.

## 2. What changed: the authority is reachable

`LANE_AUTHORITY` names Google's `generativelanguage models.list` as the authority
for the Google lanes. **That endpoint is reachable from a coder window on this
box.** `OTR_GOOGLE_API_KEY` (53 chars) and `GEMINI_API_KEY` (39 chars) are both
present in the environment; `resolve_api_key()`
(`nodes/_otr_google_api/client.py:43-57`) reads them in that precedence.

**Measured 2026-08-09, read-only, key sent as an `x-goog-api-key` header (never
a query string):** `models.list` returns **58 live ids**. Every Google-authority
slug this pack ships was checked against it.

| Lane | Slugs | Result |
|---|---:|---|
| `google_api` text (`_otr_google_api/models.py`) | 6 concrete + 2 pointers | **8/8 LIVE** |
| `google_image` (`eng_google_image.SUPPORTED_MODELS`) | 3 | **3/3 LIVE** |
| `google_tts` (`eng_google_tts._SUPPORTED_MODELS`) | 3 | **3/3 LIVE** |
| `google_lyria` (`eng_google_lyria.SUPPORTED_MODELS`) | 1 | **1/1 LIVE** |
| ~~video lanes~~ | ~~4~~ | **OUT OF SCOPE -- operator ruling** |

**Zero dead slugs.** So the honest entry for these is no longer `UNVERIFIED`; it
is a real, earned date.

## 3. The `preview` question, and the data that settles it

GO_FORWARD 6-STATUS asks for a ruling on four `preview`-marked slugs:
`gemini-3.1-flash-tts-preview`, `gemini-2.5-flash-preview-tts`,
`gemini-2.5-pro-preview-tts`, `lyria-3-clip-preview`. The framing: *"preview" in
an identifier is a LIFECYCLE PROMISE baked into an id -- the same class as
`:free`, which is what killed `tencent/hy3:free`.* Options offered: **ban, warn,
or date**.

**BAN IS WRONG, and the live catalog is why.** Filtering the 58 live ids by
capability:

| Capability | Live ids | Stable alternative? |
|---|---|---|
| TTS | `gemini-2.5-flash-preview-tts`, `gemini-2.5-pro-preview-tts`, `gemini-3.1-flash-tts-preview` | **NONE -- all three are preview** |
| Lyria / music | `lyria-3-clip-preview`, `lyria-3-pro-preview` | **NONE -- both are preview** |
| Image | `gemini-3.1-flash-image`, `gemini-3.1-flash-lite-image`, `gemini-3-pro-image` (stable) **and** `gemini-3.1-flash-image-preview`, `gemini-3-pro-image-preview` (preview twins) | **YES** |

Banning `preview` would delete the entire Google TTS lane -- the cloud audio lane
the content-addressed audio cache was built for -- and the Lyria music lane, for
a cosmetic rule. Google publishes those capabilities in **no other form**.

**But the image row is the finding.** The pack ships the three STABLE image ids
while live `-preview` twins exist for two of them. Nothing pins that. A future
edit "upgrading" `gemini-3-pro-image` to `gemini-3-pro-image-preview` would look
like a version bump and would silently move a shipping lane onto a lifecycle
promise. **That is a real trap with a real guard available.**

### Proposed ruling (the panel should attack this)

`preview` is **not banned** and **not merely warned**. Two rules:

1. **A `preview` slug may not rest at `UNVERIFIED`.** It is the highest-decay
   class, so it must carry a real verified-on date. Absence of a date is a
   failure, not a backlog row.
2. **A `preview` slug may not be shipped when a stable twin exists.** Concretely:
   if `X-preview` is shipped and `X` is also offered, that is a defect.

Rule 2 is the one with teeth. Rule 1 converts a decay risk into a dated claim.

## 4. Scope of the change

**Files:** `nodes/_otr_slug_provenance.py`, `tests/test_slug_provenance.py`.
**Not touched:** any video engine, any workflow JSON, any render path, any
profile. No behavioural change to generation -- this is a data module plus its
guard.

1. **Add two lanes** -- `google_tts` (3 slugs) and `google_lyria` (1), each with
   its authority. These are currently shipped with **no entry at all**, which the
   existing guard does not catch because `_shipped_concrete_slugs()`
   (`tests/test_slug_provenance.py:36-71`) is a hand-maintained import list, not
   a tree scan. **Both modules import cleanly in the test context (verified).**
2. **Date 9 entries** -- `google_api` (6) + `google_image` (3), `UNVERIFIED` ->
   `2026-08-09`, earned against `models.list`.
3. **Add the `preview` rules** above, with `PREVIEW_MARKERS` alongside the
   existing `FREE_MARKERS`.

Net: 27 entries -> 31; `UNVERIFIED` 21 -> 12.

## 5. Known hazards for the panel to pressure-test

* **The orphan test cuts both ways.** `test_provenance_carries_no_slug_the_pack_no_longer_ships`
  fails if a slug is added to `SLUG_PROVENANCE` without also being added to
  `_shipped_concrete_slugs()`. The two must move together.
* **A date is a claim with a shelf life.** Dating 9 entries today means they are
  "verified" forever unless something re-checks. Is a date alone enough, or does
  a dated entry need a staleness horizon? **This is the sharpest open question**
  -- it risks re-creating BUG-12.86 one year out, in a slower form.
* **Rule 2 needs a source of truth for "a stable twin exists".** The suite has no
  network. Deriving it from the shipped list only catches the case where BOTH are
  shipped -- it cannot see that Google offers a stable twin we do not ship. State
  that limit in the guard rather than implying a stronger check.
* **`is_pointer()` already exempts `-latest`.** Confirm no `preview` rule
  accidentally fires on a pointer.
* **Do not widen to the video lanes**, however tempting the symmetry. They were
  measured and are live, but the operator ruled them out of this chunk.

## 6. Baseline

Suite **9520 passed / 111 skipped / 3 deselected / 1 xfailed, exit 0** at
`a4cd217b`. The 3 deselected are a concurrent window's uncommitted
`eng_wan_i2v.py` edit (GO_FORWARD's RED-BUT-NOT-YOURS box) -- not ours, not to be
fixed here. The +4 over the recorded 9516 is the operator's untracked
`otr_sbcov_*.json` profiles: 12 collected ids reference `sbcov`, 2 per profile.
