# BUILD SPEC -- slug provenance, NON-VIDEO lanes (r4-final, ready to implement)

**Status:** SPEC COMPLETE, NOT BUILT. Full `kibitz-plugin:kibitz` r1-r4 arc closed
2026-08-09. **This file is TRACKED on purpose** -- the arc's raw artifacts live in
`kibitz-runs/2026-08-09-slug-provenance-nonvideo/`, which is **gitignored**
(`.gitignore:251`), and a spec left only there is invisible to every future doc
search. That lesson is already in GO_FORWARD's baseline table; this file is it
being applied.

**Scope (operator, 2026-08-09):** *"full four round but no video related items"* /
*"dont touch any video stuff"*. No video engine, no video registry, no mechanism
that enumerates either. Pure data + test + verifier: no node, protocol,
`NODE_CLASS_MAPPINGS`, widget, or canonical-workflow change.

**Arc accounting: 7 delivered external reviews across 4 rounds, not 8.** r3 is a
documented SINGLE-LANE round (Codex only -- agy timed out twice, zero quota
markers, no `quota_hold`; a timeout, not credits). Model drift was caught at r2
(`Gemini 3.5 Flash (High)` where CLAUDE.md specifies **3.6**) and pinned for
r3/r4. Driver anchors preceded every fan-out; the r2 anchor carries a written
correction of the driver's own withdrawn claim.

---

## 0. PREREQUISITE -- do not start until this is true

**One coder window in the code at a time (`CLAUDE.md`).** At spec close,
`git status` still showed a concurrent window's uncommitted
`nodes/_otr_video_engines/eng_wan_i2v.py` and
`config/profiles/otr_g4_wan_ti2v.json`. Those files do not overlap this chunk, so
the risk is not a conflict -- it is the staging hazard this repo has already been
bitten by, a whole-file `git add` sweeping another window's dirty work into an
unrelated commit. **Stage by name, never `git add .`**, and confirm that window
has yielded before starting.

## 0A. ESCALATED TO THE OPERATOR -- explicitly NOT built here

`gemini-3.1-flash-image-preview` is the id behind the `Nano Banana 2 (Gemini 3.1
Flash Image)` selector. Runtime sends it to a **Vertex proxy**, not the catalog
endpoint that was measured:

```
ApiEndpoint(path=f"/proxy/vertexai/gemini/{model_id}", method="POST")
```
`nodes/_otr_shared/cloud_media_invoke.py:510` -- confirmed in source.

Catalog presence therefore does **not** prove that route works. Codex reports a
public Google shutdown of **2026-06-25** with `gemini-3.1-flash-image` as the
replacement; **that claim is UNVERIFIED from this box** (no web access in the
window) and the id was still present in the 2026-08-09 catalog fetch, which is
consistent with either reading.

Repointing the selector at the stable twin (already shipped, confirmed live)
would change **which model renders stills** -- recipe-adjacent, and the standing
directive is that the recipes are not on the table. **Operator's call.** Cheapest
settlement: one still through the Nano Banana lane -- it either renders or the
proxy rejects the id.

**Consequence for this build:** that id ships `unverified` regardless of catalog
presence, so no date can legitimize an unproven route. It is the sole entry on
the named preview exception list (section 6).

## 1. The defect being fixed

`test_every_shipped_concrete_slug_has_provenance` reads as a totality claim and
is not one -- it enforces "every slug reachable from six hand-written imports"
(`tests/test_slug_provenance.py:36-71`). It is green today while real shipped
model ids are invisible to it.

**Live instance.** `cloud_media_invoke.py:477-484` resolves shipped display names
into real Google ids at invoke time:

| Shipped selector (what the guard sees) | Real id sent to Google |
|---|---|
| `Nano Banana 2 (Gemini 3.1 Flash Image)` | **`gemini-3.1-flash-image-preview`** |
| `Nano Banana 2 Lite` | `gemini-3.1-flash-lite-image` |
| anything else | identity fallthrough (`else: model_id = model_choice`) |

## 2. Build order -- ONE atomic green commit

r4 found the original order impossible: **every** shipped TTS model is
preview-labelled (`eng_google_tts.py:30-39`), as is Lyria
(`eng_google_lyria.py:27-28`), so a commit adding those lanes before dates exist
would violate its own new rule.

1. `nodes/_otr_shared/google_image_model_ids.py` -- shared selector->id map.
2. `nodes/_otr_shared/slug_inventory.py` -- `InventoryRecord` + projections.
3. Verifier pure core + `scripts/verify_google_slugs.py`.
4. **Live catalog run -> human-reviewed dates.**
5. `nodes/_otr_slug_provenance.py` schema migration + `tests/test_slug_provenance.py`.

**Steps 1-5 are ONE commit.** Values are unpacked positionally in
`unverified_slugs()` (`:120-124`) and two tests (`:101`, `:111`); a split commit
is red in between, and the dates must exist before the rule that requires them.

**Then, per `CLAUDE.md`: full suite AND the Bug Bible, then PUSH immediately.**
Per-green-chunk push is standing policy, not optional.

## 3. Leaf 1 -- shared selector mapping

Extract from `cloud_media_invoke.py:477-484`. Stdlib-only, non-video, imports
nothing from engine namespaces.

```
resolve_selector_to_model_id(selector: str) -> str
SELECTOR_TO_MODEL_ID: Mapping[str, str]
```

**Behaviour-identical, including the identity fallthrough** -- raising on an
unmapped selector would break every seedream / krea / photon row, none of which
is mapped. `cloud_media_invoke.py` imports it; the collector imports it. **No
mirrored copy** -- mirroring recreates the two-lists-that-must-agree defect this
chunk exists to kill.

## 4. Leaf 2 -- shared inventory

```
InventoryRecord(source_key, source_kind, selection_surface,
                offered_value, provider_id, authority_lane, catalog_target)
```

* `source_kind` is `engine` | `static_lane`. **A registered engine name cannot
  represent every source:** `COMFY_LLM_MODELS` and `_otr_google_api.models.*`
  are static lanes, not engines (`test_slug_provenance.py:51-58`).
* `selection_surface` distinguishes selector groups within one engine; the
  per-surface twin ban keys on it.
* `catalog_target` is SEPARATE from `authority_lane` -- a Comfy surface can
  expose a Google provider id, and conflating them would send ElevenLabs and
  Comfy display selectors into Google's catalog.
* `provider_id` is stored **without** any `models/` prefix.
* Uniqueness = exactly one record per `(source_key, selection_surface,
  offered_value)`; `provider_id` MAY repeat across sources.
  `gemini-3.1-flash-lite-image` legitimately arrives from both
  `eng_google_image.SUPPORTED_MODELS:32` and the `Nano Banana 2 Lite` mapping.
  **No `setdefault` collapse.**
* `cloud_model_ids.V3_MODEL_IDS` defaults are consistency CHECKS, not inventory
  rows (`cloud_model_ids.py:34-39` already repeats the Nano default against
  `eng_cloud_image.py:50`).
* **Checked-in constants/defaults ONLY. Never call runtime selectors** --
  `cloud_model_ids.resolve_model_id()` reads env overrides and
  `_otr_google_api/models.py:127-159` folds disk-cached ids, either of which
  makes results machine-dependent.
* Both the test and the verifier import this. Neither keeps another list.

## 5. Provenance schema

`ProvenanceRecord(authority_lane, verification_kind, verified_on: str | None)` --
a stdlib `NamedTuple`, keyed by **`(provider_id, authority_lane)`** so a
duplicate id keeps both authorities.

| kind | `verified_on` | meaning |
|---|---|---|
| `catalog_listed` | real, **non-future** ISO date | direct-authority catalog hit |
| `signal_listed` | real, non-future ISO date | upstream signal only -- the 6 comfy rows; NOT authority verification |
| `unverified` | `None` | nobody has checked |

**Do not encode kind into the date string.** Both lanes proposed
`"kind@YYYY-MM-DD"`; packing two facts into one string is the positional-parse
defect already present.

Update matrix: direct-Google hit -> `catalog_listed`; Comfy-surface hit ->
`signal_listed`; the Nano preview id -> **always `unverified`**; non-Google
targets never submitted to Google.

**MIGRATION MUST NOT SILENTLY DROP EXISTING PROTECTIONS.** Preserve and re-test
all five: the free-price-marker ban (`:105-107`, tested `:116-124`), the exact
pointer predicate (`:110-117`), lane-authority coverage, bidirectional
inventory/provenance completeness, and exclusion of pointers from version-dating.

## 6. Preview rules

1. **A `preview` slug may not rest at `unverified`** -- it must carry a real
   date. **EXCEPTION LIST, named and reasoned, currently one entry:**
   `gemini-3.1-flash-image-preview` (section 0A, unproven route). The exception
   is data with a stated reason, never a silent skip -- without it the rule is
   unshippable on day one.
2. **Per-surface twin ban; cross-surface pairs REPORTED, not failed.**
   `eng_cloud_image` bills through the Comfy partner path, `eng_google_image`
   through a BYO key (`resolve_api_key` + `create_interaction`) -- different
   auth/billing surfaces, and a global ban would delete a valid one.

**Detection is a token-boundary predicate over `-`-delimited segments, never a
substring match.** Three placements exist in one lane:
`lyria-3-clip-preview` (suffix), `gemini-2.5-flash-preview-tts` (infix),
`gemini-3.1-flash-tts-preview` (infix, reversed). Twins live in an explicit
directed `PREVIEW_TO_STABLE` table -- keys must contain a boundary-delimited
preview token, values must not, both validated against the inventory. **No
strip-and-compare derivation.**

## 7. Completeness cross-check -- FAIL-CLOSED, never vacuous

Enumerate every registered **audio + image** engine; assert each is covered by a
collector or EXPLICITLY exempt.

* **Import the packages first.** Registration happens via import-time decorators
  in `_otr_audio_engines/__init__.py` and `_otr_image_engines/__init__.py`;
  importing `registry.py` alone can see a partial or empty registry.
* **Pin the expected engine set; fail on a strict subset.** Guarded imports mean
  a missing optional dep silently drops an engine, and a loop over an empty
  collection would PASS -- **BUG-12.87**, *a gate reports success from its own
  skip path*, promoted three days ago. A vacuous pass must be impossible.
* **NEVER infer exemption from an absent `native` flag.** `eng_cloud_image.py`
  declares `native` **zero times**, so `getattr(e, "native", True)` would
  classify the entire cloud image lane (nano banana, seedream, krea, photon,
  flux_pro, ideo) as local and let it escape. Use an exhaustive
  engine-name -> collector-or-exemption map whose keys are asserted **EQUAL** to
  the audio+image registry union. Static lanes are enumerated separately.
* `sonilo` is exempt, keyed on "declares no concrete model collection" rather
  than its name (`eng_cloud_sonilo.py:38` ships `_PARTNER_ROW`).
* **No protocol change, no `register()`-time enforcement.**
  `tests/test_image_platform_c1.py:123-139` asserts `ImageEngine.__annotations__`
  is a structural superset of `AudioEngine`'s with identical annotations -- a
  protocol field would ripple into the VIDEO protocol through the sibling parity
  test, violating the operator's scope by construction. Test-time introspection
  only. Registration-time failure could also break ComfyUI node loading rather
  than failing a test.

## 8. Staleness

Age thresholds **NEVER** fail. A calendar-triggered red on an offline-first pack
would fire on a fresh clone in 2027, on a box nobody touched. **Both review lanes
independently proposed a hard failure; that makes it the intuitive answer, not
the right one.**

**Future-dated entries DO fail**, as malformed evidence. Age is a pure function
taking an injected `as_of`; the current date is read only at the script boundary.

The report lives in the **verifier**, not in pytest-captured prints --
`test_the_unverified_backlog_is_visible` (`:143-155`) prints under capture and is
invisible under the `pytest -q` baseline, which silently gutted the original
mitigation. No separate terminal-summary hook: two reporting paths drift.

## 9. Verifier

**Split for testability.** An import-safe PURE CORE holding all parsing and
classification, with injected `get_json`, key resolver, sleeper and `as_of`;
`scripts/verify_google_slugs.py` is a thin `main()` wrapper. **Tests import only
the core and inject all I/O** -- that resolves the conflict between "no test
imports the verifier" and "test pagination, retry, every exit code, and prove
zero network."

* Reuses `resolve_api_key()` and `get_json()` (`client.py:43-57,219-277`) so
  header, base override and error classification stay single-sourced.
* Retry law: three total attempts, matching `DEFAULT_MAX_RETRIES = 2`
  (`client.py:12-15`). Bounded retry for transport / 429 / 5xx only -- never for
  auth or malformed responses, and **never convert exhaustion into "missing."**
* Stable numeric exits, each tested: **0** success, **2** shipped ids missing,
  **3** no-key/not-run, **4** auth/transport, **5** malformed catalog.
* Normalization: strip exactly one `models/` prefix, reject blank/non-string
  names, hold `pageSize` across every URL-encoded `pageToken`. **Only a terminal
  page without `nextPageToken` may authorize "missing."**
* `missing` = shipped concrete ids absent from a COMPLETED catalog.
  `provenance-orphaned` = provenance identities absent from the inventory.
  Unshipped catalog entries are neither.
* Pointers audited and reported separately, never version-dated.
* **Never writes dates back.** A date is a human claim that someone looked; a
  script stamping dates unattended manufactures the evidence-shaped field this
  chunk fights. Print the delta; a human commits it.
* Prints backlog/age BEFORE `resolve_api_key()`, so a no-key exit still reports
  offline state.
* Key travels as an `x-goog-api-key` header, never a query string.
* Not collected by pytest.

## 10. Coverage statement (must be said out loud)

Both docstrings state the limit: the guard proves **"these listed lanes and
registered engines are complete"**, never "the pack has no unlisted model", and
the claim is scoped to **checked-in static/default model selections** --
`_otr_google_api/models.py:127-159` folds disk-cached ids into slot choices, so
no static enumeration covers every runtime-selectable model. Understating a guard
is the cheap failure; overstating one is how BUG-12.86 got promoted.

Also record that `nodes/_otr_slug_provenance.py` has exactly ONE importer in the
tree (`tests/test_slug_provenance.py:33`) -- that is what makes the schema change
cheap, and it will not stay true by accident.

## 11. Acceptance matrix (write before code)

Registry omission; **strict-subset registry (vacuous-pass guard)**; explicit
exemption (`sonilo`); an engine absent from the map fails; static lanes
enumerated separately; duplicate provider id across two authorities; both display
mappings reaching the invoke function; the identity fallthrough; schema kind/date
pairing; **future-dated entry fails**; preview token boundaries at prefix, infix
and suffix plus a false-positive string; preview-without-date fails; the named
exception list is honoured and is not a silent skip; per-surface twin ban vs
cross-surface report; pointer separation; pagination, repeated token, and
partial-fetch never "missing"; each of the five numeric exit codes; **the five
preserved legacy protections (free-marker ban, pointer predicate, lane-authority
coverage, bidirectional completeness, pointers excluded from dating)**; **a
fresh-process test proving collection imports neither `nodes._otr_video_engines`
nor any video registry**; and proof the unit tests perform no network access.

## 12. Gate

Full suite green by **EXIT CODE**, not an exact collected count -- untracked
operator profiles move it (+4 at spec time: 12 `sbcov` collected ids, 2 per
profile). Baseline `9520 passed / 111 skipped / 3 deselected / 1 xfailed, exit 0`
at `2c2df490`; the 3 deselected are the concurrent window's `eng_wan_i2v.py` edit
and are **not ours to fix**.

Then: **the Bug Bible** (`CLAUDE.md` requires it after every code change), the
verifier's live run (a no-key "not-run" does **NOT** count as acceptance, because
preview rows require real dates), Sonnet 5 QA on the diff, the Fable gate, then
push.
