# Model-slug curation -- AS SHIPPED (GO_FORWARD queue item 1, chunk A)

Date 2026-08-07. Branch `v2.0-alpha`. Base HEAD `403868c8`.
Full `kibitz-plugin:kibitz` r1-r4 arc. Working artifacts (anchors, per-round
reviewer files, judgments) are in `kibitz-runs/2026-08-07-model-slug-curation/`,
which is **LOCAL ONLY and gitignored** -- this file is the tracked record.

## Why

`_otr_model_catalog` offered `tencent/hy3:free`. The promo ended, the slug
stopped resolving upstream, and nothing noticed. Two things had been written
down and neither could enforce itself: a code comment saying the pin was
"temporary through 2026-07-21", and the assumption that somebody would act on
it. The date passed unnoticed by 17 days.

Operator's framing: **"the fix is fewer slugs, not a refresh chore"** -- a
periodic-refresh step is a chore that will not get run.

## Policy this establishes

1. **Prefer `~family-latest` routing pointers.** They resolve upstream at
   request time, so a new model in that family is picked up with no edit here
   and nobody is offered a stale slug. Replay stays intact because the ledger
   stamps the RESOLVED concrete model (`tests/test_openrouter_resolved.py:19-30`).
2. **Carry a concrete id only where a specific version genuinely matters**, and
   date it in `OPENROUTER_VERIFIED_ON_BY_ID`.
3. **Never carry a `:free` or promo-priced slug.** A `:free` id is a PRICE
   PROMISE baked into an IDENTIFIER; promises expire, identifiers do not.
4. **No auto-routers.** `openrouter/auto` and `openrouter/auto-beta` are real
   and ARE listed by `/api/v1/models`, but `auto-beta` routes by trailing-week
   community spend, so one config could resolve differently week to week. This
   pack is built on reproducible receipts.

## What shipped

**`nodes/_otr_model_catalog.py`**
* `OPENROUTER_FRONTIER_LATEST` -> `OPENROUTER_CURATED_ALIASES`; **10 pointers**,
  adding `~x-ai/grok-latest`.
* **Deleted:** `tencent/hy3:free`; the whole `_PINNED_CREATIVE_CONTENDER_ROWS`
  mechanism (constant, filter path, cold-cache branch);
  `OPENROUTER_NO_LATEST_AUTHORS`; `_newest_concrete_for_author`;
  `_NON_FRONTIER_MARKERS`; `_OPENROUTER_RECENT_COUNT`; the `_created` helper;
  the recent-tier loop.
* **Added:** `OPENROUTER_VERIFIED_ON_BY_ID`, dating every concrete id shipped.

`~x-ai/grok-latest` is what retired the synthesis machinery: ~30 lines existed
to pick "the newest concrete slug for an author with no `~latest`", and
OpenRouter now publishes that pointer itself. Verified behaviour-neutral -- the
pointer prices identically to `x-ai/grok-4.5`, which is what the deleted code
selected.

**`aion-labs/aion-3.0-mini` was dropped too** (operator ruling: "if it never won,
dump it"). It never did: both bake-offs were BANK contests that held the writer
constant, so no model-vs-model contest was ever run. Aion was the *instrument*
of the earlier 720 run, and its live record is PBUG-20260713-20 (whole episodes
against an effective 8,192-token window while advertising 131,072) plus an
episode-aborting `finish_reason=length` on both news-coda bridge attempts
(`_otr_openrouter_backend.py:202-206`). With both rows gone the mechanism had
nothing to carry, so it was deleted rather than left empty.

**Result, read from the node's own live `INPUT_TYPES`:** slot A **22 -> 12**
entries, slot B **21 -> 12**. Ten of twelve are `~latest` pointers; the only
concrete ids left in the pack are the two recommended defaults, both dated.

**Other files:** `OTR_LedgerScriptWriter.py` -- two stale positional comments
corrected (`[19]/[20]` -> the verified `[17]/[18]` for the OpenRouter pair;
`[21]/[22]` -> `[19]/[20]` for the Comfy pair) and both slot tooltips rewritten;
`docs/openrouter-setup.md` -- a paragraph stating exactly what the filters
reach; `tests/test_openrouter_slug_curation.py` (new guard) and
`tests/test_openrouter_catalog_rows.py` (rewritten contracts).

## The guard, and why it is written the way it is

`tests/test_openrouter_slug_curation.py` asserts: no shipped id carries
`:free`/`-free`; `OPENROUTER_VERIFIED_ON_BY_ID` keys equal EXACTLY the shipped
concrete ids (so a pin cannot be added undated, nor an entry left behind);
dates parse via `datetime.date.fromisoformat` (a regex accepts 2026-02-31);
every pointer starts with `~`; the retired symbols stay retired; and the ten
pointer spellings are pinned as INDEPENDENT literals.

Two failure modes were designed out, both caught by the review panel:
* An earlier draft iterated the module's `__all__` -- which exports no slug
  constant, so the guard would have iterated nothing and passed forever,
  reproducing the exact silent decay it exists to prevent.
* Every other contract test derives its expectation FROM
  `OPENROUTER_CURATED_ALIASES`, so it cannot see a typo inside the tuple. The
  literal spot-check is the one assertion that does not share a root with the
  thing it checks.

**Proven non-vacuous by mutation:** injecting a `:free` pointer, an undated
concrete default, the impossible date `2026-02-31`, and a resurrected
`_OPENROUTER_RECENT_COUNT` each turned the guard RED, green again after revert.

**Honestly scoped:** it forbids free-MARKER ids. It cannot detect an
arbitrarily-named promo model, and it cannot see slugs supplied at runtime via
`OTR_OPENROUTER_FAVORITES`, `OTR_OPENROUTER_SLOT_x_DEFAULT` or
`OTR_OPENROUTER_MODEL_ALLOWLIST`.

## Limits -- do not report these later as a failed fix

1. **The curation removes our OFFER, not the slug.** A dead id can still appear
   under `OTR_OPENROUTER_FULL_CATALOG=1` from a stale cache:
   `models/openrouter_models.json` is **untracked and git-ignored**
   (`.gitignore:15`), per-machine, refresh-script-owned. Running the refresh
   clears it.
2. **Every claim here is catalog-LISTING evidence.** Nothing was proven to
   resolve or generate: `openrouter_enabled()` reads `OPENROUTER_API_KEY` from
   `os.environ` only (`_otr_openrouter_backend.py:261-263`, `:345-351`), and the
   key lives in the launcher scripts, not a coder window. That is why the
   creative-default promotion and `qwen/qwen3.7-flash` are chunk B, gated on one
   live leg.
3. **Curated pointers bypass `_filter_catalog_models`,** so REQUIRE_JSON and the
   denylist do not narrow them. This is PINNED BEHAVIOUR with its own test:
   filtering them against a possibly-cold cache would empty the curated block in
   exactly the state it exists to survive. The slot-B tooltip and
   `docs/openrouter-setup.md` say so plainly.

## Corrections the panel forced, recorded so they are not re-derived

* **Reasoning is already ON for every remote call** --
  `DEFAULT_REASONING_EFFORT = "low"` (`_otr_openrouter_backend.py:293`), sent on
  every request. An early draft argued the technical default must stay concrete
  because a flash pointer would "turn reasoning on"; that was false. The
  decision stands on the independent ground that the flash tier is unqualified
  for the schema-constrained lane. Related dead field, pre-existing:
  `reasoning.default_enabled` is captured at `:918` and read nowhere.
* **`config/profiles/otr_cloud_{low,hq}.json` are TRACKED and clean** -- an
  early draft called them another window's untracked work, conflating them with
  the six untracked `otr_sbcov_*.json`. Their concrete-pin ratify gate is real
  and belongs to queue item 9, not here.
* **The bare identifier `alias` is on this repo's forbidden-symbol extinction
  list** (`tests/_s28_forbidden_sweep.py`, `\balias\b`, from the
  no-legacy-back-compat directive). OpenRouter's routing pointers are an
  unrelated homonym; loop variables here are `slug`. The plural and the
  uppercase constant do not match the marker.
