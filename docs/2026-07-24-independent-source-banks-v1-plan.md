# Independent Source Banks -- client-authored v1 (LEAN plan)

**Date:** 2026-07-24. **Status:** normative plan of record for the extensibility
feature. **Supersedes** `docs/2026-07-12-user-source-lanes-architecture.md` (the
Path-A/B architecture), now RETIRED to decision-log status. Grounded at HEAD
`d550aff8` through the full r1-r4 kibitz arc + an r5 simplification pass
(codex `gpt-5.6-sol` high + agy `Gemini 3.6 Flash (High)`; Claude sole judge).
Decision log + all grounded detail: `kibitz-runs/2026-07-24-user-source-lanes-r6*/`.

## Product goal (operator)

Make it AS EASY AS POSSIBLE for a client to add their OWN source bank, EQUAL to
the shipped six. It is NOT foolproof -- the client owns getting it right.
Streamlined code; hard-fail-loud + fix bugs over defensive subsystems.

## The model: N INDEPENDENT source banks (no Path A/B, no family)

Each bank is self-contained and EQUAL, exactly like the shipped six
(`media_archive`, `original`, `public_domain`, `shakespeare`, `scifi_news`,
`scifi_news_pro`). A client adds a 7th+ the SAME shape, from their own folder,
without editing shipped repo files. A client bank provides: the bank definition
(row: id / label / source_kind / fetcher / interpreter / default pipeline /
default story model / defaults / required_seams), its fetcher + interpreter, a
shared-writer pipeline reference, and >=1 story pack. It runs through the
TRUSTED SHARED WRITER, which builds the complete production ledger + tail. There
is no "feed-variant vs original-plugin" split.

## THE #1 REQUIREMENT: a COMPLETE, CLEAN LEDGER (operator -- the biggest key)

Downstream consumers (TTS, per-beat audio slicing, video/shot direction,
captions, credits, `obs_publish`) read FIELDS, not intentions. A client bank
MUST yield a ledger with every required field owned and filled. This is PRIMARILY
a requirements/documentation matter -- the docs state exactly what a clean,
complete ledger requires; the code HELPS the client meet it; if the ledger is
still incomplete after cleanup, the run HARD-FAILS LOUD (no silent fallback, no
partial episode).

## LEDGER CLEANUP PASS (operator suggestion -- ADOPTED)

The shared writer tail runs a deterministic + LLM ledger cleanup/completion pass
AFTER a client bank's fetch/interpret: it fills/cleans the ledger to completeness
(as many LLM passes as needed) and SANITIZES CONTENT IN PLACE. SFW is a REPAIR,
never a story-fail (SFW was dropped as a gate because it failed too many
stories). Only a STRUCTURALLY incomplete ledger -- a required field with no owner
or value after cleanup -- hard-fails. Content / length / style / quality NEVER
fail a story (THE LAW).

## Non-negotiables (never traded for ease)

- Ledger COMPLETE for every downstream consumer (the #1 key).
- Fail-loud; no silent fallback to another bank / model / feed / asset.
- ComfyUI boots: shipped-seed error = loud fail (node registration dies);
  client-bundle error = QUARANTINE (absent from every dropdown), boot survives.
- Content safety = sanitize/repair in place, NEVER a story-fail. THE LAW holds.
- No regression to the dead `science_news` / `common_writer` / universal
  `legacy_many_pass` topology. Six-row INDEPENDENT execution (grounded):
  media_archive->legacy_many_pass, original->original_multi_pass,
  public_domain & shakespeare->legacy_many_pass_adapt (inline);
  scifi_news->scifi_news_circuit, scifi_news_pro->scifi_news_pro_multipass
  (own-runner). `_RUNNER_BY_PIPELINE`=2, `_LEGACY_INLINE_PIPELINES`=3.

## In v1 (lean scope)

- **Client bank folder** `user_packs/source_banks/<bank_id>/`: `bank.json` (the
  independent bank row + fetch/interpret entry-point names), a single-file
  fetcher/interpreter Python module (stdlib + the repo contracts leaf; lazy
  imports inside functions; a missing third-party lib HARD-FAILS LOUD naming the
  module -- NO `dependencies` manifest, NO version resolution), a shared-writer
  pipeline reference, and >=1 story pack.
- **Validate + activate:** `otr_check bank <path> --activate` runs schema/
  contract checks + a bounded child-process import + fixture preflight; writes a
  content-addressed snapshot + a receipt (timestamp-free canonical digest); boot
  admits IFF authoring bytes match the receipt AND the snapshot exists; any
  failure QUARANTINES that bank ONLY, with a stored `ValidationIssue` shown in
  console + `otr_check`. Duplicate/protected id, path escape, partial bundle,
  stale receipt -> quarantine.
- **Execution = SHARED WRITER only.** A client bank supplies `fetch_source` +
  `interpret_source` + `check_compatibility`; the TRUSTED writer builds the
  ledger, runs the ledger-cleanup pass, and the shared tail. The client NEVER
  touches the ledger directly -> canonical-ledger corruption from client code is
  IMPOSSIBLE.
- **One authority** (`_otr_lane_specs`-style): a pure uncached shipped-seed
  parser, atomic publish under an UNLOADED/LOADING/READY/FAILED lock, a 2-way
  cache reset (authority + routing), function-local consumer imports. Client
  banks admit ALONGSIDE shipped in the one authority; shipped-fails-boot vs
  user-quarantines is preserved.
- **Bounded fetch** (one `_otr_feed_fetch` seam, both feed + article hops):
  https-only, connect 5s / read 10s, 3 redirects, 2 MiB decoded cap, 2 retries,
  loopback/private/link-local reject, MIME media-type parse (feed = rss/atom/xml;
  article = html/xhtml), one ~25s monotonic deadline, default User-Agent +
  charset detect. Hard-fail loud on a tripped bound.
- **Story Pack widget** stays (operator-required, append-only, canonical JSON in
  the same commit); packs resolve by OWNER (`resolve_pack_ref`, four-field
  PackRef, replay by stamped owner+digest), or a bank's manifest default covers
  it.

## Deferred past v1 (revisit on real demand -- not built now)

- Client-shipped OWN-RUNNER (a bank shipping its own `run_lane`) + staging
  ledger / atomic promotion. The shared-writer path delivers the easy 80% and
  keeps the ledger safe; this removes the only client path that could corrupt
  canonical state.
- `dependencies` manifest + import->distribution/version resolution.
- Standalone `story_rules` module (it does NOT exist; authored behavior lives in
  story-pack prompt stages; update the binding `SOURCE_BANK_GUIDE.md` +
  `SOURCE_BANK_PREFLIGHT.md` in the same wave that stops citing it).
- Randomizer/roll behavior tests (own build).

## Deliverables (docs-FIRST, per operator)

1. **REQUIREMENTS DOC (primary): `docs/EXTENDING_OTR.md`** -- exactly what a
   client must provide, the COMPLETE-LEDGER field contract (every
   downstream-consumed field + its required owner), the clean-ledger rules, the
   SFW-by-repair note, a worked example bank, and fixtures. This is where the
   client "learns to keep a clean ledger."
2. **Loading / validation / activation code** -- `otr_check bank --activate`,
   quarantine, snapshot/receipt, the one authority admitting client banks
   alongside shipped (per the grounded arc invariants).
3. **The ledger-cleanup pass** in the shared tail.
4. **The bounded-fetch seam.**

Every wave: focused tests + full Windows suite + Bug Bible + AST/JSON/BOM/
zero-byte + commit&push + `HEAD == origin/v2.0-alpha`. Re-derive every line pin
at the coder-slot HEAD (fast-moving-base precondition). Canonical JSON changes
(the story_pack widget) are append-only in the SAME commit and re-validated.

## Estimate (lean, replaces the ~21-31 day A/B figure)

Cutting client own-runner/staging + the dependency subsystem + standalone
story_rules removes the heaviest waves. Rough order: docs/requirements + one
authority + activation/quarantine + ledger-cleanup pass + bounded fetch +
story_pack widget = materially smaller than the retired A/B plan; re-estimate at
the coder slot after the requirements doc is drafted, since docs-first may
absorb part of the "coding" scope the operator flagged.
