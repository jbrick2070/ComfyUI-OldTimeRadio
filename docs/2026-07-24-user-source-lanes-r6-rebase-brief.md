# r6 REBASE BRIEF -- corrected base for the full r1-r4 re-hardening

**Date:** 2026-07-24. **Purpose:** the r4 fold of
`docs/2026-07-12-user-source-lanes-architecture.md` was grounded on a
PRE-RENAME base. The r5 confirmation pass (codex `gpt-5.6-sol` high + agy
`Gemini 3.6 Flash (High)` + Claude anchor, all grounded at HEAD `d550aff8`)
proved the base moved under the doc. This brief is the AUTHORITATIVE corrected
base. The full r1-r4 arc hardens the design AGAINST THIS -- do NOT re-derive
the stale `science_news` / `legacy_many_pass` topology; every claim below is
grounded against the real Windows files this session.

## OPERATOR RULING (2026-07-24 -- binding, supersedes the doc where they disagree)

**Every source bank is INDEPENDENT and EQUAL. There is NO "scifi_news family"
and no privileged base. `scifi_news` is ONE equal bank among the runnable six.**
Feed-variant (Path A) base eligibility is therefore a PER-BANK property (the
bank is feed-capable AND runnable AND declares the variant contract), never a
family membership. The design must not special-case any bank as "the" base;
each bank stands on its own contract.

## GROUNDED HEAD TOPOLOGY (verified d550aff8 -- supersedes every pre-rename cite)

- **Runnable banks (`nodes/story_packs/banks.json`), each independent/equal:**
  `media_archive`, `original`, `public_domain`, `shakespeare`, `scifi_news`,
  `scifi_news_pro` (+ an `+ Add Your Own` placeholder row). **There is NO
  `science_news` bank.**
- **Feed/RSS-capable lanes and their EXECUTION identity:**
  - `media_archive` -- fetcher `media_archive_rss`; INLINE execution
    (`legacy_many_pass`, no registered runner).
  - `scifi_news` -- fetcher `science_rss`; OWN-RUNNER pipeline
    `scifi_news_circuit` -> `_run_scifi_codex_lane`.
  - `scifi_news_pro` -- fetcher `science_rss`; OWN-RUNNER pipeline
    `scifi_news_pro_multipass` -> `_run_fable2_lane`.
- **`_RUNNER_BY_PIPELINE` (`OTR_LedgerScriptWriter.py:1899-1904`) = TWO entries**
  (`_run_fable2_lane`, `_run_scifi_codex_lane`); every other lane runs INLINE via
  `_LEGACY_INLINE_PIPELINES` (`:1907-1911`) = {`legacy_many_pass`,
  `original_multi_pass`}. The doc's "FIVE lazy `_run_*_lane` wrappers" and "all
  common-writer lanes reference the registered `legacy_many_pass` row" are STALE.
- **`resolve_story_pack`** at `_otr_story_routing.py:534`; falsy-coercion at
  `:539` (`story_model_id if story_model_id is not None else
  bank.default_story_model`). `SourceBank` 11-field frozen record + `_BANK_KEYS`
  present. `list_story_pack_choices` does NOT exist (L4 creates it).
- **TWO writer pack-resolution sites** (`OTR_LedgerScriptWriter.py:2051, :3461`),
  not the doc's three. `new_ledger()` precedes the dispatched runner (~`:3446`);
  dispatch hard-derefs `outline_view/canon/run_story_spine/final_title_override`
  (~`:3481-3492`), `tail_finalizer` soft `getattr` (~`:3496`).
- **`_otr_story_rules.py` / `resolve_story_rules` DO NOT EXIST** (0 content
  matches). story_rules survives only as a bank/pack concept and a run-context
  value threaded to runners. The doc's `_otr_story_rules.py` cites (S3 3-way
  clear_caches "rules", S2 rules-axis, S3 reset-seam asymmetry, L1 owned
  surface, S5.1 "runnable-lane law :274-280") are all invalid.
- **`_otr_source_payload.py` is stdlib-only at import** (future/dataclass/
  hashlib/re; heavy imports lazy) -- the SDK import-contradiction resolution
  (de-facto leaf, re-export `SourceFetchResult`) holds. `SOURCE_PAYLOAD_KEYS`
  (7 keys) + `_FETCHERS`/`_INTERPRETERS`/`registered_*_ids` present.
- **Network holes REAL** (`nodes/story_orchestrator.py`): `_fetch_full_article`
  (~`:1051`) `requests.get` + `raise_for_status`, no host allowlist / redirect
  cap / streamed size cap; `ThreadPoolExecutor(max_workers=len(shuffled_feeds))`
  (~`:1601`) unbounded; process-global `socket.setdefaulttimeout`;
  `SCIENCE_NEWS_FEEDS` hardcoded (~`:1001`); NO `_otr_feed_fetch.py`. The
  bounded-fetch seam is genuinely REQUIRED, not inherited.
- **To-be-created modules still absent:** `_otr_lane_specs.py`,
  `_otr_lane_contracts.py`, `_otr_feed_fetch.py`, `user_packs/`.

## r5 MUST-FIXES (grounded, triangulated -- the review baseline for r1-r4)

- **M1 (dominant):** rebase S1-S4 / S5.6 / S16 to the topology above. Each bank
  INDEPENDENT/EQUAL (operator ruling). A feed-variant inherits its base bank's
  EXACT execution identity -- inline (`media_archive`) OR own-runner
  (`scifi_news` -> `scifi_news_circuit`, `scifi_news_pro` ->
  `scifi_news_pro_multipass`); there is no universal "common-writer
  `legacy_many_pass`" inheritance. This is the r1/r2 arc's central redesign.
- **M2:** ONE normative `run_lane` signature using typed `LaneRunContext` +
  `SlotPort` (never raw `resolved`/`led`/`meta`/`slot_scheduler`); reconcile
  S5.3:574 with S5.3:551 + S18. Publish field-level types incl. an outline
  protocol and EpisodeCanon-compatible `canon`.
- **M3:** replay resolves the pack by stamped `owner_lane_id` (all four PackRef
  fields), NEVER `resolve_story_pack(bank, stamped_id)` (S10:909); mismatch ->
  existing `terminal_error`.
- **M4:** ONE PackRef definition (4 fields); exact typed records + JSON schema
  for per-feed url/outlet/author/rights/license/attribution + selected
  feed/article provenance.
- **M5:** bounded-fetch SSRF policy -- reject loopback/link-local/private/
  metadata/multicast/non-HTTP(S) on EVERY redirect and BOTH hops; MIME
  allowlist, missing-Content-Type, Retry-After cap, decode-failure handling
  (beyond the numeric bounds).
- **M6 (cluster):** staging promotion atomicity (rebase+persist paths IN
  staging, rename LAST); own-runner `run_lane` activation-test gap; receipt
  identity component list (activation_id excludes timestamp); test-schedule
  (drop randomizer runtime tests here, live re-baselines after L4); S13
  "carried verbatim" must cite binding sections, not a topic list.

## REVIEW MANDATE (r1-r4)

Harden the DESIGN against this corrected base. Treat each bank as
independent/equal. Do not resurrect `science_news` or a universal
`legacy_many_pass` inheritance. Line pins re-derive at the coder slot (S14).
The panel proposes; Claude grounds every claim against the real files and folds
only survivors.
