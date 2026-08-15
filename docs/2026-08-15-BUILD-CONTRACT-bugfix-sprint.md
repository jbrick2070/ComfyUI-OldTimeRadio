# BUILD CONTRACT -- the 2026-08-15 bug-fix sprint

Hardened across seven independent review lanes: Codex, Antigravity, GPT-5.6-sol,
Gemini 3.1 Pro, DeepSeek-v4-pro, a Fable structural gate and a Sonnet
implementability audit. Claude drove, grounded every claim against the real
Windows files, and remained sole judge. Working artifacts (driver anchor, raw
reviews, per-round judgments) are local-only under
`docs/2026-08-15-bugfix-sprint/` and `kibitz-runs/2026-08-15-bugfix-sprint/`,
both gitignored by repo convention. **This file is the durable contract.**

Branch `v2.0-alpha`. Baselines to hold: suite 10532/110/1, Bible 20/26/3,
variants 50/0. **No production code has been written yet.**

---

## LAWS -- reject any proposal that breaks one

1. No shims. Root cause only.
2. Code may DETECT and explain. Only a MODEL may rewrite PROSE.
3. **A source attribution is NOT prose.** It is a Python-owned fact with a receipt.
4. No content guardrails on generated episodes.
5. Story quality is CLOSED (2026-08-04). Correctness defects are carved out.
6. No count chasing. Word/beat/act counts are requests; no test may pin one.
7. **A render must not die.** A writer may refuse structurally BEFORE generation.
8. One owner per ledger field.
9. Runaway guards are code-side and stay.
10. No `INPUT_TYPES`, widget, or `workflows/otr_canonical.json` change.

---

## BUILD ORDER -- each chunk ends green and pushed

Two dependency cycles were found in review: D1 protects a fact D5 defines, and
D5's media close needs identity D6 produces. **Both break by defining the
schema first.**

| # | Chunk | Gate |
|---|---|---|
| 0 | Source-identity schema + fact schema + identity adapter + transition-receipt schema/validator | breaks both cycles; DEC-3, DEC-12 |
| 0.5 | Law-7 publish gate (`_check_g14_provenance_publish` -> eligibility receipt, block at `obs_publish`) | DEC-10; before any affected render acceptance |
| 0.75 | D4 vendor gate: all 65 sidecars generated, schema-validated, stale-checked | DEC-11; before the read-only runtime path is enabled |
| 1 | Mutation lifecycle + transaction object + shared reseal mechanics | DEC-1, DEC-2; precedes D1 and D2 |
| 2 | D1 component boundary (`clean_spoken_component`, pre-composition marker, one composer) | DEC-5; needs chunk 1 |
| 3 | D2 lane integrations (both content-owned lanes' proof surfaces) | **must precede 3.5** |
| 3.5 | **D3 `END` grammar + required-shape diagnostics** | DEC-8, DEC-9. **Was missing from this table entirely** -- as written the fix would never have been built |
| 4 | D5 non-media codas (shakespeare, public_domain) | needs chunk 0 |
| 5 | D6 media selector + reservation + durable post identity | DEC-6, DEC-7 |
| 6 | D5 media_archive close | consumes chunk 5 |
| -- | D4's *pin application* and D7 are independent, may run in parallel | D4's vendor gate is 0.75 |

**D2 -> D3 is load-bearing and non-obvious.** If the `END` grammar lands before
the reseal, `scifi_news_pro` clears the markup ladder and then dies at the
freeze cascade on a content-authorship mismatch, because the clean stage
rewrote 9 of 14 rows. Fixing the parser first only moves the corpse.

---

## D1 -- THE CLEAN STAGE ERASES PYTHON-OWNED FACTS

`nodes/_otr_ledger_clean.py`, shipped 2026-08-14, rewrites the closing announcer
row, which carries a deterministic Python-owned source attribution.

Confirmed on three live ledgers. `reel_of_mystery` b016 composed
`<bridge>": "` + *"In other news, the Library of Congress announces its film
loans for the month, including 'None But the Lonely Heart', 'Symphony of
Swing', and 'The Man With the Golden Arm'."* and SHIPPED *"Clarisse's gaze
meets the reel's enigmatic label"* -- the entire source note deleted.
`midnights_ticktock` b016 paraphrased the Python-owned `spoken_coda_line` output
while `meta` still advertises `spoken_coda_source: "provenance"`.
`ghost_of_elsinore` b016 rewrote the sign-off. Rewrite rate 9/14 voiced rows in
all three.

**The operator's "What news story???" was never the interpreter's fault.** The
`news_close_brief` was factual and the coda was correct when composed.

**ARCHITECTURE.** The cleaner receives ONLY the authored bridge, UPSTREAM of
composition. Provenance/news produces the immutable fact. ONE final composer
writes `lines[].text` exactly once. Bridge and fact are inputs owned by their
producers; the composer is the sole field writer. Law 8 satisfied without span
parsing or reattachment.

**SCOPE BY COMPOSE FLAG.** `news_coda_bridge` / `news_coda_fact_only`
(`_otr_line_composer.py:1484`). **NOT** by `speaker_role` -- announcer rows are
legitimately judged, and `tests/test_ledger_clean_stage.py:186-194` asserts
`judge_calls == 3` on a synthetic ledger whose announcer row has no flags, so a
role-keyed exemption drops it to 2 and breaks. **NOT** by the `"news_coda"`
prefix -- the codex coda row carries flag `news_coda` with NO deterministic fact
(`_otr_scifi_codex.py:3644-3678` authors the whole read, anchor-verified);
prefix matching sends that lane hunting a fact that does not exist.

**THE SEPARATOR IS CONDITIONAL.** `_otr_line_composer.py:1441-1446` returns
`" ".join((f"{bridge}: {fact}" if bridge else fact).split())` -- when the bridge
fails validation the row is JUST the fact, no `": "`. Do not recover the fact by
searching for `": "`; it misfires there and again if the fact contains a colon.
**Anchor on the durable meta value** (`meta["provenance_coda_line"]` /
`meta["news"]["news_close_brief"]`, stamped before assembly and never touched by
the clean stage) and match with `endswith`.

**`FIDELITY_BANKS` IS A RED HERRING.** `_otr_spoken_text_policy.py:83` looks like
ready-made protection and covers two of three victim banks. Its own doctrine
(`_otr_ledger_clean.py:1739-1741`): the carve-out only restricts which
pattern-detector KINDS may auto-spend a repair, but *"the judge still reads the
line and its verdict counts everywhere."* That is exactly why it did not save
these codas. D1 needs its own row-scoped authority. `_otr_ledger_clean.py` has
ZERO occurrences of `news_coda` or `source_owned` -- this is new plumbing.

---

## D2 -- THE RESEAL IS FOUR SURFACES WIDE

`scifi_news` dies `CodexPreTailAuditError: line receipt mismatch for l004`.
`l004` is the first row the clean stage rewrote; `l001`/`l002` shipped
still-unclean and passed. **The leg had 8 voiced rows, not 12, so the
act-topology change is FALSIFIED** and the named revert experiment is
deliberately skipped rather than spending a live roll.

Re-deriving `line_text_sha256` alone does not fix it:

| surface | site | if stale |
|---|---|---|
| `_CodexTailFinalizer.expected` | `_otr_scifi_codex.py:3301-3308` | `_proof` is expected-driven on BOTH prongs; `after_save` re-proofs at `:3364` |
| `meta.scifi_codex.accepted_lines` | `:4057-4058` | ships the full pre-clean TEXT dict -- text nobody says |
| `meta.content_authorship` | `:4430-4435` and `_otr_scifi_fable2.py:3026-3033` | compared to LIVE text (`_otr_content_authorship.py:194-197`), enforced at `_otr_freeze_cascade.py:803` -> `needs_full_rerun` |
| voiced-row COVERAGE set | `_otr_ledger_cleanup.py:256-263` | cleanup can blank a row out of `_voiced_rows`; coverage validation fails |

**Boundary: between `:6762` and `:6831` in `_run_writer_tail` -- AFTER
`run_ledger_cleanup`.**

`scifi_news_pro` IS audited: `Fable2TailParts` carries no `tail_finalizer`
(`_otr_scifi_fable2.py:3141-3148`; writer passes `None` via `getattr` at
`:4190-4193`, so `:6830-6831` skips `before_save`), but the lane stamps
`content_authorship` and the freeze cascade enforces it terminally. **One shared
reseal for both lanes.**

**LAW-7 CORRECTION.** "Fails loudly" must not mean "raises on the render path".
**Today's `CodexPreTailAuditError`, killing a render after 13.6 minutes of
finished work, is itself a law-7 violation and gets FIXED, not preserved.**
Reject the transaction, restore pre-transform rows, stamp a degradation receipt,
let the episode ship.

The reseal must be a **byte-identical no-op when nothing was cleaned**
(`tests/test_scifi_codex_lane.py:598-635`), must keep accepting historical
frozen ledgers (`:718-755`), and must keep the unattributed-divergence
regression failing (`:627-635`). Also restamp `meta.writer_word_delivery`
(`:6486`), which is stamped pre-clean and never re-verified.

**Not a mutator:** `stamp_text_for_tts_delivery` writes only `text_for_tts*`
(`_otr_readiness.py:317-355`). The comment at `:264-266` claiming
`phase_7_audio_readiness` rewrites canonical text is stale doc drift.

---

## D3 -- THE `END` DELIMITER

`_otr_fable2_markup.py:41` `_RE_END = re.compile(r"^END\.\s*$", re.IGNORECASE)`
demands a literal period. The model wrote bare `END`: past `_RE_END` (`:545`),
past `_RE_SPEAKER` (`:548`, needs a colon), onto `BAD_LINE_SHAPE` (`:552`) with
detail `"END"`; `p.on_end` never fires so `:566` adds `MISSING_END`. **Both
reported defects, one cause.** Reproduced offline; not length-dependent.

**Fix is ONE regex plus one message change.** Widening `_RE_END` makes `**END**`
unwrap automatically through shape 4's existing result-gate
(`:55-60,108-131`) -- three independent readers confirmed; no
`_TRANSPORT_CLASSIFIERS` change needed. Golden fixtures use `END.` and stay
green (`tests/test_fable2_markup.py:114,312,433`).

**The generalizable half:** the ladder DOES repair
(`_otr_scifi_fable2.py:2180-2192`) and still failed four times because the
message says the END is malformed AND missing without stating the required
shape. Every transport defect must state WHAT IS REQUIRED. Scope to the fable2
transport catalog, not a repo-wide rewrite.

**THE LLM-PARSE FORK IS CLOSED.** Rejected by four independent reviews plus the
driver. Decisive argument: a misspelled speaker (`MACBET:`) could be
hallucinated into structure (`SCENE:`), silently deleting dialogue -- and
mis-attribution is one of only two things this project calls a failure. The repo
already measured an attribution judge unstable (3/6 then 1/6 on identical
fixtures) and ships it disabled. The model is ALREADY in the repair loop.

---

## D4 -- VOICE GENDER

Measured per line on the delivered masters; windows verified contiguous and
non-overlapping.

**CONFIRMED, `midnights_ticktock`, reciprocal inversion across twelve lines:**
GERTRUDE (a woman) male on all six (111.9/112.4/105.0/111.1/110.1/107.5 Hz);
**LORD RONALD (a man) female on all six** (279.1/233.0/186.0/241.3/281.5/269.7
Hz) -- a second instance not originally reported. Audible in the script: the
male-voiced character is addressed *"Miss McFiggins"*, the female-voiced one
*"Lord Ronald"*.

**NOT REPRODUCED, `kindling_the_past`:** JULIANA female on all six lines
(212-300 Hz). A pooled-F0 routing-fault hypothesis was raised in review and
refuted per line. **The invented-lane half is dropped.**

**Everything below the gender decision is correct.** The picker honoured the
tags (`vz_bill_boerst` measured 126 Hz to the male slot, `vz_donor_glenn`
measured 313.9 Hz to the female slot) and the voice bank audited clean -- 41
references, zero label disagreements. **No voice-bank relabelling is needed.**

**ROOT CAUSE:** `cast_source_contract.gender_by_name` is `{}` because
`config/source_banks/public_domain_story/sources/gertrude_governess.provenance.json`
does not exist -- one sidecar for the corpus. Blind 40/40/20 roll inverted the
pair. Both slots are `source_owned` so `_repair_ensemble_names` exempts them
(`_otr_casting.py:682-684`), and GERTRUDE is in no name-pool bucket anyway.

**FIX:** the vendor-time stamper from
`docs/2026-08-05-character-gender-ladder-SPEC.md`. **Operator ruling:** determine
the gender with a model call, then pick from the matching pool -- do NOT
hand-balance the name lists. Sidecars need invalidation metadata (source content
hash, model/version, evidence span, timestamp).

**LANDMINES, measured and in-code (`_otr_casting.py:736-745`):** feeding pins as
`prior_genders` makes the allocator push the other way --
`_plan_gender_distribution(1, ['male'])` returns female on 400/400 seeds.
Re-calling with a reduced count changes shuffle stream consumption and
desynchronizes replay. The shipped design overrides IN PLACE (`:762-768`). Any
proposal that re-plans the distribution is rejected on arrival.
`_PINNABLE_GENDERS` is binary, so an undetermined gender **preserves the
existing roll and stamps an unresolved receipt** -- never fabricates a pin.

**SPUN OUT:** `kindling_the_past` c02 is named `TARIQ SCOTT` while its
description reads *"40s, HENRY BARTEL, Seasoned Craftsman"* and its speech
signature says *"Henry's sentences..."*. Needs a character-record invariant.

---

## D5 -- THE CODA MUST NAME THE WORK

`_otr_provenance.py:95-100` `_CODA_BY_STATUS` is a flat `{status: sentence}`
table; `spoken_coda_line()` (`:103-128`) reads ONLY `prov["status"]` -- zero
per-work input, so it can never name a title. Shakespeare is the same table by
ABSENCE: no `licensed_*` key, so `:128` returns `""`.

Title and author already exist on every public_domain episode in
`meta["source_meta"]` (`_otr_public_domain_sources.py:557-558`), absent only
from the `source_rights` sidecar fed to `normalize_provenance` (`:4058`).

**Operator rulings 2026-08-15:** shakespeare names the play AND Shakespeare,
never the edition or licensor; public_domain names title AND author;
media_archive names the actual post. Standing rule: **all RSS-fed lanes close by
naming the source story as a historical learning lesson, science-news style.**

**DISPLAY RULE, settled by data against the entire panel:** name `title` +
`author`, **NEVER** `label`/`unit_label`. All four reviewers recommended
`"{unit_label}, from {title}, by {author}"`; all four are wrong. 60 of 65 units
have `label != title`, but `label` is a SCENE DESCRIPTOR -- dracula *"Harker at
the castle"*, frankenstein *"The meeting on the ice"*, monkeys_paw *"Three
wishes"*. That rule would name a PASSAGE as a WORK on ~57 of 65 units. The
operator's own example, *"The Time Machine, by H. G. Wells"*, is exactly the
`title` field. Shakespeare: `play_title` + Shakespeare.
(Manifest key is `label`; `source_meta_from_unit` renames it `unit_label`.)

**GOVERNANCE:** the 2026-08-05 ruling *"A LICENSED SOURCE GETS NO SPOKEN LINE"*
is documented in-code at `_otr_provenance.py:111`. The 2026-08-15 ruling
supersedes it for AUTHOR + WORK only. **Remove the suppression explicitly and
document the supersession** -- do not bolt a branch around a live ruling.

**HARD CONSTRAINT:** `_SPOKEN_CODA_SOURCES` is a closed vocabulary
(`OTR_LedgerScriptWriter.py:262`) with a write-time raise (`:2795-2800`) and a
set-equality pin (`tests/test_announcer_close_routing_matrix.py:554-555`).
**A new `spoken_coda_source` value is a build-breaker.** The archive close is a
better-fed `news_close_brief`, not a new type.

Anonymous works: 0 of 65 sources lack title or author (schema-enforced), so the
gap is theoretical. Keep a defensive degraded coda plus receipt; never raise.

---

## D6 -- `media_archive` ALWAYS TAKES ENTRY 0

`_otr_media_archive_sources.py:221`
`payloads[_configured_index() % len(payloads)]`, index defaulting to `"0"`
(`:191-196`). Feeds arrive newest-first, so it retells the newest post forever.
**That selection path has no test -- verified.**

Copy the science-news idiom (`story_orchestrator.py` `_load_news_history`
`:1193`, `_record_news_usage` `:1236`, `_llm_rank_news_candidates` `:1272`,
`_llm_rerank_with_bodies` `:1394`, recorded `:1880-1889`): dedup against
history, a MODEL ranking candidates, and recording the choice. **Separate
history file** -- reusing `news_history.json` cross-contaminates two pools tuned
for different cadences. The second body-rerank pass is optional.

**PREREQUISITE:** the post headline is never stamped durably.
`_otr_source_payload.py:585-621` builds `source_meta` as label/url/date only,
and `news_seed_receipt` is never passed for media_archive (`:698-701`) so
`_stamp_news_seed_receipt` early-returns (`:1683-1685`). **Stamp the headline at
SELECTION time** (`_rss_source_fetch_result`, `:604-610`), not as a coda patch.

Specify before implementing: explicit-index override precedence, partial feed
failure, URL canonicalization, dedup, TTL, empty-pool fallback, ranking-failure
fallback, concurrency. Use an atomic reservation committed when the ledger
seals -- recording only after success leaves the no-repeat claim false.

Interface note: `_otr_source_payload.py:687-701` discards `load_config`/`policy`
and `fetch_media_archive_rss` does not accept them.
`tests/test_source_payload_chunk3.py:472-474` pins the exact 3-kwarg contract
and breaks by design.

---

## D7 -- THE TITLE NAMES THE WRONG PLAY

`tempests_midnight_revelations` is Macbeth end to end (cast MACBETH/BANQUO,
`source_ref "folger-macbeth:act1-scene3-witches"`, `play_title "Macbeth"`).
**Scene selection is CORRECT.** `_generate_title_from_script`
(`OTR_LedgerScriptWriter.py:1349-1526`, called `:6293-6305`) is never passed
`source_meta`/`play_title`/`play_code`/`source_ref`, so it free-associated
"Tempest" from the storm sound-world -- and The Tempest is a DIFFERENT play in
the same manifest (`curated_scenes.sample.json:27-49`).

Threading identity in is necessary but NOT sufficient: there is no semantic
post-validator. Add a positive source-identity anchor plus a **deterministic,
code-side collision check** against the other configured work titles; reject and
re-ask once. Collision semantics: normalized whole-title or explicit-title-phrase
matches, never arbitrary shared words -- otherwise "storm" false-collides.

**Law 2:** the fallback must be a previously MODEL-AUTHORED candidate that
passed the checks. Python may select or reject; it may not synthesize a title.

**TRAP:** the title prompt must not contain `"news"`, `"headline"`, `"article"`
or `"RSS"` (`tests/test_writer_title_scratchpad.py:176-179`). Categorical
wording only; **no forbidden example ever enters a prompt** ("Never name the
feared failure" -- writing "no Arkham" implants Arkham).

---

---

## r2 CORRECTIONS -- the plan was NOT codeable; these make it so

### The bridge-only boundary does not exist yet (blocks D1)

The r1 panel converged on "clean the bridge upstream of composition". **That
boundary is not a switch to flip and the scoping rule was circular.**

* `run_ledger_clean()` accepts and mutates an ENTIRE ledger
  (`_otr_ledger_clean.py:1590-1990`).
* The coda is already composed and written EARLIER
  (`OTR_LedgerScriptWriter.py:2674-2825`).
* **The compose flag is DERIVED from whether the cleaned bridge is empty**, so
  `news_coda_bridge` vs `news_coda_fact_only` **cannot identify the row before
  cleaning.** Keying the exemption on that flag is circular.

Required: a component API (`clean_spoken_component(...) -> ComponentCleanResult`),
a STABLE PRE-COMPOSITION ownership marker, a whole-ledger skip rule keyed on that
marker, and ONE post-clean call that composes canonical `lines[].text`.
Cost guard: the component bridge is judged ONCE and the composed fact-bearing row
must not then incur a second whole-row judge call (per-row ceiling documented at
`OTR_LedgerScriptWriter.py:6715-6744`).

### The reseal point is later than stated, and rollback is wider (D2)

Reconcile **immediately after `run_ledger_cleanup()` and BEFORE
`stamp_text_for_tts_delivery()`** -- the writer currently runs clean, cleanup,
TTS-delivery stamping, consistency, telemetry, and only then the lane finalizer
(`:6745-6834`).

Take a transient pre-clean snapshot and atomically reconcile or roll back ALL of:
`text`, `skip`, `tts_skip_reason`, counts, compose flags, `text_for_tts*`,
`_CodexTailFinalizer.expected`, `line_text_sha256`, `accepted_lines`,
`content_authorship`, coverage, `writer_word_delivery`. **A row-only rollback
leaves false receipts.**

**Do NOT overwrite `accepted_lines`** (`_otr_scifi_codex.py:4016-4060`) -- that
destroys the distinction between ACCEPTED MODEL TEXT and CLEANED CANONICAL TEXT.
And restamping `content_authorship.line_proofs` does not PROVE the accepted
artifact still owns the transformed text: schema v1 binds artifact and final line
hashes but has **no transition chain** (`_otr_content_authorship.py:83-207`).
Build content-authorship v2 OR a separately validated transition receipt
(pre/post hashes, authorized stage, affected ids, cleaner receipt), and
**preserve v1 validation for historical ledgers.**

**Scope the four-surface reseal to lanes that CARRY those proof surfaces.** Not
"every lane crossing the shared tail" -- legacy adaptation lanes need
coda-component protection, not fabricated codex/content-authorship state.

### D7's terminal fallback is logically impossible as written

"Must pass the collision check" + "must be model-authored" + "must never raise"
cannot all hold: the first candidate may collide, the single re-ask may collide,
and then no validated model-authored candidate exists. Current code falls back to
`outline.title` with NO collision validation (`:1349-1526, 6293-6320`).
**Pick one:** additional bounded model calls, OR an explicit degraded colliding
title shipped with a receipt, OR an authorized deterministic source-title
fallback (which relaxes Law 2 and must say so).

Also specify the collision inventory and normalization concretely -- which
manifests are scanned, article handling, punctuation/Unicode normalization,
exact phrase boundaries. "Explicit-title-phrase" is not executable.

### `[END]` was over-promised (D3)

The matrix requires `[END]`, but the canonicalizer does NOT unwrap brackets and
`_RE_END` requires `END.` (`_otr_fable2_markup.py:41-60, 108-199`). Either pin
the paired grammar explicitly -- e.g. `^(?:END\.?|\[END\.?\])\s*$` -- or drop
`[END]` from the accepted set. Test unpaired brackets and content-bearing
variants as FAILURES. Name the function that emits the required-shape diagnostic.

### D6 reservation and empty-pool (blocks the no-repeat invariant)

`fetch_media_archive_rss()` returns one payload immediately (`:191-221`),
`_rss_source_fetch_result()` has no reservation token (`:585-621`), and the
legacy history code is unlocked read-modify-write best-effort
(`story_orchestrator.py:1193-1268`). Define candidate/reservation/history
schemas, canonical-URL rules, lock or atomic-replace behaviour, lease expiry,
explicit-index precedence, and a reservation token carried through
`SourceFetchResult`. **Commit only after the terminal save and audit succeed**
(`:6890-6915`), with guaranteed release or lease expiry on every exception.

**Empty pool contradicts no-repeat:** with one unexpired candidate and two
concurrent renders the second must repeat, refuse, or break the matrix.
**Choose a typed pre-generation `MediaSelectionUnavailable`** -- consistent with
Law 7's structural-refusal allowance. Silent reuse is not. Also define
ranking-failure behaviour: `_llm_rank_news_candidates()` RAISES rather than
falling back when `load_config` is present (`:1272-1388`).

**Do NOT import `story_orchestrator`** to reuse its private selectors -- it
imports optional Transformers and runtime modules at module load (`:25-75`).
Extract a lightweight selector/history module, or inject a ranking callback.

### Sidecar fork RESOLVED -- vendor all 65

Codex overrules the on-demand proposal: the render reader is deliberately
local/read-only (`_otr_roster_gender.py:226-263`) and on-demand render caching
would need a SECOND storage and merge authority. **Vendor all 65 once**, with
stale detection from `body_sha256` plus ladder/model/prompt versions, resumable
generation, and NO render-time network or cache writes.

### Law 7 vs `_check_g14_provenance_publish` -- RESOLVED

It is part of the general freeze audit and raises a Phase-10 exception
(`_otr_ledger_freeze.py:680-738`; `tests/test_provenance_v4.py:197-237`).
**Smallest consistent fix: stamp a non-publishable eligibility receipt, permit
ledger freeze and render, and enforce the block at `obs_publish`.** A render may
complete and be non-publishable.

### "Byte-identical no-op" narrowed

An unconditional reseal receipt necessarily changes serialized bytes. The
guarantee is: a no-mutation run leaves **the four protected surfaces** unchanged
and emits **no reseal marker**; ordinary clean telemetry may still differ.

### D5 needs one identity adapter

`spoken_coda_line()` accepts only provenance (`:95-128`); public_domain identity
lives in `source_meta.title`/`author` (`_otr_public_domain_sources.py:540-580`)
while **Shakespeare uses `play_title` and has NO author field**
(`_otr_shakespeare_sources.py:428-458`). Define ONE extractor returning a typed
bibliographic identity plus a coda builder returning canonical text AND receipt,
with exact degraded behaviour for missing fields.

### Cut, confirmed

The documented whole-row-exemption fallback for D1 is **cut** -- it violates the
component-ownership architecture and invites a later shim instead of forcing the
boundary to be built. The second body-rerank pass for D6 is **cut**.

---

## DECISIONS TAKEN (r3) -- every "pick one" is now picked

The panel repeatedly and correctly refused to choose between options. Choosing
is the driver's job. These are decided; do not reopen without new evidence.

### DEC-1. The raise/degrade contradiction, resolved by LAYER

My own acceptance table said an unattributed divergence "degrades with a
receipt and does NOT kill the render", while
`tests/test_scifi_codex_lane.py:627-635` must "still raise". Both are true at
different layers:

* **The reconciliation helper RAISES a typed internal error.** `_proof` keeps
  its current behaviour, so the existing regression stays green and an
  unattributed divergence is still a hard failure at the helper boundary.
* **`_run_writer_tail` CATCHES it**, restores the complete transaction
  snapshot, stamps a degradation receipt, and continues. The render survives.

Law 7 is satisfied and the unit test is unchanged. Add separate tests for
helper rejection and for render-path survival.

### DEC-2. A TRANSACTION OBJECT, not a ledger snapshot

A transient ledger snapshot cannot restore the in-memory
`_CodexTailFinalizer.expected`. The transaction snapshots **both** ledger
surfaces **and** finalizer state before cleaning, and restores both on any
failure. Reconciliation runs immediately after `run_ledger_cleanup()` and
before `stamp_text_for_tts_delivery()`.

### DEC-3. A parent-linked TRANSITION RECEIPT, not content-authorship v2

`_otr_content_authorship.py` hardcodes `SCHEMA_VERSION = 1` and strictly
compares `proof_by_id[line_id]["text_sha256"]` to the live digest. **Keep v1
unchanged for historical ledgers** and add a narrow, separately validated
transition receipt: schema version, authorized stage, cleaner receipt hash,
affected line ids, pre/post text hashes, pre/post voiced coverage, and the
parent content-authorship receipt hash. Its validator wires into the freeze
cascade. **Do NOT restamp v1 `line_proofs` as if transformed text were directly
accepted model output**, and do NOT migrate to v2 -- that carries historical
migration risk for no gain. Schema and validator land in chunk 0; transaction
mechanics in chunk 1; lane proof surfaces in chunk 3.

### DEC-4. D7 terminal fallback = THE SOURCE WORK'S OWN TITLE

Bounded model re-asks first. If they still collide, the episode takes the
source work's own title verbatim (`play_title`, e.g. *"Macbeth"*), stamped with
a receipt naming it as the deterministic fallback.

**Why this and not a degraded non-publishable colliding title:** Python is
SELECTING an existing metadata string, not composing prose, so Law 2 holds --
the same reasoning that lets the coda name the work. It terminates, it can
never collide (it IS this work's title), and on a lane where fidelity outranks
arc a plain correct title beats a wrong evocative one. Non-publishable receipts
are reserved for genuinely unresolvable identity, not for a title we can always
supply correctly.

### DEC-5. D1 -- reject a bridge that repeats the fact

"The fact appears exactly once" fails if the cleaned bridge itself contains it.
Before composition, deterministically detect a byte-identical occurrence of the
fact in the cleaned bridge; on a hit, reject that bridge and compose FACT-ONLY
with a receipt. This is detection and selection, not Python rewriting prose --
Laws 2 and 3 both hold.

Never reconstruct a bridge from already-composed text: that contradicts the
component boundary and breaks for fact-only rows and colon-bearing facts.

### DEC-6. D6 -- reservation is PENDING-BEFORE-GENERATION

"Commit only after terminal save" leaves a dual-write hole: the ledger can save
and the history commit then fail, the lease expires, and the same URL is
selected again. **Persist a durable PENDING reservation BEFORE generation,
carry its token and canonical URL in the ledger, and reconcile pending records
against sealed ledgers before releasing expired leases.** Plain lease expiry is
insufficient. Use an inter-process lock plus atomic replace -- a thread-only
lock does not protect concurrent render processes -- and convert lock timeout
to a pre-generation `MediaSelectionUnavailable`.

**Explicit-index override:** reject malformed and out-of-range values; do NOT
modulo-wrap. The override may bypass model ranking but must still acquire a
unique reservation; if that URL is reserved or unexpired, refuse
pre-generation.

**Ranking failure:** typed pre-generation refusal. Silently falling back to
entry 0 would bypass the operator-mandated model ranking, which is the whole
defect.

### DEC-7. D6 -- the fetcher signature change is AUTHORIZED

A model cannot rank without a config/client, and `fetch_media_archive_rss`
currently takes only `bank`, `technical_model`, `source_ref` while
`_otr_source_payload.py:693` discards `load_config`/`policy`. **Expanding that
signature is explicitly authorized**, and `tests/test_source_payload_chunk3.py:472-474`
is updated in the same change -- it is the test that pins the old contract.
Prefer injecting a ranking callback over importing `story_orchestrator`, which
pulls optional Transformers at module load (`:25-75`).

### DEC-8. D3 grammar, pinned

`^(?:END\.?|\[END\.?\])\s*$`, retaining the existing bold-unwrap path so
`**END**` still reaches it as bare `END`. Explicit FAILING tests for `[END`,
`END]`, `[END] trailing` and content-bearing variants, plus an explicit test
that `**END**` unwraps rather than matching the bracket branch.

### DEC-9. D3 IS CHUNK 3.5 -- it was missing from the build order

The prose said "D2 -> D3 is load-bearing" but the table never assigned D3 a
chunk, so as written the `END` fix would never be built. **D3 is chunk 3.5,
strictly after chunk 3.**

### DEC-10. Law 7 / publish gate is CHUNK 0.5

The `_check_g14_provenance_publish` change had no chunk. It lands before any
affected render acceptance: freeze stamps and preserves a non-publishable
eligibility receipt, render continues, and `obs_publish` rejects that exact
receipt. Verify the receipt actually propagates from freeze output to
`obs_publish` -- otherwise the block is lost between save and publication.

### DEC-11. D4 sidecar generation is a PRE-RENDER GATE (chunk 0.75)

All 65 sidecars generated, schema-validated and stale-checked before the
read-only runtime path is enabled. Define sidecar naming/keying, atomic
write/resume, model/config source, bounded retries, and what constitutes a
model/prompt/ladder version. **Runtime stale or missing entries preserve the
allocator's existing roll and stamp unresolved status, with no network
access.**

### DEC-12. D5 identity adapter, versioned, in chunk 0

One versioned adapter returning a typed bibliographic identity:
`source_kind`, `work_title`, `author`, `post_headline`, `canonical_url`, plus
field-level provenance. Mappings: public_domain from `source_meta.title`/
`author`; Shakespeare from `play_title` plus the constant "William
Shakespeare" (that lane has NO author field --
`_otr_shakespeare_sources.py:428-458`); media from the selection identity.
The coda builder returns canonical text AND a receipt; it never infers
lane-specific keys ad hoc. A missing title yields a detected degraded coda plus
a non-publishable receipt, never an exception.

### DEC-13. D7 collision matching, made executable

Deterministic configured-title matching only -- no fuzzy or model judgment.
Normalization pipeline: Unicode NFC, case folding, punctuation stripped,
whitespace folded, leading articles handled, whole-title or explicit
title-phrase boundaries. The current work is excluded by **stable work
identity**, not by title text. Scanned inventory: the configured
`curated_scenes` manifest for the active bank.

---

## r3 WIRING CORRECTIONS -- where the code actually is

### DEC-5 AMENDED: the component clean happens IN PLACE, not at the tail

"One final composer writes `lines[].text` exactly once" never said WHERE, and
the natural reading -- at the shared clean boundary -- **is wrong**. The factual
close is composed and written at `OTR_LedgerScriptWriter.py:2681-2831`; TITLE
GENERATION reads the ledger at `:6285-6305`; the whole-ledger cleaner runs much
later at `:6748-6754`. Deferring final composition to the clean boundary would
make title generation AND the intervening reflection passes consume a
bridge-only or placeholder close.

**Split the I.5 path IN PLACE:** author bridge -> `clean_spoken_component()` ->
`finalize_news_coda_surface()` -> ONE `patch_line_text` -- and only then let
title and reflections run. The later whole-ledger cleaner SKIPS that row via the
marker.

Add a sequencing assertion: the factual coda is complete BEFORE
`_generate_title_from_script` and both reflection passes. Exact-once
audio/caption assertions cannot detect a temporarily incomplete ledger consumed
upstream.

### The component receipt would be DESTROYED under the obvious key

`run_ledger_clean()` CREATES AND OVERWRITES `meta.ledger_clean`
(`_otr_ledger_clean.py:1630-1678, 1843-1846`). A component receipt stamped
earlier under that same key disappears. **Store component receipts under a
separate durable key, or explicitly merge them into the later episode
receipt.**

`ComponentCleanResult` fields: final bridge text, outcome, model-call count,
before/after hashes, findings.

**Use ONE shared marker constant imported by both the I.5 emitter and the
whole-ledger skip reader** -- per `BUG_BIBLE.yaml` entry `12.86`, the
producer/consumer mismatch rule. Two independently-spelled string literals is
exactly that bug.

### ComfyUI caching: the publish decision is hidden external state

`OTRMasterAudioMux.IS_CHANGED()` hashes only `clip_manifest_json`
(`otr_master_audio_mux.py:546-549`). A publication-eligibility decision that
lives outside that hash means **ComfyUI can reuse a cached terminal result
after eligibility changes.** Include the matched ledger's
eligibility/episode digest in `IS_CHANGED`, reusing the episode/stem validation
already applied to the in-flight ledger at `:416-435`. Never gate one episode
using a stale singleton.

### The publish gate has to touch the MUX, not just the freeze

OBS publication happens UNCONDITIONALLY inside `OTRMasterAudioMux.mux()`
(`:551-569`) while G14 currently raises into freeze errors
(`_otr_ledger_freeze.py:695-697, 720-739`). The chunk: G14 stamps durable
eligibility WITHOUT a freeze error, then the mux validates the matching
in-flight ledger before `_publish_to_obs`. An ineligible episode **retains the
archival final, returns success with `obs_publish BLOCKED`, omits the OBS copy,
and stamps NO false `obs_final_path`.**

### The ranker cannot use the writer's slot scheduler at the fetch point

RSS selection happens in `_resolve_inputs()` (`:3841-3888`), but `_SlotScheduler`
is not constructed until `:3969-3980`. **Do not move all source resolution
later** without re-auditing the writer. Extract a lightweight selector/ranker
accepting `technical_model`, `load_config` and `policy` that lazily requests the
technical slot, matching the existing science fetch contract. Preserve the
bounded network seam at `_otr_feed_fetch.py:64-95`. Ranking failure is defined
BY BACKEND CLASS -- the GGUF path raises when `load_config` is present
(`story_orchestrator.py:1365-1388`); remote/non-GGUF behaviour must be stated.

### The reservation token has no path today

`SourceFetchResult` carries only payload/meta/rights/document
(`_otr_source_payload.py:134-155`); `normalize_fetch_result_with_document()`
returns a four-tuple and drops other transient state (`:228-240`);
`_resolve_inputs()` returns no reservation field
(`OTR_LedgerScriptWriter.py:2040-2109`). Add a typed transient
`selection_reservation`, propagate it through a deliberately updated normalizer
contract and `resolved`, and commit only after BOTH `led.save()` and
`tail_finalizer.after_save()` succeed (`:6894-6908`). Lease expiry must cover
every earlier exception path.

### D4: two different "undetermined" states, and the invalidation is currently inert

The vendor ladder is TOTAL and forbids `unknown` sidecar rows
(`docs/2026-08-05-character-gender-ladder-SPEC.md:126-165,417-484`). Only a
RENDER-TIME cast name that fails to JOIN a complete sidecar may preserve the
existing roll. State that distinction; they are not the same thing.

**And the proposed invalidation metadata would never affect render behaviour:**
`load_roster_characters()` reads only `characters` and ignores `body_sha256`,
model and prompt metadata (`_otr_roster_gender.py:222-252`). Add selected-unit
body-hash validation and a durable stale/missing/join-miss status reaching
`cast_source_contract` -- otherwise stale sidecars keep pinning voices and the
metadata is decoration.

Corpus gate: require one FRESH, COMPLETE sidecar per manifest unit and verify
`gender_ladder.body_sha256` against the normalized text. **A count of 65 files
permits stale or duplicated receipts.**

### D3: a dedicated defect helper, not an overload

Do NOT overload `_standalone_stage_direction_repair_note()`
(`_otr_scifi_fable2.py:1856-1944`) -- it handles different defect data. Add a
dedicated typed-defect helper that states the four accepted END forms. The
current repair message only says "TITLE through END" (`:2180-2192`).

### BUILD-ORDER AMENDMENT

**D7 is NOT independent** -- its positive identity anchor depends on chunk 0.
Corrected: *"D4's pin application is independent; D7 starts after chunk 0."*
Put the G14-to-mux relocation inside the D5 chunk and the reservation
propagation inside the D6 chunk, so **each green commit contains its complete
producer/consumer pair.**

### Two acceptance tests the matrix was missing

* A MULTI-PROCESS reservation test (not threads): two selectors race for one
  available canonical URL, exactly one reservation succeeds, and expiry/release
  then permits a later selection.
* A terminal integration test proving `research_only` yields a canonical
  episode final, NO OBS copy, no `obs_publish OK`, and a durable
  non-publishable receipt. Current tests pin only the freeze-time failure.

## ACCEPTANCE -- a green suite proves none of this

| Defect | Invariant | Artifact assertion |
|---|---|---|
| D1 | the immutable fact appears byte-identically, exactly once | delivered closing AUDIO and caption both contain the source fact |
| D2 | final hashes equal shipped text; every diff attributed | sealed ledger ships; unattributed divergence degrades with a receipt and does NOT kill the render |
| D3 | every supported END form resolves to one delimiter; no speech line touched | parsed script reaches ledger generation; lane completes |
| D4 | a source-backed pin controls the selected voice WITHOUT changing allocator traversal | GERTRUDE and LORD RONALD no longer render inverted, measured per line |
| D5 | required work identity is spoken and matches durable metadata | closing audio, captions, ledger and credits name the same work |
| D6 | selection durably identified; concurrent runs never pick the same unexpired URL | headline stamped at selection; the same post is not retold |
| D7 | no collision with another configured work title | rendered title names no different play; Macbeth-not-Tempest regression fixture |

## TEST FALLOUT -- verified

`test_scifi_codex_lane.py:598-635` (reseal must no-op), `:627-635` (unattributed
divergence must still raise), `:718-755` (grandfather historical ledgers);
`test_ledger_clean_stage.py:186-194` (flag-keyed exemption only), `:523`;
`test_public_domain_library_roll.py:65`; `test_provenance_v4.py:66-70,185,193`;
`test_spoken_citation_audit.py:119-134`;
`test_announcer_close_routing_matrix.py:554-555`;
`test_writer_title_scratchpad.py:176-179`;
`test_source_payload_chunk3.py:472-474`; `test_fable2_markup.py:114,312,433`.

**Coverage gaps to close with the work:** `Fable2TailParts` appears in ZERO
tests, so "fable2 gets `None` in production" is unproven;
`compose_news_coda`/`_assemble_news_coda_surface` have no dedicated unit test;
the media selection path has none at all.

## REJECTED ON GROUNDING -- do not reinstate

* The anthology display-title rule (all four reviewers; refuted by the manifest).
* Hand-balancing the first-name pools (superseded by operator ruling).
* Cutting model ranking from the media picker (superseded by operator ruling).
* Recasting JULIANA (refuted by per-line measurement).
* Voice-bank relabelling (bank audited clean).
* A third LLM parse rung (rejected by four reviews plus the driver).
