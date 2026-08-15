# BUILD CONTRACT -- the 2026-08-15 bug-fix sprint

**Forward-only and single-authority.** Every operative decision is merged into
the defect sections and the build order below; the review history is deleted on
purpose, because a contract with two authorities is a build risk. Working
artifacts (driver anchor, raw reviews, per-round judgments) are local-only under
`docs/2026-08-15-bugfix-sprint/` and `kibitz-runs/2026-08-15-bugfix-sprint/`,
both gitignored by repo convention.

Branch `v2.0-alpha`. Baselines: suite 10532/110/1, Bible 20/26/3, variants 50/0.
**No production code has been written yet.**

**Review provenance, stated precisely.** Four rounds ran. Codex reviewed all
four. Antigravity reviewed r1 only and was quota-held (`RESOURCE_EXHAUSTED`
429) for r2/r3/r4. The cloud panel (GPT-5.6-sol, Gemini 3.1 Pro,
DeepSeek-v4-pro) reviewed r1 and r3. A Fable structural gate, a Sonnet
implementability audit and a Haiku fresh-eyes pass ran against r1. **14 external
reviews total. This was not a clean two-reviewer four-round arc** -- Antigravity
covered one round of four.

---

## LAWS -- reject any proposal that breaks one

1. No shims. Root cause only.
2. Code may DETECT and explain. Only a MODEL may rewrite PROSE.
3. **A source attribution is NOT prose.** It is a Python-owned fact with a receipt.
4. No content guardrails on generated episodes.
5. Story quality is CLOSED. Correctness defects are carved out.
6. No count chasing. No test may pin a word/beat/act count.
7. **A render must not die.** Structural refusal is allowed BEFORE generation.
8. One owner per ledger field.
9. Runaway guards are code-side and stay.
10. No `INPUT_TYPES`, widget, node-registration or `workflows/otr_canonical.json`
    change. `OTR_MasterAudioMux` stays registered via `__init__.py:315`.

---

## BUILD ORDER -- the only authority on sequence

| # | Chunk | Contains |
|---|---|---|
| 0 | **Identity + transition schemas** | the versioned bibliographic identity adapter; the transition-receipt schema and its composite validator. No standalone "fact schema" -- nothing consumes one. |
| 0.5 | **Publication eligibility, end to end** | the eligibility aggregator (producer), `OTRMasterAudioMux` consumption, the `IS_CHANGED` cache key, and the blocked-path behaviour. One complete producer/consumer pair. |
| 0.75 | **D4 vendor gate** | all 65 sidecars generated, schema-validated, freshness-verified. |
| 1 | **Transaction mechanics** | snapshot/restore of ledger surfaces AND finalizer state; the degradation receipt. |
| 2 | **D1 component boundary** | `clean_spoken_component`, the ownership marker, the in-place composer. |
| 3 | **D2 lane integrations** | both content-owned lanes' proof surfaces. |
| 3.5 | **D3 `END` grammar + its diagnostic** | strictly after 3. |
| 4 | **D5 non-media codas** | shakespeare, public_domain. |
| 5 | **D6 selector + reservation** | includes reservation propagation. |
| 6 | **D5 media_archive close** | consumes chunk 5's identity. |
| -- | **D4 pin application** is independent. **D7 starts after chunk 0.** | |

Every chunk ends green, is pushed, and carries its own documentation/coverage
gate (see ACCEPTANCE).

---

## D1 -- THE CLEAN STAGE ERASES PYTHON-OWNED FACTS

`nodes/_otr_ledger_clean.py` (shipped 2026-08-14) has a model rewrite the
closing announcer row, which carries a deterministic Python-owned attribution.

Confirmed on three live ledgers. `reel_of_mystery` b016 composed
`<bridge>": "` + *"In other news, the Library of Congress announces its film
loans for the month, including 'None But the Lonely Heart', 'Symphony of
Swing', and 'The Man With the Golden Arm'."* and SHIPPED *"Clarisse's gaze meets
the reel's enigmatic label"* -- the source note deleted entirely.
`midnights_ticktock` b016 paraphrased the Python-owned `spoken_coda_line` output
while `meta` still advertises `spoken_coda_source: "provenance"`.
`ghost_of_elsinore` b016 rewrote the sign-off. Rewrite rate 9/14 voiced rows on
all three. **The interpreter was never at fault** -- `news_close_brief` is
factual in the artifact.

### The boundary

Split the I.5 path **IN PLACE**, at the existing composition site
(`OTR_LedgerScriptWriter.py:2681-2831`):

```
fact producer -> authored bridge -> clean_spoken_component() -> finalize_news_coda_surface() -> ONE patch_line_text
```

then title generation (`:6285-6305`) and the reflection passes run against a
COMPLETE close. **Do not defer composition to the whole-ledger cleaner at
`:6748`** -- title and reflections would consume a placeholder.
`run_ledger_clean()` later SKIPS that row by marker.

### The marker

**One shared constant**, imported by both the I.5 emitter and the whole-ledger
skip reader (two hand-spelled literals is `BUG_BIBLE.yaml` entry `12.86`,
producer/consumer mismatch). It is stamped **independently of the bridge
outcome** -- NOT `news_coda_bridge`/`news_coda_fact_only`, which are DERIVED
from whether the cleaned bridge came back empty and are therefore circular:
they cannot identify the row before cleaning, which is exactly when the skip
must fire.

Scoping by `speaker_role` is also wrong: announcer rows are legitimately
judged, and `tests/test_ledger_clean_stage.py:186-194` asserts
`judge_calls == 3` on a synthetic ledger whose announcer row has no flags, so a
role-keyed exemption drops it to 2.

### The receipt

`ComponentCleanResult`: final bridge text, outcome, model-call count,
before/after hashes, findings. Stored under a **separate line-id-keyed
component receipt key** -- `run_ledger_clean()` creates and OVERWRITES
`meta.ledger_clean` (`_otr_ledger_clean.py:1630-1678, 1843-1846`), so anything
stamped there earlier is destroyed.

### Failure and duplication

* A bridge that fails cleaning, repeats, or is invalid -> select **fact-only**
  with a receipt. Never drop the fact, never kill the render.
* Before composition, deterministically detect a byte-identical occurrence of
  the fact inside the cleaned bridge; on a hit compose fact-only. This is
  detection and selection, not Python rewriting prose -- Laws 2 and 3 hold.
* Never reconstruct a bridge from already-composed text: it breaks for
  fact-only rows and colon-bearing facts.
* The bridge is judged ONCE; the composed row must not incur a second whole-row
  judge call (per-row ceiling at `OTR_LedgerScriptWriter.py:6715-6744`).

`FIDELITY_BANKS` (`_otr_spoken_text_policy.py:83`) is NOT usable here: per
`_otr_ledger_clean.py:1739-1741` it only restricts which pattern KINDS may spend
a repair, and *"the judge still reads the line and its verdict counts
everywhere"* -- which is why it did not protect two of the three victim banks.

---

## D2 -- THE RESEAL IS FOUR SURFACES WIDE

`scifi_news` dies `CodexPreTailAuditError: line receipt mismatch for l004`.
`l004` is the first row the clean stage rewrote; `l001`/`l002` shipped
still-unclean and passed. The leg had **8 voiced rows, not 12**, so the
act-topology change is FALSIFIED and its revert experiment is deliberately
skipped.

Surfaces: `_CodexTailFinalizer.expected` (`_proof` is expected-driven on both
prongs; `after_save` re-proofs at `_otr_scifi_codex.py:3364`),
`meta.scifi_codex.line_text_sha256`, `meta.content_authorship`, and the
voiced-row COVERAGE set (cleanup can blank a row out of `_voiced_rows`).

**`meta.scifi_codex.accepted_lines` is PRESERVED, not rewritten** -- it records
what the model ACCEPTED, which is not what ships after an authorized clean.
Do not relabel cleaned text as accepted model output.

`scifi_news_pro` IS audited despite `Fable2TailParts` carrying no
`tail_finalizer` (`_otr_scifi_fable2.py:3141-3148`; `getattr(..., None)` at
`OTR_LedgerScriptWriter.py:4190-4193` so `:6830-6831` skips `before_save`) --
it stamps `content_authorship`, enforced terminally at
`_otr_freeze_cascade.py:803`. One shared contract covers both lanes.

### The composite validator

`_otr_freeze_cascade.py:249-252` calls `validate_receipt()`, which compares v1
hashes to LIVE post-clean text (`_otr_content_authorship.py:194-197`) -- so a
transition receipt sitting beside an unchanged v1 validator still fails.
**One composite validator:**

* untransitioned ledgers -> v1 unchanged (historical ledgers keep working);
* transitioned ledgers -> verify the parent v1 receipt against **pre-clean**
  hashes, validate the authorized transition, then compare live rows and
  coverage against **post-clean** hashes.

On success, update `_CodexTailFinalizer.expected` and `line_text_sha256` to
final canonical text, preserve `accepted_lines` and the v1 authorship receipt,
and **emit no transition on a no-op**. Keep v1, do not build schema v2.

Transition receipt fields: schema version, authorized stage, cleaner-receipt
digest, affected line ids, pre/post text hashes, pre/post voiced coverage,
parent content-authorship digest.

### Failure semantics, by layer

* `_CodexTailFinalizer._proof()` stays **fail-loud** -- `tests/test_scifi_codex_lane.py:627-635`
  calls that primitive with an unattributed mutation and must keep raising.
* The production transaction CATCHES a failed reconciliation, atomically
  restores the pre-clean snapshot, stamps a degradation receipt, re-proves the
  restored state, and continues. **Law 7 holds: today's behaviour -- killing a
  render after 13.6 minutes of finished work -- is itself the violation and is
  what gets fixed.**

Snapshot boundary: immediately before `run_ledger_clean()`. Reconciliation runs
after `run_ledger_cleanup()` and before `stamp_text_for_tts_delivery()`. The
snapshot is a TRANSACTION OBJECT covering row presence and order, `text`,
metrics, skip fields, compose flags, delivery fields, every proof/coverage
surface, **and the in-memory finalizer state** -- a ledger-only copy cannot
restore `expected`. Also restamp `meta.writer_word_delivery` (`:6486`).

No-op guarantee, narrowly: the protected surfaces and finalizer state are
unchanged and NO transition marker is emitted. Ordinary clean telemetry may
differ; do not assert whole-ledger byte equality.

`stamp_text_for_tts_delivery` is NOT a mutator (`_otr_readiness.py:317-355`
writes only `text_for_tts*`). Correct the stale comment at `:264-266` claiming
otherwise.

---

## D3 -- THE `END` DELIMITER

`_otr_fable2_markup.py:41` `_RE_END = re.compile(r"^END\.\s*$", re.IGNORECASE)`
demands a period. Bare `END` falls past `_RE_END` (`:545`), past `_RE_SPEAKER`
(`:548`, needs a colon), onto `BAD_LINE_SHAPE` (`:552`); `p.on_end` never fires
so `:566` adds `MISSING_END`. **Both defects, one cause.** Reproduced offline.

**Grammar, pinned:** `^(?:END\.?|\[END\.?\])\s*$`, retaining the existing
bold-unwrap path so `**END**` arrives as bare `END`. Unpaired brackets
(`[END`, `END]`), trailing content, and content-bearing variants remain LOUD
defects.

**The diagnostic:** a dedicated typed-defect helper stating the four accepted
forms. Do NOT overload `_standalone_stage_direction_repair_note()`
(`_otr_scifi_fable2.py:1856-1944`) -- different defect data. The current repair
message only says "TITLE through END" (`:2180-2192`), which is why the ladder
burned four rungs re-emitting `END`: it named the offence and never the required
shape. **Scope limited to the END diagnostic** -- no catalog-wide rewrite.

**The LLM-parse fork is CLOSED.** Rejected by four independent reviews plus the
driver: a misspelled speaker (`MACBET:`) could be hallucinated into structure
(`SCENE:`), silently deleting dialogue, and mis-attribution is one of only two
things this project calls a failure. The repo already ships its attribution
judge disabled after measuring 3/6 then 1/6 recall on identical fixtures.

---

## D4 -- VOICE GENDER

Measured per line on delivered masters, windows verified contiguous and
non-overlapping. `midnights_ticktock` is RECIPROCALLY INVERTED across twelve
lines: GERTRUDE (a woman) male on all six (111.9/112.4/105.0/111.1/110.1/107.5
Hz); **LORD RONALD (a man) female on all six** (279.1/233.0/186.0/241.3/281.5/
269.7 Hz) -- a second instance not originally reported. The script makes it
audible: the male-voiced character is addressed *"Miss McFiggins"*.

`kindling_the_past` does NOT reproduce: JULIANA female on all six (212-300 Hz).
A routing-fault hypothesis was raised and refuted per line. **The invented-lane
half is dropped.**

Everything below the gender decision is correct -- the picker honoured the tags
and the voice bank audits clean (41 references, zero label disagreements).
**No voice-bank relabelling.**

**Root cause:** `cast_source_contract.gender_by_name` is `{}` because
`gertrude_governess.provenance.json` does not exist; one sidecar exists for 65
units. The blind 40/40/20 roll inverted the pair. Both slots are `source_owned`
so `_repair_ensemble_names` exempts them (`_otr_casting.py:682-684`), and
GERTRUDE is in no name-pool bucket.

**Fix:** the vendor-time stamper from
`docs/2026-08-05-character-gender-ladder-SPEC.md`. **Operator ruling:** determine
gender with a model call, then pick from the matching pool -- do not hand-balance
the name lists.

**This SUPERSEDES the SPEC's "no render-path change" claim (`:576-614`).**
`load_roster_characters()` returns only rows and ignores all invalidation
metadata (`_otr_roster_gender.py:226-263`), so freshness fields would be
decoration. Pin a typed loader result carrying status
(fresh / stale / missing / join-miss), expected and observed hashes, and the
exact `cast_source_contract` fields they land in. Validate the selected unit's
`body_sha256` against the normalized text at read time.

**Two different "undetermined" states, not one:** the vendor ladder is TOTAL and
forbids `unknown` sidecar rows; only a RENDER-TIME cast name that fails to JOIN
a complete sidecar preserves the existing roll and stamps unresolved status.

**Staleness identity excludes `ran_utc`** -- it is audit metadata, and treating
a timestamp as invalidation state breaks idempotence. Preserve it on a no-op;
update it only when substantive sidecar content changes.

Corpus gate: one FRESH, COMPLETE sidecar per manifest unit with a verified
`body_sha256`. A count of 65 files permits stale or duplicated receipts.

**Landmines, measured and in-code (`_otr_casting.py:736-745`):** feeding pins as
`prior_genders` makes the allocator push the other way
(`_plan_gender_distribution(1, ['male'])` returns female on 400/400 seeds);
re-calling with a reduced count changes shuffle stream consumption and
desynchronizes replay. The shipped design overrides IN PLACE (`:762-768`).
Any proposal that re-plans the distribution is rejected on arrival.

**Spun out:** `kindling_the_past` c02 is named `TARIQ SCOTT` while its
description reads *"40s, HENRY BARTEL, Seasoned Craftsman"*. Needs a
character-record invariant. (PBUG-20260815-08.)

---

## D5 -- THE CODA MUST NAME THE WORK

`_otr_provenance.py:95-100` `_CODA_BY_STATUS` is a flat `{status: sentence}`
table and `spoken_coda_line()` (`:103-128`) reads ONLY `prov["status"]`, so it
can never name a title. Shakespeare is the same table by ABSENCE: no
`licensed_*` key, so `:128` returns `""`.

**Operator rulings:** shakespeare names the play AND Shakespeare, never the
edition or licensor; public_domain names title AND author; media_archive names
the actual post. Standing rule: all RSS-fed lanes close by naming the source
story as a historical learning lesson, science-news style.

**DISPLAY RULE: `title` + `author`, NEVER `label`/`unit_label`.** Settled by
data against the entire panel, all four of whom recommended an anthology form.
60 of 65 units have `label != title`, but `label` is a SCENE DESCRIPTOR --
dracula *"Harker at the castle"*, frankenstein *"The meeting on the ice"*,
monkeys_paw *"Three wishes"*. An anthology rule would name a PASSAGE as a WORK
on ~57 of 65 units. The operator's own example, *"The Time Machine, by H. G.
Wells"*, is exactly the `title` field. (Manifest key is `label`;
`source_meta_from_unit` renames it `unit_label`.)

**Identity adapter (chunk 0), versioned:** `source_kind`, `work_title`,
`author`, `post_headline`, `canonical_url`, plus field-level provenance.
Mappings: public_domain from `source_meta.title`/`author`
(`_otr_public_domain_sources.py:553-563`); Shakespeare from `play_title` plus
the constant "William Shakespeare" -- that lane has NO author field
(`_otr_shakespeare_sources.py:442-452`); media from the selection identity.
The coda builder returns canonical text AND a receipt. **Pin the exact coda
template per bank and the exact missing-field behaviour** -- "canonical text" is
not canonical while several valid sentences remain possible.

Media's Python-owned post identity and the interpreter-owned `news_close_brief`
must combine without giving one field two writers.

**GOVERNANCE:** the 2026-08-05 ruling *"A LICENSED SOURCE GETS NO SPOKEN LINE"*
(`_otr_provenance.py:111`) is superseded for AUTHOR + WORK only. Remove the
suppression explicitly and document the supersession.

**HARD CONSTRAINT:** `_SPOKEN_CODA_SOURCES` is closed
(`OTR_LedgerScriptWriter.py:262`, write-time raise `:2792-2800`, set-equality
pin `tests/test_announcer_close_routing_matrix.py:554-555`). **No new
`spoken_coda_source` value.** The archive close is a better-fed
`news_close_brief`.

Anonymous works: 0 of 65 sources lack title or author (schema-enforced), so the
gap is theoretical. Degrade with a receipt; never raise.

---

## D5a -- PUBLICATION ELIGIBILITY (chunk 0.5)

One versioned eligibility aggregator, **outside** the read-only audit:
`run_gap_audit()` is explicitly read-only (`_otr_ledger_freeze.py:666-671`), so
`_check_g14_provenance_publish()` must not stamp inside it, and D5's identity
failure must not become a second writer. **One producer, combining rights and
identity reason codes**, stamped for every ledger.

`OTRMasterAudioMux` consumes that exact receipt. Today OBS publication is
unconditional inside `mux()` (`otr_master_audio_mux.py:551-569`) and
`_stamp_terminal_paths()` ALWAYS writes an OBS path (`:493-539`).

* Ineligible: retain the archival final, return success with
  `obs_publish BLOCKED`, omit the OBS copy, and **explicitly clear/omit both
  OBS path aliases** -- stamp no false `obs_final_path`.
* Missing, malformed, or episode-mismatched eligibility -> treat as blocked.
* `IS_CHANGED()` currently hashes only `clip_manifest_json` (`:546-549`).
  **Add episode identity + eligibility digest**, reusing the episode/stem
  validation at `:416-435`. Never gate one episode using a stale singleton.

---

## D6 -- `media_archive` ALWAYS TAKES ENTRY 0

`_otr_media_archive_sources.py:221`
`payloads[_configured_index() % len(payloads)]`, index defaulting to `"0"`
(`:191-196`). Feeds arrive newest-first, so it retells the newest post forever.
**That selection path has no test.**

Copy the science-news idiom (`story_orchestrator.py` `_load_news_history:1193`,
`_record_news_usage:1236`, `_llm_rank_news_candidates:1272`, recorded
`:1880-1889`): dedup against history, a MODEL ranking candidates, recording the
choice. **The second body-rerank pass is CUT.**

**Prerequisite:** the post headline is never stamped durably.
`_otr_source_payload.py:585-621` builds `source_meta` as feed label/url/date
only, and `news_seed_receipt` is never passed for this lane (`:698-701`) so
`_stamp_news_seed_receipt` early-returns (`:1683-1685`). Stamp the headline at
SELECTION time into the chunk-0 identity.

### The contract table (pin before coding)

State-file path under `_otr_paths.otr_state_dir()`, **separate from
`news_history.json`**; schema and version; pending vs committed records;
canonical-URL algorithm (scheme/host case, default-port removal, fragment
removal, percent-encoding, tracking-query policy, original URL preserved for
attribution); history cap; TTL; reservation lease; lock timeout; partial-feed
behaviour; atomic replacement and crash recovery; malformed-state behaviour.

**Reservation is PENDING-BEFORE-GENERATION.** Commit-after-save leaves a
dual-write hole: the ledger saves, the history commit fails, the lease expires,
and the same post is retold. Persist a durable pending record before
generation, carry its token and canonical URL in the ledger, reconcile pending
records against sealed ledgers before releasing expired leases, and commit only
after BOTH `led.save()` and `tail_finalizer.after_save()` succeed
(`:6894-6908`). **Define commit/release when `tail_finalizer` is absent -- it is
absent for `media_archive`.** Lease expiry must cover every earlier exception
path.

Concurrency: an inter-process Windows-safe lock around read/reserve/commit plus
atomic replace while held. `os.replace` alone cannot serialize
read-modify-write and the existing history writer is unlocked (`:1236-1269`).
Bounded acquisition; timeout becomes pre-generation `MediaSelectionUnavailable`.

**Explicit index:** `OTR_MEDIA_ARCHIVE_ITEM_INDEX` may bypass ranking but still
requires a unique reservation. Reject malformed and out-of-range values --
**no modulo wrap**. If that URL is reserved or unexpired, refuse
pre-generation.

**Ranking failure:** typed pre-generation refusal. Silently choosing entry 0
would bypass the operator-mandated ranking, which is the defect itself. Define
behaviour per backend class -- the GGUF path raises when `load_config` is
present (`:1365-1388`); remote/non-GGUF must be stated explicitly.

### Wiring

`SourceFetchResult` carries only payload/meta/rights/document
(`_otr_source_payload.py:134-155`);
`normalize_fetch_result_with_document()` returns a four-tuple and drops other
transient state (`:228-240`); `_resolve_inputs()` returns no reservation field
(`OTR_LedgerScriptWriter.py:2040-2109`). Add a typed transient
`selection_reservation` and propagate it through a deliberately updated
normalizer contract.

**Expanding `fetch_media_archive_rss`'s signature is AUTHORIZED** -- a model
cannot rank without a config, and `_otr_source_payload.py:693` currently
discards `load_config`/`policy`. `tests/test_source_payload_chunk3.py:472-474`
pins the old contract and is updated in the same change.

**The ranker cannot use the writer's slot scheduler at the fetch point:** RSS
selection runs in `_resolve_inputs()` (`:3841-3888`) but `_SlotScheduler` is not
constructed until `:3969-3980`. Extract a lightweight selector/ranker taking
`technical_model`, `load_config`, `policy` that lazily requests the technical
slot. Do NOT import `story_orchestrator` (optional Transformers at module load,
`:25-75`). Preserve the bounded network seam at `_otr_feed_fetch.py:64-95`.

---

## D7 -- THE TITLE NAMES THE WRONG PLAY

`tempests_midnight_revelations` is Macbeth end to end (cast MACBETH/BANQUO,
`source_ref "folger-macbeth:act1-scene3-witches"`, `play_title "Macbeth"`).
**Scene selection is CORRECT.** `_generate_title_from_script`
(`OTR_LedgerScriptWriter.py:1349-1526`, called `:6293-6305`) receives no
identity, so it free-associated "Tempest" from the storm sound-world -- and The
Tempest is a DIFFERENT play in the same manifest
(`curated_scenes.sample.json:27-49`). The caller then falls back unchecked to
`outline.title` (`:6293-6315`).

**Depends on chunk 0** for the identity anchor.

**Collision check, code-side and deterministic.** Inventory: the selected bank's
configured manifest. Normalization: Unicode NFC, case folding, punctuation
stripped, whitespace folded, **leading articles -- name the set (`the`, `a`,
`an`) and strip before comparison** -- whole-title or explicit title-phrase
boundaries with positive and negative fixtures. The current work is excluded by
**stable work identity**, not title text. No fuzzy or model-judged matching.

**Terminal rule, guaranteed:** test every model-authored candidate AND
`outline.title` against the collision check. If none passes, take the source
work's own title verbatim (`play_title`) with a degradation receipt. Python is
SELECTING an existing metadata string, not composing prose, so Law 2 holds --
the same reasoning that lets the coda name the work. It terminates, and it can
never collide because it IS this work's title.

User-entered titles and lane-owned final-title overrides stay outside this
validator.

**TRAP:** the title prompt must not contain `"news"`, `"headline"`, `"article"`
or `"RSS"` (`tests/test_writer_title_scratchpad.py:176-179`). Categorical
wording only -- **no forbidden example enters any prompt** ("Never name the
feared failure").

---

## ACCEPTANCE

A green suite proves none of this. Every chunk carries: focused tests, the full
suite, variants, the Bug Bible regression, AST/BOM/zero-byte checks, and
`HEAD == origin` on `v2.0-alpha`.

| Defect | Invariant | Artifact assertion |
|---|---|---|
| D1 | the fact appears byte-identically, exactly once | **the announcer row's final `text` is byte-identical to `meta["provenance_coda_line"]`, AND `meta["ledger_clean"]["rows"]` carries NO entry for that `line_id`.** A row-edit entry means the judge touched it and the pass is LUCK regardless of how similar the output looks -- similarity cannot distinguish "protected" from "rewritten into something close". Then byte identity in captions and `text_for_tts` hashes, and the rendered slice via the project's ASR check. "Audio contains the fact" is not a binary assertion on its own. |
| D2 | final hashes equal shipped text; every diff attributed | authorized mutation succeeds; unauthorized helper call raises; the tail catches, restores every surface, stamps degradation and completes |
| D3 | supported END forms resolve to one delimiter; speech untouched | bare/dotted/bracketed/bold accepted; unpaired and content-bearing rejected; the required-shape diagnostic pinned |
| D4 | a source-backed pin controls the voice without changing allocator traversal | GERTRUDE and LORD RONALD no longer render inverted, measured per line |
| D5 | required work identity is spoken and matches durable metadata | closing audio, captions, ledger and credits name the same work |
| D5a | eligibility has one owner and survives caching | identical mux inputs with changed eligibility do not reuse a cached result; ineligible yields archival success, no OBS copy, no OBS path |
| D6 | selection durably identified; concurrent runs never pick the same unexpired URL | a real MULTI-PROCESS race proves exactly one reservation succeeds |
| D7 | no collision with another configured work title | Macbeth-not-Tempest regression fixture |

**LIVE-LEG TIMING, decided 2026-08-15 (driver + Antigravity, independently).**
**DEFER the qualifying leg until chunk 2 (D1) has landed.** Until the clean
stage is scoped, it may still rewrite the coda row, so a leg cannot qualify D5
either way: a pass is luck (non-deterministic model output) and a failure is
already known from three artifacts. Run it on `public_domain` pinned to
`gertrude_governess:main` once D1 is merged and green.

*Rejected on grounding:* a proposed "dry-run / ledger-only validation" third
option via `scripts/otr_canonical_api_run.py --dry-run`. The script and flag
both exist, but `--dry-run` only *"build[s] and dump[s] the API prompt without
POST /prompt"* (`:191`, `:222`) -- no writer runs, so it validates no ledger, no
receipt and no coda. Its GOAL is already met at zero VRAM by the seam tests in
`tests/test_source_identity_coda.py`, which mirror the writer's exact call
sequence against the real shipped meta shapes.

**Live legs through `workflows/otr_canonical.json`:** media_archive twice
(D1/D5/D6); public_domain (D1/D4/D5); shakespeare (D1/D5/D7); scifi_news (D2);
scifi_news_pro (D2/D3); plus a research-only terminal case proving an archival
final with no `obs_publish OK`, no OBS copy and a durable non-publishable
receipt.

**Documentation gate, per applicable chunk:** `docs/PROD_BUG_LOG.md` already
carries PBUG-20260815-01..08. On each green chunk, promote the covered defect to
the Bug Bible with executable coverage where automatable, append its
`otr_coverage_index.yaml` row, and commit AND push the survival-guide repo too
(`CLAUDE.md:119-135`).

## REJECTED ON GROUNDING -- do not reinstate

* The anthology display-title rule (all four reviewers; refuted by the manifest).
* Hand-balancing the first-name pools (operator ruling).
* Cutting model ranking from the media picker (operator ruling).
* Recasting JULIANA (refuted by per-line measurement).
* Voice-bank relabelling (bank audited clean).
* A third LLM parse rung (four reviews plus the driver).
* Whole-row exemption for D1 (recreates the circular flag dependency).
* content-authorship schema v2 (v1 plus one transition chain is sufficient).
* A standalone "fact schema" (nothing consumes it).
* Timestamp in the staleness predicate (breaks idempotence).
