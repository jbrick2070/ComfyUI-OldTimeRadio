# Character gender ladder -- driver anchor for the ONE review round (queue item 3a)

**Driver: Claude (Fable 5.1, Cowork), 2026-09-02. HEAD `2a977fab`, branch `v2.0-alpha`.
Every claim below was read at the real Windows files today; the line numbers are HEAD's.**
Spec under review: `docs/2026-08-28-character-gender-ladder-SPEC-v2.md` (193 lines). Prior
finding lists the reviewer must check the spec against: r2 Codex
(`kibitz-runs/2026-08-05-gender-ladder/r2/codex.md`, 11 must-fixes) and r3 Codex NO with 7 +
Antigravity yes-with-fixes with 9 (`kibitz-runs/2026-08-06-2026-08-06-gender-ladder-r3/r3/`).
The plan row: `docs/GO_FORWARD_PLAN.md` Section 1.2 -- "ONE review round, then code".

## 0. The question for the reviewer

Is each of the spec's four blockers (B1-B4) genuinely answered by the design in section 3
below, or merely restated? And are the seven driver decisions (D1-D7) the right defaults,
given what the code already does (section 1) and what is actually still broken (section 2)?
Say NO where a decision would ship a wrong voice, cost a render, or re-open a closed defect.
Do not review the deleted mechanisms (web search tier, full-text LLM extraction, the in-loop
vendor hook): they are gone from v2 and will not be rebuilt.

## 1. What already exists (re-grounded today) -- the spec was written before some of it

| piece | where | status |
|---|---|---|
| Tier 1 (roster) + tier 2 (pronoun scan) stamper for the prose lane | `scripts/otr_stamp_character_genders.py` (276 lines; `_decide` :101-129, `stamp_unit` :151) | SHIPPED. 65 units stamped; tier counts across the corpus `roster 0 / pronouns 132 / llm_web 0 / name_frequency 0`. The stamper writes `characters[]` + `gender_ladder` only, carries the fetcher's fields forward (`STAMPER_OWNED_SIDECAR_KEYS`, `_otr_roster_gender.py:232`), is idempotent on content (`ran_utc` moves only when content moves), and OMITS a declined name rather than writing `unknown`. |
| Pronoun scan | `nodes/_otr_gender_pronoun_scan.py` (`scan_gender` :261; `SCORE_FLOOR = 8`, `DOMINANCE_RATIO = 3.0`, `MENTION_WINDOW_CHARS = 240`) | SHIPPED, tested (`tests/test_character_gender_sidecars.py::TestPronounScan`). Declines below the floor or the 3x margin. |
| Render-time join | `nodes/_otr_roster_gender.py`: `resolve_roster_gender` :314 (exact / short_form / qualified / contains; ambiguity abstains), `resolve_with_supplement` :438, `gender_map_for_names` :473 -> `OTR_LedgerScriptWriter.py:3551` -> `lock_cast(source_character_genders=...)` `_otr_casting.py:1860,2031-2043` -> `precompute_ensemble_slots(gender_by_name=...)` :708 | SHIPPED. Names the join cannot pin are OMITTED and fall to the 40/40/20 roll (`_otr_casting.py:577`). |
| Shakespeare curated supplement | `config/source_banks/shakespeare/roster_gender_supplement.json` (5 plays, 10 names; loader `load_gender_supplement` :392 requires male/female + non-empty evidence; may only FILL, never overrule a confirmed sidecar) | SHIPPED. `tests/test_roster_gender.py::test_every_shipped_cast_hint_resolves_except_the_two_left_to_the_operator`: 40 of 42 cast hints resolve; ARIEL and PUCK are left to the roll BY DESIGN. |
| `RosterGenderVerdict` | `_otr_roster_gender.py:209-219`: `gender / evidence / tier / matched`; SIX construction sites, all in this file (:298, :310, :311, :334, :364, :468), zero in tests | matches the spec's B4 count. |
| Name-frequency data | `config/cast_pools.py`: `_FIRST_NAME_GENDER_INDEX` :210 (the curated first-name buckets, upper-cased), `gender_of_first_name(name)` :1392 -> `male / female / unisex / unknown` | SHIPPED, used for the invented lanes' repair path. `tmp/gender_probability_v2.json` (153 names, gemma-4-E4B `share_male` per name) is a LAB output, untracked, not data. |
| Tier 3 prototype (the LLM verdict on a character IN ITS WORK) | `scripts/otr_gender_secondopinion_lab.py`: `_ask` :62 -- system "answer questions about characters in published literature ... answer 'unsure' ... never guess", user `In "<title>" by <author>, is the character "<name>" male or female?`, constrained to `CharacterGenderOpinion(gender: male|female|unsure, reason<=200)` via `nodes/_otr_constrained_generate.make_constrained_generate_fn(entry, Model)`; model loaded with `nodes._otr_model_loader.load_llm(model_id, optimization_profile="Standard")`; default model `google/gemma-4-E4B-it`; asks twice and reports flip-flops | DIAGNOSTIC ONLY today (never writes). It is the exact question the operator ruled for tier 3, already proven to run locally. |
| Sidecar writer for Shakespeare | `scripts/otr_fetch_public_domain.py:388-400` writes `characters[]` rows `{name, roster_name, description, gender, gender_source}` from the parsed dramatis personae; an unmatched speaker gets `gender unknown / gender_source absent_from_roster` | SHIPPED. 14 Shakespeare sidecars, 85 rows, **32 `unknown`** (e.g. Comedy of Errors act 3 scene 1: ANTIPHOLUS OF EPHESUS, both DROMIOs, BALTHASAR, LUCE, ANGELO). |
| Corpus gates | `tests/test_character_gender_sidecars.py::TestVendoredCorpus` (every manifest unit has a sidecar; `test_no_row_is_ever_UNKNOWN` -- prose rows; body hash matches disk) | SHIPPED for the prose bank. No gate says a Shakespeare row may not be `unknown`. |

## 2. What is actually still broken (measured today)

1. **Prose lane, 5 of 65 units carry NO character rows** because the pronoun scan declined
   every cast hint: `alice_tea_party` (Alice 38 vs 41 across 51 windows -- the Hatter and the
   Hare pollute her windows), `pride_prejudice_proposal` (Elizabeth Bennet 11 vs 19; Darcy 53
   vs 24, under the 3x margin), `frankenstein` (Victor 0 vs 0 across 3 mentions; the Creature
   3 vs 0, under the floor), `don_quixote_windmills`, `fire_not_quenched`. Ten names. At
   render each of them is a 40/40/20 roll, so ELIZABETH BENNET can be voiced male and DARCY
   female -- the exact correctness class the operator keeps open (CLAUDE.md, story-quality
   carve-out). Tier 3 knows all ten cold; tier 4 knows Alice, Elizabeth, Victor, Oscar,
   Harriott and abstains on Sancho, Fitzwilliam, "the Creature", "the Hatter".
2. **Shakespeare, 32 `unknown` sidecar rows.** At render the 42 cast hints are covered (40 by
   roster + supplement, ARIEL/PUCK by design), so the residual exposure is a writer-extracted
   name that is a roster row but not a hint and not in the supplement (LUCE, ANGELO,
   BALTHASAR ...): `briefs.character_names` is LLM-extracted, 2-6 names
   (`_otr_public_domain_sources.py:584`), and the adaptation lanes propagate it
   (`OTR_LedgerScriptWriter.py:3532-3544`). Such a name joins the sidecar row, finds
   `unknown`, and the join ABSTAINS at that tier without falling through (`:314` docstring:
   "the source named this person and declined to gender them, which is an answer") -> roll.
3. **No confidence / rung provenance reaches the ledger.** `gender_map_for_names` emits
   `{gender, evidence, tier, roster_name}`; `tier` is the JOIN shape, not the ladder rung
   (spec B4 is right about that axis mix-up).

## 3. The design the driver proposes (decisions D1-D7; the reviewer pressure-tests these)

**D1. Tier 3 and tier 4 are added to the EXISTING stamper, stamp-time only. The render path
stays a join of committed data.** `_decide` gains two rungs after the pronoun scan:
`llm_recall` (the lab's exact question, constrained JSON, temperature 0.0, one ask; `unsure`
or unparseable -> next rung) then `name_frequency` (`config/cast_pools.gender_of_first_name`
on the honorific-stripped first token, accepting only `male|female`; `unisex|unknown` ->
DECLINE, row omitted, the roll stays). No LLM call at render, no source text to any model
(the ask carries work title + author + character name only -- B1 stays answered
structurally: the prompt builder takes three short strings and has no parameter for text).
*Alternative the reviewer may prefer:* a render-time tier-4 floor inside `gender_map_for_names`
for names absent from the sidecar. The driver says no: it makes the render depend on a name
table's coverage instead of on committed, auditable rows, and it would touch the Shakespeare
join for names the source deliberately left ungendered.

**D2. The name index is a committed per-bank file beside the supplement:**
`config/source_banks/<bank>/character_gender_index.json` =
`{"schema_version": 1, "entries": {"<work_title>|<name>": {"gender", "asked_as": "title|bare",
"model", "prompt_version", "asked_utc", "reason", "locked": false}}}`, keys normalized
upper-case, written with sorted keys. Each distinct (work, name) is asked ONCE, ever; a
`locked: true` entry is never re-asked and its gender is taken as an operator correction --
this is the "index of names" he asked for, short enough to read (about 45 rows across both
banks after the first pass). It resolves the supplement-vs-hand-edit tension r3 raised: the
supplement stays the operator's evidence file for the SHAKESPEARE join at render (unchanged);
the index is the stamper's memory; hand corrections go into the index or the supplement,
never into a sidecar (the stamper regenerates sidecars).

**D3. Shakespeare: the stamper gains the Shakespeare bank and fills ONLY `unknown` rows
(ruling 1), through tier 3 then tier 4.** KNOWN rows are byte-identical before and after
(test: hash the 53 known rows). The ask names the play ("in Shakespeare's The Comedy of
Errors, is the character BALTHASAR male or female?"). `absent_from_roster` speakers (LUCE)
are asked too -- they speak in the scene. ARIEL and PUCK: the index answers `unsure` or a
gender; if a gender comes back it is recorded with `asked_as: title` and the supplement's
"left to the roll" comment is superseded by the sidecar row -- the reviewer should say
whether the operator's earlier "leave them to the roll" is a ruling (keep rolling: mark the
two index entries `locked` with gender `""`?) or a limitation of the old mechanism. The
driver reads it as a limitation and lets tier 3 answer.

**D4. Monotonic merge, tier-ordered.** Rung order `roster > pronouns > llm_recall >
name_frequency`. A re-run REPLACES a row only when the new rung is HIGHER than the row's
`gender_source` or the row is absent; equal or lower never overwrites, so an LLM answer can
never displace the author's pronouns and a later text change is picked up only by a higher
rung. Deleting a row happens only when the candidate set no longer names it (the manifest's
`cast_hints` shrank). This answers r2 #6 / r3 #6 (stale rows, replace-vs-retain) with one
rule.

**D5. The additive fields.** `RosterGenderVerdict` gains `gender_source: str = ""` and
`gender_confidence: str = ""` with defaults (six sites untouched; `_verdict_from` :290 fills
them from the matched row). Sidecar rows carry `gender_source` (already) and new
`gender_confidence` in {`stated`, `recalled`, `inferred`}: roster/pronouns/relation/title ->
`stated`; llm_recall with a work title -> `recalled`; llm_recall bare-name or name_frequency
-> `inferred`. `gender_map_for_names` passes both through to `lock_cast` ->
`cast_source_contract` (additive dict keys; `Ledger.set_cast` drops nothing because the
contract is a nested dict the ledger already stores verbatim -- verify: `_otr_casting.py:2339`
writes `gender_by_name` into the contract). Sidecar `gender` stays `male|female` only on
prose rows; Shakespeare rows keep `unknown` only where every rung abstained.
`normalize_gender` (:100) remains the sole normalization path; the stamper never writes
`other`.

**D6. Model and cost.** Default `google/gemma-4-E4B-it` (the lab's), overridable
`--model`; loaded once per stamper run via `load_llm`, unloaded at exit; the run needs the
GPU free (CLAUDE.md section 4) and is an operator-side data pass, not a render step. About
45 asks on the first pass, zero on every later pass (index hits), so back-pressure and spend
reporting (r3 #2) reduce to a per-run `asked / cached / unsure` count in the corpus report.
The 4060 can run it (E4B is 8 GB-class).

**D7. Acceptance and the live proof.** (a) Corpus gate extended: every Shakespeare sidecar
row is `male|female` except rows whose index entry is `locked` with an empty gender (if D3's
ruling question resolves to "keep rolling"); every prose cast hint has a row or an explicit
`declined` reason naming the rung that declined it. (b) Unit tests: index cache hit makes
zero calls (stub `generate_fn` counts); `unsure` -> name_frequency; `unisex` first name ->
declined; a pronoun row is never replaced by an LLM verdict; known Shakespeare rows
byte-identical; `gender_map_for_names` carries `gender_source` + `gender_confidence`; a no-op
re-run changes zero bytes in sidecars and index. (c) Live: run the stamper on the 5080 with
the server down, commit the regenerated sidecars + the two index files, then ONE
public-domain episode on `pride_prejudice_proposal` published to `otr/obs/`, ledger cast
rows ELIZABETH BENNET female / DARCY male with `gender_source llm_recall`,
`gender_confidence recalled`. That episode is the receipt.

## 4. Blocker-by-blocker: answered or restated?

* **B1 (full-text extraction cannot run):** answered by construction -- no rung reads more
  than the sidecar row, the manifest's title/author and the name. The pronoun scan reads
  the text mechanically, in-process, as it does today.
* **B2 (shared surname must abstain):** already answered in the shipped scan
  (`_shared_forms`, `test_a_SHARED_surname_is_counted_for_NEITHER_character`) and in the
  join (`ambiguous_join`). D1 adds no new alias path. The r3-Antigravity "populate the map
  under all alias forms" fix is NOT taken: `lock_cast` joins by the writer's extracted name,
  and `resolve_roster_gender`'s four tiers already bridge SCROOGE <-> EBENEZER SCROOGE
  (`contains` / `qualified`); a map keyed by every alias would let two rows collide.
* **B3 (manifest sequencing deadlock):** answered -- the stamper is already a separate pass
  over the manifest; it stays one, and gains the Shakespeare manifest
  (`curated_scenes.sample.json`) as a second input.
* **B4 (verdict cannot carry the rung):** answered by D5, additively, at the six sites.

## 5. What the reviewer should try to break

1. D1's "no render-time floor": is there a lane where the writer routinely extracts a
   name the sidecar will never carry (so the stamp cannot help)? If yes, name it and the
   measured rate, and say whether a render-time tier 4 is the right answer or whether the
   stamp should widen its candidate set (e.g. the scene's `speakers` list, which Shakespeare
   already uses).
2. D4's monotonic rule against the r2/r3 idempotence findings: find a re-run sequence that
   changes bytes twice or resurrects a deleted row.
3. D3 on ARIEL / PUCK: ruling or limitation?
4. D5: any consumer that iterates a cast row's keys and would break on two new keys
   (`Ledger.set_cast`, `production_ledger.py:1091` region, `cast_lock.py:1159` region,
   `_otr_scifi_news_pro.py:438`)?
5. D6: temperature 0.0 with the constrained schema -- is one ask enough, or does the lab's
   flip-flop finding mean tier 3 needs two agreeing asks to PIN (cost: 2x on first pass
   only)?
6. Anything in the r2/r3 lists that v2 + this anchor neither answers nor deliberately
   drops. List the item numbers.

## 8. The review round, grounded, and what the driver took (2026-09-02, after r2)

Roster: ONE partner, Antigravity (Gemini 3.7 Flash High) via kibitz, r2 on this anchor + spec v2
+ the r2/r3 finding lists (`kibitz-runs/2026-09-02-gender-ladder-v2-review/r2/antigravity.md`,
18.7 KB, verdict NO with 7 must-fixes, 6 should-fixes). Every claim below was re-read at the
real files before it was folded. This satisfies the plan's "ONE review round, then code".

| # | the finding | grounded? | taken as |
|---|---|---|---|
| M1 | `temperature=0.0` crashes: the closure hardcodes `do_sample=True` (`_otr_constrained_generate.py:296`) | YES (:296-297; story_orchestrator.py:765 already worked around it with 0.05) | root fix in the closure: `temperature <= 0` -> `do_sample=False`, no temperature/top_p kwargs; every existing caller passes > 0 and keeps byte-identical kwargs |
| M2 | ARIEL / PUCK are a binding editorial ruling, not a limitation (`roster_gender_supplement.json:24-27`, `test_roster_gender.py:23`) | YES | index seeded with LOCKED empty-gender entries for ARIEL, PUCK and ROBIN (Folger's speech prefix for Puck); the stamper never asks and the corpus test stays 40 of 42 |
| M3 | byte-identical known rows vs a new `gender_confidence` field | YES | read-through fallback `confidence_for_source` in `_verdict_from`; the stamper writes the field only on rows it touches. NOTE the prose rows already carried `gender_confidence: "known"` from the first stamp, so the vocabulary is known / recalled / inferred (not "stated") |
| M4 | the stamper's manifest loop cannot read the Shakespeare manifest (`scenes`, not `sources`) and would strip `roster_name` / `description` | YES | a dedicated `stamp_scene`: candidates are the sidecar's own unknown rows, rows are filled in place, no `_base_identity`, LF bytes so a no-op re-dump is byte-identical (the fetcher's exact formatting) |
| M5 | the six verdict sites and `gender_map_for_names` do not carry the fields | YES | filled at `_verdict_from` and the supplement site; the map carries both keys into `lock_cast` -> `meta.cast_source_contract.evidence[<NAME>]` |
| M6 | `Ledger.set_cast` whitelists row keys; the receipt is the CONTRACT in meta, not the cast row | YES (`production_ledger.py:1039+`, `_otr_casting.py:2336-2344`) | acceptance re-stated: the fields are read from `meta.cast_source_contract.evidence`; cast rows gain no key (item-G ruling: nothing for set_cast to drop) |
| M7 | "equal or lower never overwrites" freezes stale rows; anchor on `body_sha256` | YES | merge re-anchored: text changed -> fresh; unchanged -> equal rung refreshes, lower never demotes, declined keeps the old row; `tier_counts` derived from the final rows |
| S1 | `llm_web` -> `llm_recall` breaks the corpus test and flags 65 sidecars | YES | tuple updated; the rename + `gender_ladder_v2` version bump rewrite every sidecar exactly once, on purpose |
| S2 | tier 4 is not "total" | YES | spec amended (below); declines to the roll are the honest floor |
| S3 | honorifics: "Miss" and the familial titles | YES | `strip_honorifics` public and extended |
| S4 | cut `asked_as: bare` | YES | cut; every ask carries the work title |
| S5 | the acceptance leg must patch node 1 of `otr_canonical.json` in memory (`otr_api.py:859 patch_creative`) | YES | acceptance plan below |
| S6 | `try/finally unload_llm` | YES | in `main()` |
| cut | two-ask consensus under greedy decoding | agreed | one ask at temperature 0.0 |

**Driver's own grounding find, not in the review:** the pronoun scan is WRONG on scene text. Run
on the real Folger text it called LUCE (the kitchen maid) MALE: a play's mentions are speech
prefixes, so the 240-character window after "LUCE" is her own line, whose pronouns point at the
men she is shouting at. `stamp_scene` therefore never runs the scan; an unknown scene row goes
supplement -> recall -> name pool. The supplement is consulted first (keyed by cast hint, so
"ANTIPHOLUS" fills "ANTIPHOLUS OF EPHESUS") so sidecar and supplement can never disagree at
render (`resolve_with_supplement` raises on a contradiction).

**Tests:** `tests/test_gender_ladder_recall_and_floor.py` (22) + the updated corpus gate; 219 green
across the nine gender test files. **Live proof plan:** run the stamper on the 5080 with the
server down (`--write`, gemma-4-E4B-it), commit the regenerated sidecars and both indexes, then
ONE public-domain episode on `pride_prejudice_proposal` through the canonical graph
(`patch_creative` on node 1: `source_bank public_domain`, `source_ref pride_prejudice_proposal:main`)
published to `otr/obs/`, with `meta.cast_source_contract.evidence` showing ELIZABETH BENNET female
and DARCY male from `llm_recall` / `recalled`.

## 9. Live proof (2026-09-02, 5080, canonical graph, `otr_soak_still_motion_flux2_klein`, one act)

* Stamp run: `google/gemma-4-E4B-it`, 100 asks (84 prose + 16 scene), 44 unsure, 0 unparseable,
  3 locked; corpus 202/249 decided (81.1%), 57 by recall, 132 by pronouns, 14 by the supplement
  (Lira Kell and the Comedy of Errors pair among them); a plain re-run is a byte-for-byte no-op.
* Leg 1 `signal_lost_unhand_me_sir_20260902_151527` (11:32): ELIZABETH BENNET female
  (`llm_recall`, `exact`); MR. DARCY missed the join (aliases unread) -> rolled FEMALE. The defect
  the operator described that afternoon, caught by the receipt, fixed in `_candidate_names`.
* Leg 2 `signal_lost_intensity_in_the_drawingroom_20260902_152901` (server rebooted): ELIZABETH
  BENNET female (`exact`), MR. DARCY male (`short_form` via "darcy"), COLONEL FITZWILLIAM male
  (`short_form` via "fitzwilliam" -- a given-name alias hit on a different character, right by
  coincidence; recorded as the next fork). Cast rows: female IndexTTS2 reference for her, male
  for him. Consistency audit on leg 1: 0 violations.
* Tests: `tests/test_gender_ladder_recall_and_floor.py` (28) + the updated corpus gates and
  roster tests; full suite green on the tree before the last three patches (12,882 passed), and
  the final run's line is in the commit message.
