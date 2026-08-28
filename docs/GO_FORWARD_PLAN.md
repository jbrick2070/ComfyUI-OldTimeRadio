# OTR Go-Forward Plan

**Forward-only, and now literally so.** Open work and nothing else.

* **Standing rulings, laws, review routing and the credit ladder:**
  `docs/OTR_STANDING_RULINGS.md` -- **read it, it is not optional.** The plan says
  what to do; that file says what you may not do while doing it.
* **Closed receipts:** `docs/GO_FORWARD_ARCHIVE.md` (not read to resume).
* **The highest authority is still `CLAUDE.md`**, unchanged.

Split 2026-08-23 on the operator's instruction: *"go forward should only have the
go forward plans. Only."*

**Reorganized 2026-08-28 (the lean-and-mean pass)** on the operator's
instruction: *"clean up the go forward so it's lean and mean, and then analyze
and reorganize it on the most logical [basis] so we can code the most and test
the least."* The organizing fact: most items here need a LIVE RENDER LEG to
prove, and a leg costs 1-3 HOURS on one GPU -- testing is the bottleneck,
coding is not. So the file is now three working queues plus support sections:

* **Section 1 -- NO-RENDER WORK.** Everything provable without a GPU leg,
  front-loaded so it can all ship in coding sittings.
* **Section 2 -- RENDER WORK, BATCHED BY THE LEG THAT PROVES IT.** One leg
  proves several items; each batch says so explicitly.
* **Section 3 -- WAITING ON THE OPERATOR.** Every open question in one list,
  answerable in one pass instead of interrupt-by-interrupt.
* **Section 4** -- rulings lifted verbatim from archived blocks.
  **Section 5** -- parked/deferred. **Section 6** -- standing traps.

Receipts from the 2026-08-27/28 sessions and superseded queue blocks moved to
the archive VERBATIM, nothing deleted; every operator ruling they carried was
lifted into Section 4 FIRST. The standing warning from the 2026-08-24 prune
(now itself in the archive) still binds: **recording what you just finished
inside a section headed OPEN is the easiest way to make this file lie.**
Receipts belong in `docs/PROD_BUG_LOG.md` and the archive; rulings belong in
`docs/OTR_STANDING_RULINGS.md`; only what is still TO DO belongs here.

## HOW TO READ THIS FILE (three files since 2026-08-23)

**THE PLAN IS THIS FILE AND IT IS ONLY OPEN WORK.** Operator: *"go forward
should only have the go forward plans. Only."* Two companions carry what used to
be tangled in here:

* `docs/OTR_STANDING_RULINGS.md` -- the laws, the standing operator rulings, the
  review routing, the model/credit ladder, how to talk to the operator, the
  obs-path override, window packing, tombstones and pointers. **Read it. It is
  not optional and it is not history** -- it is the set of constraints the next
  piece of work has to satisfy. When something in this plan says "THE LAW" or
  "the REVIEW ROUTING block", that is where it now lives.
* `docs/GO_FORWARD_ARCHIVE.md` -- closed receipts, verbatim. **Not read to
  resume.**

`CLAUDE.md` is unchanged and remains the highest authority.

## THE CURRENT STEP -- READ THIS FIRST

### >>> FIRST: READ THE OVERNIGHT QUEUE'S RESULTS IN `otr/obs/` -- THE LEG COST IS ALREADY PAID <<<

**WHAT IS OWED -- THE LISTENING TEST, and it is the whole reason the
code work happened.** No receipt can stand in for it. The queue state as of
the 2026-08-28 morning restart: the FOLEY control leg already PUBLISHED
(`the_curdling_of_memory`, obs #23); the remaining four run via
`scripts/_tmp_overnight_rest2.ps1` -- MIME FIRST (the lane where the
vocalization was heard, so it is the real test of the prompt work), then H3
silent, H3 audio-in, the ltx_audio_in baseline. Watch `otr/obs/`; the live
swim-lane view (`tmp/otr_swimlane.html`, self-refreshing) shows which stage
the current leg is in.
The obs path bug in the queue script itself (pointed at the repo's own empty
`otr/obs` instead of ComfyUI's real `output/otr/obs`, which would have
silently reported every leg FAILED all night) was caught and fixed before it
could do that; see `a196fd90`.

**Judge it as a whole**, not against a sync-only rubric (operator, after
judging a separate LTX 2.3 IA2V A/B the same night: *"I'm creating radio
drama, not impersonating an AI assistant"* -- colour, movement and life count,
they are not subordinate to a technical score).

**One morning pass over `otr/obs/` judges six lanes at once** -- this is
Batch R0 in Section 2. The same episodes are also a free re-observation
chance for the two parked eyeball items (announcer framing, name-splice #2;
see Batch R5), and the two H3 legs may already satisfy Batch R1's receipt
conditions -- check before spending another leg.

(The foley-audibility resolution, the named-sounds ship, the voice-bleed fix
and the H3 style-cue ordering fix that this listening test proves are all
closed receipts now -- 2026-08-28, in the archive and `docs/PROD_BUG_LOG.md`.)

---

## SECTION 1 -- NO-RENDER WORK (code the most: everything here ships without a GPU leg)

All of this is provable by the suite, a scoped review, or an offline read.
One coder window at a time; every chunk = focused tests + full suite + Bug
Bible + commit AND push + `HEAD == origin/v2.0-alpha`.

**THE CODING ORDER (2026-08-28, operator-set and REORDERED the same day on
his instruction: the 4060 set moves LAST -- "I don't want to test [the] 4060
until we are done with all other bugs." The 4060 test is the CAPSTONE: it
runs once, on a tree with every other bug already closed, so the frictionless
verdict is not polluted by known-open defects):**

1. **Judge the sanctioned-gap r2 panel and CODE it** (1.1 -- plan written,
   all three reviews returned, awaiting the driver's judgment). Closes the
   control path AND the only surviving LEMMY-era PBUG (-02).
2. **Spandrel `_resolve_model` hardening** (1.5) -- repro first; fix only if
   reproducible; one finished-diff review.
3. **Gender-ladder SPEC REWRITE** (1.3 -- now fully unblocked: both operator
   rulings recorded at its head).
4. **Local-LLM sweep Leg 0** (1.2 -- ~15-20 min, in-process, no ComfyUI).
5. **The 4060 frictionless set** (1.0 below) -- LAST coding item, then the
   republish sequence it gates: operator's node-delete click -> version bump
   + kokoro in pyproject in ONE commit -> clean publish -> the 4060 template
   test.
6. Handoff bookkeeping.

### 1.0 THE 4060 FRICTIONLESS SET (the CAPSTONE -- runs LAST, after every other bug closes; the path to "download the template, click run" on his 4060)

Four small items, all verified against the tree on 2026-08-28, plus one trap:

* **`workflows/variants/otr_4060_floor.json`: `"quant_policy": "none"` ->
  `"bnb_nf4"`.** One line. The single thing blocking the 4060 from loading
  its 10 GB writer; the old justification for "none" is provably stale
  (accelerate + bitsandbytes are declared deps now).
* **Declare `kokoro` in `requirements.txt`** and fix the stale repo hint at
  `nodes/_otr_audio_engines/eng_kokoro.py:139` (`1038lab/KokoroTTS` in the
  error message vs `hexgrad/Kokoro-82M` actually loaded at `:159`). Two
  lines; removes the biggest fresh-install voice friction.
* **Ship a template in `example_workflows/`** -- the 4060 floor graph
  already exists as a variant; it just does not ship where "download and
  click run" looks.
* **README model table** from the compatibility workbook's Baseline Combos
  tab (`outputs/20260828-ungated-models/` -- the LIVING fact sheet: edit
  cells in place, never add a changelog tab). Include the HF-token
  two-tier story: defaults need no account ever; gated upgrades = account +
  license click-through + `HF_TOKEN` once (the resolver already finds it).
* **THE TRAP: `pyproject.toml` edits AUTO-FIRE a registry publish.** The
  `kokoro` line also belongs in its static deps -- but that edit WAITS and
  rides the deliberate republish: operator deletes the flagged node entry
  (his click, not ours) -> ONE commit bumps the version + adds kokoro to
  pyproject -> clean publish -> clean 4060 install of the shipped template.

Waiting on externals, not on us: Codex deep research on the HF-token
problem statement (`docs/2026-08-28-hf-token-feature/PROBLEM_STATEMENT.md`)
and the operator's registry node-delete click.

### 1.1 THE SANCTIONED-GAP CONTROL PATH -- r2, the coding plan, is the next step

**Live-proven necessary on 2026-08-26.** Ideogram is not seed-deterministic:
after the music-card fix it went from refusing every music card to **6 of 7**,
and that ONE refusal still killed a 30-minute episode at the still-spine gate.
No amount of prompt work takes a stochastic refusal to zero.

The dispatcher already says "the episode continues" and the composite already
floors an `exists=False` row -- **nothing in between mints that row**. Spec and
r1 judgment: `kibitz-runs/2026-08-25-model-refusal-required-still/`.
Accounting for it landed 2026-08-26 (`a2837b05`) and is deliberately inert
until this exists.

**THE BLOCKER IS CLEARED -- r2 CAN RUN (2026-08-27).** r1 ended on one open
item it refused to decide for the operator: what an episode should do when
EVERY required still is sanctioned-gapped, given that node 92's success check
is `clip_count > 0`. **Asked and ruled 2026-08-27: it PUBLISHES.** The full
ruling, its reasoning and -- importantly -- what it does NOT license are in
`docs/OTR_STANDING_RULINGS.md` under *"AN ALL-REFUSED EPISODE STILL PUBLISHES"*.
Read it there before r2, because the ruling is narrower than its headline: it
permits publishing an all-refused episode, it does NOT permit REPORTING one as
a clean render, and the `required_scene_targets` ledger-completeness law is
untouched.

**STATE 2026-08-28: r2 JUDGED. The panel killed a build-breaker and the plan
is now C0-C7, ready to code.** Panel: Codex `gpt-5.6-sol` + Antigravity
`Gemini 3.7 Flash (High)` + Cursor `grok-4.6-high` (operator's best-of-3).
**All three returned NO on the same finding, verified by the driver against
the files:** `validate_and_repair_still_spine(ledger)` runs at
`otr_video_render_batch.py:533`, ONE LINE BEFORE `run_real_episode` at `:535`
-- so a refused beat kills the episode before any downstream change could
execute. The r2 plan changed five things below a gate that raises first.

**Judgment + hardened plan: `kibitz-runs/2026-08-28-sanctioned-gap-r2/r2/final.md`
(gitignored -- the structure is reproduced here so it survives).** THE EIGHT
CHANGES, in dependency order, coded in four commit chunks:

* **C0 (NEW -- everything else is dead code without it).** The spine
  validator accepts a `sanctioned_gap` receipt row: skip materialize, do not
  raise (`render_driver.py:1143-1150`), do not append a fake validated path,
  do not let `:1219-1227` copy a historical path onto the gap row. Same skip
  for jump-segment (`:1176-1183`) and HuMo-portrait (`:1205-1209`).
* **C1.** The dispatcher stamps the receipt with **THREE outcomes, not two**
  -- only `reason == "model_refusal"` is tolerated today
  (`otr_image_gen_dispatcher.py:1868-1878`); `no_engine` / `dead_path` /
  `prompt_path_guard` must keep raising. Schema: identity keys + `status` +
  evidence `{reason, engine_id, seed, prompt, prompt_hash, detail,
  image_revision}`; ok rows gain `status:"ok"` so absence is never the
  signal. `_still_index` AND the spine must both ignore image rows whose
  object_id is a this-dispatch gap.
* **C2.** Skip in the **`run_episode` loop** (`:4636-4669`), NOT the request
  builder -- `build_request_from_shot` returns a request, not a shot, and on
  the cheap families a missing still does not even raise there. Append the
  original ShotLock shot to `new_shots`; do not invent `exists` on shots.
* **C3.** `build_clip_manifest` projects the marker onto the manifest CLIP
  row -- `timeline_quality_report` indexes `manifest["clips"]` and never sees
  `video.shots`.
* **C4.** The predicate computed over `manifest["clips"]`:
  `ok = n > 0 and unsanctioned == 0`, `degraded = sanctioned > 0`. (The r2
  draft used an undefined `total` and read a counter built AFTER the report.)
* **C5.** **Was wrongly marked out-of-scope.** `_build_render_engines_payload`
  (`:212-223`) counts EVERY `exists=False` as a sanctioned gap with no
  evidence test -- correct only while no gap row can arrive. After C2 it
  would report a crashed all-missing render as publishable-degraded.
* **C6.** Downstream: credits `or` fallback (`otr_credits_roll.py:319`),
  `check_ltx_open_health` strict mode, the acceptance grader's
  `RULE_MISSING_CLIP`.
* **C7.** End-to-end through the REAL spine + composite. **Expect a hole:**
  all-gap with empty `new_shots` makes `assemble_silent_timeline` raise
  `"manifest has no renderable beats"` -- that hole IS the work.

**Commit chunks:** (1a) C0+C1 together -- C0 has nothing to read without C1;
(1b) C2; (1c) C3+C4+C5; (1d) C6+C7. Each chunk = focused tests + full suite +
push, so the tree is never half-migrated.

**The original next step was r2, the coding plan**, per r1's own stated roster
(r1 Codex+Fable -> r2 Codex -> r3 Codex+Cursor -> r4 agy Pro). Nothing else
about this row is waiting on the operator.

**r1 WAS WRITTEN 2026-08-25 AND THE TREE MOVED THE NEXT DAY. RE-GROUNDED
2026-08-27 against HEAD -- do not hand r2 the stale finding list.** `a2837b05`
landed on 2026-08-26 and added 85 lines to `otr_video_render_batch.py`, so one
of r1's four findings is already closed and one of the survivors got SHARPER.

| r1 finding | status at HEAD, verified by reading the file |
|---|---|
| 1. Nothing mints the `exists=False` row between a sanctioned dispatch and the spine | **STILL THE ROW.** This is the item itself. |
| 2. The skip branch never reaches the renderer loop | **STILL OPEN**, unverified in this pass -- r2 grounds it. |
| 3a. The manifest loop counts a gap as a delivered receipt (`:146-150`) | **ALREADY FIXED** by `a2837b05`. `_clip_delivered_motion(clip)` (`:134-153`, `exists` alone, deliberately) now routes an undelivered beat to `sanctioned_gap_shot_ids` at `:213` instead of minting a receipt for it. Do NOT re-fix this. |
| 3b. `delivered_frames_ok` is True over an absent clip (`:750-770`) | **STILL LIVE**, and here is the exact mechanism: a gap has no `source == "clip"` segment, so `segs` is empty, `status` becomes `no_clip_segment` -- and `no_clip_segment` is the one status that flips NOTHING. `ok_all` is only cleared by `held_last_frame`, or by `not positioned and segs and delivered != tgt`, whose `segs` guard is falsy for exactly this case (`otr_silent_composite.py:766-769`). |
| 4. An all-refused episode reports FAILURE (`clip_count > 0`) | **STILL LIVE, now at `otr_video_render_batch.py:640`** -- and `a2837b05` made it BITE rather than merely lurk. |

**FINDING 4 DESERVES ITS OWN PARAGRAPH, because the fix for 3a is what armed
it.** `clip_count` is `len(receipts)` (`:129`), and since `a2837b05` receipts
correctly EXCLUDE sanctioned gaps. So an all-refused episode now has a genuinely
empty receipt list and `"ok": manifest["clip_count"] > 0` genuinely evaluates
False. Before that commit the gap rows were counted as receipts, so the same
episode would have reported ok=True by ACCIDENT -- for the wrong reason, off a
receipt that lied. **The correct accounting collides head-on with the 2026-08-27
ruling that such an episode must publish**, which is not a regression in
`a2837b05` but the point at which an existing contradiction became honest enough
to see. r2 fixes the success predicate, NOT the accounting.

**The payload-never-empty guarantee already anticipates this** and is worth
reading before designing (`otr_video_render_batch.py:203-209`):
`OTR_CreditsRoll._require` rejects `{}`/`[]`/`None`/`""`, so an all-gap episode
returning an empty payload would convert a publishable degraded episode into a
hard mux-time failure -- "the exact outcome the sanctioned gap exists to
prevent", in the code's own words. Whoever writes r2 starts from there.

### 1.2 LOCAL-LLM SWEEP LEG 0 -- the in-process preflight (~15-20 min, no ComfyUI)

Leg 0 of the local-LLM acceptance sweep is NOT a render: one command,
in-process -- `request_slot` -> ~40-token generate -> `_self_unload` per row,
with `reset_peak_memory_stats()` around each. It is what fails loudly on a
dead row. Run it in a coding sitting; the four canonical legs it precedes are
Batch R3 in Section 2, and the full sweep design lives there in one piece so
it is not split across sections.

### 1.3 CHARACTER GENDER LADDER -- the SPEC REWRITE (fold r2 + r3, then r3 again; NOT r4)

**TWO OPERATOR RULINGS, 2026-08-28, and they reshape the spec:**

* **Shakespeare: fill ONLY the 32 `unknown` roster rows.** KNOWN rows from the
  parsed dramatis personae stay untouchable; the ladder's lower tiers may fill
  the blanks. (Section 3 question B is ANSWERED -- remove it on next tidy.)
* **THE WEB-SEARCH TIER IS REPLACED, not plumbed.** Operator's design, his
  words: *"just have the LLM decide -- ask what the likely gender of this
  person name is, have the LLM decide, and keep that in an index of names."*
  So tier 3 becomes an LLM VERDICT ON THE NAME (the model already knows
  Scrooge and Marley from training -- no live search needed), cached in a
  PERSISTENT name index so each name is asked once, ever. Tier 4
  name-frequency stays as the deterministic floor beneath it, keeping the
  ladder TOTAL when the LLM call fails. This dissolves both review rounds'
  biggest must-fix (the silent no-op web call): there is no web call. On his
  "is it not easy to query a search engine?" -- keyless search-engine querying
  is the fragile part (scraping is blocked/ToS; keyless APIs are thin); the
  RSS precedent covers feeds because feeds are MEANT to be fetched. The
  LLM-ask design avoids the whole problem, offline-first.
  The invented lanes (original, scifi_news_pro, media_archive) KEEP ROLLING
  by the standing ruling -- their characters do not exist, so no lookup of
  any kind applies.

Live 2026-08-05: `EBENEZER SCROOGE` = female, `JACOB MARLEY` = other,
`HENRY HARTWICK OGLETHORPE` = female. Meanwhile MACBETH, BANQUO, PROSPERO and
MIRANDA are all correct.

**The split IS the diagnosis, and it means the render code is not broken.**
Shakespeare ships 14 provenance sidecars carrying `characters` with genders; the
prose lane has ONE tracked sidecar and its `characters` key is `None`. The pin
chain already exists and is lane-neutral (`_otr_roster_gender.py`, 12.6 KB, on
disk). Shakespeare is right because the DATA is there. Prose is rolled because it
is not. **This is a vendor-time data gap with a working consumer** -- the exact
inverse of the Item 7 bug, where the value existed and nothing read it.

Spec: `docs/2026-08-05-character-gender-ladder-SPEC.md` (Fable, driver-grounded).
A four-tier TOTAL ladder -- roster -> pronouns in the source text ->
character-in-work web lookup -> name-frequency percentage -- stamping `gender`
(always populated, never `unknown`, because a voice must still be cast) plus
`gender_source` and a confidence. Operator rulings baked in: Shakespeare's KNOWN
rows are untouchable, the announcer stays randomly male/female by design, and the
invented lanes (`original`, `scifi_news`, `scifi_news_pro`, `media_archive`) keep
rolling -- their characters do not exist, so a name search there risks matching a
real person.

**Codex r2 verdict: NO. Eleven must-fixes, at `kibitz-runs/2026-08-05-gender-ladder/r2/codex.md`.**
The diagnosis survived; the mechanism did not. The three that matter:

1. **The web search would silently do nothing.** The spec passes a tools/plugins
   argument to `OpenRouterBackend.generate`, which swallows unknown kwargs through
   `**_ignored` -- no error, no search, a confident answer from a model that never
   looked. That is the same silent-no-op class as the defect above.
2. **"LLM extraction over the FULL unit text" cannot run.** `beckoning_fair_one.txt`
   is 143,176 bytes and 58 of 65 source files exceed 12,000 bytes, against a
   32,768 estimated-token per-call cap.
3. **Blanket surname aliases are identity-unsafe.** Two rows sharing a surname
   with the same gender currently produce a confident pin rather than an
   abstention.

**r3 RAN 2026-08-06 and it is STILL NO** -- Codex NO with 7 must-fixes, agy
yes-with-fixes with 9. That is three NOs across two rounds, and r3's findings invalidate
the CODING PLAN rather than its line numbers, so the standing re-ground rule applies:
**the next step is a SPEC REWRITE folding r2 + r3, then r3 again. Not r4.**

Both lanes independently found (a) a manifest sequencing deadlock -- the stamper is
specced to run per-unit inside the vendor fetch loop, but the manifest is written only
AFTER the loop, so it can never see the unit it was called for; and (b)
`RosterGenderVerdict` has no `gender_source` / `gender_confidence` fields, so the
ladder's whole output cannot be carried without changing every verdict-construction
path. Codex also confirmed the r2 finding that the OpenRouter backend still has no
web/plugin parameter, so the web-search tier would silently do nothing.

Judgment: `kibitz-runs/2026-08-06-2026-08-06-gender-ladder-r3/r3/judgment.md` (LOCAL
ONLY). **Trap: commit `496d9d57` inserted ~90 lines near the top of
`nodes/_otr_roster_gender.py`, so every cite into that file in the r3 review has
SHIFTED. Re-pin before acting.** The rewrite should also CONSUME the
`normalize_gender` boundary item 8 installed there rather than adding a second
normalization path.

**Found while grounding, and it reopens an operator ruling:** 32 of 85 Shakespeare
roster rows are `unknown` TODAY -- 38% of the lane assumed solved. Comedy of Errors
ships 7 characters, every one unknown. The narrower ruling that fits the evidence:
Shakespeare's KNOWN rows stay untouchable, but tiers 3-4 may fill only its
`unknown` rows. That fixes 32 rows without ever second-guessing a parsed
dramatis personae. **Operator decision, not a driver call** -- cross-listed as
Section 3, question B.

### 1.4 ~~RETIRE `OTR_ENABLE_LTX_I2V`~~ -- DONE 2026-08-28, tombstone

Shipped the same night it was scoped: `192d3aa2` (family = image_to_video,
declarative still requirement, both driver maps, the enum token, content
oracle) + `f9eab3f6` and `063e8c0b` (the seven test sites the full suite then
found, including two prompt-only fixtures that now SUPPLY a still instead of
disabling the requirement, and the soak rotation's text_to_video chair
re-seated with `animatediff15_v3_haunted_video`). Full suite green. The
operator's ruling is now simply how the lane works; there is no flag to
retire and no text-only LTX path to reason about.

### 1.5 ~~NARROW LEARNED-UPSCALE HARDENING~~ -- CLOSED 2026-08-28: NEITHER EDGE CASE REPRODUCES

The row was conditional -- "harden the two `SpandrelEsrgan._resolve_model`
edge cases **if still reproducible**" -- so the first step was the
reproduction, not a fix. Both were attempted on the real module (CPU, no
model load) and both behave correctly at HEAD:

| edge case | result |
|---|---|
| `folder_paths.get_folder_paths` raises a non-ImportError | Resolver survives and falls back to the repo-relative dir. **One warning across five calls**, not five -- the `_RESOLVE_WARNED` cap works. |
| An unreadable candidate (`PermissionError`) | **Propagates** rather than masquerading as absence, exactly as the docstring claims. Already pinned by `test_upscale_cache_fingerprint.py:243-249`. |
| (checked while there) shallow checkout, <5 path parents | No `IndexError`; resolves. |

The hardening this row asked for was done in earlier passes and the code
documents it -- including an explicit note that a classification pass was
written, found unreachable, and REMOVED rather than kept as reassuring dead
code.

**What the verification did surface, and it is now fixed:** the warn cap
itself had NO test. Removing the guard that keeps a broken `folder_paths`
from logging once per prompt evaluation (it is called from `IS_CHANGED`)
would have failed nothing. Three tests added in
`tests/test_upscale_cache_fingerprint.py`: warns once not once-per-call, the
set stays bounded, and the repo fallback still resolves.

The ruling attached to this row stands and is lifted into Section 4: **the
multi-GPU learned-upscale stage itself is CLOSED and must not be reopened.**

### 1.6 OPEN DEFECTS THAT ARE CODING WORK (a leg may prove some of them later; none needs a leg to FIX)

MECHANICAL defects survive story-engine churn; STORY-QUALITY judgments do not.

**EVERY LINE CITE IN THIS SECTION IS SUSPECT.** Each one checked during the
2026-07-27 triage had moved: `_is_cloud_video_engine` is `render_driver.py:1599` not
`1274-1295`; the "NO FALLBACK to text-only" refusal is `:2148` not `1801-1817`;
`_use_i2v` is `eng_ltx_video.py:583` not `559-572`. The defects are mostly still
real; their coordinates are not. **Re-pin a row's cite when you touch it.**
Path note (verified 2026-08-04): engine adapters live under
`nodes/_otr_video_engines/` (and `_otr_audio_engines/`, `_otr_image_engines/`)
-- bare `eng_*.py` cites in these rows are shorthand for those paths.

#### The P0 / source-span cluster (2026-07-30)

- **`full_text` reaches the span coordinate system carrying HTML BLOCK JOINS WITH
  NO SEPARATOR, and on the live evidence this is the DOMINANT P0 failure cause.**
  Measured in the campaign logs: `'...Field of Martian PolygonsNASA/JPL-'`,
  `'...and the School ofEngine'`, `'...what you're doing.Let's s'`, `'...(AMR).The
  resea'`. The RSS adapter strips tags without inserting whitespace, so two elements
  fuse into one token. `_normalize_span_source_text` collapses whitespace RUNS but
  cannot insert a space that was never there, so the model quotes the sentences a
  reader sees and they are not byte-exact in the stored text -- exactly the
  "non-literal source span" rejection that killed 12 of the 15 P0 legs.
  **Deliberately NOT fixed by A-3:** inserting separators is a WIDER change to the
  coordinate system `source_digest` pins -- an operator decision, and it belongs in
  the source adapter rather than the codex normalizer. Owed: which adapter builds
  `full_text`, whether a separator can be inserted at admission without breaking any
  accepted ledger, and a fixture from these four strings. (Cross-listed as
  Section 3, question D.)
- **The deterministic P0 rung PRUNES SILENTLY, which violates the plan's own
  Invariant 3.** `repair_literal_source_metadata` drops an unsupported span, then its
  evidence row, then the fact -- and emits no receipt. An accepted P0 index simply
  has fewer facts than the model wrote, and nothing says which were dropped or why.
  Under "fail loud, not fatal" the degrade is the right direction and the silence is
  not.
- **The deterministic P0 rung is ALL-OR-NOTHING across an artifact, and can poison
  its own good work.** It is handed `a0_payload` (all seven keys) while
  `_validate_fact_index` restricts spans to `allowed_source_fields` (the projection).
  A quote rehomed into a field the projection omitted makes `post_validator` reject
  the WHOLE repaired artifact -- "cites source field ... outside the supplied P0
  evidence" -- so one unlucky rehome discards every correct prune in the same pass.
  Either give the repairer the allowlist or prune per row.
- **Nothing measures whether a pruned P0 index is ACCEPTED** (recorded, no action
  owed yet). No live leg has ever run with the deterministic rung reachable (it became
  reachable at `47c554fa`, after the campaign stopped), and the rejection logs carry
  only a truncated `raw head` plus no source payload, so the question cannot be
  answered offline. A-1's instrumentation is what makes the next campaign able to
  answer it.
- **`scifi_news` P0 convergence defect** -- both 120w and 320w legs fail in P0 after
  two attempts on non-literal fact source spans; provider/model convergence, extends
  BUG-11.35. NOT a word/length gate. Blocks the last 120w receipt and the
  `scifi_news` live reverify (PBUGs 20260712-22/23/24/25, fixed in tree, reverify
  still owed).
- **`scifi_news_pro` provider capacity** -- `requested_output=2800` vs provider cap
  `512`; the whole-artifact retry contracts LANDED @ `314dd481` are the base; the
  residual fix is now unblocked. Related independent items: the P9 8K
  structured-capacity follow-up + the GGUF structured-enforcement NEWBUG. Do not
  raise the minimum word target as a capacity workaround.

#### The orphan-occupancy registry (design item -- full arc BEFORE code)

Falls out of PBUG-20260825-04, whose four landed fixes shipped in
`fb67d059` after a full kibitz r1-r4 arc (Codex r2/r3, Cursor r4, Fable r1).
The arc found a new race in each of the first two cuts of the same fix, so
**do not treat this item as mechanical** -- it is a genuine design
choice with more than one defensible answer, which per CLAUDE.md means a full
arc BEFORE code, not after. (Its sibling, the GGUF generation deadline, CLOSED
2026-08-25 -- receipt in the archive; its lesson is lifted into Section 4.)

- **THE ORPHAN-OCCUPANCY REGISTRY -- still deferred, now on its third
  independent confirmation.** `has_local_resident_llm()` reports "nothing
  resident" the instant a timeout invalidates the cache dict, even while the
  orphan worker is still actively running CUDA kernels on the model that
  entry described. `nodes/otr_shot_lock.py:1781` and
  `nodes/otr_video_render_batch.py:289` both trust that signal before
  starting visual/video work. The r1 panel (Codex + Cursor + Fable) deferred
  this unanimously; r3 and r4 each re-raised it and each time it was
  re-confirmed as correctly out of scope for the cache-bookkeeping fixes.
  Shape: a process-global, lock-protected registry of in-flight generations,
  registered before invalidation and cleared via `Future.add_done_callback`,
  with fail-fast admission on `request_slot` and the visual-entry guards
  reading real occupancy instead of the dict's cleared-or-not state.
  **What this session's fixes did and did not buy:** they close the concrete
  cache-bookkeeping windows (no abandoned publish, no laundered
  invalidation, no torn read, no unconditional teardown of a foreign live
  entry); they do NOT make orphan GPU occupancy visible to a downstream
  visual stage. That remains exactly as exposed as before.

#### Coverage, canvas and clip-contract

- **The route lock is ONE NODE TOO LATE for the image phase** (found 2026-07-25, node
  order confirmed against the canonical JSON: `87 VideoDirector -> 88 ImageDirector ->
  89 MetaBrief -> 90 ShotLock -> 91 ImageGenDispatcher -> 92 VideoRenderBatch`).
  `resolve_final_shot_engines` runs at node 92, but stills are minted at 91 and image
  PROMPTS at 89. The landed fix closed the spine-validation gap; the image phase still
  relies on its own MIRROR (`otr_meta_brief_image_prompt._effective_prompt_engine_for_role`,
  whose docstring says it "mirrors the image dispatcher's effective-engine seam").
  **Chunk 1 of the coverage block is the fix.** Note node 89 precedes node 90, so
  hoisting to ShotLock still does not put MetaBrief downstream of the authority --
  that needs a VideoDirector-time freeze and is NOT in scope. (This is also the
  "image-phase still ownership" item from the campaign queue.)
- **THREE silent coverage mechanisms exist, not one** (found 2026-07-25).
  Mechanism 1, the engine mirror/ping-pong (`wrapper_bridge.extend_frames_to_target`),
  is GONE from `eng_ltx_8gb` -- pinned behaviourally by a test that detonates the
  helper and renders successfully. It REMAINS in `eng_wan_ti2v`, deliberately and
  permanently: WAN renders a short native clip on purpose and fills the beat with it,
  which is the shipped 8GB tier contract `PBUG-20260723-02` protects. **Still open:**
  composite loop-fill (`otr_silent_composite._should_loop_fill`, which also SUPPRESSES
  its own underrun warning once it activates) and held-last-frame. For `ltx_8gb` the
  composite path is now de facto unreachable -- the adapter returns exactly the
  requested count or raises -- but not structurally impossible:
  `encode_frames_to_silent_mp4` reports the size of the array it piped into ffmpeg
  rather than re-probing what ffmpeg wrote, so an encode-side drop could still
  under-report. PRE-EXISTING; close it when the assembly boundary is next opened.
- **`_should_loop_fill` names the permanent fix and it is now being built**
  (`otr_silent_composite.py:244-266`): *"The real fix is phrase-chunking (render the
  beat's correct duration so it never underruns) -- tracked as a follow-up."* The
  coverage block IS that follow-up.
- **THE 7d-PREFLIGHT THAT "PROVED THE GPU" RAN AT THE WRONG CANVAS** (found
  2026-07-27, B5 panel; verified -- and it corrects a claim this file once made).
  `render_single` and both HTTP entry points use the older ledger-free `build_request`,
  which never reaches the canvas seam and defaults to `OTR_VIDEO_RENDER_CANVAS`
  (832x480). So the "GPU IS PROVEN" leg (`ltx_8gb`, 25 frames, 3004 MB) exercised
  832x480, not the production canvas. `render_single` parity is explicitly deferred by
  the O1 judgment; what must NOT happen is another "proof" through that harness being
  read as a production proof. (The 512x288 canvas itself HAS since rendered live --
  bench arm D, three cells -- but through a DIRECT-NODE graph, which proves the canvas
  and the recipe, not the seam.)
- **The ShotLock WRITE-side canvas validation is still owed** (O1 judgment item 1).
  `otr_shot_lock.py` stamps `video.canonical_canvas` unvalidated from a possibly-empty
  policy. B5 made this non-load-bearing for the render (the engine declares its own
  canvas now), so it is no longer urgent -- the drift guard in
  `tests/test_ltx_8gb_canonical_canvas.py` covers the disagreement that matters. Close
  it when the general canvas resolver lands.
- **Odd-canvas evenness is validated at the ENCODER, not where the canvas is chosen.**
  The stride defect itself is closed (`b1f2ee86`): `ffmpeg_silent_mp4_cmd` declares the
  REAL width/height and `encode_frames_to_silent_mp4` REFUSES an odd canvas by name,
  because yuv420p subsamples chroma 2x2 and cannot represent an odd dimension. Still
  true and NOT fixed: neither `WanInitImageMixin._dims()` nor the `Canvas` schema
  validates evenness, so an odd canvas is caught late rather than at the choice. No
  live producer builds one today (832x480, 512x288, 1472x832 are all even).
- **`CanonicalClip.frame_count` -- "the integer timing authority" -- HALF CLOSED**
  (`58e288af` + `40780b82`, count closed @ `48e3c6fb` without paying a decode). Every
  module that writes a clip now ffprobes it, and a roster gate in
  `tests/test_terminal_frame.py` fails by name for any module that writes a clip
  without proving it. What this proves and what it does not: it proves the muxer wrote
  what it was piped, which is the right question for a clip written by ONE ffmpeg
  pass; it does NOT prove decodability, which is why `assemble_beat_segments` still
  decode-counts every ASSEMBLED beat and must keep doing so. **Re-verify before
  acting:** this row's "still self-declared elsewhere" pointers (the four `viz_*` and
  four `still_*` engines) were both closed afterwards, so the remaining open surface
  may be empty.
- **KNOWN LIMIT of the widened roster gate**, recorded so it is not rediscovered as a
  surprise: the codec flag is matched as a STRING CONSTANT, so a flag assembled at
  runtime (an f-string, `"-c:%s" % stream`) or the stream-index spelling `-c:0` is
  invisible to the sweep. Nothing in the tree does that today; an encoder that ever
  needs to must be pinned in `_ENTRY_POINT_PROOFS` by hand, which the inventory test
  makes a visible decision. Separately, ONE mutant survives the round by construction:
  deleting the self-proving membership assertion is catchable only by a meta-test of
  that assertion.
- **`ltx_av` underruns long beats** (found 2026-07-25, codex; confirmed). It caps at
  `_LTX_AV_MAX_FRAMES` (`eng_ltx_av.py:58`, default 497, env-overridable) and clamps
  at `:950-953`. It is NOT "renders to target natively" as three earlier docs claimed.
- **Ping-pong on a capped HuMo beat played lip sync BACKWARDS** -- FIXED in code @
  `a1d810f1`, but the finding is STATIC (no live artifact), so it is NOT a PBUG row. A
  capped-14B leg would reproduce it. Kept here so the live proof is not forgotten
  (listed under Section 2, deferred render items).
- **`docs/ENGINE_MATRIX.md` reports the DECLARED contract only** (found 2026-07-27).
  Correct today and consistent with its own stated design (every number read from the
  live registry). But the moment a profile pins an `ltx_8gb` ceiling, the matrix keeps
  printing `9-161 step 8` for a tier whose real window is narrower, and the `--check`
  drift gate cannot notice because it diffs the registry, which the effective contract
  never touches. Owed at the prequalification step, not before.

#### Routing, env-capture and the credits card

- **`wants_talking_prompt()` escapes any routing freeze.** It calls
  `_recipe_config(self._recipe())` and `_recipe()` (`eng_ltx_av.py:402-432`) re-reads
  `OTR_LTX_AV_RECIPE` / `OTR_LTX_AV_SHARP` / the UNET name on EVERY call by documented
  design ("Read fresh every call"). So a `required="when_engine_talking"` row evaluated
  through the hook re-reads the environment after capture. S0b-core needs ONE shared
  `row_is_active(...)` evaluator over captured state, with the talking result inside
  `ltx_resolved`.
- **`provider_side` is a THREE-part rule, not an attribute.** `_is_cloud_video_engine`
  accepts a `cloud_` id prefix OR the attribute OR `node_key.startswith("cloud_")`.
  `cloud_kling_avatar` has no `provider_side` attribute and is caught by the id prefix
  alone, so an `engine_facts` builder using a bare `getattr` would classify it local
  and let the radio-host redirect send a cloud avatar to local LTX. Needs a regression
  on picked AND forced `cloud_kling_avatar`.
- **Env-read sites missing from the S0b inventory** (was four; the
  `OTR_ENABLE_LTX_I2V` site was DELETED by the 2026-08-28 retirement, so two
  remain): `render_driver.py:1176-1203` and
  `otr_meta_brief_image_prompt.py:297-300` (`OTR_ENABLE_HUMO_HOSTS`), and
  `eng_ltx_av.py:352-353,403-432` (recipe/UNET re-read outside `assert_usable`).
- **The credits card needs a SMALL-CANVAS VARIANT, and the ladder is not it.** At
  512x288 (the ltx_8gb tier) col1 is 65px past its footer even with every ledger row
  this policy may drop already dropped; at 640x360 it is 12px over. Both are drawn
  anyway (a terminal node never destroys a finished episode) and LOGGED at ERROR
  naming the canvas -- the old behaviour was drawn, clipped by PIL, silent. At 288
  lines the three-column console is already a polite fiction: col3's scrolling
  transcript is as unreadable as anything col1 clips. This is a DESIGN job -- a card
  laid out for a small canvas -- not more ladder heroics.

### 1.7 THE ADAPTATION DESIGN (hardened, NOT yet built; multi-session -- start only with room to finish step 1)

Plan of record: `kibitz-runs/2026-08-03-adaptation-fidelity/r2/final.md`.

**The keystone correction: compile source speech, do not generate it.** A ledger row
that merely POINTS at a source segment proves structure, not meaning --
`PRODUCTION_SPRINT_LESSONS.md` lesson 11 documents that exact failure class.
Source-owned text must be materialized deterministically from an authenticated
segmented artifact and verified against it. "Summarize into X words" then means
SELECTING WHICH REAL SEGMENTS FIT THE BUDGET, not paraphrasing -- which also removes
the VRAM hazard, since no model sits in the source-speech path.

Settled by arithmetic: an episode cannot exceed **1,520 words** (19 voiced beats at
act_count 7, `BEAT_WORD_HARD_MAX` 80), so full-scene performance is impossible
without redesigning beat topology. Build target is the 300-word unit.

**NEXT, IN ORDER:**

1. **The segmented source artifact** (schema, spans, hashes,
   `body[start:end] == segment.text`, omission receipts) and the pass-to-field
   ownership table -- **nothing else codes until that table exists.**
2. **Cast from the selected cut.** Real scenes carry 3-12 speakers against a
   6-character ceiling (`_otr_casting.py` 1-6, `OutlineRequest` rejects >6), so which
   speakers appear must follow from the cut that fits the word budget. Coupled hard
   to the capacity guard: at act_count 1 there are exactly THREE voiced beats, so a
   4-person cast is a mathematically guaranteed `CastVoiceCoverageError` -- the
   failure that killed `scifi_news` in the six-bank run. `compute_episode_budget`
   must also receive the TRUE locked cast.
3. **Loosen the count-match invariant** (`OTR_LedgerScriptWriter.py:4061-4067` hard-
   raises on any locked != requested) and change the pack text that tells the model
   to drop figures.
4. **Extend `_otr_provenance.py`** -- do not add a second attribution owner -- and
   bind its output to the verified body hash.
5. **Schema migration** to retire `cast_hints`; still required by the validators and
   by `public_domain_manifest_schema.json`, so manifests and tests migrate in the
   same change. (`visual_style_policy`, the other half of this item, was ripped
   2026-08-04.)

**KNOWN AND NOT FIXED:** `canonicalize_shakespeare_text` truncates at 12,000 chars
and the interpreter sees only the first 5,000, so a 3,445-word scene reaches the
brief as ~880 words, silently. Belongs with the artifact work, where each beat is
fed its own segment rather than a blind prefix.

### 1.8 STYLE / IDENTITY DECISION WORK (backlog; not the next coder window)

Grounded by the 2026-08-03 four-agent forensics; every line has a file:line in the
session traces.

1. **"Invent one and tag it"**: add a derived style/genre field to
   `run_story_brief_reflection` (`_otr_story_brief.py:513` -- proven content-loyal on
   both specimens), stamp beside `story_brief`, repoint the treatment `Style:` line
   (`video_engine.py:1762`) and the HUD (`video_engine.py:1336` -> `_build_left`
   `:1592`) at it. Highest-leverage item here: it fixes the credits line for all six
   banks uniformly.
2. **Rename `meta.style` -> `meta.story_scaffold`** (operator: too many metas; the
   field is neither scifi nor a description). Consumers move in ONE atomic change:
   writer stamps, credits `_story_style_receipt`, `visual_plan.style`,
   `video_engine.py:1336`, tests -- AND the ledger validators (r3):
   `_otr_ledger_consistency.py` pins the field in its matrix
   (`MatrixRow("style", ...)` at `:68`, `:177`) and `_otr_ledger_cleanup.py`
   reads it too; missing them fails ledger validation on the first episode.
3. **Ghost-name reconciliation fork**: pitch cast never reaches `lock_cast` (names
   are a pure pool draw; `source_character_names` deliberately None for invention
   lanes). Decide: scrub briefs after cast lock, or propagate pitch names. Evidence:
   Evelyn/Leonard as offscreen lore; Fogbound Rails bio still opens "Lizzie Gray".
   (Cross-listed as Section 3, question E.)
4. **Dead fields found**: `ending_template` computed but zero LineRequest call sites
   pass it; `seed_policy.style_seed_env` validated but unconsumed; `dramatic_state`
   derived PRE-dialogue goes stale in the treatment.
5. **`meta` is a 120-key drawer** -- the cleanup the operator keeps asking for. Scope
   as its own rip with the ledger law (every field one owner).

---

## SECTION 2 -- RENDER WORK, BATCHED BY THE LEG THAT PROVES IT (test the least)

A leg is 1-3 hours on the one GPU. Run batches SERIALLY -- *"two windows
resetting one GPU is how each kills the other's leg"* (from the archived
scheduling note; still true). Reset per `CLAUDE.md` section 4 before every
leg. A leg that does not reach `otr/obs/` did not pass. **Every canonical leg
below is also a free chance to** (a) re-observe the two parked eyeball items
(Batch R5) and (b) watch the two ledger-cleanup behaviour changes named in
Open risks that have no live receipt yet.

### Batch R0 -- ALREADY RENDERED: the six overnight 1-act legs. One morning pass over `otr/obs/` proves six lanes

See THE CURRENT STEP at the top of this file. One pass proves: the mime
prompt, the foley bed under dialogue, plain ltx25 video, H3 silent, H3
audio-in, and the ltx_audio_in baseline -- and may satisfy Batch R1 for free
(check the H3 legs' prompt receipts before spending R1's leg).

### Batch R1 -- ONE leg on `otr_w45_minimax_h3_video` proves the H3 prompt-policy fix

**The code is shipped, green and pushed (`e923a9f3`, suite 12332/121/1). What
is missing is the one thing CPU tests cannot supply: a published episode.**

A vanilla canonical run does NOT prove it -- the canonical defaults to the
still floor, so no character beat reaches H3 at all. The leg has to select
`minimax_h3_video` deliberately.

**THE COMMAND IS READY -- do not improvise a harness.**
`scripts/otr_headless_canonical.ps1` is the sanctioned wrapper: it resets
selectively, boots the UTF-8 launcher, and ALWAYS loads the real canonical.

```powershell
cd C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\otr_headless_canonical.ps1 -Profile otr_w45_minimax_h3_video -Acts 1
```

**CORRECTED 2026-08-27, after the wrong version of this command failed a live
leg.** The first version here said to pin the engine with
`-Set "OTR_VideoDirector.character_video_model=..."`. That is REFUSED by
design: `patch_creative` whitelists CREATIVE widgets only (writers, seeds,
banks), and the video-model widgets are MANAGED -- engine routing goes through
a PROFILE (`scripts/otr_api.py:831`, `CREATIVE_WHITELIST`).
`otr_w45_minimax_h3_video` is the sanctioned profile: all three roles on
`h3_low_video` AND the h3 boot contract the engine ENFORCES
(`--reserve-vram 12`, `--disable-pinned-memory`) -- a default boot would have
been refused even if the patch had landed. Writers DO ride as `-Set`
(whitelisted), e.g.
`-Set "OTR_LedgerScriptWriter.technical_model=google/gemma-4-12b-it (11.9 GB)"`.

* Reset per `CLAUDE.md` section 4 (selective kill by CommandLine, port 8000
  empty, GPU back to ~1.5 GB); the wrapper does this, but verify it happened.
* **A FRESH EPISODE ID IS MANDATORY.** `request_hash` excludes prompt bytes, so
  an existing clip is cache-eligible and an old SPEAKING clip would be reused
  -- a false pass that looks exactly like a real one.
* Publish to `C:\Users\jeffr\Documents\ComfyUI\output\otr\obs` and confirm the
  asset on disk with a current timestamp plus `obs_publish OK`.
* **Read the prompt receipt, which is the actual verdict:** nonverbal action
  and camera PRESENT; the beat's exact dialogue and any speaking / lip-sync /
  mouth anchor ABSENT.

**The BEFORE sample already exists and must be preserved:**
`signal_lost_the_caretakers_clause_20260826_155835` in `otr/obs/` -- every beat
on `minimax_h3_video`, rendered before the fix. Do not overwrite it or reuse
its episode identity; it is half of the A/B.

### Batch R2 -- ONE image-mode sweep on `otr_soak_llmsweep_02` settles SCENE + PORTRAIT `elements: []`

**THIS ROW USED TO ORDER A FULL ARC. IT NO LONGER DOES, because the premise it
rested on was never measured.** Re-grounded 2026-08-27 against the evidence
file, the PBUG and the canonical JSON, after the operator asked the reasonable
question -- *"i am not aware of this bug, maybe it has been fixed, i wonder when
it came up"* -- and the honest answer turned out to be worth the check.

**WHAT IS ACTUALLY MEASURED.** All SIX refusal events in
`docs/2026-08-26-ideogram4-card-refusal-evidence.md` are the same beat type:

```
ideogram4_local still_music_opening_001 min=79.0 std=10.5
ideogram4_local still_music_closing_001 min=80.0 std=10.5
ideogram4_local still_music_closing_001 min=78.0 std=10.2
ideogram4_local still_music_opening_001 min=80.0 std=10.2
ideogram4_local still_music_closing_001 min=80.0 std=10.3
ideogram4_local still_music_opening_001 min=87.0 std=10.5
```

Zero SCENE refusals. Zero PORTRAIT refusals. **And the music route -- the only
route that ever refused -- was FIXED on 2026-08-26 (`ae7e7b6a`) and proven on
two published episodes with zero refusals, on the weakest writer and the
strongest alike.**

**WHERE THE SCENE/PORTRAIT CLAIM CAME FROM.** `ae7e7b6a`'s own message: the two
routes are *"still `elements: []` and therefore still expected to refuse"*.
**Expected to. Not observed to.** It is an inference from structural similarity
to a route that has since been repaired, and it hardened into this row as
though it were a finding.

**TWO MORE CORRECTIONS TO WHAT THIS ROW USED TO SAY.**
* *"the three lanes production renders with"* is wrong. `ideogram4_local`
  appears **ZERO times** in `workflows/otr_canonical.json`; the canonical names
  `z_image_turbo`, three times. The engine is OPT-IN by construction --
  `default_roles = ()`, and its own comment says *"z_image_turbo stays the
  shipped default; no model is 'primary'"*. It ran in the sweep only because
  profile `otr_soak_llmsweep_02` selects it deliberately.
* The other four local engines went **91 mints, zero refusals** across that same
  sweep (flux2_klein 35/0, z_image_turbo 32/0, flux_gen1 16/0, lumina_image
  8/0). Nothing about this is a general stills defect.

**SO THE NEXT STEP IS A MEASUREMENT, NOT A PANEL. It is a RENDER item.**
Re-run the image sweep's profile `otr_soak_llmsweep_02` against post-`ae7e7b6a`
HEAD -- `scripts/otr_bank_engine_sweep.py`, image mode, which walks every bank
against both engine profiles -- and read whether SCENE and PORTRAIT beats refuse
at all now.
* **If they refuse:** the fork below is real and gets the full arc, now with
  numbers instead of an inference.
* **If they do not:** this row collapses to a documentation correction and costs
  nothing further. That is the likelier outcome and it is why no arc runs first.

**THE FORK, PRESERVED FOR THE ARC THAT MAY NOT BE NEEDED.** The lens receives
only three things -- the prose, `kind` and `role`
(`ideogram4_local.py:644-646`). There is no subject field to derive an anchor
from, and this repo has ALREADY ruled on extracting one from the prose:
`_wrapped_caption` (`ideogram4_local.py:372`) says the composer emits *"a
comma-joined five-layer string behind a style prefix, which is a convention, not
a grammar, so any attempt to re-extract subject / setting / elements from it
mis-fires."* Two defensible answers, not equal:
* **(a)** extract a subject noun from the prose -- lens-local and small, but it
  is the option the codebase already tried and wrote off, and a wrong noun
  INVENTS CONTENT, which the source-fidelity rule forbids in as many words;
* **(b)** a new metadata channel so the producer hands the lens a real subject
  -- more wiring, but the anchor is derived rather than guessed.

Evidence: `docs/2026-08-26-ideogram-music-card-PROBLEM-STATEMENT.md` and
PBUG-20260826-01.

### Batch R3 -- FOUR canonical legs prove the WHOLE local-LLM acceptance sweep (operator directive 2026-08-25)

**One sweep of four legs (after Leg 0, Section 1.2) proves all 7 surviving
local LLM rows in BOTH slots, plus the gemma Q8_0 negative probe -- and each
leg doubles as a ledger-cleanup live-watch and an eyeball re-observation
chance.**

**THE OPERATOR'S WORDS, which are this item's charter:**
*"when all this LLM coding is done we should look and do more coding and retest
all local LLMs on a 1 runthrough that should catch it"*; *"if it doesn't fit
nicely or requires Ollama rip it from the dropdown and blast radius"*; *"clean
sweep I only want easy to load LLMs"*; *"there should be an LLM preflight guide
-- preflight guides for adding all your own components"*; *"all models should
live out here `C:\ComfyUI-Models`"*; and *"all LLMs should either be able to
play creative or technical equally. If they're not, and they were not tested or
not implemented, and they serve no worth, we should rip them out."*

**ALREADY DONE (pushed, green, lockstep-verified):** the clean sweep itself
(`Qwen/Qwen2.5-14B-Instruct` ripped with its blast radius), the PASS-tier
invariant gate, `docs/LLM_PREFLIGHT_GUIDE.md`, and the `Q6_K` dropdown removal.
The ruling is recorded in `docs/OTR_STANDING_RULINGS.md` ("ONLY EASY-TO-LOAD
LLMs SHIP"). All 7 surviving local rows are verified present on disk.

**STILL OPEN -- THE SWEEP ITSELF. Design is done (11-agent fan-out, 2026-08-25),
build is not.** The honest shape, which is NOT one render:
* **7 local rows / 2 model slots = 4 canonical legs MINIMUM**, plus a **Leg 0**
  in-process preflight (no ComfyUI; `request_slot` -> ~40-token generate ->
  `_self_unload` per row, with `reset_peak_memory_stats()` around each). Leg 0
  is one command, ~15-20 min, and is what fails loudly on a dead row -- but a
  leg that never reaches `otr/obs/` did not pass, so the 4 canonical legs are
  the real proof.
* **Every leg must PIN `--source-bank` to the scifi lane.** Canonical ships
  `'roll (any eligible bank)'`, and `_otr_scifi_news_pro.py` is the only runner
  code-verified to drive BOTH slots. Unpinned, a leg can land on a lane that
  never touches the technical slot and the sweep proves nothing about that row.
* **`gguf_quant` is ONE per-run widget**, and `unsloth/Qwen3-8B-GGUF` ships
  only `Q4_K_M` -- so any leg carrying it runs Q4_K_M.
* **A KNOWN FALSE-GREEN TO DESIGN AROUND:** `meta.slot_calls_by_slot` is
  incremented ONLY inside `_SlotScheduler._account_and_get_entry`
  (`OTR_LedgerScriptWriter.py:627`). SIX `request_slot` sites live outside it
  (`story_orchestrator.py:1224`/`:1351`, `otr_shot_lock.py:1002`,
  `OTR_LedgerFreezeCascade.py:282`, `OTR_LedgerScriptWriter.py:4393`,
  `_otr_motion_clause.py:361`). The counter proves IN-WRITER generation only;
  reading it as full-row exercise is a false green.
* **The operator's creative/technical parity rule is the sweep's acceptance
  criterion.** Structurally both slots already build from the IDENTICAL
  `dropdown_choices()` list, so no row is slot-restricted; what is unproven is
  whether each row can actually do the TECHNICAL job (constrained JSON / GBNF).
  A row that cannot do both, and was never tested or implemented, is a RIP
  candidate under his rule -- but rip only on a measured failure, never on
  assumption.
* **A negative probe worth running deliberately:** the gemma GGUF row at
  `Q8_0` / `n_ctx=4096` needs ~14.70 GiB FREE against a 15.92 GiB card with
  ComfyUI resident, and `_otr_gguf_backend.py` compares against
  `mem_get_info()` FREE with "NO silent context downgrade". Either outcome is
  informative; record both.

**Full design (coverage matrix, per-row assertions, skip-reporting rules, risks)
is in the 2026-08-25 workflow result; re-derive from this row if it is lost.**

**The old LEMMY "Phases 2-4" numbering is RETIRED (2026-08-28).** The trio it
tracked collapsed on live evidence: -01 closed 08-16 (mis-attributed), -03
closed 08-18, -03 EXTENDED closed 08-28 (`a469ffb2` -- 48 post-fix
scifi_news_pro ledgers, zero empty cast contracts, and a natural-roll LEMMY
with 8 lines published to obs on 08-26). Nothing remains for a phase number
to point at; PBUG-02 below is the sole survivor and has its own exit
condition.

### Batch R4 -- ONE canonical `fastwan_8gb` leg with 60-second opening AND closing cues proves PBUG-20260811-02

Lifted verbatim from the 2026-08-25 re-triage (the rest of that re-triage, and
the closed 2026-08-11 bank-sweep trio it retired, are in the archive):

* **PBUG-20260811-02 -- the ONLY one still OPEN.** Root cause established, the
  repair is WRITTEN, and it is not a coding item: it needs a canonical
  `fastwan_8gb` leg with 60-SECOND opening AND closing cues (long enough to
  chunk at `_MUSIC_MAX_CHUNK_DUR_S = 22.0`). **That is a RENDER window, not a
  coder slot.**

Full detail: `docs/PROD_BUG_LOG.md` and the archived trio block.

### Batch R5 -- OPPORTUNISTIC: D2 fail-hunting still legs; every clean leg is also an eyeball re-observation

Run when a render window is free and nothing above needs the box.

**D2 (from the long-standing parked row -- renders have resumed):**
Reset per AGENTS.md section 4, boot headless, run **320-word `public_domain` or
`shakespeare` still legs until one fails** (~1 in 6). Three legs on 08-04 all
published, which at that rate is a ~58% chance of zero -- neither confirmed nor
cleared.

**Either outcome is valid.** A publish is a clean leg; a **fail-closed with
complete evidence is the PROOF D1 WORKS** and is the outcome you want. When it
fails the server log names the branch itself -- arm, token, index, canonical
`prompt_hash`, repr-escaped excerpt -- plus a compact JSON `MISSING_TARGET`
record emitted BEFORE the raise (the canonical runner truncates the exception at
500 chars, `scripts/otr_api.py:749`). The log survives reboot;
`scripts/otr_rotate_log.ps1` rotates instead of truncating. D3 then fixes THAT
branch at its root and `PROD_BUG_LOG.md` gets a mechanism, not a guess.

**Do NOT:** weaken the completion gate, revive the portrait-init fallback, or
rebuild the withdrawn "give the collapse guard a still owner" fix -- the 08-04
postmortem disproved that chain (70 whiffs and 69 cast-time deferrals across 11
passes that ALL published).

Record: `docs/2026-08-04-POSTMORTEM-still-unmaterialized-320w.md`,
`docs/2026-08-04-D1-SHIPPED-still-skip-evidence.md`.

**The two eyeball re-observations that RIDE these legs (and any other real
render leg -- they cost zero extra legs):**

Both were eyeball observations against a story engine that has since had its LLM
vetoes ripped, THE LAW imposed, six banks renamed onto new packs, word-fit ceilings
retired, the repair-first plan landed, and a ledger cleanup pass added. Neither has a
reproduction at current HEAD, and under the standing rule a finding with no
reproduction is not a row. **Do NOT schedule coder time against either.** They are
settled by the operator eyeballing a real render leg: still there -> re-admit
as a FRESH dated row with that leg as evidence; gone -> the LAW-era work already fixed
it, tombstone it.

- **Announcer framing defect** (`docs/2026-07-11-announcer-framing-defect.md`).
  Episodes START a story instead of admitting you into one; the announcer takes debate
  turns instead of framing. Operator eyeball 2026-07-11. If it survives re-observation
  the fix is still seam + score contract + fail-closed validator, never Python
  authorship.
- **Name-splice defect #2.** v4-campaign Phase 0 record in HANDOFF_LOG; its timebox
  predates THE LAW.

### Batch R6 -- SEVEN 45-word engine proofs (from the archived 2026-08-13 runway table, row 3 -- CONFIRM STILL OWED before spending seven legs)

Lifted verbatim from the archived table: **Run seven fresh post-change 45-word
render proofs.** Exit condition: All seven exact public engine IDs pass against
the post-bugfix/post-Lemmy HEAD with `COVERS`, `RESULT SUCCESS`, server
`Prompt executed` + `obs_publish OK`, and the canonical OBS asset on disk.

**Flagged 2026-08-28:** the row's "See WHAT IS ACTUALLY LEFT below" pointer is
one of the eleven cross-references the 2026-08-16 audit already recorded as
BROKEN (target removed before the audit; see the archive). The engine list must
be re-derived from the live registry, and whether all seven proofs are still
owed at current HEAD has not been re-verified. Confirm before rendering.

### Deferred render items (each blocked, or waiting on something else first)

- **Capped-14B HuMo leg** -- live proof of the ping-pong lip-sync reversal fix
  (`a1d810f1`); see the coverage cluster row in Section 1.6.
- **The LOCAL mistral/gemma writer matrix** (render-window judgment question,
  not a coder slot). The Sonnet arm of the creative-writer question is answered
  (`docs/2026-07-17-model-bakeoff-scoreboard.md`); the local roster comparison
  never ran.
- **`scifi_news` live reverify** (PBUGs 20260712-22/23/24/25, fixed in tree) --
  blocked by the `scifi_news` P0 convergence defect (Section 1.6), then fan-out.
- **The WAN 8-GB proof obligations** -- a render on a PHYSICAL 8 GB card is
  still owed (the four-arm bench PREQUALIFIED on a 16 GB card told to reserve
  8 GiB, which is not the same claim), and the 18-engine GPU campaign is engine
  COVERAGE, not an 8-GB qualification. Both sit behind Section 3, question F.

---

## SECTION 3 -- WAITING ON THE OPERATOR (answer these in ONE pass)

Nothing in this section needs a render or a coder until answered. Defaults are
stated where a default exists; silence keeps the default.

### The question list

* **(A) Arm `defaults.scene_coherence_check` on any bank?** The G15 vacuity fix
  shipped 2026-08-28 (`e2807dcc`; receipt in the archive) but nothing arms the
  gate today -- the fix changed a function with zero live callers in current
  production, by design, so it shipped at zero risk. GO_FORWARD's original text
  said "measure OFFLINE over the published corpus first, then arm in ONE
  change" -- that measurement was never attempted and stays open. Whoever picks
  this up next decides first whether any bank should arm it at all before
  running that measurement. (The measurement itself is no-render work once
  ruled.)
* **(B) ANSWERED 2026-08-28: YES -- fill only the unknowns.** KNOWN rows stay
  untouchable. Ruling recorded at the top of Section 1.3 together with the
  web-tier replacement (LLM decides, cached name index).
* **(C) ANSWERED BY EVENTS 2026-08-28: the numbering is retired.** The LEMMY
  trio it tracked is fully closed on live evidence (see Batch R3's note);
  nothing remains for a phase number to point at.
* **(D) The `full_text` HTML block-join separator.** Inserting separators is a
  WIDER change to the coordinate system `source_digest` pins -- it belongs in
  the source adapter, and it is the DOMINANT P0 failure cause on live evidence.
  Detail: the first row of Section 1.6's P0 cluster.
* **(E) Ghost-name reconciliation fork:** scrub briefs after cast lock, or
  propagate pitch names. Detail: Section 1.8, item 3.
* **(F) After profile retirement, who owns a tier's native render ceiling?**
  The single blocker on the code-complete WAN 8-GB block, and the question that
  also unblocks the 8-GB writer-profile fix and the A2 echo fix. Detail blocks
  below.
* **(G) The three works that refuse to vendor** (`ghost_ship` gid 11045,
  `purple_cloud` 11229, `beleaguered_city` 11521 --
  `scripts/otr_vendor_public_domain_library.py:303/341/542` against the parser
  at `:594-686`) **need one Gutenberg fetch, so it is operator-opt-in only** --
  not schedulable inside an offline sprint.
* **(H) The Bible fan-out batch** -- one operator pass clears every row marked
  "awaiting fan-out" in the promotion table below, the PBUG-20260710-07
  retirement ratification, and the duplicate-id cleanup.
* **(I) Name the first H3 video-path sprint** -- standing context below.
* Plus the standing question list carried from before, verbatim:

* A research_only source now WITHHOLDS the OBS copy instead of killing
  the finished render (chunk 0.5 behaviour change, live since 08-15).
  If the operator wants the old kill-the-render behaviour back, it is a
  one-line revert -- say so.

* **Does `media_archive` want the catalog premise at all**, or the same
  scaffold-off treatment as `original`? Found by the five-bank beat test: a
  `pirate_radio_resistance_drama` premise was drawn over a film-reel standoff
  seeded by a real Library of Congress item on 'Midnight' (1939) -- the operator
  caught it on screen. Second specimen of the content-blind-draw class. The
  scaffold-off rule so far was stated only for `original`.
* **Rename the un-namespaced `OTR_WAN_*` knobs?** `eng_wan_i2v`'s six frozen knobs
  are `OTR_WAN_STEPS` / `_CFG` / `_SHIFT` / `_SAMPLER` / `_SCHEDULER` / `_NEGATIVE`
  -- no `I2V` namespace, unlike every sibling. Default if unruled: leave them. The
  freeze already removed the power that made the missing namespace dangerous (they
  are consent-act-only now and cannot bind a production leg), and a rename would
  silently break operator muscle memory for a sweep.
* **`style_tail_policy`'s closed enum cannot express a SHIPPED path.**
  `VALID_STYLE_TAIL_POLICIES` has `full` and `minimal_clean`, but
  `build_radio_host_prompt`'s `ltx_radio_mouth` branch
  (`otr_meta_brief_image_prompt.py:394-401`) RETURNS EARLY with
  `"%s, warm dramatic lighting"`, skipping both `finish_visual_prompt(...,
  era_profile="still")` and the `image_grade_tail` append -- deliberately, per the
  2026-07-02 operator look direction. The `ltx_audio_in` bookend row nonetheless
  declares `style_tail_policy="full"`. Adding an enum token is an operator call:
  either add a third token for "canonical warm, no era tail, no grade tail", or
  ratify that the `ltx_radio_face` path is EXEMPT from the plan's style-tail
  authority. Default if unruled: the exemption, because it changes no behaviour.
* **`check_compatibility`: ratify the inert constant, or schedule the rip?** See
  Open risks.

### Standing context for question (I): MINIMAX H3 -- A SPRINT SERIES ON THE VIDEO PATHS (operator, 2026-08-09)

**THE RULING IS IN, AND IT DISSOLVES THE OLD QUESTION.** This section used to
say the next step was an operator ruling on "does H3 belong in the video
dropdown given the 4 s floor vs the sub-4 s beats". That framing is RETIRED.
The operator's 2026-08-09 direction: H3 is **"a series of sprints all to refine
the video paths"** -- scope TBA.

**What that changes for a window picking this up:**
* It is NOT a yes/no dropdown admission any more, so do not go looking for a
  verdict to record. There is nothing blocked on the operator here.
* The unit of work is a SPRINT against the video paths, not a one-shot chunk.
  Expect several, each with its own kibitz gate, each landing green and pushed
  on its own.
* The 4 s floor is now an INPUT to that refinement -- a constraint the video
  paths have to accommodate or explicitly route around -- rather than a
  disqualifier that settles admission.

**Scope is TBA and that is deliberate.** Do NOT invent the sprint list. When the
operator names the first sprint, write it into Section 1 or 2 of this file as
its own row, and leave this section as the standing context.

**Grounding that survives the reframing:**
* Problem statement `docs/2026-08-03-PROBLEM-STATEMENT-minimax-h3.md` is
  UNTRACKED and another window's working file -- never stage, edit or delete it
  from a different window. Read it; do not touch it.
* The matrix-pattern spec already names MiniMax as a churn driver
  (`docs/2026-08-06-SPEC-subsystem-matrix-pattern.md` section 5), so a video-path
  sprint will likely collide with the un-converged matrix work (section 0). Read
  that section's "what survived all four rounds" before designing anything.
* The recipes are NOT on the table (standing directive). A video-path sprint
  refines PATHS -- routing, canvas negotiation, admission, extension -- never
  the shipped render recipe.

### Detail for question (F): the 8 GB / profile cluster -- four blocks, one underlying decision

- **WAN 8-GB low-VRAM launch contract -- CODE-COMPLETE and PROOF-INCOMPLETE. It is
  not a coding item; the one thing blocking it is an OPERATOR DECISION.**

  **Already BUILT and WIRED end to end** (verified hop by hop): `otr_8gb_wan.json`
  `video.max_render_frames=17` -> `capability_profiles` optional-key validator ->
  `_otr_workflow_apply.py:532` flatten -> `workflows/variants/otr_8gb_wan.json`
  node-87 widget slot 14 = 17 -> `otr_video_director.py:423` policy stamp ->
  `otr_shot_lock.py:1722` `ledger.video.max_render_frames` ->
  `render_driver.py:3328` per-adapter policy -> `motion_common.profile_max_render_frames()`
  -> `eng_wan_ti2v._floor_length` hard cap (`:730`) and `_planned_length` refusal
  (`:785`), with `render_driver.py:3845` refusing on drift. Landed `f914f0a4`, dead
  node-87 widget repaired `7f4644a1` + `8f41af27`, WAN deliberately excluded from
  `frame_contract.PLANNING_CAP_ENGINES` by `b23fc035`, recipe frozen `71753cb4` /
  `8424f369`, whole-beat single-UNET-load `439ce8c7`. Regression net:
  `tests/test_remaining_video_contracts.py:16-194` (nine hop-by-hop tests) plus
  `tests/test_multiclip_effective_contract.py:216,234`.

  **THE ONE OPERATOR DECISION (the actual blocker).** The ceiling reaches a leg ONLY
  through a variant workflow or a hand-set widget: `otr_canonical.json` node 87 ships
  `max_render_frames=0`, so a plain canonical WAN run is UNPINNED and inherits
  `_TI2V_MAX_FRAMES = 177` -- exactly the 2026-07-23 failure shape. The obvious patch
  (pin 17 in the canonical) is WRONG: the canonical serves every tier, and 17 is the
  8-GB tier's number, so pinning it would cap LTX/HuMo 16-GB legs too. The channel
  that carries 17 today is `config/profiles/*.json`, which is on the RETIREMENT list
  -- so writing new behaviour onto it is forbidden. **Decision needed: after profile
  retirement, who owns a tier's native render ceiling?** The shape that fits the
  per-adapter-ownership doctrine is that `eng_wan_ti2v` DECLARES its own tier ceiling
  (a capability-row field), the widget becomes an operator OVERRIDE with 0 meaning
  "use the adapter's own contract", and the profile channel stops mattering. That is
  a real design change with a live-behaviour blast radius on any card with headroom
  (the VRAM predictor currently gets to ask for more than 17 and often can), so it is
  NOT being written on assumption. Ratify the shape first.

  **Also open, all PROOF obligations rather than build work:** the 18-engine GPU
  campaign is engine COVERAGE, not an 8-GB qualification; and a render on a PHYSICAL
  8 GB card is still owed -- the four-arm bench PREQUALIFIED on a 16 GB card told to
  reserve 8 GiB, which is not the same claim.

  **One untested edge, cheap to close whenever this reopens:** WAN is out of
  `PLANNING_CAP_ENGINES`, so a tier ceiling and a multi-clip plan CAN contradict by
  design, and `_planned_length` hard-refuses mid-episode when they do -- but no test
  asserts a 17-frame tier survives a multi-segment beat. `:216`/`:234` in
  `test_multiclip_effective_contract.py` pin the topology, not that outcome.

- **THE 8 GB PROFILE FAMILY CANNOT RUN ITS OWN WRITER** (found 2026-07-27;
  LIVE-REPRODUCED TWICE on two different banks, then confirmed by a two-strikes
  kibitz panel -- codex `gpt-5.6-sol` high and agy independently reached the same
  diagnosis). `config/profiles/otr_8gb_ltx.json` pairs a 12B GGUF writer
  (`gemma-4-12b-it-Q4_K_M`, 6.63 GB of weights) with `llm.gguf_n_ctx: 2048` under a
  declared `vram_ceiling_gb: 6.8`. The pipeline's own smallest prompt needs **2064
  input tokens** and P0 reserves 2800 output (`_P0_BASE_OUTPUT_TOKENS`), so the leg
  dies in `OTR_LedgerScriptWriter` before any render. Live preflight, verbatim:
  `Needed=8.13 GB (weights=6.63, kv=1.40 @ n_ctx=2048)`. **ctx is the SYMPTOM; the
  writer MODEL is the cause** -- 4096 puts it near 9.4-9.5 GB, OOM on the very card
  the tier exists for. Every 2048-ctx profile (`otr_8gb_ltx`, `otr_8gb_wan`,
  `8gb_lite`, `cpu_floor`, `otr_amd8_rocm`, `otr_cloud_lanes`) is `status=draft` and
  every one pairs 2048 with the 12B; the only `status=shipping` profile is
  `16gb_full` (4096 + Mistral-Nemo). **NOT a one-line profile edit:** the GGUF
  registry ships exactly two rows (`unsloth/gemma-4-12b-it-GGUF`,
  `unsloth/Qwen3-8B-GGUF`); `google/gemma-2-2b-it` is in the TRANSFORMERS catalog, a
  different lane -- agy proposed it and was wrong, recorded so nobody re-derives it.
  **Largely mooted by profile retirement:** with no profile passed, the canonical
  JSON's own `gguf_n_ctx=4096` / Q8_0 binds and the leg runs. **Fix the profiles or
  finish retiring them; do not leave both.**

- **A2 -- HELD pending the profile retire-now vs retire-later scope. The profile's
  `llm` section silently overrides the canonical JSON, and the applied-overrides echo
  HIDES it.** Held because its entire subject is `apply_profile_to_workflow` and the
  printed echo -- a channel directed to be retired, so building on it now may be work
  on something scheduled for deletion. The fix SHAPE is correct and ready when the
  scope is settled. The profile's `llm.*` values win over the widgets the operator set
  in `otr_canonical.json` (which ships `creative`/`technical` = `google/gemma-4-12b-it`,
  `gguf_n_ctx=4096`, `gguf_quant=Q8_0`, `llm_vram_ceiling_gb=14.5`), while
  `scripts/otr_api.py:817` flattens only `role_overrides` / `slot_overrides` /
  `features` + two `seed_policy` keys for the printed summary -- so the run reports
  "16 overrides" while ALSO having replaced the entire LLM configuration. **Causal
  chain corrected** (triage 2026-07-27, codex; grounded): the override does NOT come
  from the validator's `OTR_ACTIVE_PROFILE` export -- it happens at submission,
  `scripts/otr_canonical_api_run.py:157` -> `apply_profile_to_workflow`; and the real
  applier (`nodes/_otr_workflow_apply.py:492-540`) ALREADY flattens `llm`; only the
  printed echo is stale. **Fix: generate the echo FROM the applier's flattened map.**
  Adding `llm` to the echo by hand leaves the next drift intact.

- **The `ltx_8gb` render-length ceiling has TWO owners that only agree by
  coincidence** (found 2026-07-27, B6 panel, two lenses independently). The coverage
  PLANNER reads `config/profiles/otr_8gb_ltx.json` `video.max_render_frames`, and
  `ltx_8gb` is the sole member of `PLANNING_CAP_ENGINES`. The ADAPTER's own
  pre-render refusal reads `OTR_LTX_8GB_MAX_FRAMES`. Today both land on 161 (profile
  unpinned, env unset), so nothing breaks. But `workflows/variants/otr_8gb_ltx.env.json`
  ships `OTR_LTX_8GB_MAX_FRAMES=97` and NOTHING currently reads that file. The day a
  launcher honours it without also pinning the profile, the planner emits a 98-161
  frame segment and the adapter refuses it MID-EPISODE -- after the stills are minted
  and, on a multi-segment beat, after the 6.34 GiB checkpoint is hoisted.
  **Deliberately NOT fixed in B6:** pinning the profile to 97 changes how a 237-frame
  beat partitions, which is a production planning decision, not a cleanup. The preset
  carries a `_ceiling_note` saying do not export it alone. Compare WAN, which B3
  wired correctly: `otr_8gb_wan.json` sets BOTH `launch.env.OTR_WAN_TI2V_MAX_FRAMES`
  and `video.max_render_frames`.

---

## SECTION 4 -- STANDING RULINGS LIFTED FROM ARCHIVED BLOCKS (2026-08-28)

Lifted VERBATIM before their blocks moved to the archive. The full
standing-rulings authority remains `docs/OTR_STANDING_RULINGS.md`.

* From the archived 2026-08-13 runway table, row 4: **"The multi-GPU
  learned-upscale stage itself is CLOSED and must not be reopened."** (The
  narrow `SpandrelEsrgan._resolve_model` hardening survives as Section 1.5.)
* From the same table, row 1: **"Do NOT restart from the Story Lab."** (The lab
  is parked, read-only and being retired; the story work happens in
  production.)
* **Operator acceptance 2026-08-28: LTX 2.5 vocalizing its prompt is CLOSED --
  it is not an open investigation.** Do not re-open it from older text. The
  named-sounds cue table, the joint-AV identity/prose removal, the runtime
  identity-leak guard and the ASR stem auditor are the shipped answer; receipts
  in the archive, `docs/PROD_BUG_LOG.md` and `git log` (`d3cca496`..`5cd4dcc8`).
* From the archived GGUF generation-deadline row (CLOSED 2026-08-25), the
  reusable lesson the row was kept for: **"A reachability question answered
  against the default path only is not answered."**

---

## SECTION 5 -- PARKED / DEFERRED (out of the working queue)

### PARKED (operator ruling 2026-08-12): wire character casting to the VOICE REFERENCE BANK

**Status: PARKED, not rejected.** Operator: *"park it on go forward."* Raised
after the operator observed we should have far more voices than the writer is
being offered. He was right, by a wide margin.

#### The finding, measured live

| pool | count | what it serves |
|------|-------|----------------|
| Bark `VOICE_PROFILES` (`config/cast_pools.py`) | **10** (6M/4F) | what the writer's casting menu offers |
| Kokoro presets on disk | 4 | ANNOUNCER only, a separate namespace |
| **Voice reference bank** (`config/voice_reference_bank.json`) | **204 declared, 153 resolvable on disk** (97M/106F/1N) | IndexTTS2 / Dia / Chatterbox cloning |

`_otr_voice_bank.default_char_engine()` returns **`indextts2`**, promoted to the
shipped character-voice default on 2026-06-04. It is a zero-shot CLONING engine:
`requires_voice_ref = True`, `voice_ref_kind = "wav_path"`. Every reference clip
is a distinct voice.

So the writer casts from **10 Bark presets** while the engine that actually
speaks the characters draws from a **153-voice reference bank**.
`_otr_casting.py` states it outright -- *"Open-character voices are always drawn
from the Bark pool (VOICE_PROFILES in config/cast_pools.py), so the tts_model is
Bark by construction"* -- and `_assert_unique_bark_voices` enforces uniqueness
across those 10.

#### Why this is parked rather than done

`MAX_SPEAKING_CAST = 10` was set from the Bark pool and is therefore a Bark
artifact, not a real ceiling. But raising the constant alone achieves NOTHING:
`_deal_voice_menu` builds the menu from `VOICE_PROFILES` and refuses with
*"voice stock capacity 10 < cast size N"*. The actual work is pointing the
casting menu at the reference bank when the character engine is a cloning
engine.

**That is why it is parked and not done unilaterally.** It changes what
`voice_preset` and `tts_model` MEAN on every cast row, and cast rows are ledger
JOIN KEYS -- `cast[].name` / `char_id` / `voice_preset` / `voice_ref_id` /
`voice_engine`, joined from `lines[].speaker` and `beats[].char_id`. Under the
operator's hard rule (*the writer must OBEY the ledger for downstream content; a
hole in the ledger is a broken render*), a change of that shape needs every
field's owner enumerated BEFORE the call is moved, not after.

#### What the work is, when it is taken up

1. Enumerate every consumer of a cast row's voice fields -- casting, TTS
   dispatch, per-beat audio slicing, credits, portraits, captions, `obs_publish`
   -- and name the new owner of each field. Exactly one owner each.
2. Make the casting menu engine-aware: Bark presets when the character engine is
   Bark, reference-bank entries when it is a cloning engine. Gender and
   `commercial_clean` already exist on bank rows.
3. Replace `_assert_unique_bark_voices` with an engine-agnostic
   one-voice-per-character invariant. The rule itself is right and must survive:
   two characters sharing a voice is a correctness defect.
4. Derive `MAX_SPEAKING_CAST` from the ACTIVE engine's pool instead of a
   constant. `tests/test_cast_size_is_a_request.py` already asserts the constant
   matches the live stock, so it will report the drift rather than hide it.
5. Prove on `scifi_news_pro` (the only bank on the fable2 writer) with a cast
   larger than 10 and complete speaker-to-`char_id` equality in the ledger.

Related and already shipped: `num_characters` is now a REQUEST rather than a cap
(operator directive, all banks) -- see `tests/test_cast_size_is_a_request.py`.

### The Shakespeare verbatim executor -- do NOT start it in a single session

**Do NOT start the Shakespeare verbatim executor in this session.** It is a
multi-session structural change gated on the ownership table
(`docs/2026-08-03-fidelity-pass-ownership.md`) with four overwrite paths to close
first, and starting it half-way is worse than not starting it.

### Carried administrative rows

- **PBUG-20260710-07** -- root fix shipped; stays ROOT-OPEN in the log until ratified
  at the next operator fan-out (green codex leg `c1f3891f` is the retire candidate).
  (Cleared by Section 3, question H.)
- **Phase-2 de-naming** (module filenames, `meta[]` ledger keys, wire-schema `.v4`
  literals) -- DEFERRED, operator-flagged, from the keep-6 rename.

---

## SECTION 6 -- STANDING TRAPS AND RECORDED LIMITS (carried knowledge; no scheduled work)

### Test-harness and tooling

- **The B7 forbidden sweep cannot see an UNTRACKED file, so a new test file passes the
  gate and fails one commit later.** `tests/test_b7_forbidden_sweep.py` builds its
  input from `git diff s29-clean-slate-gate -- *.py`, which covers tracked files only.
  A new test file added and gated in the same session is green; the moment it is
  committed it enters the diff, and a forbidden runtime identifier in it turns HEAD red
  with nothing else changed. Cost one red HEAD. **Not fixed, because the fix is a
  judgment call:** sweeping the working tree instead of the diff would widen the gate
  to every untouched file in the repo. Cheap mitigation until then -- re-run the full
  suite once after the FIRST commit of any new test file.
- **MOOT since 2026-08-23 -- kept as one line so the finding is not re-derived.**
  This row noted that two `scripts/` bake-off runners aborted a whole sweep on an
  encoder count mismatch, and called it the correct direction that an operator
  should know before an overnight run. **Both runners were deleted with every
  other bake-off** ("delete any animatediff..." was the Ghost half; "I think I am
  done with all bakeoffs" was this one). The finding still generalises: a runner
  that discards the encoder's return value and recomputes the count independently
  will disagree with it silently. Worth carrying into any replacement sweep.
- **LATENT, not reachable today: the fewest-segments partitioner can accept a
  disproportionate trim on a WIDE discrete menu.** WIRE-W1 makes `partition_beat` take
  the lowest segment count that covers, including via a permitted tail trim. On a
  ladder that is always the right trade; on a DISCRETE menu whose largest entry dwarfs
  its smallest it need not be -- covering 1019 frames from a `(10, 999)` menu, two
  segments give `[999, 999]` and discard 979 frames where three give `[999, 10, 10]`
  exactly. **A bound was written, MEASURED and REVERTED, and the measurement is the
  point:** rejecting a trim of a whole smallest-clip turned `[12, 12]` into
  `[12, 4, 4]` on a `min=4 max=12 quantum=8` ladder -- a third render and a third model
  load to recover four frames -- across 4,885 cases in the sweep grid. The widest
  shipped menus are Veo's `(100, 150, 200)` and Pixverse's `(125, 200)`, whose worst
  real trim is 25 frames. Revisit only if an adapter declares a menu with an extreme
  ratio; the reasoning is recorded in `coverage_plan.partition_beat` so the next reader
  does not re-derive the bound and re-ship the regression.

---

## Bug Bible promotion field -- pending actions only

| Record | Pending action |
|---|---|
| `PBUG-20260712-22/23/24/25` | Live reverify -- blocked by the `scifi_news` P0 convergence defect, then fan-out |
| `PBUG-20260712-18/19/26` + `PBUG-20260713-15..18` + `-20` | Awaiting the next operator Bible fan-out (overlap check + approval; Section 3, question H) |
| `PBUG-20260713-19` | Live requalification pending (promoted BUG-05.11) |
| duplicate-id cleanup | Same fan-out: BUG-11.54 legacy_id -> `PBUG-20260713-21`; verify the acronym-union rule's legacy_id (both Bible rows cite `-10`; see the log's renumber note) |
| historical `PBUG-20260711-18` | Keep as a standing context/cap engineering risk; never eligible from static evidence |
| `PBUG-20260710-07` | Ratify retirement at the next fan-out (green codex leg `c1f3891f`) |
| **Seedance softener mangles authored prompts (2026-08-17)** | **CANDIDATE, not admissible yet.** A blind regex pass over authored text produced "Dial slowly sweeps wildly" and inverted "vibrates aggressively" -> "vibrates subtly" on the DEFAULT pack's most energetic beat. Provable statically and now fixed pack-side, but it conditions a CLOUD render this repo cannot observe, so it fails the admission rule. Promote only if a cloud leg ever runs and produces the artifact. Nearest existing coverage is `12.108`'s `self-veto-resolution` / `phrase-not-word-matching` tags, which do NOT cover blind-regex rewriting of authored text |

(The 12.139 / 12.140 promotions completed 2026-08-28 and the 2026-08-25 /
2026-08-18 / 2026-08-17 promotion receipts are in the archive.)

The active production-fix owner updates `docs/PROD_BUG_LOG.md`; the approval queue is
`docs/BUG_BIBLE_PROMOTION_QUEUE.md`; no plan review or invented fixture creates a row.

## Open risks

- **NO CLIENT BANK HAS EVER RUN LIVE.** Every extensibility wave is proven by the suite
  and by contract tests, and the first real client bundle is still an unproven path end
  to end (fetch -> interpret -> writer -> cleanup -> tail -> publish). Treat the first
  live client-bank leg as a qualification, not a formality. Deferred power-user tiers
  (client own-runner + staging, dependency manifest, standalone story_rules) are
  explicitly OUT of v1 and are a NEW block if the operator ever wants them.
- **CLIENT-AUTHORED PYTHON executes in-process** (wave 3). The posture that must hold in
  every future change: `--activate` is the consent act; the seam fails LOUD
  (`UserBankExecutionError`) and never substitutes; client code never touches the
  canonical ledger; owner IDENTITY is verified so a bank can only run its OWN bundle; the
  shipped fetcher/interpreter registries are never widened to admit a client id. Do not
  relax any of these for convenience.
- **The client-facing surface is LIVE TEXT, not just docs:** the `custom_source_bank`
  row's `guide_ref` is raised to the operator by `require_runnable_bank`, and the
  `source_bank` tooltip repeats it. Any future change to the activation path (folder
  name, CLI verb, restart behaviour) must update `nodes/story_packs/banks.json`, that
  tooltip and `docs/EXTENDING_OTR.md` together, or the product will confidently instruct
  clients to do the wrong thing.
- **`check_compatibility` is RESERVED, not wired** -- operator/planner decision flagged,
  with a 2-of-2 recommendation to RIP (codex and Fable independently, Claude grounded
  both). The argument that decided it: the "it reserves the name" benefit is FALSE --
  `BUNDLE_ENTRY_ATTRS` constrains what OTR-side code may request from
  `bundle_entry_point()`, it reserves nothing against clients, and activation provably
  ignores whatever a client puts under that name (`tests/test_otr_check_cli.py:335`
  asserts a bundle whose `check_compatibility` is a plain integer activates). The only
  artifact that reserves the name is the `EXTENDING_OTR.md` paragraph, which exists
  either way. Case AGAINST: churn on landed green code for zero behaviour change, and the
  plan of record already names the future consumer (randomizer eligibility). Blast radius
  if ripped: ~5 code sites, 2 test files, 3 docs; no workflow JSON, no routing, no
  source-payload consumer. **Not a coder chunk** -- either ratify the inert constant or
  schedule the rip as a planner chunk. Proposed doctrine line: a name published to
  clients before its consumer exists lives in the client-facing DOC as "reserved, no
  contract, ignored if defined" and nowhere in executable code, because code that names
  an interface is read as enforcing it.
- **The ledger-cleanup pass runs on EVERY bank, not just client banks** (`3d97a130`). It
  is a no-op on a complete ledger and costs no LLM call there, but two shipped-lane
  behaviours did change and are worth watching on the next live legs: (a) unsafe spoken
  language on a `content_owned_readonly` bank is now REPAIRED at the writer tail instead
  of reaching G9 untouched, so a leg that used to die at freeze may now ship a sanitized
  line; (b) a blank `meta.episode_title` is now filled at the tail instead of exploding
  later in `otr_credits_roll`. Both are the intended direction under THE LAW; neither has
  a live receipt yet.
- No code lands mid-sweep of an active qualification campaign (the 420-rung
  uniform-code-confound lesson).
- There is no standalone SFX provider layer to rebuild. Current video clips are
  silent and the terminal mux uses the frozen upstream master audio. The future
  direction in `ROADMAP.md` is to retain and mix selected video-generation audio
  as inexpensive ambience; do not revive the fast-moving provider/bed stack or
  claim that future path is already wired.
- Lean-mean has one current ordered campaign in `docs/LEAN_MEAN_CLEANUP.md`.
  The retired FRONT/TAIL and SW-1 execution model must not be revived.

## After this queue

One coder window at a time; every chunk = focused tests + full suite + Bug Bible
+ commit AND push + `HEAD == origin/v2.0-alpha`.

When the sections above are exhausted, continue with `ROADMAP.md`: lean-mean ->
RunPod/AMD/Mac -> install -> product docs/v2 release. That is a pointer, not
work that precedes lean-mean. Lean-mean is not an item in this queue:
`docs/LEAN_MEAN_CLEANUP.md` is its sole current scope, blast-radius,
coding-order, and verification authority.
