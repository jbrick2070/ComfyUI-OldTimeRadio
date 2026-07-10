# QA PROBLEM STATEMENT -- scifi_fable2 S1b: freeze-cascade machinery vs the fable2 lane

**Date:** 2026-07-10 (afternoon). **Repo:** ComfyUI-OldTimeRadio, branch `v2.0-alpha`,
HEAD `8e3d9228`. **Audience:** an external analyst with full repo access. Read the REAL
Windows files under `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio`.

**FROM THE OPERATOR: these are my big problems (section 2 and 3), and beyond them please
INSPECT ALL DOWNSTREAM PATHS for possible blockers -- my concern is not just today's
defect but everything coming down the line, so the FULL fable2 lane works end to end on
every media path (section 5.5 lists my specific concerns). Deliverable: write your QA
analysis to a file -- root-cause findings + concrete fixes for the open defects, plus a
verified landmine list for the runway. Do not restate what is already fixed.**

## 1. What works (proven today, all pushed)

The scifi_fable2 lane (LLM-first multipass writer: the LLM writes a whole radio play in
strict markup, python parses/assembles it into the production ledger) is LIVE:

- **Default lane (procgen video): FULL GREEN.** 30-word episode "Einstein's Echo"
  published end-to-end (writer -> freeze cascade -> CastLock -> TTS -> stills -> video ->
  captions -> credits -> obs) in 570s. Commit `ff4c226d`.
- 25 prior consecutive live-roll failures were each root-fixed the same session
  (full per-roll ledger: `docs/2026-07-10-fable2-s1b-smoke-hardening.md`; kibitz r2/r3/r4
  under `kibitz-runs/2026-07-10-fable2-s1b-hardening/`; architecture deviations recorded
  in `docs/2026-07-10-scifi-fable2-architecture.md` section 13.5).
- Suite 7448 passed / 31 skipped / 1 xfailed; Bug Bible 17/7/3; canonical workflow JSON
  NO-DIFF; OTR_WorkflowValidator OK.
- One deep defect was already root-fixed via a sonnet+opus grounded fan-out (both
  converged): `nodes/_otr_ledger_reviewer.py` `role_mismatch` repair honored the LLM cast
  auditor's `expected="character"` on NON-CHARACTER SENTINEL char_id rows (announcer /
  music_*) via a bare breadcrumb-less assignment, and the reviewer's "clean_no_edits"
  verdict only counts Pass-2 doctor edits, hiding Pass-1 repairs. Fixed at `8e3d9228`
  (symmetric sentinel guard + `role_mismatch_repair` compose-flag + regression tests in
  `tests/test_phase3_ledger_reviewer.py`).

## 2. OPEN DEFECT A (the blocker): a skip=True mutator in the freeze cascade

**Evidence (LTX-lane roll, 2026-07-10 ~13:00, server log
`C:\Users\jeffr\Documents\ComfyUI\output\otr\episodes\_shared\fable2_s1b_server_ltx10.log`,
episode ledger under `C:\Users\jeffr\Documents\ComfyUI\output\otr\episodes\` newest
`*_ledger.json`):**

- The fable2 writer completed clean (spine complete, 8 receipts). The freeze cascade's
  Stage-5B story critic named reroll targets; the Sprint-5C targeted reroll then FAILED on
  every target with `StoryPackValidationError: migrated seam 'line_composer_system'
  missing/empty in pack scifi_fable2/scifi_fable2_v1 (no fallback)` and logged "keeping
  the original line".
- Afterwards, line `shot_002_b3` (char_id `c03`, a REAL character row, scene 2) sits in
  the saved ledger with **`skip=True`, non-empty text, `tts_skip_reason` null, and EMPTY
  `compose_flags`** -> Phase 10 gap audit: "skip=True AND non-empty text (inconsistent
  state)" -> `freeze_verdict=needs_full_rerun` -> OTR_CastLock refuses -> run dead.
- The SAME fingerprint appeared in an earlier roll (ledger
  `...\pending_20260710_105235\audio\pending_20260710_105235_ledger.json`, row
  shot_002_b3) BEFORE the reviewer fix -- i.e., this skip mutator is a SECOND, separate
  hole from the role_mismatch one.

**Questions for QA:**
1. Which exact code path in the Sprint-5C targeted reroll / A2 repair-then-ship machinery
   (`nodes/_otr_reroll.py`, `nodes/_otr_freeze_cascade.py` around the "A2
   repair-then-ship" and "needs_full_rerun for the cascade A2 ship-through" warnings)
   stamps `skip=True` on a reroll TARGET row, and on which branch does its restore/cleanup
   not run when `compose_line` raises `StoryPackValidationError` (as it always will for
   fable2, whose pack legitimately has no `line_composer_system` seam)?
2. Is the correct root fix (a) the reroll never marks a row skip before a successful
   re-compose, (b) the error path restores the row, or (c) the reroll should SKIP THE
   WHOLE PASS for banks whose pack lacks the legacy compose seams (registry-detectable:
   `_otr_story_routing.resolve_story_pack(bank).prompt_stages` has no
   `line_composer_system`) -- and how does each option interact with the A2
   "ship-through" contract ("2 residual structural errors ... never refusing" yet Phase
   10 then hard-refuses: these two postures contradict each other on this evidence)?
3. Same question for the Stage-7 critic escalation path ("Wave 1 Agent C escalation ...
   fall back to legacy line re targets=0") -- does anything else in the cascade assume
   legacy-lane pack seams or legacy row shapes that the fable2 lane does not carry?

**Constraint:** the operator law is "fix at the root, no shims; the reroll/critic passes
are LEGACY-lane machinery -- fable2 has its own P4/P5/P8 loop (currently S2, not yet
built), so cascade LLM passes that require pack seams fable2 deliberately lacks should
arguably no-op cleanly for this lane rather than half-run."

## 3. OPEN DEFECT B: LTX-lane end-to-end still unproven for fable2

The operator wants one green fable2 episode on the COMPLEX media path: character lane =
`ltx_audio_in` (IA2V: still image + per-beat audio -> LTX video with lip-sync-capable
portrait conditioning), stills = `z_image_turbo`. Reproduce with:
- boot: `scripts\_otr_soak_server_launch.cmd <log> LTX`
- driver: repo-root `_tmp_fable2_ltx_smoke.py` (ad-hoc profile = `16gb_full` +
  `role_overrides.character_visual=ltx_audio_in`, then `target_words=30`,
  `source_bank=scifi_fable2`). One prior attempt got ~25 minutes in (stills + LTX beats
  rendered) and died at the last beat only because of the (now fixed) role-flip demanding
  an announcer portrait; the remaining blocker is DEFECT A.

## 4. Known accepted behaviors (do NOT flag these)

- Announcer rows carry sentinel `char_id="announcer"` by design (mutation-proofing).
- meta["news"] = None for this lane (the treatment IS the interpretation).
- Parser performs DELETE-ONLY decoration normalization (markdown emphasis, line-leading
  letters-only delivery tags), all strips flagged in `ParsedScript.normalizations`.
- Dossier entities/numbers the source text cannot corroborate are DROPPED loudly
  (delete-only), never rerolled; the P2c read validator is the strict gate (source text =
  legality authority; token-exact numerals; spelled-number equivalence; fictional cast
  names banned case-sensitively with source-name precedence).
- P8 audit is audit-only: lexicon-confirmed kills fail loud; taste classes report-only.
- K.5.5/K.5.6 tail meta passes run post-P8 by design (fail-soft).

## 5. File inventory (primary)

- Lane runner: `nodes/_otr_scifi_fable2.py`; parser: `nodes/_otr_fable2_markup.py`
- Writer dispatch/gates/tail: `nodes/OTR_LedgerScriptWriter.py` (`_RUNNER_BY_PIPELINE`
  ~:1589, entry gates ~:3205, splice ~:3568, `_run_writer_tail`)
- Pack/seams: `nodes/story_packs/scifi_fable2/scifi_fable2_v1.json` (10 seams incl.
  `fable2_news_read_system`); registries: `nodes/story_packs/banks.json`,
  `nodes/story_packs/pipelines.json`
- Freeze cascade: `nodes/_otr_freeze_cascade.py`, `nodes/OTR_LedgerFreezeCascade.py`,
  reviewer `nodes/_otr_ledger_reviewer.py`, reroll `nodes/_otr_reroll.py`, gap audit
  `nodes/_otr_ledger_freeze.py`, `nodes/cast_lock.py`, `nodes/otr_shot_lock.py`
- Tests: `tests/test_fable2_*.py`, `tests/test_phase3_ledger_reviewer.py`,
  `tests/test_structured_call_clamp.py`; fixtures `tests/fixtures/fable2/`
- Hardening ledger: `docs/2026-07-10-fable2-s1b-smoke-hardening.md`; architecture:
  `docs/2026-07-10-scifi-fable2-architecture.md` (s13.5 deviations)

## 5.5 THINK AHEAD -- pre-audit the WHOLE remaining runway, not just today's defect

The goal is the FULL fable2 lane working at production quality. Beyond Defects A/B,
audit these coming steps NOW and flag every landmine you can verify in the code, so we
fix classes not instances:

1. **Every downstream consumer that touches fable2 rows.** The lane emits row shapes the
   legacy lane never produced live (sentinel announcer char_id on outro/coda/news-read
   rows; explicit scenes[]/shots[]/beats[] hierarchies; music sentinel rows; merged
   same-speaker line rows with proof_map). Sweep EVERY consumer between writer and obs --
   freeze cascade phases, CastLock, BatchCharacterVoices/AnnouncerVoice (TTS routing by
   char_id vs speaker_role), SceneSequencer, ShotLock, ImageGenDispatcher, render driver
   (IA2V registers: TALKING vs WIDE), MusicGen (cue ids opening/inter_NN/closing),
   EpisodeAssembler, CaptionBurn, CreditsRoll (receipt _require calls -- we already hit
   cast seed + meta.style), obs_publish, and the video HUD/news_json readers -- for
   assumptions that break on these shapes (e.g. char_id-keyed lookups that miss
   'announcer', beats[]-derived logic that never saw beats before, portrait resolution
   for sentinel rows).
2. **S2 (next sprint): the full loop** -- P1 three-pitch + P2a select + P4 critic + P5
   revision (verbatim-preserve law + keep-better-draft judge) at 350 words. Pre-audit:
   the P5 revision markup ladder shares the P3 parser -- what breaks at 3 scenes / 350
   words (envelope formula, band floor interaction at >=125w, token budgets ~970 for a
   350w play vs the 4200 cap, truncation-retry arithmetic)? Does the keep-better-draft
   judge (_defect_score) exist yet (NO -- it is unbuilt) and what contracts should it pin?
3. **Long-episode ceiling**: target_words up to 900 is claimed supported post-S2; the
   act-chunked mode is deferred. Verify the entry gates + envelope + micro-episode cap
   boundaries are coherent across 30/120/350/900 words (e.g. _ONE_DRAFT_THRESHOLD=120,
   _WORD_BAND_ABS_FLOOR=25, _micro_episode_line_cap only <60 words).
4. **Other media lanes** the sweep will hit next: wan_ti2v / wan_i2v / HuMo talking /
   cloud lanes -- same IA2V register question (which registers demand portraits, and do
   fable2's cast portraits generate for c02.. rows on every lane?).
5. **The source-bank sweep + mixed-bank soak** (GO_FORWARD section 3): science_news must
   remain byte-identical -- verify no fable2 change leaked into shared code paths in a
   way that alters legacy behavior (_otr_structured_call clamp now walks nested paths:
   confirm no legacy pass relied on the old top-level-only behavior; the reviewer
   role_mismatch guard: confirm the legacy lane never legitimately re-roles a sentinel
   row).
6. **Operator-eyeball quality items already logged** (not gates, but list anything you
   can measure): few-shot example name leakage (VERA/DOKU aired once), stance/card
   variety across seeds, register distinctness in performance, the news read's wire-desk
   tone.

## 6. Deliverable format

Write a QA analysis file with: (1) the exact skip-mutator code path (file:line, verified
by reading the code); (2) the recommended root fix among 2(a)/(b)/(c) with rationale
against the A2 ship-through contract; (3) any OTHER legacy-seam or legacy-row-shape
assumptions in the cascade that will bite the fable2 lane next; (4) a regression-test
sketch per fix. Cite real line numbers; never present an unverified claim as fact.
