# OTR Go-Forward Plan

**Updated:** 2026-07-15 late night -- HEAD 4cd36761 (v2.0-alpha). Full plan-stack
baseline: every active plan doc audited against HEAD and re-headered (fan-out
audit, 3 grounded agents); this file's queue re-grounded. Session record in
HANDOFF_LOG.md (top entry).

**CAMPAIGN IN FLIGHT (2026-07-15 evening):** the three-phase bake-off campaign is
running. PHASE A (Fable final gate on the 8 _v3 promotions + source-snapshot
B7/B8) = PASSED, no build-breakers, nothing folded (general-purpose grounded
review + Claude anchor both clean; Fable out of credits + codex CLI unhealthy ->
substituted by the grounded reviews + the live renders). PHASE B (F2 live-replay
proof) = DONE: original_radio snapshot captured (sha ed1c941f8e99), triplet run
30w local under C7 -> base GREEN, _v2/_v3 content-failed IDENTICALLY on the
deterministic weapons gate (clean F2: the pack is the only causal variable);
acceptance met (REPLAY sha + cast_seed_source == "OTR_CAST_SEED override").
PHASE C (160-leg bake-off: 16 _v2/_v3 lanes x 5 tiers x 2 profiles) = 30w smokes
(32 legs) LAUNCHED, production mode, autonomous (~5h); then 120/320/420/720.
Runner tmp/_phaseC_sweep.ps1; receipts tmp/_phaseC_receipts.csv. Content-FAILs
recorded with reason, never re-rolled. See HANDOFF_LOG.md (top entry) for detail.

## CURRENT STEP (2026-07-15): Bank-Improvement Bake-off -- source-snapshot injection (B7/B8)

All 24 selectable `source_bank` lanes (8 base + 8 `_v2` + 8 `_v3`) are BUILT, green,
and pushed; zero canonical-JSON diff (the dropdown is a dynamic `list_bank_ids()`
combo, `OTR_LedgerScriptWriter.py:3106`). Detail specs (the
`2026-07-15-bank-improvement-bakeoff/` folder is gitignored -- read from disk):
r3 wiring `.../kibitz/r3-final.md`; per-bank v2/v3 content
`.../roundtable/pass01_plan.md` Sec D; chunk-4 hardened design
`kibitz-runs/2026-07-15-chunk4-v3-lanes/r2/final.md`.

Shipped:
- **CHUNK 1** (`9e0fdf9e`): `base_source_bank_id` helper + 5 family sites.
- **CHUNK 2** (`19872aa6`): 8 `_v2` rows + packs (Sec-D seams) + B1 owner_bank
  threading + B5 pins. F8: "EDNA FROST've" is model output (NOT `_otr_ledger_scrub`),
  so the ALL-CAPS-no-contraction rule lives in the `media_archive_v2` seam.
- **CHUNK 3** = CUT (r3 B2).
- **CHUNK 4** (`c32d4c04`): 5 clone pipelines + 8 `_v3` rows + 8 v3 packs +
  `run_v3_advisory` (deterministic, advisory-only, L-6/no-hole) + `_make_v3_runner`
  wrapper factory (kibitz r2: ONE factory, not 3 files -- the sci-fi runners assemble
  the ledger IN-runner, so the wrapper reads `led` uniformly; the assemble-timing
  "trap" was a misread) + `_INLINE_V3_PIPELINES` + one inline post-Phase-0 hook +
  fable2 early word-budget gate family-fix + tooltip. 24 runnable / 25 visible;
  bijection-validated (`test_custom_runner_truthfulness`); suite 7884 green, Bug
  Bible 17 green, canonical delta = none.
- **SOURCE-SNAPSHOT (B7/B8)** (`031851ce`): frozen-source replay layer. New leaf
  `nodes/_otr_source_snapshot.py` -- a process-wide manifest keyed by BASE bank
  (env `OTR_SOURCE_SNAPSHOT_MANIFEST`), loaded in `_resolve_inputs` IMMEDIATELY
  after bank resolution and BEFORE the three source branches, so a replay
  bypasses RSS/random. Base-mismatch / malformed / altered-payload envelopes
  raise `SourceSnapshotError` LOUD (never a silent fall-through). The replayed
  `source_meta` sidecar carries the same fields a live branch would (spark_atoms
  for original, cast_hints for adaptation), so every downstream owner is fed.
  B8 seeds: `OTR_FABLE2_SEED=42` pinned under C7 + a manifest echo in
  `_otr_soak_server_launch.cmd`. Canonical delta = none. Dry registry-load = 24
  runnable / 25 visible.
- **kibitz r4 CONVERGED + hardened (`c28af5f4`):** Codex (gpt-5.5 high) + Claude
  anchor (agy flaked). Folded one CONFIRMED footgun -> the source-snapshot is now
  **strict-by-default**: a configured manifest that lacks the selected bank's base
  now RAISES (opt-in `"allow_partial": true` for freeze-some/source-rest-live),
  never a silent live source. Added a LOUD C7-replay warning (source frozen but
  cast/style unpinned). +6 net tests -> suite 7907 green, Bug Bible 17. Artifacts:
  `docs/2026-07-15-bank-improvement-bakeoff/kibitz/r4-convergence-plan.md` +
  `kibitz-runs/2026-07-15-bank-bakeoff-r4/r4/{claude_anchor,codex,final}.md`.

NEXT (in order):
1. **Fable final gate** -- DONE 2026-07-15 evening: PASSED, no build-breakers,
   nothing folded (Fable out of credits + codex CLI unhealthy -> substituted by the
   general-purpose grounded review + Claude anchor + the live renders; see the
   CAMPAIGN block above + HANDOFF_LOG top entry).
2. **Live replay proof (F2)** -- DONE 2026-07-15 evening: original_radio triplet at
   30w local under `OTR_C7=1` + captured manifest (sha ed1c941f8e99). Acceptance
   met: server log `source-snapshot REPLAY ... sha=ed1c941f8e99` + ledger meta
   `cast_seed_source == "OTR_CAST_SEED override"` on all 3. base GREEN; _v2/_v3
   content-failed identically (weapons gate) -> pack is the only causal variable.
   NOTE: the literal `[launch]` C7/manifest echoes route to the hidden console (not
   the server log); the writer's REPLAY line + cast_seed_source are the ground-truth
   proofs -- a one-line launcher echo->%1 fix would satisfy the literal-echo wording.
3. **Phase C bake-off (IN FLIGHT):** 30w smokes (32 legs) running; then
   120/320/420/720 x {local,aion}. Then the durable report + World Cup scoreboard.
   Code-state verify unchanged (no fold): 24 runnable / 25 visible; canonical clean.
4. **After phase C -- operator decisions:** (a) roster trim (operator may remove
   weak lanes; verdict + phase-C receipts are the evidence); (b) which `_v2`/`_v3`
   seams promote into their base packs; (c) the 720-verdict IMPROVE passes (below).
   The F2 finding (original_radio `_v2`/`_v3` seam steers toward weapons content vs
   base) feeds the same seam-tuning pass. `tencent/hy3:free` panel seat expires
   2026-07-21.
5. **Unblock the mistral/gemma creative-writer matrices** (verdict open item 3) --
   the smoke-blocked local writer matrices are the only path to "best model" rather
   than "best bank on aion". Render-window work, not coder-slot.

Note: the v3 packs currently carry the v2 seam text; the STRUCTURAL v3 delta is the
advisory diagnostic (kibitz-final ruling). If deeper per-lane v3 seam text is wanted
(Sec-D one-liners), the packs are in place to edit -- a bake-off tuning follow-on.

r3 rulings that MUST hold (all satisfied in the shipped chunks):
- **B1:** provenance/owner_bank uses the ACTUAL variant id (never base-mapped);
  `base_source_bank_id` is FAMILY behaviour only.
- **B2:** adaptation + original v3 are INLINE (clone pipeline in
  `_LEGACY_INLINE_PIPELINES`); only the 3 sci-fi v3 are own-runner (wrapper factory
  in `_RUNNER_BY_PIPELINE`).
- **B5:** every variant row is BEFORE `custom_source_bank`; pinned registry-order
  tuples updated per chunk.
- Every chunk: full suite + Bug Bible GREEN -> commit AND push `v2.0-alpha` -> HEAD==origin.

## THE LAW (operator, 2026-07-13 -- supersedes every plan below that disagrees)

> **AN AUDIT MAY IMPROVE A STORY. IT MAY NEVER FAIL ONE.**
> Only DETERMINISTIC validators may end an episode. An LLM verdict may trigger a
> bounded rewrite; it may never raise.

### Done 2026-07-13 (all pushed to `v2.0-alpha`)

Every LLM veto in the codebase is gone. `original_codex56sol` lost P4 (fair play),
P7/P8 (blind listener + retake), P9 (final contract audit) and the P5 score-intent
anchor patch: 9 LLM passes -> 4 plus one per-line script patch. Fair play is now a
deterministic contract -- the device anchor is SPOKEN on a clue line before the
reveal line -- repaired by the bounded line patch. `scifi_gemini` lost `sfw_pass`,
the P5-recheck and `SciFiGeminiRewriteExhaustedError`; its spoken-text check moved
INTO the P4/P6 ladder with a cast-name/source-acronym exemption. `scifi_sonnet`
lost the `severity`/`invented_fact_flags`/`sfw_pass` veto, replaced by
`ungrounded_lines` (a factual line must cite a real dossier fact and may speak only
numbers the source states -- a proof). `scifi_fable2` lost the P8 LLM ledger audit,
which raised AFTER `_assemble` and two saves. `original_radio`'s corroboration was
a raw substring scan over the judge's OWN prose -- now word-boundary, script-only.

**G9** in `_otr_ledger_freeze.run_gap_audit` is the first working SFW ship-stop: a
word-boundary DEFAULT_PROFANITY_TERMS scan over spoken ledger text, on the one path
every lane crosses, raising at Phase 10. The lanes that had any profanity check had
it as an LLM opinion; codex and fable2 had none.

(Historical note: `scifi_gemini` and `original_codex56sol` were subsequently ripped
from the roster entirely @ `3312aec7`, on the 720-verdict LEAVE ruling.)

### Live proof

| Leg | Prompt | Result |
|---|---|---|
| `original_codex56sol` 30w | `fb34bf4f` | SUCCESS + obs_publish + 54.5 MB asset |
| `original_codex56sol` 120w | `9874b749` | SUCCESS + obs_publish + 84.3 MB asset |
| `original_codex56sol` 420w | `b9c49e0d` | SUCCESS + obs_publish + 58.6 MB asset |
| `scifi_gemini` 30w | `12f7ecde` | SUCCESS + obs_publish + 46.0 MB asset |

PBUG-20260713-15..18 logged and closed by `fb34bf4f`; -14 marked SUPERSEDED.

**Bakeoff follow-up note (2026-07-14):** the Shakespeare and likely
`public_domain_story` source banks currently pass through the shared
story-grammar/style adaptation layer, so their live episode framing is not
source-native in the same way the source text is. Accept this for the current
evidence run; after the bakeoff, consider source-specific dramatic framing
modes and compare them against the shared adapter without changing the frozen
receipts. Preserve source character names and roles by default in those banks;
do not mint a fresh cast unless an explicit adaptation mode requests it.

### The bug class that cost twelve live rolls -- "the lost anchor"

A pass hands an LLM an IMMUTABLE string Python already owns -- a constraint-draw
field, a dealt card, a locked speaker, a coordinate from an accepted artifact --
and asks for it back verbatim. The model paraphrases. Python compares exactly and
kills the episode over a copy of its own input.

**Restoring an input is not authoring.** Restore when the correction is FORCED
(exactly one value possible); return it to the model when it is not. Three further
laws proven live: a repair prompt that does not fit is worse than no repair
(PROMPT_GUARD truncates the contract silently); a bounded repair must ask for the
unit the model can deliver (batch a patch and a partial success becomes a total
failure); and "it is broken" is not a repair prompt -- name the missing object, the
unassigned clue, the exact string.

### Per-lane live-proof ladder

DONE -- superseded by the executed campaigns: the 720 bake-off (all ten
then-rostered banks, verdict `docs/2026-07-15-720-bakeoff-verdict.md`) and the
in-flight phase-C sweep supply the per-lane live proofs; receipts in
`tmp/_phaseC_receipts.csv` and HANDOFF_LOG.

---

## Re-grounded coder queue (2026-07-15 baseline)

Replaces the 2026-07-12 sprint table (prior full revision: 2026-07-12 21:04 PDT,
in git history). Same discipline: one coder window at a time; every chunk =
focused tests + full suite + Bug Bible + commit AND push + HEAD==origin. Dated
plans are implementation evidence, not competing queues; each doc now carries a
2026-07-15 status header stating exactly what it needs before execution.

```text
finish bake-off campaign (render window; operator verdicts)
  -> quick-wins block (reverify tail, cliche excision, announcer contract,
     watchdog, fable2 C5, interstitial-audio rip, ENGINE_MATRIX, context/cap)
  -> LEAN-MEAN RIP (dedicated window; drift-check header first)   [operator may
  -> user source lanes / extensibility (gated: sec-16 + r5)        swap these two]
  -> Randomizer A -> dynamic_story
  -> ROADMAP (SFX campaign after Timeline Cue Ledger gate)
```

### Quick-wins block (~4-9 coder-days, small chunks, any order inside the block)

| # | Chunk | Gate | Est |
|---:|---|---|---:|
| 1 | Sci-Fi Codex reverify tail | PBUGs 20260712-22/23/24/25 are FIXED IN TREE, LIVE REVERIFY PENDING; the campaign's canonical `scifi_codex` legs (phase C 120w) are the natural reverify vehicle -- confirm receipts, mark the log, release the coder slot formally. | 0.25 d |
| 2 | Cliche-span excision (X1-X4) | `docs/2026-07-10-llm-first-story-edit-pass.md` Wave 3: `repair_cliche_span` (`_otr_line_composer.py` ~:2632/:2676) + `cliche_replacements` in all 8 story_rules JSONs still rewrite SPOKEN lines -- a standing violation of the LLM-first directive. Excise; suite + Bible. | 0.5-1 d |
| 3 | Announcer framing contract | `docs/2026-07-11-announcer-framing-defect.md` -- still fully OPEN. Pack seam + score contract + fail-closed STRUCTURAL validator (lawful under THE LAW); `original_radio_v2`'s billboard/sign-off seam is prior art. Fold into the same pass as the 720-verdict IMPROVE seam work (below) so packs are touched once. | 0.5-1 d |
| 4 | 720-verdict IMPROVE passes | shakespeare: confirm which seam version produced judged leg `c42700e1`, second prompt pass if the fix didn't take; scifi_sonnet: seam consolidation (nine seams -> fewer; it owns the set's only outright FAIL); original_radio: clarity/throughline without losing the noir mood + the F2 weapons-steering seam finding; science_news: constrain the concept, keep the steady 18-beat template. Seam/prompt work, no Python authorship. | 1-3 d |
| 5 | Canonical watchdog support | Unchanged scope: runner heartbeats, watchdog recognizes canonical `RESULT`, pinned failure/stall paths -- plus the campaign follow-up: launcher echoes C7/manifest vars into `%1` (one line). Harness defect, not a PBUG. | 0.5 d |
| 6 | Fable2 C5 consumers | Captions and credits use alias-aware cast lookup; HuMo stale guard uses role/source-family/ShotLock identity. | 0.5-1 d |
| 7 | Rip interstitial audio only | Remove synthesis, insertion, timing, and dead tests; retain `music_inter` story/visual semantics. | 0.5-1 d |
| 8 | `docs/ENGINE_MATRIX.md` | Extend `scripts/build_variants.py --check` to emit + diff the matrix from the three live CAPABILITIES registries; link from README. PRECONDITION for Lean-Mean W6. | 0.5-1 d |
| 9 | Context/cap foundation | One provider-effective cap/count/reservation/must-fit authority feeding preflight, invocation, receipts; no silent truncation, no blind cap raise. Partially advanced by the static-row ctx fix (`32e680b2`, PBUG-20260713-20, live-reverified at ctx=131072); the authority itself is still open. Carries the diagnostic-gap class from SUPERSEDED PBUG-20260712-17 (target lane ripped @ `3312aec7`): if attempt capture is needed, re-target the PARKED telemetry seam (generic `_otr_structured_call` callback; reconcile with the existing `on_attempt_complete` hook) at a surviving lane. | 1-3 d |
| 10 | Operator backlog (render tuning) | Kokoro ALL-CAPS pre-TTS normalization (TTS-only copy, never the ledger `spoken_text` or captions; confirm the canonical default engine first) + ending credits ~1.5x faster (`nodes/otr_credits_roll.py`; if a widget, the canonical JSON changes in the SAME commit). Ideal filler during render campaigns. | 0.5-1 d |

RETIRED from the old table: **item 5, Codex56 attempt telemetry** (target lane
ripped @ `3312aec7`; plan `docs/2026-07-12-codex56sol-llm-telemetry-plan.md` is
SUPERSEDED, its lane-agnostic pieces PARKED for portability -- see its header);
**item 6, PBUG-20260712-17 root fix** (entry SUPERSEDED in the log, same rip);
**item 10, the "fresh two-matrix bakeoff"** (superseded by the executed 720
bake-off + the in-flight phase-C campaign; its hy3-creative 420 design dies with
the 2026-07-21 hy3 expiry).

### Big blocks (in order; the first two may be swapped by operator call)

1. **LEAN-MEAN RIP** -- `docs/2026-07-10-lean-mean-rip-final.md`, D-1..D-6
   RATIFIED, CLEARED TO EXECUTE **after its 2026-07-15 drift-check header is
   satisfied**: SW-1 full seam re-survey (writer now 7,703 LOC; `_run_writer_tail`
   already extracted; `_otr_source_snapshot` now lives inside `_resolve_inputs`),
   SW-3 news_ingest re-survey, W6 keep-list adds + the ENGINE_MATRIX precondition
   (quick-win 8), W7 tombstone re-triage, R-7 re-grep of all line cites. Kill
   lists + W5's positional obligation re-verified LIVE 2026-07-15 and intact;
   nothing was double-ripped. Dedicated window; multi-day.
2. **User source lanes / extensibility** -- `docs/2026-07-12-user-source-lanes-architecture.md`
   (supersedes the vibe-coder r2 plan). GATED: operator ratifies its section 16
   (nine flags) + one r5 confirmation kibitz pass; THEN fold into this plan and
   claim the coder slot. **~21-31 coder-days** (not the old "4-7"). The
   ratification + r5 are docs/panel work and can run DURING the lean-mean window.
3. **Randomizer Rolls Design A** -- `docs/2026-07-12-randomizer-rolls-r2-coding-plan.md`,
   AFTER extensibility (its `_otr_lane_specs` authority is ABSORBED by the
   extensibility build; this build shrinks to `_otr_bank_roll` + eligibility).
   Re-ground per its 2026-07-15 header (24-lane roster, factory-wrapped _v3
   runners, drifted pins). 1-2 d + 1 GPU day.
4. **`dynamic_story` visual direction** -- rev-5 FINAL, do not rerun panels;
   re-checked 2026-07-15: roster-agnostic, wiring snapshot still matches live
   canonical. After extensibility + randomizer; re-derive IDs at build.
   5-9 coder-days + 2-4 GPU days.

**Ranges:** quick-wins ~4-9 coder-days; lean-mean = a dedicated multi-day window
(see its doc); extensibility ~21-31; randomizer 1-2; dynamic_story 5-9. Combined
~31-51 coder-days AFTER the lean-mean window, plus campaign GPU days. ROADMAP
runway (SFX campaign etc.) excluded as before.

## Parallel planning only -- one future SFX campaign

Spend **0.5-1 docs day** re-grounding the local generated-SFX R4 candidate into a
tracked current-HEAD R4.1 plan. This does not claim the coder slot and does not
pull implementation forward.

The result updates the single later campaign in `ROADMAP.md`: Timeline Cue
Ledger C0/C1 and its blind noun-detector gate come first; if that gate passes,
the generated-SFX design supplies the renderer/mix/canonical work and supersedes
the obsolete CC0-library renderer. No second SFX implementation queue and no
library fallback survive.

The generated path retains: a static selector for Stable Audio 3 Small-SFX and
Medium, selected-profile hard failures with no fallback, ledger-bound semantic
cue authoring, in-SceneSequencer mixing, and no post-video or Whisper/alignment
lane. R4.1 must pin writer/consumer/lifecycle ownership, prompt-schema-fixture-
validator-repair parity, context budgets, receipts, and the 30/120/720
qualification ladder before roadmap coding starts.

## Bug Bible promotion field -- separate from coding status

Production admission, implementation status, and portable-rule promotion are
different facts. Do not encode promotion state inside a bug's fix status.

| Record | Production/fix status | Promotion status |
|---|---|---|
| `PBUG-20260712-17` | **SUPERSEDED 2026-07-15** -- target lane ripped @ `3312aec7`; diagnostic-gap class carried by quick-win 9 | Never eligible from this record |
| `PBUG-20260712-18` / `19` / `26` | FIXED AND LIVE VERIFIED (per log) | Overlap check + operator approval at the next fan-out |
| `PBUG-20260712-20` / `21` | FIXED AND LIVE VERIFIED | Already generalized into BUG-11.50 |
| `PBUG-20260712-22` / `23` / `24` / `25` | FIXED IN TREE; LIVE REVERIFY PENDING (quick-win 1: phase-C `scifi_codex` legs) | After reverify + fan-out |
| `PBUG-20260713-15..18` | CLOSED (live leg `fb34bf4f`) | Bible-worthy flags set; awaiting the next operator fan-out |
| `PBUG-20260713-19` | FIXED `c25d63c6`; live requalification pending | Promoted BUG-05.11 |
| `PBUG-20260713-20` | FIXED `32e680b2`; LIVE REVERIFIED (ctx=131072) | Queued for operator fan-out |
| historical `PBUG-20260711-18` | Analysis-only 720w context risk; the `32e680b2` ctx fix removed its named mechanism, but real closure is the quick-win-9 authority | **Never eligible from static evidence** |

Log hygiene folded in this baseline: the duplicate id `PBUG-20260713-10` is
resolved (the P1-overlong-question entry renumbered to `-21`; `-10` stays with
the P9-audit entry). BUG_BIBLE.yaml carries two `legacy_id: -10` rows
(~:4357/:4379) -- reconcile both at the next fan-out (see the log's renumber
note). No 07-14/15 bake-off-era PBUGs are logged yet; the campaign window owns
admitting any (e.g. the F2 weapons-steering finding, if operator admits it).

The active production-fix owner updates `docs/PROD_BUG_LOG.md`. The thin approval
queue remains `docs/BUG_BIBLE_PROMOTION_QUEUE.md`; no plan review or invented
fixture creates a row.

## Validation and handoff law

- Current whole-tree receipt (kibitz r4 fold, `c28af5f4`): **7,907 passed / 31
  skipped / 1 xfailed**; Bug Bible **17 passed**; canonical workflow **23 nodes /
  57 links**, delta = none; dry registry-load 24 runnable / 25 visible.
- Every code chunk: focused tests, full Windows suite, Bug Bible, AST/JSON/BOM/
  zero-byte checks, commit, push, and verify `HEAD == origin/v2.0-alpha`.
- Every node/widget/link/schema change edits `workflows/otr_canonical.json` in
  the same commit and runs `OTR_WorkflowValidator`, JSON round-trip, strict
  link/input, live widget-vector, and generated-variant audits.
- Reset selectively before every headless run; never blanket-kill Python. Every
  run loads the canonical workflow and writes directly to canonical episode/OBS
  paths. Asset existence, not resident VRAM, proves completion.
- One coder edits code or `workflows/otr_canonical.json` at a time; read-only
  audits and documentation may run in parallel. **Two windows are active around
  this baseline** (render campaign + this planning window): the campaign window
  should RE-READ this file before its wrap-up edit -- the lower half was
  rewritten 2026-07-15 late night.

## Open risks

- Extensibility is gated on operator section-16 ratification + the r5 pass; its
  ~21-31-day estimate is latent scope, not creep. Until ratified, it holds no
  slot and blocks randomizer + dynamic_story sequencing only.
- Lean-mean vs extensibility ORDER is an operator call. Baseline recommendation:
  lean-mean first (it deletes ~32-33k LOC the extensibility build would
  otherwise have to comprehend and preserve; the extensibility gate work --
  ratification + r5 -- is panel/docs work that runs in parallel). Both rip into
  the writer: NEVER interleave them.
- Phase C may surface new lane defects (the F2 weapons-steering finding already
  did for original_radio seams); the campaign window owns admitting PBUGs.
- User extensibility and `dynamic_story` both touch the writer, visual-style
  authority, and canonical workflow. They remain serial and each re-derives the
  live JSON.
- Generated-SFX R4 stays local/ignored evidence until the tracked R4.1 refit
  lands; it is not an executable queue.

## Pointers

- `ROADMAP.md` (current, 2026-07-12; lean-mean pin self-declares stale cites)
- `docs/PRODUCTION_SPRINT_LESSONS.md`
- `docs/PROD_BUG_LOG.md` / `docs/BUG_BIBLE_PROMOTION_QUEUE.md`
- `docs/2026-07-15-720-bakeoff-verdict.md` (KEEP/IMPROVE/LEAVE + open items)
- `docs/2026-07-12-user-source-lanes-architecture.md` (extensibility successor)
- `docs/2026-07-10-lean-mean-rip-final.md` (drift-check header 2026-07-15)
- `docs/2026-07-12-randomizer-rolls-r2-coding-plan.md`
- `docs/2026-07-12-dynamic-story-visual-scope.md`
- `docs/2026-07-10-llm-first-story-edit-pass.md` (X1-X4 live remainder)
- `docs/2026-07-11-announcer-framing-defect.md` (OPEN)
- `docs/2026-07-11-720-bakeoff-kickoff.md` / `docs/2026-07-11-timeline-cue-ledger.md`
- `docs/2026-07-11-sfx-engine-architecture/roundtable/pass04_final.md`
- `workflows/otr_canonical.json`
