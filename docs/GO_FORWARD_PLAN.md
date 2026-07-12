# OTR Go-Forward Plan

**Updated:** 2026-07-12 13:45 PDT

**Branch:** `v2.0-alpha`

**Snapshot base:** `e80c63f13675`, equal to `origin/v2.0-alpha` before the C4b ship commit

**Scope:** current qualification sprint plus the immediately following sprint only

**Longer runway:** `ROADMAP.md`

This is the only active go-forward file. Dated specifications remain supporting
evidence, not competing queues. Git history is the archive; completed sprint
narratives do not belong here.

## Coordination snapshot

- **C4b is GREEN and ships in the commit containing this receipt.** C4a
  already shipped at `56806091`. The Fable2 collision surfaces release after
  that commit is pushed and HEAD is verified equal to origin.
- **Generated SFX engine lane is queued for later triage, not coding.** Do not
  take its code or canonical-workflow collision surfaces while C4b runs. The
  current R4 candidate is a local, ignored Kibitz artifact and must be
  re-grounded against then-current HEAD before it becomes an active sprint.
- The 120-word queue completed its available sequence. `media_archive`,
  `public_domain_story`, `shakespeare`, and `original_radio` returned canonical
  `RESULT SUCCESS`; `scifi_fable2` then stopped at its deliberate 120+ gate.
  C4b now opens that gate with the real full loop; the bakeoff owner may rerun
  `scifi_fable2` after the C4b push. Receipt: `docs/_bakeoff_queue.log`.
- The latest `original_codex56sol` 120-word rerun failed at
  `P6_grounding_patch` after two attempts. It is a real live incident and is
  staged as `PBUG-20260712-17`; its earlier successful same-seed qualification
  remains evidence, not a waiver for this new failure.
- Ten banks are currently runnable; always re-derive the roster from
  `nodes/story_packs/banks.json`. `custom_source_bank` remains non-runnable.
- Port 8000 is still held by the resident server from the finished queue. The next
  live owner must perform the selective reset in `AGENTS.md`; never blanket-kill
  Python.
- Unrelated dirty work owned by another window must be preserved and excluded:
  `nodes/_otr_model_catalog.py`, `tests/test_openrouter_catalog_rows.py`,
  `docs/2026-07-11-cue-ledger-r1-codex-prompt.md`, and local bakeoff logs.
- **dynamic_story is DESIGN-FINAL and code-ready, not an in-flight docs arc.**
  `docs/2026-07-12-dynamic-story-visual-scope.md` is tracked at rev 5 FINAL;
  R1-R4 have already converged and all 60 review artifacts are tracked under
  `kibitz-runs/2026-07-12-dynamic-story-visual/{r1,r2,r3,r4}/`. Do not rerun
  R2-R4. No implementation exists yet. Codex owns the build after the bakeoff,
  Engine Matrix, and Randomizer release their collision surfaces (see Immediate
  next sprint).
- **README.md refreshed by a second (docs-only) window, 2026-07-12 13:14 PDT:**
  operator-directed refresh on `v2.0-alpha` — source-bank table (10 runnable banks),
  sfx/b-roll role-rip and VRAM-tier-rip corrections, portability-variant note,
  survival-guide cross-link. Committed with a pathspec limited to `README.md` +
  this file; no code, tests, or workflow JSON touched. Informational only — if a
  README commit appears in your rebase, this is it.

## Ordered current sprint

A coder-day below means roughly one focused engineering day. GPU qualification
wall time is listed separately and is not counted as coding time.

| Order | Chunk | Hard completion gate | Coding estimate |
|---:|---|---|---:|
| 1 | C4b Fable2 full mode -- GREEN / shipping | P2a/P4/P5 wired; real P3/P5 traces; winner assembled once; 120+ gates flipped atomically; suite + Bible green | complete |
| 2 | Canonical watchdog support | Canonical runner emits heartbeats; watchdog recognizes canonical `RESULT`; healthy long runs never false-dead | 0.5 day |
| 3 | Fable2 C5 consumers | Caption and credits use alias-aware cast lookup; HuMo stale guard uses role/source-family/ShotLock identity | 0.5-1 day |
| 4 | Rip interstitial **audio** only | Remove synth/insertion/timing path and tests; retain `music_inter` story/visual semantics | 0.5-1 day |
| 5 | Context/cap foundation | One provider-effective cap/count/must-fit authority; measured repair envelopes; no silent truncation and no blind 8192->16384 raise | 1-3 days |
| 6 | Fable2 qualification | 30 words on two local families + one declared cloud lane, same pairings at 120, then one 720 leg with complete ledger/episode/OBS proof | 0.5-1 coding day + 1-3 GPU days |
| 7 | Codex56 attempt telemetry | Land the code-ready five-file plan; prove raw vs projected records, fail-open behavior, directory rename safety, abort retention, and a 30-word live smoke | 2-4 days |
| 8 | One bakeoff, not two | Resolve/build the already-ratified contender D if still applicable, revalidate only stale receipts, freeze code, run one all-bank 720 batch, blind verdict, operator listen | 1-3 coding days + 2-5 GPU days |

**Current-sprint planning range:** about **7-14 coder-days**, or **8-15** with
normal live-failure root-fix margin. The qualification/event runs add roughly
**3-8 elapsed GPU days** but are not full-time coding.

### 1. C4b activation -- green ship receipt

Plan of record: `docs/2026-07-10-fable2-720-bakeoff-runway.md`, C4b only.

Landed surfaces:

- `nodes/_otr_scifi_fable2.py`
- `nodes/OTR_LedgerScriptWriter.py`
- `nodes/story_packs/banks.json`
- `nodes/story_packs/pipelines.json`
- `nodes/story_packs/scifi_fable2/scifi_fable2_v1.json`
- Fable2 artifact/runner tests

Receipt: 119 words executes compact mode; 120 and 420 execute the real full
three-pitch/P2a/P4/P5 path; 900 executes full mode; 901 rejects before a model
call. P3/P5 attempt traces come from observed calls and seal the selected
immutable FinalDraft. Draft2-win and draft1-tie retention both pass the exact
same winner object through P6/P7/P8; live and on-disk seals agree after
incremental saves and the shared writer tail. Focused Fable2 gate: 276 passed.
Full repository gate: 7774 passed, 31 skipped, 1 expected failure, 5 warnings.
Bug Bible: 17 passed, 11 skipped, 3 expected failures. Canonical workflow SHA256
`fb5c75801a5013e189c685dd9d1fbdf069ff22b3843d7ce9adf727efe3c5a830`;
OTR_WorkflowValidator, JSON round-trip, strict link/input, widget-vector, and
zod audits are green with **no canonical JSON diff**. No GPU render was started;
the bakeoff window retains the sequential GPU queue.

### 2. Long-run truth before long runs

Fix `scripts/otr_canonical_api_run.py` and
`scripts/otr_render_watchdog.ps1` before another long qualification. This is a
review-proven harness defect, not a production-admitted PBUG. Add focused tests
for canonical heartbeat progress, `RESULT SUCCESS`, explicit failure, and a
stalled/down-server verdict.

### 3. Fable correctness closeout

Land C5, then remove operator-banned interstitial audio as a separate green
chunk. Likely surfaces:

- `nodes/_otr_captions.py`
- `nodes/otr_credits_roll.py`
- `nodes/_otr_video_engines/render_driver.py`
- `nodes/stable_audio_theme.py`
- `nodes/scene_sequencer.py`
- `nodes/_otr_music_prompt.py`
- focused caption/credits/HuMo/cue tests

The interstitial rip must preserve story/visual cue rows. Only synthesis,
manifest audio placement, master-timeline insertion, and their dead tests go.

### 4. Context fit and qualification law

Before 720, measure the real base and repair envelopes for every whole-artifact
pass. Keep the 8192 local guard until evidence proves a larger per-profile cap is
safe; current VRAM evidence says a blind 16K change can cost 2-3 GiB. Prefer
must-fit failure and localized typed patches when a whole artifact cannot fit.

`docs/PRODUCTION_SPRINT_LESSONS.md` owns the ladder. Older references to
`120 -> 320`, `350 -> 720`, or a single-model 120 run are retired as qualification
laws. The only production ladder is:

1. tests and full gates;
2. 30 words on two local model families plus one configured cloud/frontier lane;
3. the same pairings at 120 words;
4. then 720 words.

### 5. Telemetry before the final event

Plan of record:
`docs/2026-07-12-codex56sol-llm-telemetry-plan.md`.

Follow its internal green chunks exactly: pending-retention guard, pure recorder,
shared structured-call callbacks alone, scheduler metadata, Codex56 lane wiring,
then live proofs. It intentionally changes no node, widget, ledger byte, or
canonical workflow JSON.

### 6. One authoritative bakeoff

Use `docs/2026-07-11-720-bakeoff-kickoff.md` as the self-regrounding event
driver. Do not also run the older four-bank C6/C7 event as a second campaign.
Re-derive the bank roster and reuse a receipt only when no relevant structured
prompt/schema/validator changed after it. Every leg loads
`workflows/otr_canonical.json` and proves ledger, episode asset, `obs_publish OK`,
and final OBS file.

## Immediate next sprint

1. **Generate `docs/ENGINE_MATRIX.md`.** Extend `scripts/build_variants.py` so
   `--check` emits and diffs the matrix from all three live CAPABILITIES
   registries; link it from README. Estimate: **0.5-1 day**.
2. **Randomizer Rolls Design A.** Follow
   `docs/2026-07-12-randomizer-rolls-r2-coding-plan.md`: first establish
   `_otr_lane_specs` as the one dispatch/compatibility authority, then add the
   source-bank roll, then three canonical 30-word proofs. Estimate: **1-2 days
   coding + 1 GPU day**.
3. **dynamic_story visual direction -- Codex build.** Follow the already-final
   rev-5 plan at `docs/2026-07-12-dynamic-story-visual-scope.md`; no more design
   rounds are scheduled. Capture and commit named-pack byte baselines first,
   then land the new `OTR_DynamicStoryDirection` node, shared context/receipt
   helpers, all three consumers, registration, tests, canonical node 96/link
   284 delta, and regenerated variants as the plan's single activation chunk.
   Run the section-11 VERIFY-AT-BUILD checklist and the full 30/120 model-family
   ladder. This starts only after Engine Matrix and Randomizer are green and the
   coder slot is released. Estimate: **5-9 coder-days + 2-4 elapsed GPU days**.

**Parallel docs-only track; not on the serial coder critical path:**

- **Generated SFX engine lane -- triage only.** Re-ground the local R4
  candidate at `kibitz-runs/2026-07-11-generated-sfx-engine-lane/r4/final.md`
  (with the earlier local architecture evidence under
  `docs/2026-07-11-sfx-engine-architecture/roundtable/`) and first materialize
  a tracked, current-HEAD plan. Decide whether the required R4.1
  production-readiness revision changes scope before any implementation.
  Preserve the accepted intent: an ungated static selector for Stable Audio 3
  Small-SFX and Medium; selected-profile hard failures with no fallback;
  ledger-bound LLM semantic cue authoring; in-SceneSequencer mixing; and no
  post-video SFX lane or Whisper/alignment dependency. Triage exit requires an
  authoritative writer/consumer/lifecycle/receipt map, prompt-schema-fixture-
  validator-repair parity, a failure-class repair ladder, measured base and
  repair context caps, and a canonical 30/120/720 multi-family live
  qualification recipe with episode and OBS receipt proof. Estimate:
  **0.5-1 day plan refit only**.

Randomizer waits until after the bakeoff so its dispatch refactor does not stale
receipts in the one-variable bank experiment.

**Immediate-next-sprint range:** about **7-12 coder-days + 3-5 elapsed GPU
days**. Generated-SFX planning may run in parallel and does not claim the coder
slot.

## Appended 2026-07-12 (evening): vibe-coder extensibility -- NEW QUEUE CANDIDATE, coder must re-prioritize

Added by the planning window after this file's 13:45 snapshot; it deliberately does NOT
renumber or reorder the queues above -- the next coder window folds it in.

- **What it is:** operator-directed extensibility build -- user content packs on ANY
  runnable lane (NO tiers), `user_packs/` overlay with quarantine (external junction
  root sanctioned for update survival), `story_pack` selection widget with two-channel
  consumer threading + replay sha stamps, `otr_check` CLI (one validator, two entry
  points, CP-1..CP-7 receipt contract), templates + generated `docs/EXTENDING_OTR.md`,
  three README recipes, and the writer `VALIDATE_INPUTS` suffix-flip fix.
- **Plan of record (code-ready):**
  `docs/2026-07-12-vibe-coder-extensibility-r2-coding-plan.md` @ `97d4f9eb` --
  full kibitz r1-r4 arc converged same day (artifacts under
  `kibitz-runs/2026-07-12-vibe-coder-extensibility/`, gitignored; judgment logs in each
  round's `final.md`). Scoper: `docs/2026-07-12-vibe-coder-extensibility-r1.md`.
- **ACTIVATION GATE (in-plan, hard):** claim the sole coder slot here first; obtain
  clean-or-released ownership receipts for `nodes/OTR_LedgerScriptWriter.py`,
  `nodes/_otr_story_routing.py`, `nodes/_otr_story_pack.py`,
  `nodes/_otr_visual_styles.py`, `nodes/_otr_model_catalog.py` (W3 only), `README.md`,
  and `workflows/otr_canonical.json`.
- **Shape:** ACTIVATION GATE -> W0 overlay/quarantine -> W1 story_pack widget +
  threading (the ONE canonical-JSON change) -> W2 otr_check -> W4 templates/docs;
  ownership receipt -> W3 (VALIDATE_INPUTS + shipped-ID manifest); {W3, W4} -> W5
  (suite + Bible + one canonical 30-word overlay-selection smoke with full
  episode/obs receipt chain). Estimate: **4-7 coder-days + <=1 GPU day** (W5 smoke).
- **Collision facts for re-prioritization (the coder's call, not this window's):**
  1. W1 touches the writer + `workflows/otr_canonical.json` -- it must land either
     BEFORE the bakeoff code-freeze (current-sprint item 8) or wait until after the
     event; landing mid-freeze is forbidden by the freeze itself.
  2. W0 reshapes `_otr_story_routing` internals (PackRecord map) -- Randomizer Design A
     establishes `_otr_lane_specs` as the dispatch/compatibility authority in the same
     neighborhood; whichever lands second rebases (cheaper: extensibility after
     Randomizer, or before Randomizer starts).
  3. dynamic_story's build also appends canonical surfaces (node 96 / link 284) -- no
     shared node with W1 (node 1), but both re-derive slots from live JSON; land
     serially, re-derive each time.
  4. W3 shares `nodes/_otr_model_catalog.py` with the unowned dirty-edit risk already
     listed in Open risks; the plan gates W3 on that receipt independently, so W0-W2/W4
     never wait on it.
  5. README.md was refreshed today by a docs window -- W4's recipes rebase on that text.
- **Suggested slots (pick one, or re-derive):** (a) immediately after current-sprint
  item 5 (context/cap) and before Fable2 qualification if the operator wants the
  extensibility surface demoable early -- accepts a rebase risk against item 8's
  freeze; (b) between the bakeoff (item 8) and Engine Matrix -- zero freeze risk,
  recommended default; (c) after Randomizer Design A -- cheapest `_otr_story_routing`
  rebase, longest wait. dynamic_story keeps its stated position (after Engine Matrix +
  Randomizer) in every option.

## Validation and incident rules

- Latest completed clean-code receipt available to this audit: 7,764 passed,
  31 skipped, 1 expected failure; Bug Bible 17 passed, 7 skipped, 3 expected
  failures. The unrelated dirty model-catalog pair is not covered by that receipt.
- Every code chunk: focused tests, full Windows suite, Bug Bible, AST/JSON/BOM/
  zero-byte checks, commit, push, and verify HEAD equals origin.
- Every workflow/node/widget/link change updates
  `workflows/otr_canonical.json` in the same commit and runs the full workflow
  validation ritual.
- A live failure enters `docs/PROD_BUG_LOG.md`. Only a production-proven,
  root-caused, generalizable, operator-approved rule enters
  `docs/BUG_BIBLE_PROMOTION_QUEUE.md`; static findings never do.

## Open risks

- `PBUG-20260712-17`: latest Codex56 P6 grounding-patch exhaustion; root cause
  still open.
- The old `PBUG-20260711-18` record describes a predicted context risk, not a
  live-admitted bug. Treat it as an engineering gate and never promote it unless
  production evidence later satisfies the admission rule.
- The current model-catalog/test edits have no owner/test receipt in this file.
  Preserve them and claim the coder slot only after their owner lands or releases
  them.
- The generated-SFX R4 candidate and its Roundtable evidence are ignored local
  artifacts. Triage must preserve the accepted decisions in a tracked plan
  before implementation can begin.

## Pointers

- `ROADMAP.md`
- `docs/BUG_BIBLE_PROMOTION_QUEUE.md`
- `docs/PRODUCTION_SPRINT_LESSONS.md`
- `docs/PROD_BUG_LOG.md`
- `docs/2026-07-10-fable2-720-bakeoff-runway.md`
- `docs/2026-07-11-720-bakeoff-kickoff.md`
- `docs/2026-07-12-codex56sol-llm-telemetry-plan.md`
- `docs/2026-07-12-randomizer-rolls-r2-coding-plan.md`
- `docs/2026-07-12-dynamic-story-visual-scope.md`
- `workflows/otr_canonical.json`
