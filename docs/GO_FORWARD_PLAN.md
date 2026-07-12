# OTR Go-Forward Plan

**Updated:** 2026-07-12 13:00 PDT

**Branch:** `v2.0-alpha`

**Snapshot:** `48b5511abc9d`, equal to `origin/v2.0-alpha` when this plan was rebuilt

**Scope:** current qualification sprint plus the immediately following sprint only

**Longer runway:** `ROADMAP.md`

This is the only active go-forward file. Dated specifications remain supporting
evidence, not competing queues. Git history is the archive; completed sprint
narratives do not belong here.

## Coordination snapshot

- **C4b is in coding review now.** Do not restart it or take its collision
  surfaces. C4a already shipped at `56806091`.
- The 120-word queue completed its available sequence. `media_archive`,
  `public_domain_story`, `shakespeare`, and `original_radio` returned canonical
  `RESULT SUCCESS`; `scifi_fable2` then stopped at its deliberate 120+ gate, which
  C4b is meant to open. Receipt: `docs/_bakeoff_queue.log`.
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
| 1 | Finish C4b review and land Fable2 full mode | P2a/P4/P5 wired; real P3/P5 traces; winner assembled once; 120+ gates flip atomically; suite + Bible green | 0.5-1.5 days remaining |
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

### 1. C4b activation -- active review

Plan of record: `docs/2026-07-10-fable2-720-bakeoff-runway.md`, C4b only.

Expected surfaces:

- `nodes/_otr_scifi_fable2.py`
- `nodes/OTR_LedgerScriptWriter.py`
- `nodes/story_packs/pipelines.json`
- Fable2 artifact/runner tests

No canonical workflow diff is expected. The review must recheck the real HEAD,
then land one green commit and push it. Do not count C4a's pure mode objects as an
open 120-word gate; current code still hard-selects compact mode.

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

Randomizer waits until after the bakeoff so its dispatch refactor does not stale
receipts in the one-variable bank experiment.

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

## Pointers

- `ROADMAP.md`
- `docs/BUG_BIBLE_PROMOTION_QUEUE.md`
- `docs/PRODUCTION_SPRINT_LESSONS.md`
- `docs/PROD_BUG_LOG.md`
- `docs/2026-07-10-fable2-720-bakeoff-runway.md`
- `docs/2026-07-11-720-bakeoff-kickoff.md`
- `docs/2026-07-12-codex56sol-llm-telemetry-plan.md`
- `docs/2026-07-12-randomizer-rolls-r2-coding-plan.md`
- `workflows/otr_canonical.json`
