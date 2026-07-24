# OTR Go-Forward Plan

**Updated:** 2026-07-24 -- **INDEPENDENT SOURCE BANKS v1 IS DONE. All seven
waves LANDED @ `30358ad1`; the CODER E slot is FREE.** A client authors a
bundle under `user_packs/source_banks/<id>/`, runs
`scripts/otr_check.py bank <path> --activate`, restarts ComfyUI, and their bank
is a selectable peer of the shipped six: discovered and integrity-checked,
admitted in the ONE routing authority, free to own `fetch_source` /
`interpret_source` via the reserved `"self"` entry point, reaching the network
only through the bounded `nodes/_otr_feed_fetch.py` seam, and handed a COMPLETE
ledger by the shared tail's cleanup pass (`nodes/_otr_ledger_cleanup.py`).
Wave 7 was ASSESSED, not built: the `source_bank` COMBO already reads its
choices live from the registry and a bank's own `default_story_model` picks its
pack, so no node, widget, link or canonical-JSON change was ever needed --
canonical is still byte-identical at `A66A416B` after all seven waves. Suite
6403 / Bible 17. `docs/EXTENDING_OTR.md` is the client contract of record.
**OPERATOR RESCOPE 2026-07-24 (supersedes the older queue everywhere in this
file):** the 45-word scene matrix, the 54-case visual-style sweep and the
WHOLE quick-wins block are CUT -- the operator wants coding, not matrices, and
will triage bugs as a batch later. The order is now **WAN 8-GB contract ->
LEAN-MEAN FRONT -> Randomizer A -> `dynamic_story` -> LEAN-MEAN TAIL -> SFX ->
re-observe the parked story bugs.** ENGINE_MATRIX survives the cut as a W6
sub-step, not a standalone chunk. Two story-shaped defects are PARKED, not
closed (see OPEN BUGS).

This file contains only go-forward work, open bugs, and standing operator
contracts. Completed work is NEVER re-described here -- it moves to
`docs/HANDOFF_LOG.md` (history) and `docs/PROD_BUG_LOG.md` (bugs) the session
it ships. Doctrine lives in `docs/PRODUCTION_SPRINT_LESSONS.md`.

## CURRENT VERIFIED HANDOFF -- 2026-07-24

This block is the current source of truth for the overnight qualification.
Nothing in this file is an instruction to reset, stash, delete, or overwrite
user changes.

- Branch: `v2.0-alpha`; HEAD and origin are `30358ad1` (the seven extensibility
  waves are `66e214ec`, `cc69e683`, `84945bc4`, `c97a0e91` + `8c45172d`,
  `1504bb4c` + `3d97a130`, `30358ad1`; the six-bank no-prose-gate retirement
  chunk is `314dd481` below). Per-wave detail lives in
  `docs/HANDOFF_LOG.md`. The worktree is
  CLEAN of task-owned changes -- what remains is `tmp/` scratch (including
  another window's modified `tmp/_chain_720.ps1`, `tmp/_rearm_gate.ps1`,
  `tmp/_status_bake.ps1` -- PRESERVE) and untracked campaign receipts (plus
  `config/profiles/otr_sbcov_1..6.json`, intentionally untracked
  coverage-campaign scratch; nothing in-repo references them, and untracked
  `docs/_bakeoff_*.log.err` + `docs/otr-*.pdf` from an earlier window).
- LANDED @ `30358ad1` (2026-07-24; suite 6403 passed / 27 skipped / 1 xfailed;
  Bible 17; AST/JSON/BOM/zero-byte/UTF-8 clean; canonical byte-identical;
  pushed, HEAD == origin): wave 7, CLOSED BY ASSESSMENT. The `source_bank`
  COMBO on `OTR_LedgerScriptWriter` already reads its choices live from the
  routing registry, and waves 1-3 already made that registry own pack
  directories BY BANK (`_Registry.pack_dirs`), so an activated client bank is
  selectable with no new node, widget, link or canonical change. The chunk
  shipped the missing PROOF (three pins that an activated bundle reaches
  `INPUT_TYPES()["optional"]["source_bank"]`, that its widget value resolves to
  a pack inside its own bundle, and that admitting a bank leaves the canonical
  34-slot positional widget vector untouched) and fixed the one real defect:
  `guide_ref` had NO runtime consumer, so the `+ Add Your Own` row answered a
  click with a dead end while its own text still claimed "the simple_4 pass
  runner does not exist yet". `require_runnable_bank` now appends the row's
  `guide_ref`, and that text names the folder, the CLI, the restart and
  `docs/EXTENDING_OTR.md` (new section 6: what the client sees in ComfyUI).
- All seven extensibility waves and their per-wave findings are recorded in
  `docs/HANDOFF_LOG.md`; `docs/EXTENDING_OTR.md` is the landed client contract.
- Prior root fix at `f150213f`: `nodes/_otr_video_engines/render_driver.py` requires
  an authoritative scene-target manifest only for scene/mesh-consuming shots;
  visualizer-only `viz_mxc_cpu`, `viz_mxc_mandala`, and `viz_camera` lanes may
  execute without one. Regression coverage:
  `tests/test_ledger_cleanup_contracts.py`.
- Verification: full Windows OTR suite `6403 passed, 27 skipped, 1 xfailed`;
  Bug Bible `17 passed, 24 skipped, 3 xfailed`.
- Canonical workflow byte-identical at SHA-256
  `A66A416BFBCAD127356047043C8C07637BC50CACE2CD7D4E0436C7CD80B09CB4`.
- Live media proof: isolated `media_archive@120w` passed with `RESULT SUCCESS`,
  `obs_publish OK`, and non-zero episode/OBS assets. In the monitored run
  `tmp/six_bank_sweep_20260723_205002_331`, `original`, `public_domain`,
  `shakespeare`, and `scifi_news_pro` passed at 120 words. `scifi_news` failed
  closed on provider/context capacity and produced no publish artifact. The
  `scifi_news_pro@120w` pass does not clear its known `requested_output=2800`
  versus provider cap `512` blocker.
- WAN is already canonically qualified and remains closed. LTX remains
  untouched/unqualified until its explicit cases run.
- Overnight monitoring automation is active in the Codex app as
  `otr-overnight-qualification-monitor`. It must continue from the live logs,
  preserve canonical assets, and report terminal receipts or reproduced bugs.
- LANDED @ `314dd481` (2026-07-24; suite 6182 passed / 27 skipped / 1
  xfailed; Bible 17; AST/BOM/zero-byte/canonical-hash gates passed; pushed,
  HEAD == origin): word-fit ceilings /
  candidate ownership retired (length = non-gating telemetry on all six
  routes); provider-capacity whole-artifact output contracts with preserved
  list-subclass markers; `scifi_news` P1/P2/P3/P5 + `scifi_news_pro`
  pitch/treatment/news/script/casting migrated to provider-capacity output (no
  target-derived cap, no +25% missing-END branch); `scifi_news_pro` markup
  acceptance now structural delimiter/order/roster only; placeholder G13 fully
  retired; campaign receipt truth hardened (no PASS without canonical
  `RESULT SUCCESS`); the repair-first plan (explicit P0 slice identity, bounded
  tagged repair context, one direct alternate owner, original post-validator
  reuse, journaled owner/backend/rung/nonce/disposition).

### Immediate next actions

1. Preserve the completed run artifacts and record its 4/5 120-word receipt
   result; do not rerun the known provider-capacity failure as a workaround.
2. Open a coder window on the WAN 8-GB low-VRAM launch contract. It is the
   first item of the rescoped order and needs no GPU to write.
3. For any reproduced failure, fix the owning producer/receipt boundary,
   re-run focused tests, the full Windows suite, and Bug Bible, then commit and
   push the green code chunk to `v2.0-alpha` and verify `HEAD == origin`.
4. Never add fallback assets, truncation, silent resizing, arbitrary provider
   caps, or prose-quality rejection.

## MODEL & CREDIT BUDGET (operator, 2026-07-24 -- read this EVERY window)

Every window states, in its first reply, which rung of this ladder it is on
and why. Pick the cheapest tool that can win; escalate only when the cheaper
rung cannot decide.

**Reset state 2026-07-24: Claude weekly credits FRESH; Codex credits FRESH
(reset taken today). Both pools reset weekly -- front-load heavy coder windows
and the big Codex spends early in the credit week; late-week, drop to the $0
rungs instead of grinding a paid pool dry.**

| Rung | Model / tool | Cost | Use for | Never for |
|---:|---|---|---|---|
| 1 | Local Qwen on the 4060 (`10.55.0.2:1234`, LM Studio/ACPX): `qwen3-coder-30b-a3b-instruct` now; `Qwen2.5-Coder-14B Q4_K_M` as the fast tier once installed | $0 | Read-only FIRST-PASS triage of failures, logs, diffs before any credit spend | Final diagnosis, patches, tests, live qualification (Codex/Claude own those); NEVER loaded on the 5080 (ComfyUI renders only) |
| 2 | agy / Antigravity, `KIBITZ_AGY_MODEL="Gemini 3.6 Flash (High)"` (operator 2026-07-24: 3.6 > 3.5; DISPLAY name exactly -- a wrong id silently kills agy and the arc runs codex-only; check antigravity.log per round) | $0 | Default grounded reviewer for ALL mechanical review; second panelist on every kibitz | -- |
| 3 | Codex CLI `gpt-5.6-sol` (high) | weekly credits | The second opinion of record: two-strikes law (mandatory 3rd-attempt panel), sec-16 + r5 extensibility confirm, pre-execution grounding of big blocks, live-failure kibitz, HANDOFF_CODEX grind delegation | Mechanical review agy can do alone. Verify `codex_model_selected.txt` every arc (stale skill cache once drifted to gpt-5.5 mid-arc unnoticed) |
| 4 | Claude (Cowork, this) | weekly credits | The actual work: planner + coder windows, anchor/judge on every panel, live-run drive | Babysitting renders (the Codex-app overnight monitor owns that); single-small-item windows (batch per the Window packing rules) |
| 5 | Cloud roundtable (OpenRouter) | real $ | Genuine R1 ideas passes only; <$20 autonomy rule applies | Mechanical/grounding review (that is rungs 2-3) |
| 6 | Fable | scarce | Single final gate on a lean-mean epoch commit only (section-9 reality exception) | Anything else |

Production (in-pipeline, all $0/local, offline-first): writers = Mistral-Nemo
(ctx cap 16384) + `gemma-4-12b` (saved runtime-qualified local default);
stills/video-init = `z_image_turbo` (Qwen-Image engine is REMOVED -- keep
Qwen3/Qwen2.5 LLM support and Z-Image's `CLIPLoader(type="qwen_image")`
encoder, unrelated). Cloud writers (Sonnet-4.5 etc.) stay opt-in bake-off
arms, never the default.

Per-window model mapping:

- RENDER / qualification windows: local production models + the Codex-app
  monitor; Claude only to launch and wrap.
- CODER windows (quick-wins, lean-mean): Claude codes; rung-1 Qwen triages
  every failure first; Codex only via the two-strikes law.
- PLANNER window: Claude; the sec-16 + r5 kibitz (codex + agy) is THIS WEEK's
  scheduled Codex spend while both pools are fresh -- it is the operator
  bottleneck on the critical path.
- CODER E extensibility (21-31 d): spans multiple credit weeks -- plan wave
  boundaries at the weekly resets; mid-build Codex only via two-strikes.

## THE LAW (operator, 2026-07-22 -- supersedes anything that disagrees)

> **AN AUDIT MAY IMPROVE A STORY. IT MAY NEVER FAIL ONE FOR LENGTH, LANGUAGE,
> STYLE, VISUAL VOCABULARY, OR QUALITY.**

The sole terminal spoken-prose policy is the shared whole-word safety
authority: profanity, explicit guns/knives/weapons, and explicit
sexual/nudity content. Smoking and benign substrings such as `begun` pass.
Structural JSON/schema/IDs/roster/source-proof/rights/graph/markup/nonempty/
provider-integrity failures remain fail-closed because they protect a usable
ledger rather than judge prose. Across all six banks, requested word length,
actual word count, drift, one-breath estimates, visual/world vocabulary,
noun/POS heuristics, casing/title/honorific style, craft, and quality are
guidance or telemetry only -- they may never reject, reroll, retire, replace,
or block an episode. Same-story LLM cleanup is allowed.

## CURRENT STEP -- WAN 8-GB contract, then the lean-mean front

Operator rescope 2026-07-24. The coder slot is FREE and the order is fixed:

1. **WAN 8-GB low-VRAM launch contract** -- make the 8-GB profile carry its
   actual 832x480 / 17-frame launch contract instead of falling back to the
   177-frame default. Pure coding; no GPU needed to write it.
2. **LEAN-MEAN FRONT** (`W0 -> W1 -> W2 -> W3 -> W4a -> W4b -> W7 -> W6 ->
   W5+SW4 -> C1-C5`), with `docs/ENGINE_MATRIX.md` folded in as a W6 sub-step.
3. **Randomizer A -> `dynamic_story`.**
4. **LEAN-MEAN TAIL** (`SW1/SW2/SW3 -> C6 -> C7 -> W8`).
5. **SFX** (still behind the Timeline Cue Ledger C0/C1 gate).
6. **Re-observe the parked story bugs** -- after SFX, see whether they still
   occur at that HEAD (see OPEN BUGS).

Standing constraints, unchanged by the rescope: keep the RTX 5080 free for
ComfyUI; the 4060 Qwen endpoint is a read-only QA reviewer, not a production
ComfyUI slot; six-bank requalification (canonical `RESULT SUCCESS`,
`obs_publish OK`, exact episode/OBS assets, and the archival final's parent
equal to the ledger-owned episode root -- PBUG-20260720-05 acceptance) is
still owed whenever a render window next opens, and was NOT cut.

## OPERATOR CAMPAIGN QUEUE -- 2026-07-23 (PAUSED)

The overnight media qualification was aborted after the WAN lane and the LTX
visual-style sweep stalled at case 6/54. No new GPU run is authorized while
confirmed bugs are being closed. Failure inventory / staging record:
`docs/2026-07-23-video-failure-inventory.md`.

Bug-first order before resuming:

1. Requalify receipt truth against the captured six-bank stdout and confirm
   the old false PASS is now a terminal FAIL (fix LANDED @ `314dd481`;
   needs live confirmation only).
2. Make the image phase own every required scene-still, mesh-fodder, and
   opening-still target, with a complete target/path receipt before video
   dispatch; no text-only or dark-floor degradation for a missing required
   still. (`f150213f` fixed the no-still visualizer spine handoff; the
   scene/mesh-consuming ownership contract is the remaining piece.)
3. Make the WAN 8-GB profile carry its actual 832x480/17-frame low-VRAM
   launch contract instead of falling back to the 177-frame default.
4. Then provider-capacity and SciFi News markup-repair residuals.

Remaining media qualification (CUT DOWN by the operator rescope 2026-07-24 --
the 45-word model-coverage matrix and the 54-case visual-style sweep are
DELETED, not deferred; reviving either is a new operator decision):

1. Six 120-word canonical runs in bank order `media_archive`, `original`,
   `public_domain`, `shakespeare`, `scifi_news`, `scifi_news_pro`:
   `google/gemma-4-12b-it` both writer slots, `viz_mxc_cpu` /
   `viz_mxc_mandala` / `viz_camera` video slots, `z_image_turbo` all three
   image slots. (4/5 of the 120w receipts are already banked from
   `tmp/six_bank_sweep_20260723_205002_331`; `scifi_news` is the open FAIL.)
   This is the ONLY surviving matrix.

The coordinator keeps one canonical API prompt active at a time, reloads
`workflows/otr_canonical.json` for every case, and records each prompt and
receipt under `tmp/`.

## OPEN BUGS / DEFECTS (live, not yet closed)

MECHANICAL defects survive story-engine churn; STORY-QUALITY judgments do not.
That split is why the two eyeball-era entries below are PARKED rather than
listed as live.

- **`scifi_news` P0 convergence defect** -- both 120w and 320w legs fail in P0
  after two attempts on non-literal fact source spans; provider/model
  convergence, extends BUG-11.35. NOT a word/length gate. Blocks the last 120w
  receipt and the `scifi_news` live reverify (PBUGs 20260712-22/23/24/25, fixed
  in tree, reverify still owed).
- **`scifi_news_pro` provider capacity** -- `requested_output=2800` vs
  provider cap `512`; the whole-artifact retry contracts LANDED @ `314dd481`
  are the base; the residual fix is now unblocked. Related independent items: the P9 8K
  structured-capacity follow-up + the GGUF structured-enforcement NEWBUG. Do
  not raise the minimum word target as a capacity workaround.
- **WAN 8-GB low-VRAM launch contract** -- FIRST item of the rescoped order.
- **Image-phase still ownership** -- bug-first item 2 above.

**PARKED -- unverified at HEAD, re-observe AFTER SFX (operator 2026-07-24).**
Both were eyeball observations against a story engine that has since had its
LLM vetoes ripped, THE LAW imposed (2026-07-22), six banks renamed onto new
packs, word-fit ceilings retired, the repair-first plan landed, and a ledger
cleanup pass added. Neither has a reproduction at current HEAD, and under the
standing rule a finding with no reproduction is not a row. Do NOT schedule
coder time against either. They are settled by the operator eyeballing a real
render leg after SFX: still there -> re-admit as a FRESH dated row with that
leg as evidence; gone -> the LAW-era work already fixed it, tombstone it.

- **Announcer framing defect** (`docs/2026-07-11-announcer-framing-defect.md`)
  -- PARKED. Episodes START a story instead of admitting you into one; the
  announcer takes debate turns instead of framing. Operator eyeball
  2026-07-11. If it survives re-observation the fix is still seam + score
  contract + fail-closed validator, never Python authorship.
- **Name-splice defect #2** -- PARKED. v4-campaign Phase 0 record in
  HANDOFF_LOG; its timebox predates THE LAW.

- **PBUG-20260710-07** -- root fix shipped; stays ROOT-OPEN in the log until
  ratified at the next operator fan-out (green codex leg `c1f3891f` is the
  retire candidate).
- **Phase-2 de-naming** (module filenames, `meta[]` ledger keys, wire-schema
  `.v4` literals) -- DEFERRED, operator-flagged, from the keep-6 rename.

## Coder queue (re-grounded 2026-07-24)

One coder window at a time; every chunk = focused tests + full suite + Bug
Bible + commit AND push + `HEAD == origin/v2.0-alpha`.

```text
WAN 8-GB low-VRAM launch contract
  -> LEAN-MEAN FRONT (W0->W1->W2->W3->W4a->W4b->W7->W6->W5+SW4->C1-C5)
       (ENGINE_MATRIX.md is a W6 SUB-STEP now, not a standalone chunk)
  -> Randomizer A -> dynamic_story  (UNBLOCKED: extensibility landed)
  -> LEAN-MEAN TAIL (SW1/SW2/SW3 -> C6 -> C7 -> W8)
  -> SFX campaign (after the Timeline Cue Ledger C0/C1 gate)
  -> re-observe the PARKED story bugs; batch-triage whatever is left
```

CUT by the operator 2026-07-24 and NOT to be re-derived by a later window: the
45-word scene matrix, the 54-case visual-style sweep, and the entire
quick-wins block. Image-phase still ownership and the six-bank requalification
were not cut -- they stay in OPEN BUGS / the campaign queue and get picked up
whenever a render window opens.

### Quick-wins block -- CUT 2026-07-24 (operator)

The whole block is gone. The operator's call, verbatim in intent: "we will
triage more bugs later" -- the block was a schedule, and ripping a schedule
does not rip the underlying defects. Everything in it that was a real bug
still lives in OPEN BUGS above; everything in it that was a nice-to-have is
simply not being built. Do NOT re-derive this table from git history.

ONE item survived the cut, folded into LEAN-MEAN W6 as a sub-step rather than
kept as a standalone chunk:

- **`docs/ENGINE_MATRIX.md`** -- emit from the three live CAPABILITIES
  registries per the existing generator pattern (`build_variants.py`
  ~:276-338): write during `--all` / explicit emit; `--check` regenerates in
  memory and FAILS on drift without writing. Columns + stable ordering; link
  from README. The lean-mean doc (`:301-304`) only needs W6's README policy
  line to link it, so this is an ordering preference the operator set on
  2026-07-10 -- NOT a hard technical dependency. W6 executes without it; the
  README link is what suffers. Estimate 0.5-1 d.

Also recorded so a later window does not re-open them: quick-win 6
(`scifi_news_pro` C5 consumers) was already CLOSED IN CODE under
PBUG-20260720-04. The `scifi_news` live reverify (PBUGs 20260712-22/23/24/25)
is not lost either -- it moved into the `scifi_news` P0 convergence row in
OPEN BUGS, which is what actually blocks it.

### Big blocks (in ROADMAP-ratified order)

1. **LEAN-MEAN FRONT** (`W0 -> W1 -> W2 -> W3 -> W4a -> W4b -> W7 -> W6 ->
   W5+SW4 -> C1-C5`) -- `docs/2026-07-10-lean-mean-rip-final.md`, D-1..D-6
   RATIFIED. Execute after its 2026-07-15 drift-check header is satisfied
   (SW-3 news_ingest re-survey, W6 keep-list adds, W7 tombstone re-triage,
   R-7 re-grep; SW-1 writer re-survey waits for the TAIL). ENGINE_MATRIX is
   now a W6 SUB-STEP of this block, not a separate precondition to satisfy
   first. Dedicated window; multi-day. THIS IS THE SECOND ITEM IN THE
   RESCOPED ORDER, after the WAN 8-GB contract.
2. **Randomizer Rolls Design A** --
   `docs/2026-07-12-randomizer-rolls-r2-coding-plan.md`. NO LONGER GATED --
   extensibility landed, and its `_otr_lane_specs` authority was ABSORBED by
   that build, so this shrinks to `_otr_bank_roll` + eligibility. Re-ground per
   its 2026-07-15 header, and re-derive the bank list against the LIVE registry
   rather than a six-row literal: `list_bank_ids()` can now return a client
   bank, and eligibility must treat one as an ordinary peer. 1-2 d + 1 GPU day.
3. **`dynamic_story` visual direction** -- rev-5 FINAL, do not rerun panels;
   roster-agnostic; re-derive IDs at build. After the randomizer.
   5-9 coder-days + 2-4 GPU days.
4. **LEAN-MEAN TAIL** (`SW1/SW2/SW3 -> C6 -> C7 -> W8`) -- the writer/widget
   structural split, REQUIRED by ROADMAP to come after blocks 2-3. SW-1 full
   seam re-survey happens here, against the then-current writer.

Open judgment question (render-window, not coder-slot): the LOCAL
mistral/gemma writer matrix -- the Sonnet arm of the creative-writer question
is answered (record: `docs/2026-07-17-model-bakeoff-scoreboard.md`); the local
roster comparison never ran.

## Window packing (credit discipline -- one line starts any window)

Starting any window costs the same boot context, so BATCH chunks per window
and never open one for a single small item. Every window starts by pasting
its one-line kickoff -- the `otr-handoff` skill reads this file + git and
states the current step. No manual context handoff, ever. This planner window
keeps GO_FORWARD + HANDOFF_LOG current; coder windows never write plans
(window-roles rule).

| Window | Scope | Model rung (see MODEL & CREDIT BUDGET) | Gate | Size |
|---|---|---|---|---|
| RENDER | finish the six-bank 120w wrap ONLY (the 45w matrix and 54-case sweep are CUT); fillers: cpu-tier smoke + nv50 re-soak | local production + Codex-app monitor | opens whenever the operator wants a live leg | GPU days |
| CODER A "seams" | **the WAN 8-GB low-VRAM launch contract** -- first item of the rescoped order, no GPU needed to write. Then image-phase still ownership if the slot is still open. | Claude codes, Qwen triages, codex on 3rd strike | UNGATED | ~1-2 d |
| ~~CODER B~~ | quick-wins harness window -- **DISSOLVED** by the 2026-07-24 rescope (its whole scope was quick-wins) | -- | -- | -- |
| ~~CODER C~~ | quick-wins foundations window -- **DISSOLVED** by the 2026-07-24 rescope; ENGINE_MATRIX moved into CODER D's W6 | -- | -- | -- |
| CODER D "lean-mean front" | drift-check re-verifies, then W0 .. C1-C5, with ENGINE_MATRIX as a W6 sub-step | same | after A | multi-day |
| PLANNER | extensibility hardening + `docs/EXTENDING_OTR.md` DONE 2026-07-24; NEXT = Bug Bible operator fan-out + the `check_compatibility` fork; plan upkeep | rungs 2-4 | parallel with D | docs |
| ~~CODER E~~ | independent client-authored source banks v1 -- **ALL SEVEN WAVES DONE @ `30358ad1`**; slot RETIRED, do not reopen (deferred power-user tiers are a NEW block, not this one) | -- | -- | -- |
| CODER F | Randomizer A -> `dynamic_story` | Claude + Qwen triage | UNGATED (E is done; re-pin at HEAD first) | ~6-11 d |
| CODER G "lean-mean tail" | SW1-SW3, C6, C7, W8 | Claude; Fable single final epoch gate | after F | multi-day |

Kickoff lines (paste as the FIRST message of the new window; swap the letter):

> resume the OTR build -- you are CODER WINDOW A per GO_FORWARD "Window
> packing"; execute your scope in order, one green pushed chunk at a time,
> and state your MODEL & CREDIT BUDGET rung first.

## Parallel lane -- no coder slot required

- **Bug Bible operator fan-out** -- 9+ closed candidates + the
  duplicate-legacy_id cleanup waiting on one fan-out session.
- **Render-window fillers:** cpu-tier smoke (needs the google image lane or
  stills) + nv50 re-soak -- the two open portability remainders; release QA
  validation time, not coding.
- **SFX R4.1 re-ground** (0.5-1 docs day): re-ground the local generated-SFX
  R4 candidate into a tracked current-HEAD R4.1 plan. Sequencing + scope
  contract live in `ROADMAP.md` (Timeline Cue Ledger C0/C1 gate first; no
  second SFX queue, no library fallback).
- **Operator-promotable option:** SFX C0 (per-line WAV stems + transcript
  drift report) is independently shippable per ROADMAP but stays parked
  unless explicitly promoted.

## Bug Bible promotion field -- pending actions only

| Record | Pending action |
|---|---|
| `PBUG-20260712-22/23/24/25` | Live reverify -- blocked by the `scifi_news` P0 convergence defect, then fan-out |
| `PBUG-20260712-18/19/26` + `PBUG-20260713-15..18` + `-20` | Awaiting the next operator Bible fan-out (overlap check + approval) |
| `PBUG-20260713-19` | Live requalification pending (promoted BUG-05.11) |
| duplicate-id cleanup | Same fan-out: BUG-11.54 legacy_id -> `PBUG-20260713-21`; verify the acronym-union rule's legacy_id (both Bible rows cite `-10`; see the log's renumber note) |
| historical `PBUG-20260711-18` | Keep as a standing context/cap engineering risk (its quick-win-9 home was cut 2026-07-24); never eligible from static evidence |
| `PBUG-20260710-07` | Ratify retirement at the next fan-out (green codex leg `c1f3891f`) |

The active production-fix owner updates `docs/PROD_BUG_LOG.md`; the approval
queue is `docs/BUG_BIBLE_PROMOTION_QUEUE.md`; no plan review or invented
fixture creates a row.

## Validation and handoff law

- Current whole-tree receipt (2026-07-24): full Windows suite `6403 passed /
  27 skipped / 1 xfailed`; Bug Bible `17 passed / 24 skipped / 3 xfailed`.
  Detail in HANDOFF_LOG.
- Every code chunk: focused tests, full Windows suite, Bug Bible,
  AST/JSON/BOM/zero-byte checks, commit, push, verify
  `HEAD == origin/v2.0-alpha`.
- Every node/widget/link/schema change edits `workflows/otr_canonical.json`
  in the same commit and runs `OTR_WorkflowValidator`, JSON round-trip,
  strict link/input, live widget-vector, and generated-variant audits.
- Reset selectively before every headless run; never blanket-kill Python.
  Every run loads the canonical workflow and writes directly to canonical
  episode/OBS paths. Asset existence, not resident VRAM, proves completion.
- One coder edits code or `workflows/otr_canonical.json` at a time; read-only
  audits and documentation may run in parallel. HANDOFF_LOG + this file are
  the only tracking surfaces (the otr-build-tracker artifact is RETIRED).

## Open risks

- Extensibility v1 is DONE, so it no longer constrains randomizer /
  dynamic_story sequencing. Deferred power-user tiers (client own-runner +
  staging, dependency manifest, standalone story_rules) are explicitly OUT of
  v1 and are a NEW block if the operator ever wants them -- not a reopening of
  CODER E. NO CLIENT BANK HAS EVER RUN LIVE: every wave is proven by the suite
  and by contract tests, and the first real client bundle is still an unproven
  path end to end (fetch -> interpret -> writer -> cleanup -> tail -> publish).
  Treat the first live client-bank leg as a qualification, not a formality.
- CLIENT-AUTHORED PYTHON executes in-process (wave 3). The posture that must
  hold in every future change: `--activate` is the consent act; the seam fails
  LOUD (`UserBankExecutionError`) and never substitutes; client code never
  touches the canonical ledger; owner IDENTITY is verified so a bank can only
  run its OWN bundle; the shipped fetcher/interpreter registries are never
  widened to admit a client id. Do not relax any of these for convenience.
- The client-facing surface is now LIVE TEXT, not just docs: the
  `custom_source_bank` row's `guide_ref` is raised to the operator by
  `require_runnable_bank`, and the `source_bank` tooltip repeats it. Any future
  change to the activation path (folder name, CLI verb, restart behaviour) must
  update `nodes/story_packs/banks.json`, that tooltip and
  `docs/EXTENDING_OTR.md` together, or the product will confidently instruct
  clients to do the wrong thing.
- **`check_compatibility` is RESERVED, not wired (wave-4 decision, kibitz
  r3 codex `gpt-5.6-sol` high + r4 agy Gemini 3.6 Flash High, Claude judge).**
  No request type, no decision type, no runtime consumer exists, so activation
  does not inspect it -- not even for callability -- and `EXTENDING_OTR.md`
  now calls it a reserved name instead of "NOT YET WIRED". `COMPAT_ENTRY_ATTR`
  is left INERT in `BUNDLE_ENTRY_ATTRS` with a comment saying so. **Operator /
  planner decision flagged, NOW WITH A 2-of-2 RECOMMENDATION TO RIP
  (2026-07-24, operator-directed consult; codex `gpt-5.6-sol` high and Fable,
  independently, no shared context; Claude grounded both against the tree):**
  the argument that decided it is that Option A's stated benefit is FALSE --
  `BUNDLE_ENTRY_ATTRS` constrains what OTR-side code may request from
  `bundle_entry_point()`, it reserves nothing against clients, and activation
  provably ignores whatever a client puts under that name
  (`tests/test_otr_check_cli.py:335` asserts a bundle whose
  `check_compatibility` is a plain integer activates). The only artifact that
  reserves the name is the `EXTENDING_OTR.md` paragraph, which exists either
  way; the constant's sole executable effect is to legalize a call nobody
  makes. Verified blast radius if ripped: ~5 code sites, 2 test files, 3 docs;
  no workflow JSON, no routing, no source-payload consumer. Case AGAINST,
  stated by both: churn on landed green code for zero behaviour change, the
  constant is loudly commented inert and a test documents the inertness, and
  the plan of record already names the future consumer (randomizer
  eligibility), so it may be re-added within a wave or two. STILL NOT A CODER
  CHUNK -- the rip touches landed wave-3/4 code and the plan of record's
  "fetch_source + interpret_source + check_compatibility" line. Either ratify
  the inert constant or schedule the rip as a planner chunk. (The one piece
  already fixed @ `8c45172d`, correct under either answer: the `missing_module`
  quarantine message demanded a `check_compatibility` the code has never
  required. Both panelists found it independently. Proposed doctrine line: a
  name published to clients before its consumer exists lives in the
  client-facing DOC as "reserved, no contract, ignored if defined" and nowhere
  in executable code, because code that names an interface is read as
  enforcing it.)
- **The ledger-cleanup pass now runs on EVERY bank, not just client banks**
  (wave 6, `3d97a130`). It is a no-op on a complete ledger and costs no LLM
  call there, but two shipped-lane behaviours did change and are worth watching
  on the next live legs: (a) unsafe spoken language on a
  `content_owned_readonly` bank is now REPAIRED at the writer tail instead of
  reaching G9 untouched, so a leg that used to die at freeze may now ship a
  sanitized line; (b) a blank `meta.episode_title` is now filled at the tail
  instead of exploding later in `otr_credits_roll`. Both are the intended
  direction under THE LAW; neither has a live receipt yet.
- Lean-mean front/tail drift: the tail's SW-1 re-survey is mandatory against
  the then-current writer. Never interleave the two campaigns in one window.
- No code lands mid-sweep of an active qualification campaign (uniform-code
  confound -- the 420-rung lesson).
- The active campaigns may surface new lane defects; the campaign window owns
  admitting PBUGs (new-bug problem-statement rule applies).
- `dynamic_story` touches the writer, the visual-style authority and the
  canonical workflow; it re-derives the live JSON at build. It is now the only
  claimant on those surfaces (extensibility has released them).
- Generated-SFX R4 stays local/ignored evidence until the tracked R4.1 refit
  lands; it is not an executable queue.

## Tombstones (do not re-derive; records in HANDOFF_LOG + PROD_BUG_LOG)

Keep-6 bank rename (six de-versioned banks; default `scifi_news`,
local/offline-first) -- LLM veto rip + THE LAW -- roster trim + Sonnet-bake-off
rip (science_news family, `_v2` lanes, scifi_sonnet retired) -- v4 improvement
campaign banks #2-#5 PARKED (superseded by the rename + THE LAW; revive only
by operator decision; plan of record `docs/2026-07-17-v4-campaign/final.md`) --
codex56sol attempt telemetry + PBUG-20260712-17 root fix -- fresh two-matrix
bakeoff -- Qwen-Image still engine (removed 2026-07-23) -- word-fit ceilings /
candidate campaigns -- style-dropdown four-surfaces -- otr-build-tracker
artifact -- `tencent/hy3:free` panel seat (expired 2026-07-21) --
**the 45-word scene matrix, the 54-case visual-style sweep, and the entire
quick-wins block (CUT by the operator 2026-07-24: coding over matrices, bugs
triaged as a batch later; ENGINE_MATRIX survived as a Lean-Mean W6 sub-step,
CODER B and CODER C dissolved with the block)** --
**independent client-authored source banks v1 (all seven waves, CODER E,
2026-07-24 @ `30358ad1`; contract `docs/EXTENDING_OTR.md`; w7 closed by
assessment -- no widget was needed and none was invented)** -- the retired
Path-A/B user-source-lanes architecture.

## Pointers

- `ROADMAP.md` (dependency edges; lean-mean pin self-declares stale cites)
- `docs/PRODUCTION_SPRINT_LESSONS.md` (incl. lesson 24 lost-anchor; 25 bank-teardown)
- `docs/SOURCE_BANK_PREFLIGHT.md` -- add-a-bank gate + the Teardown protocol
- `docs/PROD_BUG_LOG.md` / `docs/BUG_BIBLE_PROMOTION_QUEUE.md`
- `docs/HANDOFF_LOG.md` (all completed-work history, newest at top)
- `docs/2026-07-23-video-failure-inventory.md` (campaign staging record)
- `docs/2026-07-15-720-bakeoff-verdict.md` (KEEP/IMPROVE + open items)
- `docs/2026-07-17-model-bakeoff-scoreboard.md` (writer-model verdict)
- `docs/EXTENDING_OTR.md` (LANDED client contract: add your own source bank)
- `docs/2026-07-24-independent-source-banks-v1-plan.md` (extensibility plan -- DELIVERED)
- `docs/2026-07-12-user-source-lanes-architecture.md` (SUPERSEDED -- Path-A/B decision log)
- `docs/2026-07-10-lean-mean-rip-final.md` (drift-check header 2026-07-15)
- `docs/2026-07-12-randomizer-rolls-r2-coding-plan.md`
- `docs/2026-07-12-dynamic-story-visual-scope.md`
- `docs/2026-07-10-llm-first-story-edit-pass.md` (X1-X4 live remainder)
- `docs/2026-07-11-announcer-framing-defect.md` (OPEN)
- `docs/2026-07-11-timeline-cue-ledger.md`
- `workflows/otr_canonical.json`
