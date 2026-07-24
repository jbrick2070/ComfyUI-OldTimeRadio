# OTR Go-Forward Plan

**Updated:** 2026-07-24 -- **CODER E OPEN: independent source banks waves 1-4
LANDED @ `84945bc4`.** Client bundles are discovered, integrity-checked,
admitted ALONGSIDE the shipped six in the one routing authority, may OWN their
`fetch_source` / `interpret_source` via the reserved `"self"` entry point, and
are now VALIDATED AND ACTIVATED by `scripts/otr_check.py bank <path>
--activate` -- which runs the same admission contracts boot runs, imports the
bundle in a bounded child process, binds the writer's keyword sets, and
publishes the snapshot before the receipt. Suite 6294 / Bible 17; canonical
byte-identical. NEXT for CODER E = wave 5 (the bounded `_otr_feed_fetch` seam,
BOTH hops). SIX-BANK REQUALIFICATION + 45-WORD SCENE MATRIX still NEXT for the
render track; bug-first items still open for CODER A.

This file contains only go-forward work, open bugs, and standing operator
contracts. Completed work is NEVER re-described here -- it moves to
`docs/HANDOFF_LOG.md` (history) and `docs/PROD_BUG_LOG.md` (bugs) the session
it ships. Doctrine lives in `docs/PRODUCTION_SPRINT_LESSONS.md`.

## CURRENT VERIFIED HANDOFF -- 2026-07-24

This block is the current source of truth for the overnight qualification.
Nothing in this file is an instruction to reset, stash, delete, or overwrite
user changes.

- Branch: `v2.0-alpha`; HEAD and origin are `84945bc4` (CODER E wave 4; waves
  1-3 are `66e214ec` + `cc69e683`; the
  six-bank no-prose-gate retirement chunk is `314dd481` below). The worktree is
  CLEAN of task-owned changes -- what remains is `tmp/` scratch (including
  another window's modified `tmp/_chain_720.ps1`, `tmp/_rearm_gate.ps1`,
  `tmp/_status_bake.ps1` -- PRESERVE) and untracked campaign receipts (plus
  `config/profiles/otr_sbcov_1..6.json`, intentionally untracked
  coverage-campaign scratch; nothing in-repo references them, and untracked
  `docs/_bakeoff_*.log.err` + `docs/otr-*.pdf` from an earlier window).
- LANDED @ `84945bc4` (2026-07-24; suite 6294 passed / 27 skipped / 1 xfailed;
  Bible 17; canonical byte-identical; pushed, HEAD == origin): wave 4, the
  `otr_check bank <path> [--activate]` CLI (`scripts/otr_check.py` +
  `otr_check.bat`). It owns no format -- `_otr_user_banks` gained
  `preflight_bundle` / `write_activation` / `activation_status` (and
  `_validate_bundle` split so the authoring half runs on a bundle with no
  receipt, boot's check ORDER unchanged), and `_otr_story_routing` gained
  `shipped_bank_seed()` + `validate_client_bundle_contract()`. The last one is
  the wave's real find: `_admit_user_banks` runs `_sweep_pack_dir` +
  `_crossref_bank` AFTER `discover`, so validating with the row parser alone
  would have handed receipts to banks that quarantine at boot.
  `check_compatibility` stays UNWIRED by decision (see the note under Open
  risks).
- LANDED @ `66e214ec` + `cc69e683` (2026-07-24; suite 6264 passed / 27 skipped
  / 1 xfailed; Bible 17; AST/BOM/zero-byte/canonical-hash gates passed; pushed,
  HEAD == origin): independent source banks waves 1-3. `nodes/_otr_user_banks.py`
  owns client bundle integrity (content-addressed digest, activation receipt +
  snapshot, path/symlink containment, protected/duplicate id refusal) and NEVER
  raises for a bundle problem -- plus the wave-3 EXECUTION seam, which is loud
  by design (`UserBankExecutionError`) because discovery already quarantined the
  broken bundles. `_otr_story_routing.py` admits client rows alongside shipped
  via the SAME `_parse_bank` and cross-refs, routes packs by OWNER
  (`pack_dirs`), and unlocks the reserved `"self"` entry point on an explicit
  `is_client` flag. `_otr_source_payload.resolve_fetcher/resolve_interpreter`
  take an owner bundle and verify owner IDENTITY; client results still cross
  `normalize_fetch_result` / `validate_interpreter_result` unchanged.
  `docs/EXTENDING_OTR.md` carries the landed contract.
- Prior root fix at `f150213f`: `nodes/_otr_video_engines/render_driver.py` requires
  an authoritative scene-target manifest only for scene/mesh-consuming shots;
  visualizer-only `viz_mxc_cpu`, `viz_mxc_mandala`, and `viz_camera` lanes may
  execute without one. Regression coverage:
  `tests/test_ledger_cleanup_contracts.py`.
- Verification: full Windows OTR suite `6264 passed, 27 skipped, 1 xfailed`;
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
2. After a fresh selective reset, run the bounded 45-word scene-consuming
   qualification in `still_word`, `mesh_stage`, `ltx_video`, and `ltx_audio_in`
   order using `workflows/otr_canonical.json`; stop at the first shared failure.
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

## CURRENT STEP -- six-bank requalification + bug-first fixes

The retirement chunk is LANDED (`314dd481`); the coder slot is FREE. In
order:

1. Requalify the captured six-bank leg against the landed code: require
   canonical `RESULT SUCCESS`, `obs_publish OK`, exact episode/OBS assets,
   AND the archival final's parent equal to the ledger-owned episode root
   (PBUG-20260720-05 acceptance).
2. Close the remaining bug-first items below, one green pushed chunk each.
3. Keep the RTX 5080 free for ComfyUI; the 4060 Qwen endpoint is a read-only
   QA reviewer, not a production ComfyUI slot.
4. Keep GPU media qualification paused until the bug-first items are closed.

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
4. Then provider-capacity and SciFi News markup-repair residuals, followed by
   a fresh 45-word visual-style qualification.

Deferred media qualification order (unchanged; exact word counts and matrices
ARE the queue -- any new length/provider tier is a new operator decision):

1. Six 120-word canonical runs in bank order `media_archive`, `original`,
   `public_domain`, `shakespeare`, `scifi_news`, `scifi_news_pro`:
   `google/gemma-4-12b-it` both writer slots, `viz_mxc_cpu` /
   `viz_mxc_mandala` / `viz_camera` video slots, `z_image_turbo` all three
   image slots. (4/5 of the 120w receipts are already banked from
   `tmp/six_bank_sweep_20260723_205002_331`; `scifi_news` is the open FAIL.)
2. Local 45-word model coverage: one video model across all three video slots
   and one local image model across all three image slots per case, banks
   rotated; covers local viz/still/mesh/word engines, LTX, HuMo, Wan. Cloud
   video/image providers excluded (external/billable).
3. Fifty-four 45-word `ltx_audio_in` runs: every live visual style (`anime`,
   `archival_documentary`, `cartoon`, `paper_origami`, `recur_frac`,
   `sci_fi_radio`, `shakespeare_stage_realism`, `storybook_engraving`,
   `video_art`) across all six banks.

The coordinator keeps one canonical API prompt active at a time, reloads
`workflows/otr_canonical.json` for every case, and records each prompt and
receipt under `tmp/`.

## OPEN BUGS / DEFECTS (live, not yet closed)

- **`scifi_news` P0 convergence defect** -- both 120w and 320w legs fail in P0
  after two attempts on non-literal fact source spans; provider/model
  convergence, extends BUG-11.35. NOT a word/length gate. Blocks the last 120w
  receipt and quick-win 1.
- **`scifi_news_pro` provider capacity** -- `requested_output=2800` vs
  provider cap `512`; the whole-artifact retry contracts LANDED @ `314dd481`
  are the base; the residual fix is now unblocked. Related independent items: the P9 8K
  structured-capacity follow-up + the GGUF structured-enforcement NEWBUG. Do
  not raise the minimum word target as a capacity workaround.
- **WAN 8-GB low-VRAM launch contract** -- bug-first item 3 above.
- **Image-phase still ownership** -- bug-first item 2 above.
- **Announcer framing defect** (`docs/2026-07-11-announcer-framing-defect.md`,
  OPEN) -- quick-win 3.
- **Name-splice defect #2** -- OPEN per its timebox (v4-campaign Phase 0
  record in HANDOFF_LOG).
- **PBUG-20260710-07** -- root fix shipped; stays ROOT-OPEN in the log until
  ratified at the next operator fan-out (green codex leg `c1f3891f` is the
  retire candidate).
- **Phase-2 de-naming** (module filenames, `meta[]` ledger keys, wire-schema
  `.v4` literals) -- DEFERRED, operator-flagged, from the keep-6 rename.

## Coder queue (re-grounded 2026-07-24)

One coder window at a time; every chunk = focused tests + full suite + Bug
Bible + commit AND push + `HEAD == origin/v2.0-alpha`.

```text
six-bank requalification (chunk landed @ 314dd481)
  -> bug-first items (receipt truth live, still ownership, WAN contract)
  -> 45w scene matrix + 54-case visual-style qualification
  -> quick-wins block (coder windows A/B/C)
  -> LEAN-MEAN FRONT (W0->W1->W2->W3->W4a->W4b->W7->W6->W5+SW4->C1-C5)
  -> independent client-authored source banks (lean v1; w1-w2 @ 66e214ec,
       w3 @ cc69e683; CODER E owns w4..w7 -- runs PARALLEL to the lean-mean front)
  -> Randomizer A -> dynamic_story
  -> LEAN-MEAN TAIL (SW1/SW2/SW3 -> C6 -> C7 -> W8)
  -> ROADMAP (SFX campaign after Timeline Cue Ledger gate)
```

### Quick-wins block (~5-11 coder-days; small chunks, any order inside)

| # | Chunk | Gate | Est |
|---:|---|---|---:|
| 1 | `scifi_news` reverify tail | PBUGs 20260712-22/23/24/25 FIXED IN TREE, LIVE REVERIFY PENDING. The 120w sweep leg is the natural reverify vehicle but is itself blocked by the P0 convergence defect above -- fix that first, then one green base `scifi_news` 120w leg closes both; mark the log and release the slot formally. | 0.25-0.5 d |
| 2 | Cliche-span excision (X1-X4) | `docs/2026-07-10-llm-first-story-edit-pass.md` Wave 3: `repair_cliche_span` (`_otr_line_composer.py` ~:2632/:2676) + `cliche_replacements` in the story_rules JSONs still rewrite SPOKEN lines -- a standing LLM-first violation. Excise deterministically. Do not land mid-sweep of an active qualification campaign (uniform-code confound). | 0.5-1 d |
| 3 | Announcer framing contract | `scifi_news_pro` already contracts announcer intro/outro-only; the structural gap is the `scifi_news` dedicated story graph. Ship the `scifi_news` structural-contract chunk (seam + score contract + fail-closed validator, lawful under THE LAW), then AUDIT the other banks' frames before touching them. Fold into the same pass as quick-win 4 so packs are touched once. | 0.5-1 d |
| 4 | Bank IMPROVE passes (720-verdict survivors) | Only the kept banks: `shakespeare` (confirm which seam version produced judged leg `c42700e1`; second prompt pass if the fix didn't take) + `original` (clarity/throughline without losing the noir mood + the F2 weapons-steering finding). The scifi_sonnet and science_news rows are RETIRED-BY-RIP. Standing consideration: source-native dramatic framing for shakespeare/public_domain vs the shared adapter. Seam/prompt work, no Python authorship. | 0.5-2 d |
| 5 | Canonical watchdog support | Runner heartbeats, watchdog recognizes canonical `RESULT`, pinned failure/stall paths; launcher has TWO missing-log echoes (C7 `scripts/_otr_soak_server_launch.cmd:38`, manifest `:48`) -- redirect both to quoted `%~1` and prove both appear in the server log. Harness defect, not a PBUG. | 0.5 d |
| 7 | Rip interstitial audio only | Node 83's cue audio/manifest fans to SceneSequencer via links 280/281 and to EpisodeAssembler via 282/283; SceneSequencer inserts interstitials at `scene_sequencer.py` ~:794-951. Remove ONLY the SceneSequencer side (links 280/281, its two cue inputs, insertion path, retired timing + mirrored-ledger fields -- enumerate them); RETAIN 282/283 + opening/closing synthesis + `music_inter` story/visual semantics. Canonical JSON updated + validated in the SAME commit. If shipped before lean-mean, mark the plan's standalone interstitial rip SATISFIED at re-ground. | 0.5-1 d |
| 8 | `docs/ENGINE_MATRIX.md` | Emit from the three live CAPABILITIES registries per the existing generator pattern (`build_variants.py` ~:276-338): write during `--all`/explicit emit; `--check` regenerates in memory and FAILS on drift without writing. Columns + stable ordering; link from README. PRECONDITION for Lean-Mean W6. | 0.5-1 d |
| 9 | Context/cap foundation | One provider-effective cap/count/reservation/must-fit authority feeding preflight, invocation, receipts; no silent truncation, no blind cap raise. Owner module must be CREATED (none exists; cap logic scattered across writer + backends -- both r4 panelists hallucinated `_otr_generation_budget.py`). Enumerate migrations at build: `_otr_openrouter_backend.py`, `_otr_comfy_backend.py`, `_otr_google_api/llm.py`, `_otr_gguf_backend.py`, `_otr_model_loader.py`, writer preflight. Acceptance: preflight and invocation provably make the SAME decision; must-fit overflow fails loud; receipts show provider, resolved model, cap source, counts, reservation, effective output. RE-GROUND against the dirty-tree provider-capacity contracts + the 16384 Mistral-Nemo cap fix before scoping. Carries the diagnostic-gap class from SUPERSEDED PBUG-20260712-17. | 1-3 d |
| 10 | Operator backlog (render tuning) | (a) Kokoro ALL-CAPS pre-TTS normalization: kokoro serves the ANNOUNCER bus, indextts2 the character bus (canonical nodes ~81/82 -- confirm at build); normalize a TTS-only copy, never ledger `spoken_text` or captions. (b) Credits ~1.5x faster = CONSTANT-ONLY: `_SCROLL_PPS = 60.0` (`otr_credits_roll.py:70`) -> 90.0, duration/`_MAX_HOLD_S`/no-truncation tests, NO canonical JSON change; the node reads a filesystem path + global ledger with no `IS_CHANGED` -- add a change key or force rerun. Ideal filler during render campaigns. | 0.5-1 d |

(Quick-win 6, `scifi_news_pro` C5 consumers, is CLOSED IN CODE under
PBUG-20260720-04; only the live six-bank qualification remains and is covered
by the campaign queue.)

### Big blocks (in ROADMAP-ratified order)

1. **LEAN-MEAN FRONT** (`W0 -> W1 -> W2 -> W3 -> W4a -> W4b -> W7 -> W6 ->
   W5+SW4 -> C1-C5`) -- `docs/2026-07-10-lean-mean-rip-final.md`, D-1..D-6
   RATIFIED. Execute after its 2026-07-15 drift-check header is satisfied
   (SW-3 news_ingest re-survey, W6 keep-list adds + ENGINE_MATRIX
   precondition = quick-win 8, W7 tombstone re-triage, R-7 re-grep; SW-1
   writer re-survey waits for the TAIL). Dedicated window; multi-day.
2. **Independent source banks (client-authored)** --
   `docs/2026-07-24-independent-source-banks-v1-plan.md` (PLAN OF RECORD;
   supersedes the retired Path-A/B `docs/2026-07-12-user-source-lanes-architecture.md`).
   Operator reframe 2026-07-24: N INDEPENDENT banks (no Path A/B, no family);
   client adds one from a folder equal to the shipped six; TRUSTED shared writer
   builds the COMPLETE ledger (the #1 key); a ledger-cleanup LLM pass fills/cleans
   it; content by REPAIR never a story-fail (SFW dropped as a gate); broken
   bundle QUARANTINES. DEFERRED past v1: client own-runner + staging, dependency
   subsystem, standalone story_rules. DOCS-FIRST: `docs/EXTENDING_OTR.md`
   (complete-ledger field contract) is the primary deliverable. Full r1-r4 arc +
   r5 simplification DONE + CONVERGED (codex gpt-5.6-sol high + agy Gemini 3.6
   Flash High; Claude judge; `kibitz-runs/2026-07-24-user-source-lanes-r6*/`).
   `docs/EXTENDING_OTR.md` DRAFTED; waves 1-2 (bundle integrity + admission in
   the one authority) LANDED @ `66e214ec`; wave 3 (client-owned
   `fetch_source`/`interpret_source` via the reserved `"self"` entry point)
   LANDED @ `cc69e683`. REMAINING WAVES, in order:
   wave 4 (the `otr_check bank --activate` CLI) LANDED @ `84945bc4`;
   `check_compatibility` was NOT wired -- `EXTENDING_OTR.md` now calls it a
   reserved name with no contract. REMAINING: **w5** the bounded
   `_otr_feed_fetch` seam, BOTH hops (feed + article scrape); **w6** the
   ledger-cleanup pass in the shared tail -- ALSO the right home for the
   client-interpreter fallback gap below; **w7** story_pack widget / canonical
   JSON if a surface changes. Re-estimate at each wave boundary.
3. **Randomizer Rolls Design A** --
   `docs/2026-07-12-randomizer-rolls-r2-coding-plan.md`, AFTER extensibility
   (its `_otr_lane_specs` authority is ABSORBED by the extensibility build;
   this shrinks to `_otr_bank_roll` + eligibility). Re-ground per its
   2026-07-15 header. 1-2 d + 1 GPU day.
4. **`dynamic_story` visual direction** -- rev-5 FINAL, do not rerun panels;
   roster-agnostic; re-derive IDs at build. After extensibility + randomizer.
   5-9 coder-days + 2-4 GPU days.
5. **LEAN-MEAN TAIL** (`SW1/SW2/SW3 -> C6 -> C7 -> W8`) -- the writer/widget
   structural split, REQUIRED by ROADMAP to come after blocks 2-4. SW-1 full
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
| RENDER (running now) | finish six-bank wrap -> 45w scene matrix -> 54-case style sweep; fillers: cpu-tier smoke + nv50 re-soak | local production + Codex-app monitor | bug-first items closed per campaign queue | GPU days |
| CODER A "seams" | bug-first items 1-3 (receipt-truth live confirm, still ownership, WAN contract), then quick-wins 1 + 2 + 3 + 4 | Claude codes, Qwen triages, codex on 3rd strike | no code mid-sweep | ~2-4 d |
| CODER B "harness" | quick-wins 5 + 10, then one canonical `scifi_news_pro` full-media qualification leg | same | after A | ~1-2 d |
| CODER C "foundations" | quick-wins 7 + 8 + 9 | same | after B | ~2-5 d |
| CODER D "lean-mean front" | drift-check re-verifies, then W0 .. C1-C5 | same | after C (W6 needs quick-win 8) | multi-day |
| PLANNER | extensibility hardening + `docs/EXTENDING_OTR.md` DONE 2026-07-24; NEXT = Bug Bible operator fan-out; plan upkeep | rungs 2-4 | parallel with D | docs |
| CODER E | independent client-authored source banks (lean v1); waves 1-4 LANDED @ `84945bc4`; NEXT = **wave 5** (bounded `_otr_feed_fetch` seam, BOTH hops), then w6 ledger cleanup, w7 surfaces | Claude; Qwen triage; codex via two-strikes | UNGATED (re-pin every line at HEAD first) | 1 wave per window |
| CODER F | Randomizer A -> `dynamic_story` | Claude + Qwen triage | after E | ~6-11 d |
| CODER G "lean-mean tail" | SW1-SW3, C6, C7, W8 | Claude; Fable single final epoch gate | after F | multi-day |

Kickoff lines (paste as the FIRST message of the new window; swap the letter):

> resume the OTR build -- you are CODER WINDOW A per GO_FORWARD "Window
> packing"; execute your scope in order, one green pushed chunk at a time,
> and state your MODEL & CREDIT BUDGET rung first.

## Parallel lane -- no coder slot required

- **Extensibility hardening -- DONE 2026-07-24 (planner window).** The r1-r4
  kibitz arc + an r5 simplification pass ran and CONVERGED; the operator
  reframed to N independent client-authored banks (lean v1, docs-first). Plan of
  record: `docs/2026-07-24-independent-source-banks-v1-plan.md`.
  **`docs/EXTENDING_OTR.md` DRAFTED + linked from README (same session):** the
  complete-ledger requirements contract, grounded per-consumer (TTS / slicing /
  shot direction / captions / credits / mux+publish) against the live code.
  CODER E is now UNGATED. (The old sec-16 nine-flags gate is retired with the
  A/B doc.)
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
| `PBUG-20260712-22/23/24/25` | Live reverify (quick-win 1), then fan-out |
| `PBUG-20260712-18/19/26` + `PBUG-20260713-15..18` + `-20` | Awaiting the next operator Bible fan-out (overlap check + approval) |
| `PBUG-20260713-19` | Live requalification pending (promoted BUG-05.11) |
| duplicate-id cleanup | Same fan-out: BUG-11.54 legacy_id -> `PBUG-20260713-21`; verify the acronym-union rule's legacy_id (both Bible rows cite `-10`; see the log's renumber note) |
| historical `PBUG-20260711-18` | Keep as the quick-win-9 engineering risk; never eligible from static evidence |
| `PBUG-20260710-07` | Ratify retirement at the next fan-out (green codex leg `c1f3891f`) |

The active production-fix owner updates `docs/PROD_BUG_LOG.md`; the approval
queue is `docs/BUG_BIBLE_PROMOTION_QUEUE.md`; no plan review or invented
fixture creates a row.

## Validation and handoff law

- Current whole-tree receipt (2026-07-24): full Windows suite `6294 passed /
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

- Extensibility hardening is DONE + CONVERGED and `docs/EXTENDING_OTR.md` is
  drafted, so CODER E is executing. Materially smaller than the retired
  ~21-31-day A/B scope; it still constrains randomizer + dynamic_story
  sequencing. Deferred power-user tiers (client own-runner + staging, deps,
  story_rules) are explicitly out of v1, not forgotten. Waves 1-4 are LANDED
  (@ `84945bc4`); the remaining waves still touch the writer hot path.
- CLIENT-AUTHORED PYTHON now executes in-process (wave 3). The posture that
  must hold in every later wave: `--activate` is the consent act; the seam
  fails LOUD (`UserBankExecutionError`) and never substitutes; client code
  never touches the canonical ledger; owner IDENTITY is verified so a bank can
  only run its OWN bundle; the shipped fetcher/interpreter registries are never
  widened to admit a client id. Do not relax any of these to make a later wave
  easier.
- **`check_compatibility` is RESERVED, not wired (wave-4 decision, kibitz
  r3 codex `gpt-5.6-sol` high + r4 agy Gemini 3.6 Flash High, Claude judge).**
  No request type, no decision type, no runtime consumer exists, so activation
  does not inspect it -- not even for callability -- and `EXTENDING_OTR.md`
  now calls it a reserved name instead of "NOT YET WIRED". `COMPAT_ENTRY_ATTR`
  is left INERT in `BUNDLE_ENTRY_ATTRS` with a comment saying so. **Operator /
  planner decision flagged:** codex argued for deleting the constant outright
  (it names a pseudo-ABI in executable code); that edit touches landed wave-3
  code AND the plan of record's "fetch_source + interpret_source +
  check_compatibility" line, which a coder window does not own. Either ratify
  the inert constant or schedule the rip as a planner chunk.
- **Client-interpreter fallback gap (known, deliberate, w6):**
  `_otr_source_payload.build_source_interpreter_fallback` switches on the four
  SHIPPED interpreter ids and raises `UnknownInterpreterError` otherwise, so a
  client interpreter that raises `SourceInterpretError` with an
  `.attempts`-carrying cause reaches it and gets a confusing message. Failing
  loud is correct meanwhile; inventing a generic client fallback belongs to the
  w6 ledger-cleanup pass, not to a patch.
- Lean-mean front/tail drift: the tail's SW-1 re-survey is mandatory against
  the then-current writer. Never interleave the two campaigns in one window.
- No code lands mid-sweep of an active qualification campaign (uniform-code
  confound -- the 420-rung lesson).
- The active campaigns may surface new lane defects; the campaign window owns
  admitting PBUGs (new-bug problem-statement rule applies).
- User extensibility and `dynamic_story` both touch the writer, visual-style
  authority, and canonical workflow: serial, each re-derives the live JSON.
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
artifact -- `tencent/hy3:free` panel seat (expired 2026-07-21).

## Pointers

- `ROADMAP.md` (dependency edges; lean-mean pin self-declares stale cites)
- `docs/PRODUCTION_SPRINT_LESSONS.md` (incl. lesson 24 lost-anchor; 25 bank-teardown)
- `docs/SOURCE_BANK_PREFLIGHT.md` -- add-a-bank gate + the Teardown protocol
- `docs/PROD_BUG_LOG.md` / `docs/BUG_BIBLE_PROMOTION_QUEUE.md`
- `docs/HANDOFF_LOG.md` (all completed-work history, newest at top)
- `docs/2026-07-23-video-failure-inventory.md` (campaign staging record)
- `docs/2026-07-15-720-bakeoff-verdict.md` (KEEP/IMPROVE + open items)
- `docs/2026-07-17-model-bakeoff-scoreboard.md` (writer-model verdict)
- `docs/2026-07-24-independent-source-banks-v1-plan.md` (extensibility PLAN OF RECORD)
- `docs/2026-07-12-user-source-lanes-architecture.md` (SUPERSEDED -- Path-A/B decision log)
- `docs/2026-07-10-lean-mean-rip-final.md` (drift-check header 2026-07-15)
- `docs/2026-07-12-randomizer-rolls-r2-coding-plan.md`
- `docs/2026-07-12-dynamic-story-visual-scope.md`
- `docs/2026-07-10-llm-first-story-edit-pass.md` (X1-X4 live remainder)
- `docs/2026-07-11-announcer-framing-defect.md` (OPEN)
- `docs/2026-07-11-timeline-cue-ledger.md`
- `workflows/otr_canonical.json`
