# Sprint S32 -- Final QA Review

**Branch:** `s32-helper-per-subpass-routing`
**Cut from:** `s31p5-legacy-residue-cleanup` @ `7c1a2ea` (S31.5 B7 close)
**Closed:** 2026-05-14

## Summary

Sprint S32 completes the dual-LLM theme: per-sub-pass routing
inside the four writer helpers (`pick_style`, `lock_cast`,
`compose_line`, `build_news_briefs`). Nine commits land on
`s32-helper-per-subpass-routing`, each pushed to origin. The
headline B1 (atomic, hard rule R4) refactored all four helper
signatures from single `generate_fn` to paired `creative_fn` +
`technical_fn` kwargs in one commit, alongside writer wiring +
10 new tests + 36 collateral test refactors. B2-B5 landed the
per-sub-pass dispatches one at a time; B4 took the no-widget
drift (architectural rejection of per-beat T-dispatch baked into
code instead of gated by an opt-in widget). B6 added per-helper
+ per-phase forensic meta stamping. B7 empty (no round-robin
findings). B8 close.

S32 sequenced after S31.5 (residue cleanup) per the locked
sprint order S31 -> S31.5 -> S32 -> S33 -> Sprint C. The S32 B0
orphan commit (`655dd6a`, pre-S31.5) was reverted at S31.5 start
(`4837ed7`); the branch was reset to the S31.5 tip and re-landed
fresh B0 against the clean baseline.

Hard rules respected at every commit boundary: audio C7 byte-
identical pytest proxy (default config) held; no legacy back-
compat reintroduced; one generate surface preserved; lifecycle
helpers distinct from generate surface; Bug Bible 23/1/2xf;
forbidden-pattern sweep 0 runtime hits; no version-label bumps.

Pytest-only gates per the autonomous-run handoff. Runtime 5080
verification (operator-driven) deferred to post-feature-set --
audio C7 byte-identical pytest proxy stood in throughout, plus
the new differing-slots baseline established at B5.

## Commit log

| # | Hash | Subject |
|---|---|---|
| B0 | `fcab6e1` | branch cut + S32 plan re-landing (post-S31.5 baseline, drift from original projection documented) |
| B1 | `af74458` | **ATOMIC HEADLINE** -- helper signatures refactored to paired generators + writer wired end-to-end (no dispatch yet) |
| B2 | `1c2d903` | pick_style pass 2 (chooser) dispatches to technical_fn |
| B3 | `f3e6d84` | lock_cast schema validation dispatches to technical_fn; fail-fast no internal retry |
| B4 | `d1a88f4` | compose_line critic stays creative-slot; no-widget rule applied (drift from plan, plan D1 architectural rejection baked in) |
| B5 | `9ca425e` | build_news_briefs paired-contract + outline retry + differing-slots audio baseline captured |
| B6 | `c6dc745` | slot transition accounting + per-helper meta stamping |
| B7 | `8236cbf` | empty, skipped |
| B8 | (this commit) | Sprint S32 close -- per-sub-pass routing shipped, meta forensic surface extended, dual-LLM theme complete |

## Acceptance table (21 rows from plan + drift documentation)

| # | Check | Plan target | Actual |
|--:|---|---|---|
| 1 | Full pytest count (canonical subset) | ~302 / 7 / 2 | **2103 / 10 / 0** (wide walk) -- plan projection was authored pre-S31.5 and is stale. Actual is the wide walk, dramatically higher because S31.5 B1 closed the wide-walk regression set; canonical subset post-S32 B8 close has all routing + meta + audio tests green. |
| 2 | Bug Bible regression | 23 / 1 / 2 | **23 / 1 / 2** PASS at every commit boundary |
| 3 | Audio C7 default (pytest proxy) | holds B1->B8 | **PASS** at every commit boundary |
| 4 | Audio differing-slots (pytest proxy) | baseline B5; holds B5->B8 | **PASS** -- `TestDifferingSlotsBaseline` class added at B5, holds B5->B8 |
| 5 | Audio C7 default (runtime 5080) | confirmed B5, B6 | **DEFERRED** per autonomous-run handoff |
| 6 | Audio differing-slots (runtime 5080) | new baseline confirmed B5, B6 | **DEFERRED** per autonomous-run handoff |
| 7 | Forbidden sweep | 0 runtime hits | **PASS** at every commit boundary |
| 8-11 | Helper signatures (4) accept paired generators | ✅ | **PASS** -- 4/4 signature-acceptance tests at B1 |
| 12 | pick_style pass 2 -> technical_fn | ✅ | **PASS** (B2) |
| 13 | lock_cast schema validation -> technical_fn | ✅ | **PASS** (B3); fail-fast `CastValidationLLMError` per D2 |
| 14 | compose_line critic-via-technical opt-in available, default OFF | ✅ | **n/a (no-widget decision at B4)** -- widget rejected; critic always creative per D1. Replaced with sweep marker + 2 deletion-guard tests. |
| 15 | build_news_briefs all V0-V3 on technical_fn | ✅ | **PASS** (B5 verified) |
| 16 | Writer wires paired generators end-to-end | ✅ | **PASS** (B1 atomic) |
| 17 | `meta.slot_calls_by_helper` populated | ✅ | **PASS** (B6) |
| 18 | `meta.slot_transitions_by_phase` populated | ✅ | **PASS** (B6) |
| 19 | Default config slot_transitions == 0 | ✅ | **PASS** -- verified by `test_default_config_zero_transitions` |
| 20 | VRAM warning logged on opt-in + differing-slots | ✅ | **n/a (no-widget decision at B4)** -- opt-in path removed; no warning needed. |
| 21 | Writer optional widget count | 16 (was 15 at S31) | **n/a (no-widget decision at B4)** -- count stays 15; bump dropped with the widget. |

## Architectural decisions (settled, locked at B0)

* **D1.** `compose_line` critic per-beat dispatch in differing-slots
  REJECTED for ~3.3 hr VRAM-thrash overhead per episode. Originally
  planned to be gated by an opt-in widget; **drift at B4** dropped the
  widget per Jeffrey's no-widget rule. Decision baked into code with
  a sweep marker locking against reintroduction.
* **D2.** `lock_cast` schema validation single-attempt technical,
  fail-fast `CastValidationLLMError`. Implemented at B3; writer-side
  caller can branch on the subclass to trigger creative regen.
* **D3.** Outline retry stays creative (schema validation is pure
  pydantic, no LLM; retry is content regeneration). Verified at B5
  via structural test pinning the single-fn signature.

## Drift from plan

1. **B0 baseline projection drift.** Plan projected S32 baseline
   ~282/7/2 and B8 target ~302/7/2. Actual baseline at S32 B0
   was 251/7/2 (canonical subset; gap from S31 close's 243/7/2
   plus S31.5's net +8 from triage). Plan projection was authored
   pre-S31.5 and didn't account for the residue-cleanup sprint
   inserted between S31 and S32.

2. **B4 no-widget drift (major).** Plan called for a
   `use_technical_critic` opt-in widget (default OFF), writer
   widget count bump 15->16, conditional dispatch, and VRAM-
   warning logging. Per Jeffrey's no-widget rule -- "if a
   feature is useful, it's on; if not, extract from code" -- the
   widget was DROPPED. A widget defaulting OFF that gates an
   architecturally-rejected path (per-beat T-dispatch per D1) is
   the maintenance debt that rule guards against. Drift documented
   in:
   * B4 commit message
   * `docs/2026-05-14-S31-S32-cowork-execution-plan.md` B4 section
     rewritten to reflect the no-widget code path.
   * Forbidden-pattern sweep marker `\buse_technical_critic\b`
     added (S32 B4 extinction marker).
   * Acceptance rows 14 + 20 + 21 marked "n/a (no-widget decision
     at B4)".

3. **Test count drift.** Plan B1 projected 10 new tests + 9 from
   B2 + 5 from B3 + 6 from B4 + 4 from B5 + 5 from B6 = 39 new.
   Actual at S32 B8 close: ~26 new tests (B1=10 - test infra
   collateral refactors not new; B2=3, B3=4, B4=2 not 6, B5=4,
   B6=3 not 5). The audio-canary tests counted in the plan are
   the same `tests/test_audio_byte_identical.py` -- not "new" per
   se but renamed structurally for each commit.

## Forward work

* **Operator runtime 5080 verification (S32 post-close gates).** All
  gates from the plan's S32 post-close runtime release gate are
  deferred to post-feature-set per the autonomous-run handoff.
  Audio C7 byte-identical pytest proxy stood in throughout (both
  default and differing-slots). Jeffrey runs the runtime verification
  when a feature sprint that requires it ships.

* **S33 pending decisions.** S33 B1's `polish_announcer_beats`
  widget has the same architectural question that S32 B4 resolved
  no-widget. Decision deferred to S33 kickoff. The S31+S32 plan
  document's S33 section will need a drift-pass when S33 opens.

* **UNGATED_PASS_RECOMMENDATION (post-soak).** Carried from S31
  forward work. No catalog change in S32.

* **Loader API consolidation.** Carried from S31 forward work.
  Out of S32 scope. Post-port hygiene; behavior unchanged.

* **Sprint sequence ahead:** S33 (editor-only cleanup -- retire
  cascade Phase 1 + Phase 9 auditors, restore announcer polish
  per the locked sprint order) -> Sprint C (`meta.story_brief`
  v2) -> Sprint A (public-facing polish).

## B7 buffer

Empty. No round-robin findings; no adjacent-commit folds emerged.

## Sources

* Plan: `docs/2026-05-14-S31-S32-cowork-execution-plan.md` (canonical;
  B4 section rewritten at B4 commit to reflect the no-widget drift).
* Branch: `s32-helper-per-subpass-routing` (origin synced through B8).
* Parent S31.5 close: `docs/2026-05-14-S31p5-final-qa-review.md`.
* Parent S31 close: `docs/2026-05-14-S31-final-qa-review.md`.
* BUG_LOG: no new entries during S32 (the sprint is architectural
  refactor, not bug-fix).
* S32 B0 re-land artifact: `4837ed7` on the same branch (the
  pre-S31.5 orphan B0 revert), `fcab6e1` (the re-landed B0).
