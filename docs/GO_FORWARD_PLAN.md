# OTR Go-Forward Plan

**Updated:** 2026-07-17 night4 -- HEAD `1fd7743d` (v4 Phase 2 bank #1 `scifi_codex_v4` code
shipped; P3 cap-restatement + bank plan pending commit with the live proof), branch
`v2.0-alpha`. Truly forward-only by
operator directive: completed work lives in `docs/HANDOFF_LOG.md` (history) and
`docs/PROD_BUG_LOG.md` (bugs); doctrine lives in
`docs/PRODUCTION_SPRINT_LESSONS.md` (the "lost anchor" class is now lesson 24
there). Dated plan docs are implementation evidence, not competing queues; each
carries a 2026-07-15 status header stating what it needs before execution.
Hardened by a kibitz r4 confirm pass (codex `gpt-5.6-sol` + agy `Gemini 3.5
Flash (High)` + Claude anchor/judge; grounded survivors folded 2026-07-16
pre-dawn, run record local under `kibitz-runs/2026-07-15-gfp-baseline/r4/`).

## THE LAW (operator, 2026-07-13 -- supersedes anything below that disagrees)

> **AN AUDIT MAY IMPROVE A STORY. IT MAY NEVER FAIL ONE.**
> Only DETERMINISTIC validators may end an episode. An LLM verdict may trigger a
> bounded rewrite; it may never raise.

Standing enforcement: the deterministic G9 SFW gate in
`_otr_ledger_freeze.run_gap_audit` (Phase 10, the one path every lane crosses).
Veto-rip record: HANDOFF_LOG + PROD_BUG_LOG (PBUG-20260713-15..18).

## CURRENT STEP -- v4 improvement campaign (post roster-trim)

The **roster trim LANDED @ `499386aa`** (2026-07-17). The `source_bank` roster is
now **10 INDEPENDENT lanes + custom** (media_archive(+_v3), original_radio,
scifi_fable2(+_v3), scifi_codex(+_v3), public_domain_story_v3, shakespeare_v3,
scifi_sonnet_v3): the science_news family, ALL `_v2` lanes, and the orphan bases
were retired; each kept lane owns its pack + `story_rules` by EXACT id (the
`base_source_bank_id` family-map is severed -- no lane depends on another).
Default bank = `scifi_fable2`. Full record: HANDOFF_LOG 2026-07-17 afternoon;
design in `docs/2026-07-17-roster-trim-rip-plan.md`.

**NEXT (operator 2026-07-17): a v4 improvement campaign** to lift the kept banks
-- produce a **v4** for `scifi_codex` (improve on v1), `shakespeare`,
`public_domain`, `media_archive`, and `original_radio`, each a fully INDEPENDENT
bank (own row + pack + story_rules by exact id + pipeline; no base-map).

**ARC COMPLETE (2026-07-17 evening):** the full kibitz arc r1-r4 ran and CONVERGED
(operator routing: Codex @ gpt-5.6-sol + agy @ Gemini 3.1 Pro (High); Claude
anchor+judge; $0 local -- the cloud roundtable was skipped per operator "also
/kibitz GPT-5.6-sol and Gemini Pro"). Plan of record =
`docs/2026-07-17-v4-campaign/final.md` (+ LESSONS_GATE_BRIEF.md, r1..r3_plan.md,
r1..r4_judgment.md; raw panels under `kibitz-runs/2026-07-17-v4-campaign/`). Every
folded panel claim was grounded CONFIRMED against the real files.
**IN EXECUTION (coder window, 2026-07-17 night).** Plan of record =
`docs/2026-07-17-v4-campaign/final.md`. Phase 0 -> Phase 1 (8 shared fixes, each
its own green pushed chunk, canary per execution family) -> Phase 2 (5 v4 banks
serialized, atomic per-bank chunk). Ship gate = green+live+pushed; "strictly
better" = a POST-BUILD blind A/B.

- **Phase 0 DONE** (detail in HANDOFF_LOG 2026-07-17 night). PBUG-20260710-07
  root-caused STATICALLY = the D3 pre-freeze coerce sweep; already closed by
  sentinel-mint + name-exclusion + the `role_coerce` compose_flags breadcrumb,
  pinned by `tests/test_d3_role_coercion.py`. No coerce change (a shim). Durable
  v4 protection = per-lane announcer-sentinel minting invariant (Phase 2 + a live
  leg retires the PBUG, kept ROOT-OPEN in the log until then). Defect #2
  (name-splice) stays OPEN per the timebox.
- **P1(i) PUSHED @ `c3a9d420`** -- validated scalar bank defaults
  (`style_pool_class`|`require_science_floor`|`propagate_adaptation_cast`) replace
  the 3 hardcoded exact-id sets; all 10 runnable rows migrated; `select_style`
  hash keys preserved -> byte-identical slugs (C7). Suite 7974 / Bible 17.
- **P1(ii) PUSHED @ `f859036c`** -- named regression pinning PBUG-20260710-07
  (cast-keyed-mutation class): reason-stamp-on-every-coercion + announcer-sentinel
  protection. Test-only (root fix shipped pre-campaign; no coerce shim). PBUG stays
  ROOT-OPEN until a live v4 leg.
- **P1(iii) PUSHED @ `e7ba2627`** -- bank-aware genre/spoken-text guard: new
  `_otr_genre_guard` boundary matcher + writer authored-repair + deterministic G10
  terminal in `run_gap_audit`. OPT-IN via `defaults.genre_guard_spoken` (default
  False -> INERT for all 10 current banks; v4 flips in Phase 2). Suite 8018.
- **P1(iv) PUSHED @ `90ed495e`** -- `beat_bounds` structural contract in
  `_otr_episode_budget` (WORDS_PER_BEAT=40 SOFT/recorded; floor 3) + deterministic
  G11 floor terminal (opt-in via `meta.beat_bounds`); writer stamps it. Only the
  structural floor gates (operator: length recorded-not-gated); MAX + word->beat
  derivation deferred to Phase-2 live. Suite 8031 / Bible 17.
- **P1(v)/(vii)/(viii)/(vi) PUSHED** @ `0066f5ab`/`e7bfb1fe`/`4f8bd7aa`/`d29ba920`
  -- outro cast-completeness (G12), literal-placeholder-token (G13), provenance
  normalizer + research_only publish gate (G14), header<->scene structural coherence
  (G15). Each a SELF-CONTAINED module + a deterministic terminal in run_gap_audit,
  OPT-IN via a validated scalar bank default -> INERT for the 10 current banks.
- **PHASE 1 COMPLETE** (7 shared fixes ii-viii, each a green pushed chunk; suite
  8134 / Bible 17). New opt-in flags a v4 bank can set: `genre_guard_spoken`,
  `require_outro_cast_complete`, `placeholder_guard`, `provenance_normalize`,
  `scene_coherence_check`; plus the recorded `beat_bounds` contract (floor-only gate).
- **PHASE 2 -- bank #1 `scifi_codex_v4` CODE SHIPPED @ `1fd7743d`** (full suite 8139 /
  Bible 17 / `otr_canonical.json` byte-unchanged / HEAD==origin). Fully INDEPENDENT bank:
  banks.json row + pack + `story_rules/scifi_codex_v4.json` (exact id) + pipeline
  `scifi_codex_circuit_v4` mapped **DIRECTLY** to `_run_scifi_codex_lane` (no v3 advisory
  wrapper) + roster/bijection tests. Proof-pressure delta is pack-seam-only (a want, the
  gating proof, a mandatory cost beat, one reversal). Gates ON: `require_science_floor` +
  `placeholder_guard` (G13) + `scene_coherence_check` (G15). Gates DEFERRED: `genre_guard_spoken`
  (G10) + `require_outro_cast_complete` (G12) -- the dedicated codex runner does NOT cross the
  inline I.7/I.8 authored-repair boundary, so they would be no-repair hard gates until that
  boundary is wired for the codex family (vetoable). Full record + lessons:
  `docs/BANK_PLAN_scifi_codex_v4.md` (tracked; the campaign folder is gitignored).
- **LIVE-LEG LESSON (operator-flagged 2026-07-17; CORRECTED by the two-strikes kibitz r2):** the codex
  **P3** failures (`string_too_long`, then `beat_count`) are **MODEL-INDEPENDENT** (both Mistral-Nemo and
  gemma-4-E4B) and belong to the **unstated DETERMINISTIC-CONTRACT class (PBUG-20260713-02 + -06)** -- NOT
  a simple "unstated cap". The kibitz (Codex `gpt-5.6-sol` + Antigravity Gemini 3.1 Pro, grounded) proved
  `_RADIO_SCORE_DRAFT_SURFACE_INSTRUCTION` ALREADY injects tighter caps into every P3 prompt, so the seam
  cap-restatement was first reverted -- but a LIVE leg OVERTURNED that (reverting regressed
  `string_too_long` on `premise`; the text-patch never clips prose), so it was **RE-ADDED** as load-bearing
  salience. ROOT FIX (also) = make the whole compiler contract model-visible
  (`unused_shot` / `cast_coverage` / `cue_id`-unique / `cue_anchor<beat_count` added to the shared surface
  instruction + a 12-beat distribution clause), KEEP the v4 beat-count harmonization, and enrich the
  `beat_count` receipt (observed-vs-expected). `string_too_long` recovery stays the text-patch seam. Full
  record: `docs/BANK_PLAN_scifi_codex_v4.md` + `kibitz-runs/2026-07-17-p3-beatcount/`. A codex leg still
  retires PBUG-20260710-07 via the announcer-sentinel mint. Pending the live 30w Mistral-both proof.
- **NEXT:** finish the `scifi_codex_v4` live leg (Mistral + restated caps), then bank #2
  `shakespeare_v4` -> `public_domain_story_v4` -> `media_archive_v4` -> `original_radio_v4`. Each
  writes its OWN idiom (never sci-fi); the non-codex lanes are INLINE (`legacy_many_pass_v4` /
  `original_multi_pass_v4`) and DO cross the authored-repair boundary -- so genre+outro gates ARE
  safe to enable there. Pre-emptively restate any capped schema field in each lane's authoring seam.

Open operator decisions (defaulted, vetoable at the consuming chunk):
WORDS_PER_BEAT=40 (soft target; length recorded-not-gated); media_archive_v4 ships
its OWN drama_seeds; public_domain `research_only` BLOCKS publish; scifi_codex_v4 genre+outro
gates deferred pending the codex authored-repair boundary.

The 24-lane bake-off campaign that produced the scoreboard is **CONCLUDED** (its
verdict drove this trim); records live in HANDOFF_LOG +
`docs/2026-07-17-variant-scoreboard.md`. The old B1/B2/B5 variant-family
invariants are RETIRED with the family mechanism.

Standing campaign notes (kibitz r4 fold, grounded):

- **`science_news_v3` + local profile is a recorded live FAIL** (news-coda
  bridge LLM failed both attempts; the resume sweep hard-skips that cell --
  `tmp/_phaseC_KNOWN_BUGS.md`). The triage note's own "deterministic pool /
  arc-bridge fallback" suggestion is UNLAWFUL (canned spoken content -- the
  2026-07-03 no-fallback rip + LLM-first both forbid it). The fix is a
  model/prompt/budget-contract root fix or an explicit lane/profile
  disqualification -- never Python-authored dialogue, never a blind retry bump.
  Count the cell FAIL in the report; campaign window owns the PBUG admission
  (it is a live production failure). Fix owner: next coder window.
- **Evidence discipline for seam promotion:** phase C runs production mode
  (no C7, fresh source per leg) and generation payloads pin temperature but
  not a model-sampling seed -- so phase C supports reliability + blind-quality
  ranking, NOT causal seam attribution. Promotion of a `_v2`/`_v3` seam into a
  base pack needs a matched frozen-source A/B per family (the B7/B8 snapshot
  layer exists for exactly this), or is an explicit operator quality-judgment
  call. Say "pack-associated", not "pack-caused", outside the F2-style
  controlled triplets.
- **Phase C is story-quality evidence first.** Known full-media consumer
  defects (quick-win 6: captions exact `char_id` map, credits raw-ID voice
  receipts, HuMo literal-`announcer` stale guard) ship after it -- so any
  full-media promotion claim waits for quick-win 6 plus one canonical fable2
  full-media qualification leg.
- **Cost option (campaign call):** after the 30w structural smoke, the `_v2`
  arms duplicate `_v3` content (v3 = v2 seam text + read-only advisory);
  dropping `_v2` at the higher tiers roughly halves the remaining legs.

Then, in order:

1. **Durable report + World Cup scoreboard** from the phase-C receipts: named
   scoring axes, every intended cell explicitly SUCCESS / FAIL / DISQUALIFIED /
   NOT RUN (no silent omissions), receipts carrying matrix id + commit +
   resolved models. At wrap-up, move the still-authoritative bake-off contracts
   (r3 rulings, Sec-D seam rules, acceptance criteria) out of the gitignored
   `docs/2026-*/` + `kibitz-runs/` folders into a tracked doc -- a clean clone
   currently cannot see the campaign's design authorities.
2. **Operator decisions:** roster trim (which lanes go), which `_v2`/`_v3` seams
   promote into their base packs (per the evidence discipline above), and the
   IMPROVE passes (quick-win 4 below). The F2 finding -- original_radio
   `_v2`/`_v3` seam steers toward weapons content vs base -- feeds the same
   seam-tuning pass. `tencent/hy3:free` panel seat expires 2026-07-21.
3. **Unblock the mistral/gemma creative-writer matrices** (verdict open item 3)
   -- the only path from "best bank on aion" to "best model". Render-window
   work, not coder-slot.
4. Optional tuning follow-on: the v3 packs still carry v2 seam text (the
   structural v3 delta is the advisory diagnostic); per-lane Sec-D one-liners
   are in place to edit if wanted.

## Coder queue (2026-07-15 baseline, re-grounded)

One coder window at a time; every chunk = focused tests + full suite + Bug
Bible + commit AND push + `HEAD == origin/v2.0-alpha`.

```text
finish bake-off campaign (render window; operator verdicts)
  -> quick-wins block (coder windows A/B/C -- see Window packing)
  -> LEAN-MEAN FRONT (W0->W1->W2->W3->W4a->W4b->W7->W6->W5+SW4->C1-C5)
       [sec-16 ratification + r5 kibitz run in PARALLEL, $0, planner window]
  -> user source lanes / extensibility
  -> Randomizer A -> dynamic_story
  -> LEAN-MEAN TAIL (SW1/SW2/SW3 -> C6 -> C7 -> W8)
  -> ROADMAP (SFX campaign after Timeline Cue Ledger gate)
```

The lean-mean/extensibility ordering question DISSOLVES on ROADMAP's ratified
dependency edges (ROADMAP.md section 1): the front waves have no extensibility
dependency, while "user extensibility, Randomizer, and dynamic_story [come]
before the writer/widget structural split" -- so the SW tail was always
sequenced after them. Split the campaign; interleave nothing.

### Quick-wins block (~6-13 coder-days; small chunks, any order inside the block)

| # | Chunk | Gate | Est |
|---:|---|---|---:|
| 1 | Sci-Fi Codex reverify tail | PBUGs 20260712-22/23/24/25 are FIXED IN TREE, LIVE REVERIFY PENDING. Phase C runs only `_v2`/`_v3` lanes, so a variant leg exercises the same runner/transport seams but is NOT the literal "same canonical bank" condition: either run ONE base `scifi_codex` 120w leg after phase C, or record explicit operator acceptance of variant-leg coverage. Then mark the log and release the coder slot formally. | 0.25-0.5 d |
| 2 | Cliche-span excision (X1-X4) | `docs/2026-07-10-llm-first-story-edit-pass.md` Wave 3: `repair_cliche_span` (`_otr_line_composer.py` ~:2632/:2676) + `cliche_replacements` in all 8 story_rules JSONs still rewrite SPOKEN lines -- a standing violation of the LLM-first directive. Excise deterministically. **Do not land while phase C is mid-sweep** (uniform-code confound -- the 420-rung lesson). | 0.5-1 d |
| 3 | Announcer framing contract | `docs/2026-07-11-announcer-framing-defect.md` -- still fully OPEN, but the blast radius is narrower than the doc assumed (kibitz r4, grounded): fable2 already contracts "ANNOUNCER speaks ONLY in the intro and the outro" (`_otr_scifi_fable2.py:1741-1743`) and sonnet builds a cold open + sign-off; the structural gap is the CODEX lane (`CastPlanRowV4`/`ScriptLineV4`/`make_advisory_word_blueprint` live only in `_otr_scifi_codex.py`). Ship a codex structural-contract chunk (seam + score contract + fail-closed validator, lawful under THE LAW; `original_radio_v2`'s billboard/sign-off seam is prior art), then AUDIT the other lanes' existing frames before touching them. Fold into the same pass as quick-win 4 so packs are touched once. | 0.5-1 d |
| 4 | 720-verdict IMPROVE passes | shakespeare: confirm which seam version produced judged leg `c42700e1`; second prompt pass if the fix didn't take. scifi_sonnet: seam consolidation (nine seams; owns the set's only outright FAIL). original_radio: clarity/throughline without losing the noir mood + the F2 weapons-steering finding. science_news: constrain the concept, keep the steady 18-beat template. Related standing consideration (2026-07-14): source-native dramatic framing modes for shakespeare/public_domain vs the shared adapter -- compare without changing frozen receipts; preserve source names/roles by default. Seam/prompt work, no Python authorship. | 1-3 d |
| 5 | Canonical watchdog support | Runner heartbeats, watchdog recognizes canonical `RESULT`, pinned failure/stall paths -- plus the campaign follow-up: the launcher has TWO missing-log echoes, C7 at `scripts/_otr_soak_server_launch.cmd:38` and manifest at `:48`; redirect both to quoted `%~1` and prove both appear in the server log. Harness defect, not a PBUG. | 0.5 d |
| 6 | Fable2 C5 consumers | Captions and credits use alias-aware cast lookup; HuMo stale guard uses role/source-family/ShotLock identity. Current gaps (kibitz r4, grounded): captions use an exact `char_id` map (`_otr_captions.py` ~:180), credits resolve display names alias-aware but voice receipts by raw ID (`otr_credits_roll.py` ~:402), HuMo guard requires literal `char_id == "announcer"` (`render_driver.py` ~:1262). PRECEDES any full-media promotion claim from phase C (see campaign notes). | 0.5-1 d |
| 7 | Rip interstitial audio only | The exact surgery (kibitz r4; link endpoints re-derived at build): node 83's cue audio/manifest fans to SceneSequencer via links 280/281 and to EpisodeAssembler via 282/283; SceneSequencer inserts interstitials at `scene_sequencer.py` ~:794-951. Remove ONLY the SceneSequencer side (links 280/281, its two cue inputs, the insertion path, timing + mirrored-ledger fields being retired -- enumerate them); RETAIN 282/283 and opening/closing synthesis; RETAIN `music_inter` story/visual semantics. Canonical JSON updated + validated in the SAME commit. | 0.5-1 d |
| 8 | `docs/ENGINE_MATRIX.md` | Emit the matrix from the three live CAPABILITIES registries following the existing generator pattern (`build_variants.py` ~:276-338): write during `--all`/an explicit emit mode; `--check` regenerates in memory and FAILS on drift without writing. Define columns + stable ordering; link from README. PRECONDITION for Lean-Mean W6. | 0.5-1 d |
| 9 | Context/cap foundation | One provider-effective cap/count/reservation/must-fit authority feeding preflight, invocation, receipts; no silent truncation, no blind cap raise. The owner module must be CREATED -- none exists at HEAD (both r4 panelists cited `nodes/_otr_generation_budget.py` as existing; grounded: it does not -- cap logic is scattered across the writer + backends). Migrations to enumerate at build: `_otr_openrouter_backend.py`, `_otr_comfy_backend.py`, `_otr_google_api/llm.py`, `_otr_gguf_backend.py`, `_otr_model_loader.py`, writer preflight. Acceptance: preflight and invocation provably make the SAME decision; must-fit overflow fails loud; receipts show provider, resolved model, cap source, counts, reservation, effective output. Partially advanced by the static-row ctx fix (`32e680b2`, PBUG-20260713-20). Carries the diagnostic-gap class from SUPERSEDED PBUG-20260712-17: if attempt capture is needed, re-target the PARKED telemetry seam (generic `_otr_structured_call` callback; reconcile with the existing `on_attempt_complete` hook) at a surviving lane. | 1-3 d |
| 10 | Operator backlog (render tuning) | Two SEPARATE fixes (kibitz r4, grounded). (a) Kokoro ALL-CAPS pre-TTS normalization: kokoro serves the ANNOUNCER bus, indextts2 the character bus (canonical nodes ~81/82 -- confirm at build); normalize a TTS-only copy, never the ledger `spoken_text` or captions. (b) Credits ~1.5x faster is a CONSTANT-ONLY change: no speed widget exists; scroll rate is `_SCROLL_PPS = 60.0` (`otr_credits_roll.py:70`) -> 90.0, duration/`_MAX_HOLD_S`/no-truncation tests, NO canonical JSON change. Note: the node reads a filesystem path + global ledger with no `IS_CHANGED` -- add a change key or conservatively force rerun. Ideal filler during render campaigns. | 0.5-1 d |

Retired 2026-07-15 (do not re-derive): codex56sol attempt telemetry + the
PBUG-20260712-17 root fix (target lane ripped @ `3312aec7`; the telemetry plan
doc is SUPERSEDED with its portable pieces parked in its header) and the old
"fresh two-matrix bakeoff" item (superseded by the executed campaigns).

### Big blocks (in ROADMAP-ratified order)

1. **LEAN-MEAN FRONT** (`W0 -> W1 -> W2 -> W3 -> W4a -> W4b -> W7 -> W6 ->
   W5+SW4 -> C1-C5`) -- `docs/2026-07-10-lean-mean-rip-final.md`, D-1..D-6
   RATIFIED. Execute **after its 2026-07-15 drift-check header is satisfied**
   (SW-3 news_ingest re-survey, W6 keep-list adds + the ENGINE_MATRIX
   precondition = quick-win 8, W7 tombstone re-triage, R-7 re-grep; the SW-1
   writer re-survey can wait for the TAIL). Kill lists + W5's positional
   obligation re-verified LIVE 2026-07-15 and intact; nothing was double-ripped.
   If quick-win 7 shipped first, mark the plan's standalone interstitial-audio
   rip SATISFIED at re-ground (ROADMAP note). Dedicated window; multi-day.
2. **User source lanes / extensibility** -- `docs/2026-07-12-user-source-lanes-architecture.md`
   (supersedes the vibe-coder r2 plan). GATED: operator ratifies its section 16
   (nine flags) + one r5 confirmation kibitz pass ($0, planner window, runs in
   parallel with block 1); THEN fold into this plan and claim the coder slot.
   **~21-31 coder-days** (not the old "4-7").
3. **Randomizer Rolls Design A** -- `docs/2026-07-12-randomizer-rolls-r2-coding-plan.md`,
   AFTER extensibility (its `_otr_lane_specs` authority is ABSORBED by the
   extensibility build; this build shrinks to `_otr_bank_roll` + eligibility).
   Re-ground per its 2026-07-15 header. 1-2 d + 1 GPU day.
4. **`dynamic_story` visual direction** -- rev-5 FINAL, do not rerun panels;
   re-checked 2026-07-15: roster-agnostic, wiring snapshot still matches live
   canonical. After extensibility + randomizer; re-derive IDs at build.
   5-9 coder-days + 2-4 GPU days.
5. **LEAN-MEAN TAIL** (`SW1/SW2/SW3 -> C6 -> C7 -> W8`) -- the writer/widget
   structural split, REQUIRED by ROADMAP to come after blocks 2-4 (preserve
   overlay quarantine, pack replay hashes, and the final live widget layout).
   SW-1 full seam re-survey happens here, against the then-current writer.

**Ranges:** quick-wins ~6-13 coder-days; lean-mean front+tail = ROADMAP's 12-16;
extensibility ~21-31; randomizer 1-2; dynamic_story 5-9. Combined ~45-71
coder-days through the lean-mean tail, plus campaign GPU days. ROADMAP items
2-5 (SFX campaign, product expansion, RunPod/install, release docs) excluded.

## Window packing (credit discipline -- one line starts any window)

Starting any window costs the same boot context, so BATCH chunks per window and
never open one for a single small item. Every window starts the same way: open
a fresh Cowork chat and paste its one-line kickoff -- the `otr-handoff` skill
reads this file + git and states the current step. **No manual context
handoff, ever**; this planner window keeps GO_FORWARD + HANDOFF_LOG current,
coder windows never write plans (window-roles rule).

| Window | Scope | Gate | Size |
|---|---|---|---|
| RENDER (running now) | phase C tiers -> durable report -> operator decisions; fillers between tiers: cpu-tier smoke + nv50 re-soak (release QA, render time not code) | -- | GPU days |
| CODER A "seams" | base `scifi_codex` 120w reverify leg (quick-win 1), then quick-wins 2 + 3 + 4 | AFTER phase C completes (no mid-sweep code) | ~2-4 d |
| CODER B "harness + consumers" | quick-wins 5 + 6 + 10, then one canonical fable2 full-media qualification leg | after A | ~1.5-3 d |
| CODER C "foundations" | quick-wins 7 + 8 + 9 (canonical surgery, generator, new budget owner) | after B | ~2-5 d |
| CODER D "lean-mean front" | drift-check re-verifies, then W0 .. C1-C5 | after C (W6 needs quick-win 8) | multi-day |
| PLANNER (this window) | sec-16 ratification session + r5 kibitz ($0), Bug Bible operator fan-out, plan upkeep | parallel with D | docs |
| CODER E | extensibility (user source lanes) | after sec-16 + r5 | 21-31 d |
| CODER F | Randomizer A -> `dynamic_story` | after E | ~6-11 d |
| CODER G "lean-mean tail" | SW1-SW3, C6, C7, W8 | after F | multi-day |

Kickoff lines (paste as the FIRST message of the new window; swap the letter):

> resume the OTR build -- you are CODER WINDOW A per GO_FORWARD "Window
> packing"; execute your scope in order, one green pushed chunk at a time.

Credit rules: kibitz local ($0) for ALL mechanical review -- the two-strikes
law stands; cloud roundtable only for genuine R1-ideas passes; Fable only as
the single final gate on a lean-mean epoch commit (section-9 reality
exception); codex-CLI delegation via a HANDOFF_CODEX file remains available
for grind chunks (2026-07-13 precedent).

## Parallel lane -- no coder slot required

- **sec-16 ratification + r5 confirm** on the extensibility architecture --
  the operator bottleneck on the critical path; planner window, $0.
- **Bug Bible operator fan-out** -- the promotion table above has 9+ closed
  candidates + the duplicate-legacy_id cleanup waiting on one fan-out session.
- **Render-window fillers:** cpu-tier smoke (needs the google image lane or
  stills) + nv50 re-soak -- the two open portability remainders; release QA
  validation time, not coding.
- **SFX R4.1 re-ground** (0.5-1 docs day): re-ground the local generated-SFX R4
  candidate into a tracked current-HEAD R4.1 plan. Sequencing + retained-scope
  contract live in `ROADMAP.md` (Timeline Cue Ledger C0/C1 gate first; no
  second SFX queue, no library fallback).
- **Operator-promotable option:** SFX C0 (per-line WAV stems + transcript
  drift report) is independently shippable per ROADMAP but stays parked with
  its campaign unless explicitly promoted.

## Bug Bible promotion field -- pending actions only

Production admission, implementation status, and portable-rule promotion are
different facts; the log is the record, this table is only what still needs an
action.

| Record | Pending action |
|---|---|
| `PBUG-20260712-22/23/24/25` | Live reverify (quick-win 1), then fan-out |
| `PBUG-20260712-18/19/26` + `PBUG-20260713-15..18` + `-20` | Awaiting the next operator Bible fan-out (overlap check + approval) |
| `PBUG-20260713-19` | Live requalification pending (promoted BUG-05.11) |
| duplicate-id cleanup | At the same fan-out: BUG-11.54 legacy_id -> `PBUG-20260713-21`; verify the acronym-union rule's legacy_id (both Bible rows cite `-10`; see the log's renumber note) |
| historical `PBUG-20260711-18` | Keep as the quick-win-9 engineering risk; never eligible from static evidence |

No 07-14/15 bake-off-era PBUGs are logged yet; the campaign window owns
admitting any (e.g. the F2 weapons-steering finding, if operator admits it).
The active production-fix owner updates `docs/PROD_BUG_LOG.md`; the approval
queue is `docs/BUG_BIBLE_PROMOTION_QUEUE.md`; no plan review or invented
fixture creates a row.

## Validation and handoff law

- Current whole-tree receipt (`f58ed6e6`, 2026-07-16): **7,967 passed / 31
  skipped / 1 xfailed**; Bug Bible **17 passed**; canonical workflow **23 nodes /
  57 links**, delta = none. (Qwen3-8B GGUF writer row PROMOTED UNKNOWN->PASS this
  session -- an orthogonal model-roster task per `docs/2026-07-16-gguf-row-registry.md`,
  NOT a forward-order step; first GGUF build roster is now gemma-4-12b + Qwen3-8B.
  Detail in HANDOFF_LOG.)
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
  should RE-READ this file before its wrap-up edit -- it was rewritten and then
  leaned 2026-07-15 late night. The otr-build-tracker artifact is RETIRED
  (tombstoned 2026-07-15); HANDOFF_LOG + this file are the only tracking
  surfaces.

## Open risks

- Extensibility is gated on operator section-16 ratification + the r5 pass; its
  ~21-31-day estimate is latent scope, not creep. Until ratified it holds no
  slot and only constrains randomizer + dynamic_story sequencing.
- Lean-mean/extensibility ordering is RESOLVED by ROADMAP's ratified edges
  (front waves first, SW tail after extensibility/randomizer/dynamic_story);
  the residual risk is drift between the front and tail windows -- the tail's
  SW-1 re-survey is mandatory against the then-current writer. Never
  interleave the two campaigns inside one window.
- No code lands while phase C is mid-sweep -- landing mid-sweep re-creates the
  uniform-code confound that made the 420 rung unjudgeable.
- Phase C may surface new lane defects; the campaign window owns admitting
  PBUGs.
- User extensibility and `dynamic_story` both touch the writer, visual-style
  authority, and canonical workflow. They remain serial and each re-derives the
  live JSON.
- Generated-SFX R4 stays local/ignored evidence until the tracked R4.1 refit
  lands; it is not an executable queue.

## Pointers

- `ROADMAP.md` (current, 2026-07-12; lean-mean pin self-declares stale cites)
- `docs/PRODUCTION_SPRINT_LESSONS.md` (incl. lesson 24, the lost-anchor class)
- `docs/PROD_BUG_LOG.md` / `docs/BUG_BIBLE_PROMOTION_QUEUE.md`
- `docs/HANDOFF_LOG.md` (all completed-work history, newest at top)
- `docs/2026-07-15-720-bakeoff-verdict.md` (KEEP/IMPROVE/LEAVE + open items)
- `docs/2026-07-12-user-source-lanes-architecture.md` (extensibility successor)
- `docs/2026-07-10-lean-mean-rip-final.md` (drift-check header 2026-07-15)
- `docs/2026-07-12-randomizer-rolls-r2-coding-plan.md`
- `docs/2026-07-12-dynamic-story-visual-scope.md`
- `docs/2026-07-10-llm-first-story-edit-pass.md` (X1-X4 live remainder)
- `docs/2026-07-11-announcer-framing-defect.md` (OPEN)
- `docs/2026-07-11-720-bakeoff-kickoff.md` / `docs/2026-07-11-timeline-cue-ledger.md`
- `docs/2026-07-11-sfx-engine-architecture/roundtable/pass04_final.md` (local, gitignored)
- `workflows/otr_canonical.json`
