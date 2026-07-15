# OTR Go-Forward Plan

**Updated:** 2026-07-15 night -- HEAD c28af5f4 (v2.0-alpha)

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
1. **Fable final gate** on the shipped structural bake-off change (v3 promotions +
   source-snapshot) -- HELD for operator go (operator-gated spend, CLAUDE.md
   section 9). kibitz r4 already converged with no residual code MUST-FIX.
2. **Live replay proof (render window):** populate a manifest for one base bank,
   run its base/_v2/_v3 triplet under `OTR_C7=1`, and confirm the pack is the only
   variable (F2). ACCEPTANCE (Codex r4): the server log shows BOTH the
   source-snapshot manifest echo AND the C7 seed echo, and ledger meta shows
   `cast_seed_source == "OTR_CAST_SEED override"`; record payload sha + seed per row.
3. **Final verify** (code done): dry registry-load 24 runnable / 25 visible;
   `git diff --exit-code otr_canonical.json` clean; JSON round-trip 23 nodes / 57
   links; OTR_WorkflowValidator covered by the green suite.

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

### Done today (all pushed to `v2.0-alpha`)

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

### Next

Remaining live proof: `scifi_sonnet` (in flight), `scifi_fable2`, `scifi_codex`,
`original_radio`, and the four legacy banks -- 30w, then 120w. Then the frozen
bakeoff.

---

**Previously updated:** 2026-07-12 21:04 PDT

**Branch:** `v2.0-alpha`

**Grounded base:** `7c806eec`, equal to `origin/v2.0-alpha` when the active
P3 transport replacement began

**Scope:** the active qualification/release-prep sprint and the immediately
following feature sprint only

**Longer runway:** `ROADMAP.md`

This is the only active go-forward file. Dated plans are implementation evidence,
not competing queues. Git history owns completed sprint narratives.

## Ordered coding path

```text
active Sci-Fi Codex fix
  -> watchdog -> C5 -> interstitial-audio rip
  -> Codex56 telemetry -> PBUG-17 root fix -> context/cap authority
  -> user LLM/content-pack extensibility
  -> Fable2 qualification -> one frozen bakeoff
  -> Engine Matrix -> Randomizer A -> dynamic_story
  -> ROADMAP
```

The important re-prioritization is deliberate: **user extensibility lands before
final qualification and the bakeoff**, not after them. It changes story-pack
resolution and validation, writer execution, ledger replay stamps, and canonical
JSON. Landing it after the event would stale relevant receipts and risk creating
the second bakeoff this plan forbids.

## Coordination and ownership

- C4b is complete and pushed at `95582643`. Its full receipt remains in git
  history; it is no longer a queue item.
- The filtered OpenRouter catalog work is pushed at `961e095a`; the old dirty
  model-catalog warning is retired.
- The exact Sci-Fi Codex repair-envelope fix is pushed at `7aa3140e`; the
  compact P3 repair-context follow-up is pushed at `915c8314`. Both await live
  reverify with the same canonical `scifi_codex` leg.
- The all-visualizer effective-consumer root fix (`PBUG-20260712-19`) is pushed
  at `496689da`; the shared P0 bounded-evidence repair (`PBUG-20260712-21`) is
  pushed at `e1d3a035`. **The sole coder slot is currently held by the live
  Sci-Fi Codex qualification follow-on (`PBUG-20260712-24`).** The bounded
  direct-score correction failed again at live prompt
  `edbbac48-9aa8-4907-8086-f63134604604`; root commit `c942b2ae` replaces only
  P3 and P3-rewrite model transport with `RadioScoreDraftV4` plus a fail-closed
  compiler returning the same final `RadioScoreV4`. It is pushed and equals
  origin after focused/full/Bible/workflow gates. Its reverify reached a new
  shared P0 failure at prompt `81e0b0c9-2f20-4085-9fd0-e7f8034f75da`: an exact
  literal quote exceeded the finite cap, the generic clamp retained stale
  coordinates, and the raw metadata repair could not accept it. The active
  root fix moves only that proven literal metadata case into the shared Sci-Fi
  P0 repair and disables generic clamp at all three P0 boundaries. That rerun
  live-proved P0, then exposed P3's compact contract omitting nested literal
  semantics: numeric `arc_phase` values and descriptive cue names survived its
  typed repair. The active root fix restores those exact model-visible rules at
  the compact P3/P3-rewrite seam without moving authorship to Python; canonical
  live reverify remains the next action before the coder slot releases. It
  preserves the canonical workflow and shared all-visualizer no-image gate.
  The full four-round Kibitz campaign (8 local reviews) and four-round frontier
  panel ($0.8618 actual) are recorded under the dated transport-recovery artifacts.
  The RSS parity audit found no safe schema copy/paste: Gemini, Sonnet,
  Fable2, Media Archive, and original lanes remain artifact-specific follow-ups.
- Preserve the untracked cue-ledger prompt, Sci-Fi Codex repair plan, and local
  bakeoff logs. They are not owned by this planning rewrite.
- One coder edits code or `workflows/otr_canonical.json` at a time. Read-only
  audits and documentation preparation may run in parallel.

## Current sprint -- stabilize once, then run one final event

A coder-day is one focused engineering day. GPU time is elapsed qualification
time and is listed separately.

| Order | Chunk | Hard completion gate | Estimate |
|---:|---|---|---:|
| 1 | Close active Sci-Fi Codex production fixes | Land the `PBUG-20260712-22` compact P3 draft/compiler root fix, `PBUG-20260712-23` exact-oversized-source-span root fix, and `PBUG-20260712-24` compact nested-literal-contract root fix; focused tests, full suite, Bible, canonical validator/audits, commit/push; canonical 120-word `scifi_codex` rerun reaches `RESULT SUCCESS`, makes no visual-authoring LLM call or image objects under all-visualizer policy, and proves ledger, episode, `obs_publish OK`, and final OBS asset. The same rerun supplies pending live reverifies for shipped PBUGs 18-24. | 0.5-1 day + <=1 GPU day |
| 2 | Canonical watchdog support | Canonical runner emits heartbeats; watchdog recognizes canonical `RESULT`; healthy long runs never false-dead; explicit failure and stalled/down-server paths are pinned. This is a harness defect, not a production-admitted PBUG. | 0.5 day |
| 3 | Fable2 C5 consumers | Captions and credits use alias-aware cast lookup; HuMo stale guard uses role/source-family/ShotLock identity. | 0.5-1 day |
| 4 | Rip interstitial **audio** only | Remove synthesis, insertion, timing, and dead tests; retain `music_inter` story/visual semantics. | 0.5-1 day |
| 5 | Codex56 attempt telemetry | Land the code-ready retention, recorder, pure callback seam, scheduler metadata, lane wiring, reader, and 30-word live proof without changing ledger bytes or canonical JSON. | 2-4 days + <=1 GPU day |
| 6 | Root-fix `PBUG-20260712-17` | Reproduce with retained raw/projected/error records; fix the owning model/seam/validator boundary, never retries or a shim; canonical rerun proves closure. If the evidence identifies context fit as the cause, merge this work into item 7 rather than fixing twice. | 0.5-3 days + live proof |
| 7 | Context/cap foundation | One provider-effective cap/count/reservation/must-fit authority feeds preflight, invocation, and receipts; measure base and repair envelopes; no silent truncation and no blind 8192->16384 raise. | 1-3 days |
| 8 | User extensibility -- local LLMs and content packs | Execute the converged W0-W5 plan: `user_packs/` overlay/quarantine, selectable content/story packs inside any existing runnable source lane, replay SHA integrity, `otr_check`, templates/generated schema docs/README recipes, and local causal-LM selection safety. New source lanes remain the expert preflight path. | 4-7 days + <=1 GPU day |
| 9 | Fable2 qualification | 30 words on two local families and one declared cloud lane; same pairings at 120; then one frozen 720 leg with ledger/episode/OBS proof. | 0.5-1 day + 1-3 GPU days |
| 10 | Fresh two-matrix bakeoff | After item 1 commits/pushes and all ten 120-word canonical legs are individually green with receipt/ledger/OBS proof, update the durable report and World Cup scoreboard. Then run all ten 420-word legs sequentially with OpenRouter creative `tencent/hy3:free` and local technical Mistral-Nemo; fail loud if HY3 is unavailable. No 320- or 720-word legs. | 2-5 GPU days |

**Current-sprint planning range:** about **11-25 coder-days + 4-11 elapsed
GPU days**. The top of the range covers the still-unknown `PBUG-20260712-17`
root cause; it is not permission to bypass that gate.

### Current-sprint plans of record

- Watchdog: `scripts/otr_canonical_api_run.py` and
  `scripts/otr_render_watchdog.ps1`; add focused heartbeat/result/failure/stall
  tests before any new long run.
- C5 and audio rip: preserve story/visual interstitial cues while removing only
  the banned audio path.
- Telemetry: `docs/2026-07-12-codex56sol-llm-telemetry-plan.md`; follow its
  internal pushed-green-chunk order. It precedes context/cap changes so the
  failing PBUG-17 boundary is captured before a foundation change can mask it;
  later attempts automatically record the new effective cap.
- User extensibility:
  `docs/2026-07-12-vibe-coder-extensibility-r2-coding-plan.md` at `97d4f9eb`.
  Claim clean-or-released ownership for every activation surface before W0.
  Execute `W0 -> W1 -> W2 -> W4`, the independently gated W3 branch, then W5.
  W1 is the one canonical-JSON activation change; re-derive live input/widget
  positions and prove the headless harness resolves by widget name/schema rather
  than a hard-coded index.
- Qualification/event: `docs/2026-07-10-fable2-720-bakeoff-runway.md` and
  `docs/2026-07-11-720-bakeoff-kickoff.md`. Every live leg loads
  `workflows/otr_canonical.json` and proves the canonical asset paths.

## Immediate next sprint -- expose, randomize, then direct

1. **Generate `docs/ENGINE_MATRIX.md`.** Extend `scripts/build_variants.py` so
   `--check` emits and diffs the matrix from all three live CAPABILITIES
   registries; link it from the already-rebased README. Estimate: **0.5-1 day**.
2. **Randomizer Rolls Design A.** Establish `_otr_lane_specs` as the one
   dispatch/compatibility authority, then add the source-bank roll and three
   canonical 30-word proofs. `banks.json` and `_otr_story_routing.py` remain
   unchanged by the ratified plan; re-ground writer/import assumptions after
   extensibility. Estimate: **1-2 days + 1 GPU day**.
3. **`dynamic_story` visual direction.** The rev-5 design is FINAL; do not rerun
   R2-R4. Capture named-pack byte baselines, then land the node, shared context
   and receipt helpers, all consumers, registration, tests, canonical node/link
   delta, and regenerated variants as one serial activation. Re-derive every
   canonical ID/slot from the then-current workflow after extensibility. Estimate:
   **5-9 coder-days + 2-4 GPU days**.

**Immediate-next range:** about **7-12 coder-days + 3-5 elapsed GPU days**.

**Combined GO_FORWARD range:** about **18-37 coder-days + 7-16 elapsed GPU
days**. Adding the non-optional ROADMAP runway yields roughly **46-83
coder-days** from the current dirty fix through core v2 release work; optional
product-expansion campaigns are excluded.

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
| `PBUG-20260712-17` | Live failure; root cause open until telemetry captures it | **Not eligible** -- no proved reusable law yet |
| `PBUG-20260712-18` | Live-proven exact repair-envelope failure; fix pushed at `7aa3140e`; live reverify pending | After reverify: overlap check + operator approval, then add to the promotion queue |
| `PBUG-20260712-19` | Live-proven all-visualizer authoring failure; fix pushed at `496689da`, live reverify pending | After live reverify: overlap check + operator approval, then add to the promotion queue |
| `PBUG-20260712-20` / `21` / `22` | Live-proven P3/P0 structured-capacity failures; fixes use compact repair context and bounded artifact surfaces | Already generalized into BUG-11.50; live reverify remains required |
| historical `PBUG-20260711-18` | Analysis-only 720-word context risk, historically mislabelled as a PBUG | **Never eligible from static evidence**; keep only as the item-7 engineering risk unless a future live artifact independently admits it |

The active production-fix owner updates `docs/PROD_BUG_LOG.md`. The thin approval
queue remains `docs/BUG_BIBLE_PROMOTION_QUEUE.md`; no plan review or invented
fixture creates a row.

## Validation and handoff law

- Active P3 transport receipt: focused suite **139 passed, 1 skipped**;
  final whole-tree rerun **7,807 passed, 31 skipped, 1 xfailed**. Bug Bible
  **17 passed, 11 skipped, 3 xfailed**. The canonical validator, JSON round-
  trip, live widget-vector, link, and input audits report **23 nodes / 57
  links** with no canonical workflow delta.
- Every code chunk: focused tests, full Windows suite, Bug Bible, AST/JSON/BOM/
  zero-byte checks, commit, push, and verify `HEAD == origin/v2.0-alpha`.
- Every node/widget/link/schema change edits `workflows/otr_canonical.json` in
  the same commit and runs `OTR_WorkflowValidator`, JSON round-trip, strict
  link/input, live widget-vector, and generated-variant audits.
- Reset selectively before every headless run; never blanket-kill Python. Every
  run loads the canonical workflow and writes directly to canonical episode/OBS
  paths. Asset existence, not resident VRAM, proves completion.
- Multi-platform engines, profiles, registries, and generated variants are
  already implemented. Representative platform smokes are release acceptance,
  not a coding campaign unless they prove a real defect.

## Open risks

- The active dirty Sci-Fi Codex P3 owner must land/release before any new coder
  claims the slot.
- `PBUG-20260712-17` can expand item 6 only after telemetry identifies the real
  boundary; retries and cap inflation are forbidden substitutes.
- User extensibility and `dynamic_story` both touch the writer, visual-style
  authority, and canonical workflow. They remain serial and each re-derives the
  live JSON.
- Generated-SFX R4 is local/ignored evidence until the tracked R4.1 refit lands;
  it is not an executable queue.

## Operator backlog (2026-07-14 night) -- render tuning, not bugs

Two operator asks captured mid-bake-off. Both are audio/video RENDER tuning (SFW, no
word-count involvement, no story-text rewrite). File pointers located this session.

- **Kokoro TTS spells out ALL-CAPS.** Symptom: Kokoro reads ALL-CAPS tokens
  letter-by-letter (as initialisms), so acronyms / emphasis-caps in the script get
  spelled aloud in narration. Fix direction: a pre-TTS text-normalization pass that
  Title-cases (or lower-cases) ALL-CAPS words >=2 letters BEFORE they reach the engine,
  with an allowlist for genuine acronyms that SHOULD be spelled. Normalize a TTS-only
  COPY of the line -- never the ledger `spoken_text` of record or the captions (keeps
  "no Python rewrite of story text": this is render normalization, not a script edit).
  Where: `nodes/_otr_audio_engines/eng_kokoro.py` (engine); the shared voice seam
  `nodes/_otr_voice_node_common.py` is the likely normalization site. FIRST confirm
  which engine is the canonical default (memory says indextts2 is default; Kokoro may be
  an alt lane) -- fix the engine actually in `workflows/otr_canonical.json`.
- **Ending credits ~50% faster.** Increase the credits-roll scroll rate by ~1.5x (or cut
  the roll duration to ~0.667x). Where: `nodes/otr_credits_roll.py`. If the rate is a
  node widget, the change lands IN `workflows/otr_canonical.json` in the SAME commit
  (CLAUDE.md section 0 -- unwired code is dead).

## Pointers

- `ROADMAP.md`
- `docs/PRODUCTION_SPRINT_LESSONS.md`
- `docs/PROD_BUG_LOG.md`
- `docs/BUG_BIBLE_PROMOTION_QUEUE.md`
- `docs/2026-07-12-codex56sol-llm-telemetry-plan.md`
- `docs/2026-07-12-vibe-coder-extensibility-r2-coding-plan.md`
- `docs/2026-07-12-randomizer-rolls-r2-coding-plan.md`
- `docs/2026-07-12-dynamic-story-visual-scope.md`
- `docs/2026-07-11-720-bakeoff-kickoff.md`
- `docs/2026-07-11-timeline-cue-ledger.md`
- `docs/2026-07-11-sfx-engine-architecture/roundtable/pass04_final.md`
- `workflows/otr_canonical.json`
