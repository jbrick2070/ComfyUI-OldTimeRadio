# OTR Go-Forward Plan

**Updated:** 2026-07-12 15:14 PDT

**Branch:** `v2.0-alpha`

**Grounded base:** `42f1af82`, equal to `origin/v2.0-alpha` when this re-plan began

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
- The exact Sci-Fi Codex repair-envelope fix is pushed at `7aa3140e`; its live
  reverify remains open.
- **The sole coder slot is currently held by the Sci-Fi Codex visual-policy
  production fix (`PBUG-20260712-19`).** Its dirty ownership is:
  `docs/PROD_BUG_LOG.md`, `nodes/otr_image_gen_dispatcher.py`,
  `nodes/otr_meta_brief_image_prompt.py`, `nodes/otr_shot_lock.py`,
  `nodes/_otr_video_engines/render_driver.py`, and the corresponding
  image-platform, still-spine, video-platform, and credits tests.
  No next chunk starts until that owner lands or releases the files. The task ID
  is not recorded in the repository.
- Preserve the untracked cue-ledger prompt, Sci-Fi Codex repair plan, and local
  bakeoff logs. They are not owned by this planning rewrite.
- One coder edits code or `workflows/otr_canonical.json` at a time. Read-only
  audits and documentation preparation may run in parallel.

## Current sprint -- stabilize once, then run one final event

A coder-day is one focused engineering day. GPU time is elapsed qualification
time and is listed separately.

| Order | Chunk | Hard completion gate | Estimate |
|---:|---|---|---:|
| 1 | Close active Sci-Fi Codex production fixes | Land the `PBUG-20260712-19` root fix; focused tests, full suite, Bible; canonical 120-word `scifi_codex` rerun reaches `RESULT SUCCESS`, makes no unused visual-authoring LLM call or image objects, and proves ledger, episode, `obs_publish OK`, and final OBS asset. The same rerun supplies the pending live reverify for the shipped `PBUG-20260712-18` repair. | 0.5-1 day + <=1 GPU day |
| 2 | Canonical watchdog support | Canonical runner emits heartbeats; watchdog recognizes canonical `RESULT`; healthy long runs never false-dead; explicit failure and stalled/down-server paths are pinned. This is a harness defect, not a production-admitted PBUG. | 0.5 day |
| 3 | Fable2 C5 consumers | Captions and credits use alias-aware cast lookup; HuMo stale guard uses role/source-family/ShotLock identity. | 0.5-1 day |
| 4 | Rip interstitial **audio** only | Remove synthesis, insertion, timing, and dead tests; retain `music_inter` story/visual semantics. | 0.5-1 day |
| 5 | Codex56 attempt telemetry | Land the code-ready retention, recorder, pure callback seam, scheduler metadata, lane wiring, reader, and 30-word live proof without changing ledger bytes or canonical JSON. | 2-4 days + <=1 GPU day |
| 6 | Root-fix `PBUG-20260712-17` | Reproduce with retained raw/projected/error records; fix the owning model/seam/validator boundary, never retries or a shim; canonical rerun proves closure. If the evidence identifies context fit as the cause, merge this work into item 7 rather than fixing twice. | 0.5-3 days + live proof |
| 7 | Context/cap foundation | One provider-effective cap/count/reservation/must-fit authority feeds preflight, invocation, and receipts; measure base and repair envelopes; no silent truncation and no blind 8192->16384 raise. | 1-3 days |
| 8 | User extensibility -- local LLMs and content packs | Execute the converged W0-W5 plan: `user_packs/` overlay/quarantine, selectable content/story packs inside any existing runnable source lane, replay SHA integrity, `otr_check`, templates/generated schema docs/README recipes, and local causal-LM selection safety. New source lanes remain the expert preflight path. | 4-7 days + <=1 GPU day |
| 9 | Fable2 qualification | 30 words on two local families and one declared cloud lane; same pairings at 120; then one frozen 720 leg with ledger/episode/OBS proof. | 0.5-1 day + 1-3 GPU days |
| 10 | One bakeoff, not two | Resolve/build ratified contender D if still applicable; freeze code; revalidate only receipts that remain current; run one all-bank 720 event and blind verdict. The frozen Fable2 720 leg may serve both item 9 and its bakeoff leg only when SHA, settings, and event evidence are identical. | 1-3 days + 2-5 GPU days |

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
| `PBUG-20260712-19` | Live-proven all-visualizer authoring failure; fix is currently another owner's dirty work | After land + live reverify: overlap check + operator approval, then add to the promotion queue |
| historical `PBUG-20260711-18` | Analysis-only 720-word context risk, historically mislabelled as a PBUG | **Never eligible from static evidence**; keep only as the item-7 engineering risk unless a future live artifact independently admits it |

The active production-fix owner updates `docs/PROD_BUG_LOG.md`. The thin approval
queue remains `docs/BUG_BIBLE_PROMOTION_QUEUE.md`; no plan review or invented
fixture creates a row.

## Validation and handoff law

- Latest fully documented clean baseline: C4b at `95582643` -- full repo
  **7,774 passed, 31 skipped, 1 expected failure**; Bug Bible **17 passed,
  11 skipped, 3 expected failures**. Later pushed commits do not carry a newer
  full-suite receipt in this handoff, so the active Sci-Fi Codex closeout must
  establish the next whole-tree baseline.
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

- The active dirty Sci-Fi Codex owner must land/release before any new coder
  claims the slot.
- `PBUG-20260712-17` can expand item 6 only after telemetry identifies the real
  boundary; retries and cap inflation are forbidden substitutes.
- User extensibility and `dynamic_story` both touch the writer, visual-style
  authority, and canonical workflow. They remain serial and each re-derives the
  live JSON.
- Generated-SFX R4 is local/ignored evidence until the tracked R4.1 refit lands;
  it is not an executable queue.

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
