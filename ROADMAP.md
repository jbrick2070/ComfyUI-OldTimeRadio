# OTR Roadmap

**Updated:** 2026-08-22

**Branch:** `v2.0-alpha`

**Purpose:** ordered release runway after `docs/GO_FORWARD_PLAN.md`

This is the only roadmap. It contains future work only. Completed source-pack,
Google TTS, portability-build, and prior sprint narratives live in git history
and dated evidence documents, not in the forward queue.

## North star

Ship OTR v2 as a friendly, fail-loud ComfyUI pack that generates complete
old-time-radio episodes with coherent stories, distinct voices, story-native
visuals, captions, credits, and a final publishable video across practical local
and cloud hardware paths.

## Ordered release runway

| Order | Campaign | Exit condition |
|---:|---|---|
| 1 | Lean-mean/dead-code campaign | The ordered plan in `docs/LEAN_MEAN_CLEANUP.md` lands in independently green, pushed chunks; protected surfaces and accepted losses match its current matrix. |
| 2 | RunPod + AMD/Mac platform tests | Representative-machine acceptance smokes actually run, not just declared available. |
| 3 | Install path | Clean install/bootstrap/profile smoke/log collection succeeds on representative machines. |
| 4 | Product docs and v2 release | First-render guide, troubleshooting, and README match observed behavior; all release gates are green; operator controls tag/promotion. |

Every executable row in `docs/GO_FORWARD_PLAN.md` precedes row 1; its final
handoff row only points here. Completed, retired, and cut campaigns belong in
git history and dated evidence rather than this table. The multi-GPU
learned-upscale stage is already shipped and is not future work.

## Portability status -- already implemented

The multi-platform workflow, profile, registry, and generated-variant features
are already in place. They are not a future coding campaign.

Release QA still runs representative CPU/no-GPU, NVIDIA, cloud, AMD, and Mac
acceptance smokes where matching hardware or credentials are available. That is
validation time, not planned coding time. A platform smoke creates coding work
only if it proves a real defect that must be root-fixed. Generated workflows
remain under `workflows/variants/` and derive from
`workflows/otr_canonical.json`; the generated `docs/ENGINE_MATRIX.md` (emitted by
`tools/engine_matrix.py`, drift-gated in the suite) makes the already-built
engine/backend support visible to users.

There is no standalone SFX provider layer to revive. Current runtime video
clips are deliberately silent and the terminal mux uses the frozen upstream
master audio. If SFX work resumes, the operator-selected direction is to retain
useful audio produced by video generation and mix it as inexpensive ambience,
not rebuild the retired provider/bed subsystem. That direction is future design
work: dedicated SFX provider technology moves too quickly to justify another
maintenance layer.

## 1. Optional product expansion candidates

These are unscheduled possibilities, not release-runway dependencies:

- a generated-video-audio ambience path: decide which native video audio is
  retained, aligned, trimmed, levelled, and mixed without disturbing dialogue,
  music, ledger timing, or the frozen master-audio contract;
- a non-gating episode content rating, displayed in the opening and credits,
  derived from the final episode and never allowed to reject, reroll, rewrite,
  or block publication;
- a re-check of brief-driven music-cue still prompts, but only if current
  `visual_storybased` output still makes music beats look generic;
- a one-shot continuous multi-speaker segment role, treated as a distinct
  conversation-mode surface rather than forced into the per-line voice engine
  contract. It requires a fresh design for alignment, ledger timing, sidecar
  isolation, speaker limits, and licensing before promotion.

Each candidate needs its own current design, ownership table, tests, and
qualification ladder before it can enter GO_FORWARD.

## 2. Lean-mean campaign

Current scope and coding order: `docs/LEAN_MEAN_CLEANUP.md`.

This campaign starts only after the active queue in
`docs/GO_FORWARD_PLAN.md` is exhausted. ROADMAP owns that scheduling edge; the
cleanup document owns the target matrix, blast radius, dependency order, and
per-chunk verification. Do not copy its kill list back into this file or into
GO_FORWARD.

### Entry gate

Run the operator-pinned full `r2 -> r3 -> r4` arc against the committed HEAD
that will be edited. No deletion lands before r4 converges. If current
reachability or removal loss differs from the cleanup matrix, update the matrix
and the round artifacts before coding.

### Dormant-3D dependency

Before any dormant-3D deletion, move generic unknown-engine rejection to the
post-freeze `OTR_VideoDirector` boundary and push it green. The live
`mesh_stage` lane remains protected.

After the gate, execute the numbered order in `docs/LEAN_MEAN_CLEANUP.md`, one
independently green commit and push at a time. Writer/orchestrator splitting,
widget-schema work, broad script/test diets, and OpenRouter pruning remain late
because they have the largest or not-yet-qualified blast radius.

## 3. RunPod and installation

Build deployment and installation only from proven profiles:

- RunPod bootstrap/start/environment-check/smoke/log collection;
- Windows and Linux installers;
- deterministic model/download and install verification;
- `INSTALL.md`, `FIRST_RENDER.md`, `TROUBLESHOOTING.md`, and machine-profile
  guidance that match the generated workflows and engine matrix.

The install path must answer which workflow to load, which models or keys are
needed, what costs money, what stays local, and how to act on every loud failure.

## 4. README and v2 release

Feature recipes update README when their features land. This section is the
final README accuracy and product-release pass, after the workflows, installers,
and hardware claims are real. Ship v2 only when:

- canonical and generated workflows validate;
- representative local, cloud, and RunPod paths complete tiny episodes;
- source/story/visual lanes have current qualification receipts;
- full repo suite and Bug Bible are green;
- install and first-render docs match observed behavior;
- pending production-proven Bible candidates are resolved or explicitly held;
- the operator has completed the final listen/eyeball and authorizes the tag or
  promotion.

Pushes to `v2.0-alpha` remain normal green-chunk workflow. Tags and promotions
remain operator-gated.

## References

- `docs/GO_FORWARD_PLAN.md`
- `docs/BUG_BIBLE_PROMOTION_QUEUE.md`
- `docs/LEAN_MEAN_CLEANUP.md`
- `workflows/otr_canonical.json`
- `workflows/variants/`
