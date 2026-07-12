# OTR Roadmap

**Updated:** 2026-07-12

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

A coder-day is one focused engineering day. Live renders, provider waits, and
operator listening are separate elapsed time.

| Order | Campaign | Exit condition | Coding estimate |
|---:|---|---|---:|
| 1 | `dynamic_story` visual direction | Current story becomes a typed post-freeze `vd-1` direction; canonical node/link activation; variants and 30/120 ladder green | 5-9 days |
| 2 | Lean-mean/dead-code campaign | Re-grounded deletion/consolidation waves land green; no dormant interstitial audio or duplicate authorities | 12-16 days |
| 3 | Product expansion | Visual-pack roll, richer cue stills, cue-ledger/SFX work, provider additions, system-agnostic upscale as separately gated campaigns | 5-10 days per selected set |
| 4 | RunPod and install path | Clean install/bootstrap/profile smoke/log collection on representative machines | 6-10 days |
| 5 | Product docs and v2 release | First-render guide, troubleshooting, accurate README, all release gates green; operator controls tag/promotion | 2-5 days |

**Release-runway planning range after GO_FORWARD:** roughly **25-40 coder-days**
before optional product-expansion campaigns, or about **5-8 one-coder weeks**
with normal integration margin. Hardware/provider qualification can extend
calendar time without consuming equivalent coding time.

## Portability status -- already implemented

The multi-platform workflow, profile, registry, and generated-variant features
are already in place. They are not a future coding campaign.

Release QA still runs representative CPU/no-GPU, NVIDIA, cloud, AMD, and Mac
acceptance smokes where matching hardware or credentials are available. That is
validation time, not planned coding time. A platform smoke creates coding work
only if it proves a real defect that must be root-fixed. Generated workflows
remain under `workflows/variants/` and derive from
`workflows/otr_canonical.json`; the planned `docs/ENGINE_MATRIX.md` makes the
already-built engine/backend support visible to users.

## 1. Story-derived visual direction

Plan of record: `docs/2026-07-12-dynamic-story-visual-scope.md`.

Build only after the GO_FORWARD context and telemetry foundations plus
Randomizer Design A have landed. Re-run its VERIFY-AT-BUILD checklist against the
then-current code. Preserve named-style byte identity, capture pre-feature
baselines first, and activate node 96/link 284 atomically with code, tests,
canonical JSON, and generated variants.

The current design name is `dynamic_story`; old `llm_creative` wording is
retired. Dynamic direction remains downstream of the frozen story and may never
feed visual taste back into story authorship.

## 2. Lean-mean campaign

Plan of record: `docs/2026-07-10-lean-mean-rip-final.md`, but its line pins are
stale and must be re-grounded once after feature surfaces stop moving.

Execute the ratified dependency order, one pushed green chunk at a time:

`W0 -> W1 -> W2 -> W3 -> W4a -> W4b -> W7 -> W6 -> W5+SW4 -> C1-C5 -> SW1/SW2/SW3 -> C6 -> C7 -> W8`

Required edges:

- W1 before SW3;
- W2 before C2;
- W2-W4 before W7;
- generated ENGINE_MATRIX before registry deletion/consolidation;
- cloud smoke before W8;
- Randomizer and `dynamic_story` before the writer/widget structural split.

Do not re-delete the old context helper already removed at `1a6ae8f1`, and mark
the earlier standalone interstitial-audio rip as satisfied when re-grounding.

## 3. Product expansion candidates

Choose these as separate campaigns after the core surface is stable:

- Randomizer Design B: visual-pack roll, only after `dynamic_story` establishes
  one visual-style authority;
- richer brief-driven music-cue still prompts;
- ASR-anchored cue ledger and later SFX spotting;
- direct BYO Google music/image/video paths, followed by other providers only
  when they keep provider identity explicit and fail loud;
- a new system-agnostic multi-GPU upscale stage built against profile and
  registry contracts, never resurrection of the retired NVIDIA-only node.

Each candidate gets its own scoped design, exact ownership table, tests, and
qualification ladder before entering GO_FORWARD.

## 4. RunPod and installation

Build deployment and installation only from proven profiles:

- RunPod bootstrap/start/environment-check/smoke/log collection;
- Windows and Linux installers;
- deterministic model/download and install verification;
- `INSTALL.md`, `FIRST_RENDER.md`, `TROUBLESHOOTING.md`, and machine-profile
  guidance that match the generated workflows and engine matrix.

The install path must answer which workflow to load, which models or keys are
needed, what costs money, what stays local, and how to act on every loud failure.

## 5. README and v2 release

README is the final product pass, after the workflows, installers, and hardware
claims are real. Ship v2 only when:

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
- `docs/2026-07-12-dynamic-story-visual-scope.md`
- `docs/2026-07-10-lean-mean-rip-final.md`
- `workflows/otr_canonical.json`
- `workflows/variants/`
