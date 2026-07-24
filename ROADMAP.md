# OTR Roadmap

**Updated:** 2026-07-22

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
| 1 | Lean-mean/dead-code campaign | Re-grounded deletion/consolidation waves land green; no dormant interstitial audio or duplicate authorities | 12-16 days |
| 2 | Timeline Cue Ledger / generated spotted SFX | `cue-1` survives its blind noun-detector gate; generated-SFX selector, renderer, timeline mix, canonical wiring, receipts, and byproduct-bed retirement are green | provisionally 8-15 days after the R4.1 refit |
| 3 | Other product expansion | Visual-pack roll, richer cue stills, provider additions, and system-agnostic upscale as separately gated campaigns | 5-10 days per selected set |
| 4 | RunPod and install path | Clean install/bootstrap/profile smoke/log collection on representative machines | 6-10 days |
| 5 | Product docs and v2 release | First-render guide, troubleshooting, accurate README, all release gates green; operator controls tag/promotion | 2-5 days |

**Release-runway planning range after GO_FORWARD:** roughly **28-46 coder-days**
before optional product-expansion campaigns, or about **6-9 one-coder weeks**
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

`dynamic_story` is owned by the immediate next sprint in
`docs/GO_FORWARD_PLAN.md`; it is not duplicated in this longer-range queue.

## 1. Lean-mean campaign

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
- user extensibility, Randomizer, and `dynamic_story` before the writer/widget
  structural split, preserving overlay quarantine, pack replay hashes, and the
  final live widget layout.

Do not re-delete the old context helper already removed at `1a6ae8f1`, and mark
the earlier standalone interstitial-audio rip as satisfied when re-grounding.

## 2. Timeline Cue Ledger / generated spotted SFX

Foundation plan: `docs/2026-07-11-timeline-cue-ledger.md`.

Generated-SFX architecture evidence:
`docs/2026-07-11-sfx-engine-architecture/roundtable/pass04_final.md`.

**Status:** one roadmap campaign; no code; not yet code-ready. Implementation is
parked until the 720-word runway and lean-mean campaign land. The immediate
GO_FORWARD work is only a tracked current-HEAD R4.1 refit.

The two dated designs are not separate implementation queues. Timeline Cue
Ledger owns C0/C1 and the blind usefulness gate. If that gate passes, the
generated-SFX design owns the renderer/mix/canonical continuation and
**supersedes the older curated-CC0-library renderer; no library fallback
survives**. SFX remains a derived cue artifact after spoken performance exists;
never resurrect `speaker_role="sfx"`, pseudo-dialogue cues, or the retired
pre-render subsystem.

Planned sequence:

1. **C0:** persist per-line WAV stems and ship the transcript drift report.
2. **C1:** define `cue-1`, build the timestamp-blind intent/placement passes,
   and run the budget-matched blind noun-detector control. Indistinguishable
   results stop the campaign before any audio engine work.
3. **R4.1/C2:** after a passing gate, ratify the current generated-SFX plan:
   ungated static selection between Stable Audio 3 Small-SFX and Medium;
   selected-profile hard failures with no fallback; ledger-bound semantic cue
   authoring; complete request/attempt/commit receipts; onset-exact generated
   stems written directly to canonical episode paths.
4. **C3a-C3c:** mix inside SceneSequencer at the raw scene-audio seam, preserve
   room tone, rebase/persist final ledger timing before `audio_done`, retire the
   accidental Google-video byproduct bed and any double-mix path, and wire the
   canonical workflow in the same activation commit. There is no post-video or
   Whisper/alignment lane.
5. **C4 later:** add VFX rows only after the SFX lane is green; VFX genuinely
   needs a post-video picture-aware pass.

The earlier C0-C3c estimate remains a provisional **8-15 coder-days + 2-4
elapsed live-evaluation days** until R4.1 re-grounds the generated renderer's
full ownership and receipt surface. R4.1 must replace this range if the grounded
scope differs. C0 is technically independently shippable, but remains parked
with the campaign unless the operator explicitly promotes it earlier.

## 3. Product expansion candidates

Choose these as separate campaigns after the core surface is stable:

- a non-gating episode content rating on the opening sequence and repeated in
  the credits, derived from the final canonical episode on a G-through-XXX
  scale; a future design may combine transcript/audio analysis with video-frame
  moderation, but the rating is advisory display metadata only and must never
  reject, reroll, rewrite, or block publication;
- Randomizer Design B: visual-pack roll, only after `dynamic_story` establishes
  one visual-style authority;
- richer brief-driven music-cue still prompts;
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
- `docs/2026-07-10-lean-mean-rip-final.md`
- `docs/2026-07-11-timeline-cue-ledger.md`
- `docs/2026-07-11-sfx-engine-architecture/roundtable/pass04_final.md`
- `workflows/otr_canonical.json`
- `workflows/variants/`
