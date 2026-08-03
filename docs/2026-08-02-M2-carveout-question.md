# Does the M2 VRAM ladder count as evidence, or only as a hint?

**A narrow governance question. 2026-08-02, HEAD `eb7ca9f7`.**
The operator asked for this to be broken down plainly and second-opinioned.

## The rule

`CLAUDE.md` section 0: EVERY API / headless / soak run MUST LOAD
`workflows/otr_canonical.json`. The stated reason is that code not wired into
that file is dead, and measuring through a different graph can prove something
production does not do.

`CLAUDE.md` section 0A (operator ruling 2026-07-31, decision O6, labelled
**NARROW**) is the only exemption. It names exactly two runners --
`scripts/run_video_arm_bakeoff.py` and `scripts/run_wan_ti2v_bakeoff.py` -- and
only the API graphs under `scripts/bench_graphs/`, each pinned by SHA-256. It
says in terms: the exemption covers MEASUREMENT ONLY, no production render, no
soak, no published episode, no engine/profile/tier change may ride a bench
graph, and any behaviour a bench discovers must be re-proved through the
canonical workflow before it ships.

## What was actually run

`scripts/otr_humo_vram_ladder.py` (new) drives
`scripts/_otr_single_engine_smoke.py` (pre-existing), which POSTs a ONE-NODE
`OTR_VideoRenderBatch` graph with `mode=single` to the live headless server.

The distinction that matters, and it cuts both ways:

* **Argues FOR counting it.** That node calls
  `render_driver.render_single()`, which walks the REAL production dispatch --
  `assert_usable` -> `prepare` -> `render_clip` -> `canonicalize` -> `teardown`.
  `VramPeakProbe` fires exactly where it fires in production. This is a
  materially stronger path than the stock-node bench graphs section 0A ALREADY
  permits: those bypass OTR's adapters entirely, this one is OTR's adapter.
* **Argues AGAINST counting it.** It is still not `otr_canonical.json`, it is
  not named in 0A, and 0A was labelled NARROW on purpose. A one-node graph also
  omits everything the canonical graph does around the render -- writer, audio,
  compositing, publish -- along with episode scheduling, repeated beats, real
  `host_caps`/profile policy inputs, and multi-segment session lifetime.

  **CORRECTION (kibitz r1):** an earlier draft justified this by saying
  cross-phase residency is "the mechanism the LTX measurement concluded drives
  peak". That is stale and backwards.
  `docs/2026-08-02-MEASUREMENT-ltx-av-vram-vs-frames.md:99-113` explicitly
  REFUTES cross-phase residue -- the breached campaign logged successful
  pre-render cleanup and started from nearly identical free VRAM -- and leaves
  same-engine residency, allocator fragmentation and cleanup drift unresolved.
  The argument for canonical proof is that it exercises episode scheduling,
  repeated beats, policy inputs and session lifetime; **not** that it exposes a
  mechanism that document rejected.

## The result at stake

`docs/2026-08-02-MEASUREMENT-M2-humo-vram-ladder.md`: across 16 cells, HuMo peak
VRAM DECLINES monotonically as frame count rises (about -1 GB from 49 to 97
frames, in all four independent series), and the two orientations match to
within measured repeatability. The practical readings are that the 97-frame cap
bounds quality rather than memory, and that coverage splitting should prefer
LONGER segments because peak is per-segment.

Nothing has been changed on the strength of it. The document is marked
MEASUREMENT ONLY.

## The three options

1. **MEASUREMENT ONLY (status quo).** The numbers inform thinking; no cap, tier
   or profile moves until re-proved through the canonical workflow. Costs
   nothing, decides nothing.
2. **EXTEND the 0A carve-out** to name this runner, on the argument that driving
   the real adapter is stricter than the stock-node graphs already exempted.
   Cheapest path to acting on the finding; widens a rule the operator
   deliberately narrowed.
3. **RE-PROVE through `otr_canonical.json`.** About an hour of GPU for a
   reduced ladder. Unambiguous, and also the only option that exposes
   cross-phase residency.

## What the panel is asked

- Which option, and what is the strongest argument AGAINST it?
- Is "drives the real adapter" genuinely stronger than the stock-node bench
  graphs 0A already allows, or is that reasoning smuggling in an exemption by
  analogy?
- Does the one-node graph's omission of writer/audio/composite phases
  invalidate the specific claims made -- peak falls with frames, orientations
  match -- or only limit their scope?
- Is there a cheaper fourth option that gets canonical-grade confidence without
  a full re-run?
