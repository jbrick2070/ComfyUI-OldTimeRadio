# Kibitz judgment -- four-arm clamped video bench (r2 -> r3 -> r4)

**Doc under review:** `docs/2026-07-31-four-arm-clamped-video-bench-SPEC.md`
**Date:** 2026-07-31. **HEAD at review:** `db0fa304`.
**Arc:** r2 (coding) -> r3 (wiring) -> r4 (convergence). r1 skipped -- the arc
was settled by the kickoff, so r1 would have burned a round.
**Driver / anchor panelist / sole judge:** Claude (Cowork), `--driver claude`.
**Panel seat:** codex `gpt-5.6-sol`, reasoning `high` (verified per round from
`codex_model_selected.txt`; the pin has silently drifted to 5.5 on past arcs).
**Profiles:** `kibitz/references/profiles/comfyui.md` + `.kibitz/comfyui.local.md`.

## The one-seat caveat, stated up front

**Antigravity failed all three rounds** on a Gemini 429
`RESOURCE_EXHAUSTED` quota wall, first detected 12:00:02 local, suggested retry
13:00. Every round is therefore ONE SEAT plus the anchor. The same failure hit
the previous arc's r4.

This weakens the arc in a specific way worth naming: cross-seat disagreement is
the main defence against a confident wrong claim, and there was no second seat to
disagree. The mitigation actually applied was grounding -- every codex claim was
checked against the real Windows files before acceptance, and the arithmetic
claims were recomputed rather than trusted. **A second-seat re-run after the
quota window is available if the operator wants it; the spec's conclusions do not
depend on it, but its confidence would improve.**

Kibitz run artifacts live under `kibitz-runs/2026-07-31-four-arm-video-bench/`,
which is gitignored -- hence this tracked summary.

## Outcome

Codex returned **"no -- not ready to build"** in all three rounds, and after
grounding I agreed each time. The spec was rewritten, not patched.

**Zero misreads across three rounds.** That is unusual, and I am recording it
rather than manufacturing a discard to look balanced. Every claim I checked held
against the tree. The round's value was almost entirely in things the anchor had
missed, plus one finding that refuted a synthesis the anchor itself introduced.

## The findings that changed the design

| # | finding | round | effect on the spec |
|---|---|---|---|
| 1 | **The bench's stock-node graph conflicts with `CLAUDE.md` s0 and `PRODUCTION_SPRINT_LESSONS` s7** ("EVERY API / headless / soak run MUST LOAD this real JSON"). I had missed this entirely. It is not new -- `run_wan_ti2v_bakeoff.py` has loaded an ad-hoc bench graph since 2026-07-08 and shipped -- so an unwritten carve-out exists. | r2, escalated r3, blocking r4 | Became gate **G1** and operator decision **O6**, a BUILD BLOCKER sequenced before any code. A coder window must not resolve it quietly. |
| 2 | **A measurement node pack already exists**: `otr_bakeoff_helper` registers `OTR_BakeoffVramReset` (LATENT passthrough, `reset_peak_memory_stats`, always-dirty), `OTR_BakeoffVramProbe` (IMAGE passthrough, `max_memory_allocated` + `max_memory_reserved`), and `OTR_BakeoffReclaim`, all printing to the server log the harness already parses. | r2 | My proposed `OTR_VramStageMark` was reinventing an installed, better-designed wheel. Dropped. |
| 3 | **...but that pack is in no git repository** and sits outside the OTR repo, so a pushed harness would depend on unshipped local code. | r3 | Became gate **G3**: vendor it into the repo as a tracked bench-only package. |
| 4 | **Forcing marker order changes the thing being measured.** The conditioning / image-latent / model branches are independent until `KSampler` (Bug Bible **BUG-05.05**), so stage attribution needs forced ordering -- but forcing it changes ComfyUI's loader schedule, so the peak is no longer the peak we care about. | r3 -- **refuting the anchor's own r2 synthesis** | The estimator-refit claim was **withdrawn** (spec 9.1). The bench answers "which engine" only; production calibration needs the real adapter path instrumented. |
| 5 | **`--reserve-vram` is not the whole clamp.** `minimum_inference_memory() = 0.8 GiB + EXTRA_RESERVED_VRAM`, and `maximum_vram_for_weights() = total*0.88 - minimum_inference_memory()`. `--reserve-vram N` REPLACES the Windows default rather than adding to it. | r2 (existence), r3 (exact formula) | My "7.92 GiB of loader headroom" was **wrong**. Recomputed: this box under `--reserve-vram 8` allows **5.21 GiB** of weights; a real 8 GB card allows **5.65 GiB**; exact emulation would be **R = 7.56 GiB**. Keeping 8 is now a justified conservative choice (~0.44 GiB stricter) rather than an unexamined literal. |
| 6 | **`-1` passes the greenlight bar.** `PeakSampler` records `-1` when NVML fails, and `-1 <= 7168` is TRUE, so a telemetry failure would have silently PASSED an arm. | r2 | A correctness bug in my bar, not a robustness nit. Fail-closed telemetry is now a pass condition. |
| 7 | **Production removes the UNet from the graph entirely.** `prepare()` hoists it and `_build_graph` pops `external_results` nodes, then runs `free_after_use=True`. The bench does neither. | r2, unfixed-in-r3 caught by r3 | "Mirrors production 1:1" corrected to "a structural surrogate for the UNPREPARED graph through decode"; `adapter_hoist`/`free_after_use`/encode path stamped per cell; production-lifetime claims prohibited. |
| 8 | **A vs B is not a one-variable comparison** -- it changes UNet AND encoder packaging together, against `PRODUCTION_SPRINT_LESSONS` s8. | r2 | The best design finding of the arc. The progression became **A -> B-partial -> B**, and B-partial was promoted from optional fallback to **MANDATORY** -- it is the only cell isolating the encoder bundle. |
| 9 | **Arm D has no submit-ready graph.** `eng_ltx_8gb._build_graph` emits abstract aliases, not an API prompt; `otr_ltx_av_q_bakeoff_distilled_native.json` is litegraph format and the LTX-AV family. | r2, r3, r4 | Arm D **BLOCKED**, with no `ArmSpec` branch written -- a blocked arm must not add dead branching. |
| 10 | **The mandatory matrix and repeat law were lost between r2 and r3** -- I accepted them and then dropped them from my own decision list. | r4 | Re-instated as spec 7.3 / 7.4, with an explicit denominator rule and a defined winner ranking. |
| 11 | Latent harness bugs surfaced incidentally: hard-coded port 8000 vs the launcher's `OTR_HEADLESS_PORT`; `COMFYUI_OUTPUT` vs the launcher's hard-coded output root; global `STEPS=30` corrupting s/it for an 8-step recipe; a bare `offload` spill regex that fires on healthy runs; glob-based asset validation that can select a stale mp4; `total_vram_gb()` silently returning a fictional 16.0. | r3, r4 | Gate **G4** plus harness delta items D6, D9, D10, D11, D14. |

## What the anchor contributed that the panel did not

- The stage-order problem's existence (A3), independently, before r3 escalated it.
- The B-partial prediction being written down in advance, which is what makes it
  a test rather than a fishing trip.
- The observation that the bench bypasses `compute_real_frame_budget` entirely,
  so **no diagnostic guard override is needed** -- the report's "the present
  guard blocks calibration" is true on physical 8 GB hardware and through the
  adapter, but not for this bench.
- The resolution in spec 9.3: the `Wan22ImageToVideoLatent -> reset -> KSampler
  -> VAEDecodeTiled -> probe` chain is a strict data dependency, so the
  **sample+decode segment is measurable deterministically with no forced ordering
  and no new nodes.** Codex's r4 recommendation was to cut per-stage measurement
  from the first build; that would have traded away an operator hard requirement.
  Splitting it into the order-safe part (ships) and the perturbing part
  (operator decision **O7**) keeps the requirement intact and honest.

## Where I did not follow the panel

**r4 CUT 1 recommended cutting per-stage torch measurement from the first
build.** Declined as stated, because per-stage `max_memory_allocated` with
`reset_peak_memory_stats` between stages was an explicit operator hard
requirement, and a panel is not authority to drop one. It is instead surfaced as
**O7** with the technical constraint explained, for the operator to rule. The
order-safe half ships regardless.

**r4 SHOULD-FIX 1 recommended renaming away from "four-arm."** Partially taken:
the title keeps the operator's commissioned name -- the doc does specify all four
arms and their status -- but gains an explicit locked-scope line so no builder
reads it as four buildable arms.

## Spend

$0.00. Local seats only (codex CLI + antigravity CLI), no OpenRouter, no Fable,
no GPU. Antigravity contributed nothing due to the quota wall.
