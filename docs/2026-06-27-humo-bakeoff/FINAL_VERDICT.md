# HuMo bakeoff -- FINAL VERDICT (2026-06-27, operator eyeball)

## Operator verdict (across the full arc)
**14B fp8 wins, 100%.** Ranked: (1) 14B fp8 = final-usable; (2) 1.7B = usable preview only
(rejected for final -- bluer / mushier / framing clutter); (3) 17B-GGUF = rejected (even after
the cfg fix it loses to the 14B on quality AND is ~5x slower AND NVML-pool-bound).

## What was exhausted (no lever left that changes the model landscape)
- Two-stage encoder evict (OTR_BakeoffReclaim): only ~217 MB off the 14B peak.
- Allocator A/B (Step A): the 14B rides ~15.86 GB REAL (NVML stable across configs); torch
  under-reports (--cuda-malloc loads weights outside its stats). Not cache; not reclaimable.
- GGUF weight-floor lever (Step B): NO HuMo-14B GGUF exists (14B = Kijai fp8 repack; official
  HuMo = 1.7B/17B). The only fitting quantized HuMo is the 17B GGUF; its true demand (11.5 GB)
  fits, audio cross-attn works, and the cfg-1.0 "mush" was a config bug (distill-only cfg) --
  fixed at cfg 5.0 (B-R +9.2 ~= 14B). But the operator rejected its quality, and it is ~5x
  slower (20 steps + CFG doubling vs the 14B 6-step distill) and still NVML 15.8 (pool).
=> On THIS hardware + the models that exist today, "the 14B look" and "comfortable VRAM
   headroom" are mutually exclusive.

## The load-bearing empirical fact
The 14B fp8 **rendered EVERY leg with ZERO OOM** on the 16 GB card: single (15996), two-stage
(15779), and the SENTINEL with LTX-AV resident first (15974). It rides ~15.9 GB -- thin
headroom that FAILS the conservative 14.5 GB single-resident gate, but it COMPLETES, including
under cross-engine residency. The original BUG-265 demotion was older code at heavier settings;
at 832x480 / <=49f with the two-stage evict it holds ~15.9 GB and finishes.

## This is now a RISK-TOLERANCE decision, not engineering. Options:
A. **Promote the 14B fp8 and accept thin headroom.** Gets the look the operator wants.
   Mitigations: keep HuMo the SINGLE resident heavy engine (the AS-3 lease already does this),
   bound beat length (<=~49-81f @ 832x480 -- the tested-safe envelope), keep the two-stage
   encoder evict. Risk: an OOM on an unusually long beat or unexpected cross-engine co-residency
   (the sentinel still completed, so the common case is covered). This REVERSES the 16gb_full
   `humo_1.7B` pin -> `humo_14B_169` (the eyeballed wide tier).
B. **Keep humo_1.7B** (safe, but the operator rejected its look) + harden its de-blue as the
   best available SAFE option.
C. **Park HuMo as-is; revisit when a better-fitting model lands** (a distilled/quantized
   14B-class talking-head, or a newer lip-sync model that passes a Blackwell dep probe).

## Promotion is a SEPARATE operator-gated coder task
This window is diagnostic; nothing was promoted. If the operator picks A, a coder window flips
`config/profiles/16gb_full.json` other_beats_visual humo_1.7B -> humo_14B_169 via
`widget_mapping.json` + the workflow node, re-expresses through wrapper_bridge.run_graph,
re-validates, suite + Bug Bible + B7, commits v2.0-alpha. (Per Codex r3: NOT node 92; episode
mode renders from the ShotLock ledger.)
