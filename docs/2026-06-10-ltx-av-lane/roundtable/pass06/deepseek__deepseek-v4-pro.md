<!-- requested_model: deepseek/deepseek-v4-pro | resolved_model: deepseek/deepseek-v4-pro-20260423 -->

VERDICT: yes-with-fixes — the plan lacks concrete hardware gate thresholds, encoder phasing discipline, and RAM monitoring specifics; these must be nailed down before the M0 probe can be executed safely.

MUST-FIX BEFORE BUILD:
1. [Point 1 – Lane Decision Gate] The plan does not define the M0 decision table with explicit PASS/FAIL criteria. It also fails to state that Q4_K_M (16.5 GiB ≈ 17.7 GB) cannot fit fully under the 14.5 GB ceiling, leaving only Q3_K_S (13.0 GiB ≈ 13.96 GB) and Q3_K_M (13.7 GiB ≈ 14.71 GB) as realistic full-resident GGUF candidates, with Q3_K_M borderline. Fix: Add a table with columns for L1 (fp8_scaled block-swap), L2 (GGUF per quant), and L3 (NVFP4). For each, measure NVML peak, wall-time/clip at 1472×832×~6s, and quality vs 2B baseline. PASS criteria: NVML peak ≤ 14.5 GB sustained; wall-time ≤ 12 min/clip (proposed); quality ≥ 2B. Explicitly note that Q4_K_M full residency is dead; Q4_K_M tests must use per-layer offloading and measure VRAM peak with offload settings. Q3_K_S and Q3_K_M are the full-resident picks; Q3_K_M may breach 14.5 GB due to runtime overhead and should be flagged as borderline.

2. [Point 2 – Encoder Phasing] The plan does not specify the phase discipline for the 13.2 GB text encoder. It must detail: acquire AS-3 lease → load text encoder (preferably CPU-offloaded via the GGUF Q3 encoder variant, which fits in ~12 GB RAM) → encode prompts → call `reclaim_idle_models` (mirroring BUG-291 in wrapper_bridge.py) to evict the encoder → load the transformer → render → release lease. The lease wraps the entire render. Fix: Add a subsection “Encoder Phasing” that mandates this sequence, defaults v1 to CPU-offloaded text encoding, and references `reclaim_idle_models` and the AS-3 lease from gpu_residency.py.

3. [Point 3 – System RAM] The plan mentions recording RAM size and peak commit but does not name the failure mode or gate. Fix: Require the M0 sheet to record total system RAM and peak commit during render. The gate is the wall-time/clip ceiling: if insufficient RAM causes paging, the wall time will blow up and exceed the PASS bar. Explicitly state that a wall-time > 15 min/clip (see Point 4) indicates RAM paging and renders the lane operationally unusable.

4. [Point 4 – Episode Time Budget] The plan lacks a concrete per-clip PASS bar and episode budget. Fix: Propose a per-clip wall-time PASS bar of ≤ 12 minutes (double the 2B baseline) and an episode budget of ≤ 36 minutes for 3 clips. Define > 15 min/clip as a hard fail (lane unusable). Add these thresholds to the M0 decision table.

5. [Point 7 – L3 NVFP4] The 21.7 GB NVFP4 file exceeds the VRAM ceiling, requires offloading, is DEV-only (not distilled → more steps), and has an open loading-failure report (Comfy issue #11864). It is too risky for M0. Fix: Cut L3 from the M0 probe lanes; note it as a future stretch goal. The operator can add it later if the loading issue is resolved and offloading proves viable.

6. [Point 5 – Two-Stage] The plan should confirm that v1 remains base-only at 1472×832. Fix: Add a statement that base-only is confirmed; upscaling would double cost and risk the VRAM ceiling, so no hardware reason to revisit.

7. [Point 6 – FLUX Co-Residency] The plan should confirm sequential phases. Fix: Add a note that the existing pipeline order (image batch then video batch) and the AS-3 lease guarantee no co-residency; no change needed.

SHOULD-FIX:
- Specify the exact GGUF quant for the CPU-offloaded text encoder (e.g., Q3_K_M) and verify that its file size (~12 GB) fits comfortably in system RAM.
- Add a pre-M0 check: system RAM should be ≥ 32 GB to accommodate block-swap offloading of the 23.5 GiB transformer.
- Define the quality eyeball metric more concretely (e.g., side-by-side A/B comparison with the 2B output, or a simple 1–5 rating scale).

OPTIONAL / NICE-TO-HAVE:
- None.

CUT THESE (over-engineering):
- L3 NVFP4 from the M0 probe (already covered in MUST-FIX #5). Safe to cut because the file size, open issues, and lack of distillation make it an unlikely winner; the operator can revisit later.

[ASSUMPTION] The 14.5 GB ceiling is 14500 MB (decimal GB), and the judge-verified file sizes are in GiB (1 GiB = 1.07374 GB). All comparisons use this conversion.