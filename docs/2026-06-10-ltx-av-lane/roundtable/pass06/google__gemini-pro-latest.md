<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: no. The plan ignores the 13.2GB Gemma text encoder and System RAM limits, guaranteeing immediate OOMs or catastrophic paging blowups on a 14.5GB VRAM ceiling.

MUST-FIX BEFORE BUILD:

1.  **[Open questions / ARCHITECTURE] Text Encoder Phasing (Item 2):** The 13.2GB fp8 Gemma encoder and a 12-23GiB transformer mathematically cannot co-reside under the 14.5GB NVML ceiling. 
    *   **Fix:** Explicitly define the phase discipline inside the `render_clip` execution: Load Gemma -> Encode Text -> Force CPU offload (via ComfyUI model management or `wrapper_bridge.reclaim_idle_models(reason="text encode")`) -> Load Transformer. The AS-3 lease (`MotionEngineBase.prepare`) must wrap the *entire* render phase to prevent sidecars from stealing the GPU, but VRAM must be multiplexed *inside* that lease.
2.  **[Milestones] L2 GGUF Arithmetic & Decision Gate (Item 1):** Q4_K_M is a 16.5 GiB file. The NVML ceiling is 14.5 GB. Full residency arithmetic is already dead for Q4. 
    *   **Fix:** Explicitly state in the M0 PROBE that Q4 full-residency is dead. Change the L2 realistic full-resident target to Q3_K_M (13.7 GiB) or Q3_K_S (13.0 GiB), OR explicitly state L2 relies on ComfyUI-GGUF per-layer offload to measure under the ceiling. Define the M0 PASS criteria columns: NVML <= 14.5GB sustained, Wall-time <= 15 min/clip, Quality >= 2B baseline.
3.  **[Milestones] System RAM & Paging Blowup (Item 3):** Block-swap/weight-streaming a 23.5GiB fp8 file + a 13.2GB encoder will eat massive System RAM. If the laptop has 32GB RAM, it will page to disk. 
    *   **Fix:** Add "System RAM size" and "Peak Commit Charge" columns to the M0 sheet. Name the failure mode: RAM shortage causes disk paging, leading to a wall-time blowup. This is caught by the 15 min/clip wall-time ceiling gate.
4.  **[Milestones / Open questions] Cut L3 NVFP4 (Item 7):** The L3 NVFP4 file is 21.7GB, DEV-only (not distilled, requiring more steps/time), and has an open loading-failure issue (#11864). 
    *   **Fix:** Cut L3 from the M0 PROBE entirely to save sprint time. It is not viable for v1 on this hardware.

SHOULD-FIX:

5.  **[Milestones] Episode Time Budget (Item 4):** A 30-word episode requires ~3 clips (2 talk, 1 music). The 2B baseline takes ~6 min/clip (18 min/episode). 
    *   **Fix:** Set the operational usability ceiling at 15 min/clip (45 min/episode). Anything >15 min/clip means the lane is operationally unusable even if it renders without OOM.
6.  **[ARCHITECTURE] Two-Stage Confirmation (Item 5):** 
    *   **Fix:** Confirm v1 remains base-only at 1472x832. A latent upscale stage would require loading a second heavy model, blowing the 15-minute time budget and risking VRAM fragmentation. There is no hardware headroom to revisit this.
7.  **[ARCHITECTURE] FLUX Co-residency (Item 6):** 
    *   **Fix:** Confirm sequential phases (FLUX image batch THEN video batch) mean no co-residency is required. The `MotionEngineBase.teardown` (which calls `wait_until_below_mb(14500)`) and the AS-3 lease guarantee FLUX is cleared before LTX-AV starts.

OPTIONAL / NICE-TO-HAVE:
*   Add a specific `OTR_LTX_TEXT_ENCODER_NAME` env var to the loader names to explicitly pin the 13.2GB Gemma file, mirroring the HuMo config pattern in `eng_humo.py`.

CUT THESE (over-engineering):
*   None. The current plan is under-engineered regarding memory management.

[ASSUMPTION] Assuming the operator's laptop has at least 64GB of System RAM. If it has 32GB, the L1 fp8_scaled lane (23.5GiB + 13.2GB) will almost certainly fail the wall-time gate due to pagefile thrashing.