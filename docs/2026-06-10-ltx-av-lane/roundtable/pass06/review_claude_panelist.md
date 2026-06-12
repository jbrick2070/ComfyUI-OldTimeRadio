# pass06 hardware -- Claude panelist review (before reading the panel)

MUST-FIX

1. (lane gate arithmetic) State the dead ends explicitly in the plan:
   full residency at the 14.5 GiB NVML ceiling is ARITHMETICALLY DEAD
   for Q4_K_M (16.5 GiB file) and everything above it, and for L1
   fp8_scaled (23.5 GiB) -- those lanes are viable ONLY via measured
   per-layer offload / block-swap under the ceiling. The realistic
   full-resident GGUF picks are Q3_K_S (13.0 GiB; ~1.5 GiB headroom for
   activations + VAE) and MAYBE Q3_K_M (13.7; tight). The M0 table
   therefore probes: Q3_K_S resident, Q3_K_M resident, Q4_K_M
   offloaded, L1 fp8 block-swap; each row records NVML sustained peak,
   wall/clip, RAM commit peak, quality eyeball vs the 2B baseline clip.
2. (PASS bars) NVML <= 14.5 sustained (existing machine-NVML gate);
   wall-time <= 10 min/clip at 1472x832 x ~6s (vs ~6 min today on the
   2B opens; talking beats are shorter); quality >= 2B baseline on the
   P1 eyeball. A lane FAILING wall at <= 15 min may still ship as
   opt-in with a documented cost; > 15 min/clip = lane parked (a 3-clip
   30w episode would add ~45 min -- operationally dead for look-QA
   iteration).
3. (encoder phasing) The 13.2 GB fp8 gemma encoder cannot co-reside
   with any transformer pick. The adapter relies on SEQUENTIAL phases:
   text encode -> FREE the encoder (BUG-291 reclaim_idle_models, never
   unload_all) -> load transformer -> sample -> decode. The AS-3 lease
   wraps the WHOLE render_clip (single heavy ENGINE discipline);
   intra-phase swapping is ComfyUI model management + explicit reclaim
   between phases (mirror the humo pattern in grounding). M0 measures
   BOTH encode-on-GPU-then-reclaim AND encode-on-CPU (the community
   lever); v1 default = whichever passes wall+NVML, preferring GPU
   encode (a 12B CPU forward may cost minutes per clip -- measure,
   don't assume).
4. (system RAM row) M0 sheet records system RAM size + peak commit
   during the block-swap row. Failure mode: RAM exhaustion -> paging ->
   wall-time blowup; the wall bar catches it, the sheet explains it.
   Also record free disk on the C:\ComfyUI-Models volume before pulls
   (Q3_K_S 13 GiB + encoder 13.2 GB + VAEs ~2 GiB if absent).
5. (L3 NVFP4 demoted) CUT NVFP4 from the default M0 evening: DEV-only
   weights (not distilled -> more steps -> slower), 21.7 GiB file
   (offload-only anyway), and the open #11864 loading-failure class.
   It remains a DOCUMENTED stretch column the operator may run later;
   it must not gate the lane decision.

SHOULD-CONSIDER

6. Two-stage stays base-only v1 (no hardware reason to revisit; the
   upscaler would double wall against an already-tight budget).
7. FLUX co-residency: none required -- the pipeline is sequential
   (image batch completes before the video batch; image_done is
   non-blocking in v1). Confirm nothing in gpu_residency contradicts.
8. Download hygiene: extend the existing download_ltx_2_3.ps1 pattern
   (HF cache + symlink, ~22 GB warning) for the GGUF + encoder pulls
   rather than ad-hoc downloads; everything lands under
   C:\ComfyUI-Models (never bare C:).

OPEN-QUESTIONS

9. Does ComfyUI-GGUF dequant-on-the-fly hold Q4_K_M under 14.5 NVML in
   practice on sm_120? (That single measurement decides whether Q4
   quality is reachable at all; otherwise Q3_K_S is the lane.)
10. taeltx2_3 (23.5 MB) as the preview/decode VAE vs the full 1.45 GB
    video VAE: does the IA2V template default to taeltx for speed, and
    is full-VAE decode worth the VRAM/wall delta for finals? (M0 cell.)
