# pass06 (hardware) judgment -- Claude, judge + panelist

Cleanest convergence of the campaign: 4/4 on every major call.

## ACCEPTED (grounded vs gpu_residency/wrapper_bridge/motion_common +
## judge-verified HF file sizes)

- ARITHMETIC DEAD-ENDS stated in the plan: full residency under the
  14.5 (14500 MB NVML) ceiling is DEAD for Q4_K_S (15.6 GiB), Q4_K_M
  (16.5), Q5+ and L1 fp8_scaled (23.5) -- those rows are offload/block
  -swap rows ONLY. Full-resident GGUF candidates: Q3_K_S (13.0, ~1.5
  GiB headroom) and Q3_K_M (13.7, BORDERLINE -- flagged; total NVML
  decides, never file size).
- M0 DECISION TABLE (GPT's column set adopted): lane id / artifact /
  file size / aux artifacts in phase / encoder placement / offload
  setting / NVML idle/preload/peak/sustained/post (machine-wide,
  gpu_residency.probe_used_mb semantics; mid-render peak via
  motion_common.assert_vram_within_ceiling) / wall-time 1472x832x~6s /
  frames / quality vs 2B / PASS-WARN-FAIL / notes. Units: NVML MiB,
  files GiB (DeepSeek's conversion note).
- PASS BARS (merged GPT/DeepSeek/Gemini): NVML peak+sustained <= 14500
  MB; wall <= 10 min/clip PASS, 10-15 WARN (ship-able opt-in with
  documented cost), > 15 FAIL = lane parked. Episode (~3 clips/30w):
  <= 30 min PASS, > 45 min operationally dead. Quality >= the 2B
  baseline clip, A/B side-by-side (DeepSeek's metric note).
- ENCODER PHASING (all 4): 13.2 GB fp8 gemma encoder can never
  co-reside with any transformer pick. Sequence INSIDE one lease:
  acquire AS-3 -> text encode -> wrapper_bridge.reclaim_idle_models(
  reason="ltx_av text-encode phase") [BUG-291 mechanism; _soft_free is
  NOT enough -- GPT grounded] -> load transformer -> sample -> decode
  -> teardown reclaim -> release + wait_until_below_mb(14500).
  GPT MF4 refinement ACCEPTED: the lease wraps the WHOLE render incl.
  the encode phase (MotionEngineBase.prepare/teardown bracketing);
  never release between phases. M0 measures BOTH encoder modes --
  GPU-encode-then-reclaim AND CPU/RAM-offloaded encode -- v1 default =
  the passing mode, preference GPU-encode (a 12B CPU forward may cost
  minutes; measure, don't assume -- judge over GPT/DeepSeek's
  CPU-default lean). GGUF Q3 encoder variant = optional row ONLY if
  fp8 fails both modes.
- SYSTEM RAM ROW: M0_RESULTS.md records physical RAM, pagefile
  status, peak commit + working set per lane; failure mode named
  (paging -> wall blowup; the wall gate catches it); pre-M0 check RAM
  >= 32 GB for any block-swap row; disk-free check on C:\ComfyUI-
  Models before pulls.
- L3 NVFP4 CUT FROM M0 (4/4): DEV-only (more steps), 21.7 GB
  (offload-only regardless), open loading-failure class (#11864).
  Documented stretch column the operator may run later; never gates
  the lane decision.
- TWO-STAGE: base-only v1 CONFIRMED (no headroom; upscaler doubles
  wall + loads a second heavy model).
- FLUX CO-RESIDENCY: sequential by pipeline order + AS-3 + teardown
  wait_until_below_mb; ADD one M0 verification row (GPT SC3): after
  the FLUX phase, lease released + below-ceiling before video begins
  (ordering is asserted, not assumed).
- CONFIG ENVS (Gemini optional, humo pattern): OTR_LTX_AV_CKPT +
  OTR_LTX_AV_TEXT_ENCODER path overrides (config, not behavior knobs).
- MISSING_MODEL messages name the EXACT missing artifact (encoder vs
  transformer vs text_projection vs VAE) -- M0 failures unambiguous
  (GPT SC4).
- download_ltx_2_3.ps1 disk note stale (~22 GB vs 23.5 GiB): comment/
  warning bump to >= 24 GiB + the GGUF/encoder pulls reuse the same
  cache+symlink pattern (additive script touch).

## REJECTED / NOTES

- Gemini's 64 GB RAM assumption: not assumed -- the sheet records
  reality; 32 GB is the plausible floor and the wall gate arbitrates.
- No new pip into cu130 reaffirmed; ComfyUI-GGUF loader presence is an
  M0 inventory row (it's a custom node pack, not a pip dep -- if
  absent, the GGUF lane needs the pack installed via Manager and the
  pip-freeze sandwich still must hold; if it cannot, L2 is sidecar
  -blocked and L1/Q3-resident-only is the lane).

## VERIFY-AT-BUILD / M0

- ComfyUI-GGUF per-layer offload behavior for Q4 on sm_120 (the single
  measurement that decides whether Q4 quality is reachable).
- taeltx2_3 vs full video VAE on the decode path (speed/VRAM vs final
  quality -- one M0 cell).
- Exact split-graph support: can text embeddings materialize without
  retaining the gemma model (node IO shapes, M0 sheet).
- Production FLUX->video ordering row (above).
