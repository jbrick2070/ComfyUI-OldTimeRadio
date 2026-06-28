CLAUDE ANCHOR -- HuMo r3 (standalone bakeoff, zero production touch)

VERDICT: achievable with NEW FILES ONLY -- and the key realization is that the bakeoff measures VRAM
EXTERNALLY, so it needs NO production VramPeakProbe and NO eng_humo.py edit.

ISOLATION (the r3 crux):
- The LTX-AV bakeoff already proves the pattern: scripts/build_ltx_av_q_bakeoff_workflow.py (emits a
  STANDALONE graph JSON) + scripts/run_ltx_av_q_bakeoff.py (boot-per-leg headless, reads VRAM
  EXTERNALLY via nvidia-smi / the watchdog, fail-loud manifest). Clone that shape for HuMo.
- VRAM PEAK is read by the HARNESS from OUTSIDE the server (nvidia-smi polling / otr_render_watchdog),
  NOT an in-process probe. So the standalone bakeoff does NOT need the r2 "add VramPeakProbe to
  eng_humo" production change -- that probe is only for the LATER promotion, not the diagnostic.
  -> the bakeoff touches ZERO production code.
- The TWO-STAGE graph (conditioning -> reclaim TE -> sampler) is the one thing a single static JSON
  can't express, because reclaim_idle_models is a wrapper_bridge call, not a graph node. Cleanest
  production-untouched options:
  (a) the harness issues TWO /prompt calls per two-stage leg -- graph-1 conditioning (save latents +
      conditioning), then a reclaim, then graph-2 sampler/decode fed those as inputs; OR
  (b) a bakeoff-ONLY helper node living UNDER scripts/ (not nodes/) that the standalone graph wires to
      do the mid-graph reclaim -- registered only for the bakeoff, never in the canonical pack.
  Prefer (a) if the API can round-trip latents/conditioning; else (b). Either keeps eng_humo.py +
  otr_scifi_16gb_full.json untouched. Reuse HuMoEngine._build_graph READ-ONLY for the node templates.

LEGS (minimal, answers "5/21 quality AND <=13.5 GB"):
(i) humo_14B_169 single-graph 6-step distill = the 5/21 baseline (today's path, the control for "did
    we lose quality"); (ii) humo_14B_169 TWO-STAGE 6-step distill (TE-evicted) = the candidate;
(iii) humo_1.7B = the current-shipping control. (iv) optional no-LoRA ~25-step upper-bound reference.
Fixed still+audio+seed; reuse the LTX pair (c02_466a19906ccb.png + c02_b002_line.wav).

METRICS: external VRAM render-window peak (the gate number); s/it + wall-clock; side-by-side clips ->
otr/episodes/_bakeoff_humo/<leg>.mp4 for the eyeball; objective proxies ONLY if OpenCV/face libs exist
(verify first). Fail-loud manifest asserts the id/unet/steps/cfg/shift that ACTUALLY ran (LTX #1 risk).

AV-PRESSURE: clean boot-per-leg gives the cleanest QUALITY + a clean peak, but r1 noted it HIDES
cross-engine residency. Add ONE no-reboot sentinel leg (load LTX-AV + Whisper, then the two-stage 14B
in the SAME resident session) -> the production-true peak. That sentinel is what actually answers the
gate; the clean legs answer quality/speed.

[VERIFY-AT-BUILD] can the ComfyUI /prompt API round-trip HuMo conditioning+latents between two calls
(option a), or is a bakeoff-only reclaim node (option b) required? Check the LTX bakeoff for whether it
ever splits a graph across calls.
