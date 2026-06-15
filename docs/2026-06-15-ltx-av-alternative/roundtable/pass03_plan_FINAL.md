# LTX AUDIO-INPUT (A2V) ALTERNATIVE PATH -- CONVERGED (pass03 FINAL)

> ## >>> M0 OUTCOME (2026-06-15): **PARKED.** Lane A stands as production. <<<
> The M0 graph spike RAN and is decisive. The A2V topology is fully grounded (live `/object_info` @ :8000 +
> the official bundled template `video_ltx2_3_ia2v.json`): the model is a **22B fp8 FULL checkpoint
> `ltx-2.3-22b-dev-fp8.safetensors` (~23 GB)** via `CheckpointLoaderSimple` + a **Gemma-3-12B fp4** encoder
> via `LTXAVTextEncoderLoader` (~8.8 GB) + a distilled motion LoRA + a two-stage base/`LTXVLatentUpsampler`
> sampler. The frozen-audio terminal the plan needed is **confirmed real**:
> `LTXVSeparateAVLatent(av_latent) -> video_latent -> VAEDecodeTiled` (drop the `LTXVAudioVAEDecode` audio
> branch). **VERDICT = PARK (operator, Route A):** ~23 GB fp8 + ~8.8 GB Gemma cannot be single-resident under
> 14500 MB on a 16 GB 5080; only block-swap/CPU-offload could cap peak, and it streams weights every step ->
> too slow for per-beat production (the offload-thrash PARK condition). The fp8 dev checkpoint is not even on
> disk; spending the 23 GB download for an empirical receipt of an arithmetic near-certainty is low ROI, so
> no heavy forward was run. **Lane A (golden prompt-only `ltx_video`) is untouched and remains production.**
> Only future lead for a 16 GB fit: a **GGUF-Q3 (~9-11 GB)** quant (not on disk; needs an `UnetLoaderGGUF`
> graph adaptation + audio/Gemma-path verification) -- a separate, uncertain probe, pursue only on explicit
> revival. Full capture + asset inventory: `docs/2026-06-15-ltx-av-alternative/M0_GRAPH_SPIKE_FINDINGS.md`
> (+ `m0_object_info_full.json`, `m0_template_ia2v.json`). M1-M4 below remain un-started (no M0 GO).


> CONVERGED after 3 live panel passes (GPT-5.5 + Gemini-3.1-pro + DeepSeek-v4) + Claude panelist/grounding,
> ~$0.39 total. Verdict: the plan is SOUND; the remaining work is (1) DEFER to the locked 8-pass sprint plan
> for internal contracts, (2) gate ALL of Lane B behind the M0 graph spike. This FINAL is a thin REFRESH
> LAYER over the authoritative detail spec.

## AUTHORITATIVE DETAIL SPEC (do NOT re-litigate these -- they were 8-pass-converged 2026-06-10)
`docs/2026-06-10-ltx-av-lane/LTX_AV_SPRINT_PLAN.md` owns the locked internals. In particular the panel
re-confirmed and CORRECTED my pass01/02 drift back to it:
- **Frame math = snap-UP.** `next_8n1(n) = ((n+6)//8)*8 + 1`; render = `min(next_8n1(T), LTX_AV_MAX_FRAMES)`.
  DELIBERATELY diverges from Lane A's snap-DOWN (`eng_ltx_video._ltx_frame_length`). Do NOT copy Lane A's math.
- **NO Director / workflow-JSON edit.** V-6 AUTO-DROPDOWN surfaces a registered dark engine automatically;
  `@register` unconditional + roles is enough. (This resolves the dropdown-visibility question -- it's
  registry-driven, so flag-off stays visible and fails closed at render. NO `otr_scifi_16gb_full.json` edit.)
- **canonicalize is the timing authority:** `timing.target_frame_count` = T; canonicalize TRIMS to exactly T
  or PADS-BY-LAST-FRAME to T (cap case), stamping pad_tail. (Audio may be padded BEFORE generation too, but
  canonicalize still enforces exactly T -- both.)
- The assert_usable ordered gate (existing `EngineUsabilityReason` only; `assert_ltx_dims` RAISES, caught +
  wrapped as `EngineUnusable`); the `gpu_residency.nvml_available()` / `probe_used_mb` helpers; the
  encoder->`reclaim_idle_models`->transformer phasing; the `_ref_path(request.audio_ref)` extraction copied
  from eng_humo; `schemas.py FAMILIES`(+`audio_conditioned_video`)+`FAMILY_REQUIRED_INPUTS`+role_compat;
  `OTR_LTX_AV_CKPT/_TEXT_ENCODER/_VAE` envs.

## REFRESH DELTAS (what THIS pass updates over the 6/10 spec)
1. **Re-ground vs HEAD `9633e1e`.** The spec was grounded vs `56caa5b` (pre-cleanbreak). ALL its line refs
   (render_driver :281/:387/:418, eng_humo :366-383, etc.) are STALE -- re-locate before any delta. The
   prompt-only engine is now `eng_ltx_video.py` (not batch_ltx_render.py).
2. **M0 is a GRAPH SPIKE and the UNIVERSAL GATE (probe-or-park).** The LTX-2.3 A2V topology is UNKNOWN: M0
   captures from the official LTX-2.3 ComfyUI template + a live `/object_info` (Desktop AND headless diff):
   the exact node classes (audio loader, conditioning, sampler, and the TERMINAL video decode/separation),
   the viable low-VRAM artifact, the VAE decode floor at 384x216 / 512x288, and peak NVML on THIS 5080.
   **Do NOT prescribe terminal/decode node names in the plan** (my `LTXVSeparateAVLatent` was ungrounded; the
   spec's "video VAEDecode" may also be wrong for a joint-AV latent) -- M0 records them; M2 wires whatever M0
   captured; node-gate fail-closed on the captured set. **M1+ are GATED behind M0 GO; OOM/thrash -> PARK,
   write the finding, Lane A stands.**
3. **Model/VRAM reality (corrects my pass01).** "distilled v1.1 / 8-step" is the existing **2B TEXT** recipe,
   NOT the A2V model. A2V is 22B-class; fp8/distilled-22B are likely DEAD under 14.5 GB. M0 ranks GGUF
   Q3_K_S/Q3_K_M + Gemma-encoder CPU-offload + block-swap. **NVFP4 CUT** (exceeds 16 GB). Invariant tightened:
   "single heavy resident ONLY if M0 proves <= 14500 MB peak/sustained on the 5080; else PARK."
4. **Lane A is prod NOW / Lane B is DARK experimental** (operator's fact-checked framing). Lane B's real
   unique value = CHARACTER lip-sync vs HuMo (announcer/music motion is already handled by today's shipped
   ksampler+music_open+boomerang). Graduation (M4) = lip-sync-vs-HuMo operator A/B on identical
   audio/still/seed + N short clips no-OOM (N=3 for the 30-word smoke).
5. **Boomerang isolation.** NO boomerang in `eng_ltx_av.py`; do NOT touch `OTR_LTX_LOOP_VIA_REVERSE` /
   `_LOOP_VIA_REVERSE_DEFAULT` / `eng_ltx_video.py` (audio defines length; mirroring desyncs).
6. **Singleton core dispatch.** ONE module-level lazy `_ltx_av_core` (lock-guarded) exposing
   `render_talk(plan)` (I2V) and `render_music(plan)` (t2v) -- the I2V-vs-t2v branch is INTERNAL to the core
   so talk+music share ONE resident load. State machine: unloaded->loaded/rendering->idle; on OOM/cancel ->
   POISONED -> forced reclaim + release AS-3 lease + classified restart-required error (with a
   `wait_until_below_mb(14500)` TIMEOUT -> same restart error). No double-core creation (mocked-factory test).
7. **Sequencing precision.** Pre-M0: CPU-only skeleton + wiring + tests with graph-node-gate tests
   SKIPPED/fixture (no placeholder A2V class names hardcoded). Post-M0 GO: fill the captured node list, enable
   node-gate + the heavy forward. CAPABILITIES `vram_estimate` + per-artifact size floors are placeholders
   pre-M0, filled from the M0 measurement.

## TICKETS (each: suite + Bug Bible + audio-byte-identical GREEN per chunk)
- **M0 GRAPH SPIKE (probe-or-park; NO engine code).** Capture the A2V GRAPH SPEC (node classes + artifact +
  VAE floor + peak NVML at 384x216 & 512x288 / batch1 / 4-6 s / Gemma offloaded). OOM/thrash -> PARK.
- **M1 skeleton (CPU; AFTER M0 GO for graph-dependent parts).** `eng_ltx_av.py` singleton core + 2 adapters +
  the ordered assert_usable + schemas.py family + role_compat + CAPABILITIES placeholder + cold-import/AST/
  no-double-core tests.
- **M2 frozen-audio V-1.** terminal = the M0-captured video-only decode; engine return has NO audio path +
  the clip has ZERO audio streams (ffprobe test); 8n+1-padded audio-slice input; canonicalize to exactly T;
  `test_audio_byte_identical` green.
- **M3 wiring.** schemas/role_compat/fallback re-grounded vs HEAD (verify `humo`/`humo_1.7B`/`latentsync`/
  `still_kenburns` registry IDs); NO JSON edit (V-6 auto-dropdown); explicit force-map smoke
  `announcer_visual=ltx_av_talk,character_video=ltx_av_talk,music_visual=ltx_av_music`.
- **M4 graduation.** lip-sync-vs-HuMo A/B + N=3 no-OOM. Promote dark->selectable or PARK.

## CUTS (panel consensus -- keep Lane B lean)
NVFP4; the audio-reactive ledger->prompt verbs (separate Lane-A ticket); the `_slice_master_audio`
mtime_ns+size cache-key fix (its OWN ticket, do not bundle); the storm-line / episode-summary instrumentation
fields + the pre-delta semantic-projection goldens (defer to a post-M4 observability ticket -- a single
known-clip-hash + audio-byte-identical smoke suffices for the dark lane); the full optical-flow hard gate for
announcer/music.

## INVARIANTS
test_audio_byte_identical GREEN (V-1); single heavy resident ONLY if M0 proves <=14500 MB on the 5080 else
PARK; 100% local; determinism; LOUD fallbacks; UTF-8 no BOM; SFW; ADDITIVE only (ltx_video/humo/latentsync
untouched); no new static widgets (V-11); NO workflow-JSON edit (V-6 auto-dropdown).
