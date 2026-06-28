# HuMo r3 -- STANDALONE bakeoff harness (compare quality/speed/VRAM; touch NOTHING in production)

Advancing the r2 plan. HARD OPERATOR CONSTRAINT: this is a DIAGNOSTIC bakeoff only -- do NOT touch
`workflows/otr_scifi_16gb_full.json`, do NOT flip the profile, do NOT change production eng_humo.py
behaviour for normal runs. Build a STANDALONE harness that MEASURES, so the operator can eyeball
clips + metrics BEFORE any production change is ever proposed.

## Carried from r2 (verified)
Target id `humo_14B_169` (no `humo_14B`). 5/21 quality target = 14B fp8 + lightx2v 480p distill LoRA +
shift 8 @ 6 steps / cfg 1.0. The fit lever = a TWO-STAGE graph (run conditioning -> reclaim_idle_models
to evict umt5+whisper -> run the 14B sampler), NOT free_after_use (eng_humo.py:344-348: known OOM via
allocator fragmentation). HuMo has no render-window VRAM probe today (only an instantaneous post-reclaim
guard). The honest risk: the fp8 14B weights (~14 GB) may ride >13.5 GB even after TE eviction.

## r3 questions -- ground in the real harness + engine, give an IMPLEMENTABLE wiring answer
1. ISOLATION: how does the harness measure the TWO-STAGE graph + the VRAM peak WITHOUT editing
   production code or the canonical JSON? Pick the cleanest of:
   (a) the harness BUILDS its own standalone graph variant (mirror
       scripts/build_ltx_av_q_bakeoff_workflow.py -> a humo bakeoff JSON) that wires the two-stage
       split + a peak read, leaving eng_humo.py + otr_scifi_16gb_full.json untouched; OR
   (b) the harness imports HuMoEngine and wraps/monkeypatches for measurement (test-only, no prod edit); OR
   (c) a bakeoff-ONLY env flag in eng_humo that is default-OFF and never affects production.
   Which keeps production truly untouched AND still measures the real two-stage peak? Name the files.
2. LEGS (one lever per leg, fixed still+audio+seed): (i) humo_14B_169 single-graph 6-step distill =
   the 5/21 baseline (today's render path); (ii) humo_14B_169 TWO-STAGE 6-step distill (TE-evicted);
   (iii) humo_1.7B control. Optional (iv) humo_14B_169 no-LoRA ~25-step for an upper-quality reference
   (NOT a promotion candidate). What is the minimal leg set that answers "5/21 quality AND <=13.5 GB"?
3. METRICS to record per leg: VRAM render-window peak (the gating number); s/it + wall-clock (speed);
   quality = side-by-side clips to otr/episodes/_bakeoff_humo/<leg>.mp4 for the eyeball, PLUS objective
   proxies ONLY IF the GPU host has the libs (face-detect confidence, mouth-landmark motion vs audio
   energy, lip-area SSIM, blue-cast delta vs the source still -- verify OpenCV/face stack first).
   Fail-loud per-leg manifest records the resolved unet/tier/steps/cfg/shift + the id that ACTUALLY ran
   (the LTX #1 risk: measuring the wrong graph). Reuse the LTX bakeoff metric helpers where they exist.
4. RESET/BOOT discipline: reuse the LTX boot-per-leg pattern (_otr_soak_server_launch.cmd, selective
   CIM kill, OTR_HEADLESS_RESERVE_VRAM_GB). For the TWO-STAGE peak to mean anything for PRODUCTION, do
   we ALSO need a no-reboot "AV-stack-resident" sentinel leg (LTX-AV + Whisper loaded first), or is the
   clean-boot peak + a known AV-stack delta enough? (r1 said clean boot hides cross-engine residency.)

## Constraints / invariants
100% local; single resident <= 14.5 GB (target the heavy engine <= ~13.5 GB); selective box reset per
leg; LOUD; UTF-8 no BOM; SFW; assets straight to otr/episodes/_bakeoff_humo/ (never tmp). NOTHING in
this build edits otr_scifi_16gb_full.json or changes a production render -- diagnostic only.

## Deliver
The standalone-harness wiring answer (which isolation path + the exact new files to add, mirroring the
LTX-AV bakeoff scripts), the minimal leg set, the metric/manifest contract, and the boot/reset plan --
all provably leaving the key workflow + production HuMo path untouched.
