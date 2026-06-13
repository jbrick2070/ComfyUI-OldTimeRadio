# Roundtable pass01 -- judgment log (Claude is the judge; grounded vs real code)

Panel: openai/gpt-5.5-20260423, google/gemini-3.1-pro-preview-20260219,
deepseek/deepseek-v4-pro-20260423 (3 models). Spend: pass01 ~$0.0746 (DeepSeek full;
GPT empty on reasoning-token-length; Gemini truncated) + pass01b ~$0.2318 (GPT +
Gemini re-run with --reasoning-effort none --max-tokens 12000) = ~$0.306 total.
Raw reviews: pass01/ (DeepSeek) + pass01b/ (GPT, Gemini). Every claim below was
grounded against the real code at HEAD 134f8e2 before folding.

Convergence: all three independently led with Q1 (the GATE-A sweep is blind to
silent fallbacks) -- strong agreement on the headline. The panel added 6 grounded
items I had missed (M2,M4,M5,M6,M7,S7) and one cleaner fix for Q1 (Gemini). Treated
as CONVERGED for this hardening pass; a 2nd live pass would mostly re-confirm.

## CONFIRMED -> folded (grounded)
- M1 Q1 sweep blind to fallback. otr_coverage_sweep.py passes expect_engine="" ->
  _otr_soak_capstone.py:464-465 logs an informational histogram, no assert. All 3
  models. FIX (Gemini's, better than my per-leg expect_engine which false-fails
  0-beat slots): assert ZERO runtime fallbacks across the whole trace (any shot with
  final_engine != attempts[0] fails), with an opt-out for known-degrade experiment
  legs. Grounded: trace rows carry attempts[0]+final_engine (render_driver.py:1204).
- M2 (GPT) sweep returns GREEN on empty results. `return 0 if passed==len(results)`
  -> 0==0 when --only/--exclude filter everything or wan_ti2v is unregistered.
  Grounded (read main()). FIX: fail on empty results; GATE-A fails unless wan_i2v
  AND wan_ti2v are present with PASS.
- M3 (GPT) acceptance sweep must preflight the enable flags + model files. availability()
  is pure profile-fit, never reads OTR_ENABLE_WAN_I2V (its own --exclude help says so)
  -> a Wan leg enumerates "run", assert_usable gates it off, it falls back, and with
  M1 unfixed scores PASS. This IS the R2 gated_by_flag history (commit 5231d31). FIX:
  preflight OTR_ENABLE_WAN_I2V=1 (+ future OTR_ENABLE_WAN_TI2V=1) + model files; forbid
  --exclude of core Wan engines on the acceptance run.
- M4 (GPT) V-3 VRAM gate fails OPEN. `driver_peak = int(report.get("vram_peak_mb") or
  -1)`; only fails if `> CEILING`; missing/0/negative -> -1 -> PASSES. Contradicts the
  <=14.5GB invariant. Grounded (_otr_soak_capstone.py:~534). FIX: fail closed when
  vram_peak_mb is absent or <=0.
- M5 (GPT) Wan render-phase VRAM assert is skipped under OTR_TEST_MODE
  (`if not os.environ.get("OTR_TEST_MODE")` wraps assert_peak_within_ceiling).
  Grounded (eng_wan_i2v.render_clip). FIX: Phase-2 acceptance must run with
  OTR_TEST_MODE UNSET; harness preflight fails if it is set.
- M6 (GPT) assert_usable preflights only the ckpt, not the CLIP/VAE loaders.
  Grounded. FIX: verify UNET+CLIP+VAE present + match the sha/license manifest before
  any forward (offline/no-fetch invariant).
- M7 (GPT) "silent mp4 ... has_audio False" is self-declared in _clip_from_raw, not
  asserted; the soak only checks the obs final's audio. FIX: ffprobe the emitted Wan
  mp4 (has_audio False + h264/yuv420p/bt709/fps25) before mux, or a real-path test.
- M8 Q2 (+GPT,DeepSeek) wan_ti2v VAE must be fail-closed distinct from the 2.1 VAE.
  eng_wan_i2v defaults VAE to wan_2.1_vae.safetensors. FIX: own VAE env; raise
  EngineUnusable if the resolved VAE basename is empty or == the 2.1 basename; do not
  inherit _loader_names() unchanged.
- M9 Q5 (+GPT,DeepSeek) CS-3 reframe: wan_i2v ~14GB + humo_1.7B ~7GB cannot co-reside
  under 14.5GB by construction; the real proof is per-beat peak <= ceiling + the
  inter-beat reclaim drains the previous heavy engine (incl. the retained Wan unet
  patcher) before the next loads. UNBLOCKS Phase-2 scoping. DeepSeek's "guaranteed
  OOM from per-beat reload" is the failure mode this rules out (reclaim exists:
  wrapper_bridge.reclaim_idle_models, BUG-291) -- folded into M9, not standalone.

## CONFIRMED should-fix -> folded
- S1 Q3 vram_estimate 14000 optimistic + free_after_use load-bearing. CORRECTED per
  GPT: the 14499 figure was the WITHOUT-free_after_use smoke peak (my Q3 overstated
  it as the observed peak). Set estimate to the measured Phase-2 peak or 14500; note
  free_after_use=True is mandatory.
- S2 (Gemini) add a concrete wan_ti2v CAPABILITIES row now (vram_class medium,
  vram_estimate_mb ~8000 DRAFT pending the 8GB probe -- 5B VAE decode may push higher,
  model_requirements ["wan2.2-ti2v-5b"]).
- S3 Q4 single-expert MoE motion risk -> explicit Phase-2 risk; Path B (two-expert
  handoff) is the mitigation if the eyeball motion bar fails.
- S4 (DeepSeek) sweep leg isolation: one resident server, no teardown between legs ->
  prior heavy-engine residue corrupts the next leg's VRAM peak/availability (ties to
  CS-2 + the CLAUDE.md "aggressively reset before every headless run" directive).
  Reclaim/restart between legs that swap heavy engines.
- S5 Q6 stale label model_requirements ["wan2.1-i2v"] -> real Wan2.2 I2V id.
- S6 Q8 item-4 matrix breadth: otr_coverage_sweep.py only enumerates the visual-engine
  leg-set; writer-LLM + voice-variation sets aren't in it -> point each at its real
  harness (run_combo_matrix.py?) or mark TODO + drop from the GREEN surface until built.
- S7 (GPT) _materialize_init_image writes a FIXED basename otr_wan_init_<w>x<h>.png ->
  same-dim renders overwrite. FIX: add shot_id/seed/uuid (keep determinism). Lower risk
  (driver is sequential per beat).
- S8 Q7 (+GPT) spell scripts/otr_coverage_sweep.py + the exact --only substring that
  matches a Wan leg + required env (confirmed --exclude landed in THIS script).
- S9 (GPT) Phase-2 post-reset verification: assert PID/start-time changed, Sage NOT
  active, OTR_TEST_MODE unset, env visible, before submitting the leg.
- S10 (DeepSeek) _materialize_init_image Pillow-fallback relies on WanImageToVideo
  cover-resize (N9) -> require Pillow + fail loud, or confirm the node resize is
  non-stretching. Verify-at-build.

## CUTS (panel consensus -- avoid over-engineering)
- C1 (GPT) Don't build a broad VRAM-budget-aware scheduler to close CS-3 before Phase
  2; the minimal sequential-residency/reclaim assertion in the real episode path is
  enough. Scheduler waits for a measured reclaim failure.
- C2 (GPT+DeepSeek) Don't implement wan_ti2v by subclassing all of WanI2VEngine;
  share only small pure helpers (dims/aspect/materialize/canonicalize); keep loader
  names + node candidates + graph SEPARATE.
- C3 (GPT) Keep the GATE-A sweep ADDITIVE, not a visual x writer x voice cross-product.

## OVERSTATED / corrected by grounding
- Q3 softened (see S1). DeepSeek "guaranteed OOM" merged into M9 (reclaim exists).

## UNVERIFIABLE -> verify-at-build
- M1 trace-schema: expect_engine/attempts[0] could false-match if the trace records
  aliases or reordered attempts (GPT). Verify the trace field is a stable requested-id
  when implementing M1.
- TI2V-5B exact core node class + topology must be captured from /object_info before
  coding (GPT); the plan's "_node_candidates incl. the 5B latent node" is underspecified.

## Side note (not a plan item)
4 soak-fix commits (a31fc24,d33c51f,5231d31,134f8e2) are committed but UNPUSHED on
v2.0-alpha (ahead 4). Per the 2026-06-10 GIT POLICY, push them.
