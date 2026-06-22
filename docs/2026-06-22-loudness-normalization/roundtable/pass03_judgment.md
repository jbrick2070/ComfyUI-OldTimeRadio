# R3 judgment log (Claude as judge)

Panel: gpt-5.5, gemini-3.1-pro, deepseek-v4-pro, grok-4.3 (live, ~$0.13) + Claude grounding.
R3 found REAL new material (not a trivial converge) -- folded below. Two opens resolved by reading
`tests/_run_baseline.py` + `scene_sequencer.py`.

## ACCEPTED -- MUST-FIX (CONFIRMED, grounded)

- **Env must be in the ComfyUI SERVER process, not the capture shell.** `_run_baseline.py` drives a
  RUNNING ComfyUI at `http://127.0.0.1:8000` (POST `/prompt`, poll `/history`); the nodes read
  `os.environ` server-side. Re-baseline procedure MUST restart the server with `OTR_SEGMENT_LOUDNORM`,
  `OTR_SEGMENT_TARGET_RMS_DBFS`, `OTR_MASTER_MAKEUP_DB`, `OTR_CAST_SEED`, `OTR_STYLE_SEED` + a preflight
  log line asserting the effective mode/target/makeup. (GPT#2, Gemini#1, DeepSeek#4, Grok#1 -- unanimous.)
- **Calibration must compensate for the dropped +4 dB master makeup.** Today's output includes +4 dB
  makeup; rms mode sets makeup->0. So `target_rms_dbfs = measured_pre_master_dialogue_RMS + 4.0` (or
  measure final dialogue-only output) -- else the episode lands ~4 dB quieter. (GPT#5, Gemini S#1,
  DeepSeek#5.)
- **One calibration reference point**: dialogue-only clips, pre-master, active windows (above silence),
  aggregated per clip -- NOT the master WAV (mixes SFX/music/silence + makeup). (GPT#4, DeepSeek#2.)
- **Prove the peak path is byte-identical before landing.** The byte-compare skips by default, so the
  call-site swap could silently drift. Add an IN-SUITE test: `_level_dialogue_clip(x)` in peak mode ==
  `_normalize_clip(x)` byte-for-byte; AND run the runtime byte-identical once (`OTR_REGRESSION_RUNTIME=1`,
  peak) before Chunk 1 lands. (GPT#6, DeepSeek#1.)
- **Fixture / CI trap**: keep `baseline_v1.5` as the PEAK golden; do NOT overwrite it with rms fixtures
  while the default is peak (else the gated byte-compare generates peak bytes vs an rms golden -> false
  fail). Add a skip-guard: byte-compare skips when `OTR_SEGMENT_LOUDNORM` not in {unset, peak}. Promoting
  rms to default = a FUTURE separate commit (re-baseline + default flip together). (GPT#1, Gemini#3, Grok opt.)
- **`OTR_MASTER_MAKEUP_DB` rule precision**: explicit value wins; else rms->0.0; else 4.0. Pin
  `OTR_MASTER_MAKEUP_DB=0` explicitly during rms calibration/capture so a stale 4.0 can't contaminate.
  (DeepSeek#5, GPT#3, Grok#2.)

## ACCEPTED -- SHOULD-FIX

- Commit a reproducible `tools/measure_dialogue_rms.py` (float full-scale, mono/channel handling,
  active-window RMS) instead of a throwaway. (GPT opt, DeepSeek opt, Grok S#1.)
- Fold env-knob docs into Chunk 1 (live knobs land with the code). (GPT S#4 / CUT#1.)
- Acceptance pass/fail: no clipping (peak <= ceiling), no pumping/noise-floor lift, target-RMS tolerance,
  named sample episodes. (GPT S#6, Gemini opt.)
- Copy old baseline aside before any capture; ffmpeg `loudnorm` read for the A/B. (GPT S#1, DeepSeek S#7.)
- SFX safety: :726 calls `_normalize_clip` DIRECTLY (never the router), and `_normalize_clip` never reads
  the env -> SFX is peak by construction regardless of mode. Add a test asserting it. (Gemini S#2.)

## RESOLVED (by grounding -- removed from "open")

- `import os` + `import math` are ALREADY module-level in `scene_sequencer.py` (lines 28-29) -> no
  import-time risk. (GPT#7, Grok S#2 -> moot.)
- `_run_baseline` runs against a server (not in-process); `FIXED_SEEDS={}` so the env seeds ARE the
  determinism mechanism. (DeepSeek#3 -> resolved.)

## REJECTED / DOWNGRADED

- Rename `baseline_v1.5` fixture (GPT S#5) -> cosmetic; document it as historical, don't rename.
- Cut multi-episode listen from the procedure (Grok CUT#1) -> KEEP as operator RELEASE acceptance, just
  not a build/wiring gate (consistent with R1/R2).
